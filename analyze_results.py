"""Analyse step-3 responses: parse JSON, merge synonyms via LLM, visualise."""

from __future__ import annotations

import json
import os
import random
import re
import time
import warnings
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path

from llm_api import ChatMessage, LLMBackend, OpenAICompatibleBackend

# LLM 同义归并：只对原始频次最高的前 N 个**不同**名称调用一次聚类；其余名称保持自身（identity）。
MERGE_LLM_TOP_N = 200

CLUSTER_PROMPT_TEMPLATE = """你是一个学术概念归类专家。以下是从上一轮聚类中提取出的前 {count} 个高频学术概念名称。由于大模型的局部视野限制，这些名称中依然存在实质上同构的别名或高度重叠的子概念。

任务：将这些概念进一步归并为标准的“介观层级”学术概念（即具有明确数学结构或因果机制的基础数理模型）。

【输入概念列表】（共{count}个）：
{name_list}

【归并法则（Few-Shot 示例）】：
1. **介观层级**：比具体的操作技巧更抽象，但比整个学科更具体。如：属于 game theory（太宽泛）下的 signaling game（正确层级）
2. **同义词与子类向上合并**：将本质相同但表述不同的理论合并为主流术语。
   - 示例：`negative feedback control system`, `feedback control stabilization`, `closed loop control system` -> 统一归并为 `feedback control`
3. **强制剥离修饰语**：无情地删除所有具体的应用场景、约束条件或介词短语。
   - 砍掉 `with...`, `under...`, `via...`, `in...` 等后缀。
   - 示例：`constrained optimization with adaptive parameter search` -> 提取为 `Constrained Optimization`。
   - 示例：`bayesian belief updating under missing info` -> 提取为 `Bayesian Updating`。` 
4. **不要将两个概念进行拼接**：如：`constrained optimization` 和 `adaptive parameter search` 不能合并为 `constrained optimization with adaptive parameter search`。
5. **层级校验**：如果合并后的名字变成了 `psychology` 或 `optimization`，说明合得太粗了；如果是2-4个单词，说明层级合适。

请以JSON格式输出，不要输出任何JSON之外的内容：
```json
{{
  "clusters": [
    {{
      "canonical": "学术界公认、可检索的标准英文名称，绝不要包含括号、连字符、破折号",
      "members": ["原始名称1", "原始名称2", ...]
    }}
  ]
}}
```

要求：
- 每个输入名称必须且只能出现在一个cluster的members中
- canonical使用简洁的英文学术标准名称
- 如果某个名称本身就是标准名称且独立，它自己构成一个cluster
- 服务端会对 canonical / members 做规范化（全小写、多空格合一、连字符统一），你仍应尽量输出规范拼写。"""


CLUSTER_NAMING_PROMPT_TEMPLATE = """你是一个严谨的学术概念本体（Ontology）构建专家。以下是一组被向量相似度聚类算法归入同一簇的数理模型/理论名称，它们在语义空间中高度相关。
请为这个簇提炼出一个最纯粹、最核心的**标准学术名词（Canonical Name）**。

【簇成员】（共{count}个）：
{member_list}

【核心命名法则】：
1. **维基百科准则**：你输出的名字必须像是一个英文维基百科的独立词条名（Wikipedia Article Title），或者一本经典教科书目录中的小节名称。
2. **强制剥离修饰语**：无情地删除所有具体的应用场景、约束条件或介词短语。
   - 砍掉 `with...`, `under...`, `via...`, `in...` 等后缀。
   - 不要使用将两个概念进行拼接
   - 示例：`constrained optimization with adaptive parameter search` -> 提取为 `Constrained Optimization`。
   - 示例：`bayesian belief updating under missing info` -> 提取为 `Bayesian Updating`。
3. **字数高压线**：Canonical name 绝大多数情况下应该是 2-3 个单词，绝对不能超过 4 个单词。
4. **介观层级**：比具体的操作技巧更抽象，但比整个学科更具体。如：属于 game theory（太宽泛）下的 signaling game（正确层级）。

请以JSON格式输出，不要输出任何JSON之外的内容：
```json
{{
  "canonical": "学术界公认的标准英文名称，全小写，绝不要包含括号、连字符、破折号",
  "reason": "一句话说明为何选择该名称"
}}
```"""


# ─── JSON parsing ────────────────────────────────────────────────────


def _extract_json_block(text: str) -> str | None:
    """Extract JSON from LLM output, handling markdown code fences."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0).strip()
    return None


def extract_model_names_from_json(text: str) -> list[str]:
    """Parse the step3 JSON response and return a list of model/law `name` values."""
    json_str = _extract_json_block(text)
    if not json_str:
        return []
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return []

    names: list[str] = []
    models = data.get("models", [])
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict):
                name = item.get("name", "").strip()
                if name:
                    names.append(name)
    return names


_NAME_SOURCE_TO_FIELD = {
    "step1": "step1_response",
    "step2": "step2_response",
    "step3": "step3_response",
    "step4": "step4_response",
}


def extract_names_from_response(text: str, *, name_source: str = "step3") -> list[str]:
    """Extract labels used for counting and merge clustering.

    - ``step3`` / ``step4``: parse JSON ``models[].name`` (structured output; ``step4`` only for legacy rows that still have ``step4_response``).
    - ``step1`` / ``step2``: one label per record — the full response text (no JSON
      parsing, no normalization); embedding clusters these passages directly.
    """
    if name_source in ("step1", "step2"):
        t = text.strip()
        return [t] if t else []
    return extract_model_names_from_json(text)


# ─── LLM-based synonym merging ───────────────────────────────────────

# Unicode hyphen / dash characters → ASCII hyphen-minus (U+002D).
_HYPHEN_LIKE = re.compile(
    r"[\u002D\u2010\u2011\u2012\u2013\u2014\u2212\uFE58\uFE63\uFF0D]"
)


def normalize_llm_merge_label(name: str) -> str:
    """Normalize labels from LLM merge output (and canonical keys for counting).

    - Lowercase ASCII letters.
    - Strip ends; collapse internal whitespace to a single space.
    - Unify Unicode dashes to ``-``; collapse ``-`` runs to one; remove spaces around hyphens
      (e.g. ``foo  -  bar`` → ``foo-bar``).
    Idempotent for typical academic English names.
    """
    s = (name or "").strip()
    if not s:
        return ""
    s = _HYPHEN_LIKE.sub("-", s)
    s = s.lower()
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\s*-\s*", "-", s)
        s = re.sub(r"-{2,}", "-", s)
        s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_llm_merge_map_values(merge_map: dict[str, str]) -> dict[str, str]:
    """Apply :func:`normalize_llm_merge_label` to every canonical value; empty → keep raw key."""
    out: dict[str, str] = {}
    for raw, can in merge_map.items():
        c = normalize_llm_merge_label(str(can))
        out[raw] = c if c else raw
    return out


def _build_cluster_prompt(names: list[str], counts: dict[str, int] | None = None) -> str:
    lines: list[str] = []
    for i, n in enumerate(names, 1):
        if counts is not None and n in counts:
            lines.append(f"{i}. {n}\t（出现 {counts[n]} 次）")
        else:
            lines.append(f"{i}. {n}")
    numbered = "\n".join(lines)
    return CLUSTER_PROMPT_TEMPLATE.format(name_list=numbered, count=len(names))


def _merge_cluster_backend_from_env(
    *,
    model_override: str | None = None,
    base_url_override: str | None = None,
    api_key_override: str | None = None,
) -> LLMBackend:
    """OpenAI-compatible client for merge-only (与探测阶段解耦).

    优先使用归并专用：``OPENAI_API_MERGE_KEY``、``OPENAI_BASE_URL_MERGE``、``LLM_MODEL_MERGE``；
    未设置时回退 ``OPENAI_API_KEY``、``OPENAI_BASE_URL``、``LLM_MODEL``。
    同义聚类请求的 ``temperature`` 固定为 ``0.2``（与 ``MERGE_LLM_MAX_TOKENS`` 等无关）。

    可选 ``*_override``：非空时覆盖对应字段（用于 ``embedding_llm_llm`` 的 Stage 2 等）。
    """
    from llm_api import _load_dotenv_from_project

    _load_dotenv_from_project()
    if api_key_override is not None and str(api_key_override).strip():
        key = str(api_key_override).strip()
    else:
        key = (os.getenv("OPENAI_API_MERGE_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "merge_method=llm 需要配置 OPENAI_API_MERGE_KEY（或回退 OPENAI_API_KEY），"
            "可在项目根目录 .env 中设置。"
        )
    if base_url_override is not None:
        bo = str(base_url_override).strip()
        base = bo if bo else None
    else:
        base = (
            (os.getenv("OPENAI_BASE_URL_MERGE") or os.getenv("OPENAI_BASE_URL") or "")
            .strip()
            or None
        )
    if model_override is not None and str(model_override).strip():
        model = str(model_override).strip()
    else:
        model = (os.getenv("LLM_MODEL_MERGE") or os.getenv("LLM_MODEL") or "gpt-4o").strip() or "gpt-4o"
    max_tok = int(os.getenv("MERGE_LLM_MAX_TOKENS", "8192"))
    return OpenAICompatibleBackend(
        api_key=key,
        base_url=base,
        model=model,
        temperature=0.2,
        max_tokens=max_tok,
    )


def _merge_stage2_backend_from_env_if_configured() -> LLMBackend | None:
    """Optional dedicated client for ``embedding_llm_llm`` Stage 2 (second LLM merge).

    If **none** of the Stage-2-specific env vars below are set, returns ``None`` so
    :func:`merge_synonyms_via_llm` uses the default merge client
    (:func:`_merge_cluster_backend_from_env` with no overrides).

    Env (all optional; set any one to activate overrides for Stage 2 only):

    - ``LLM_MODEL_MERGE_STAGE2`` — model id (unset → same as ``LLM_MODEL_MERGE`` / ``LLM_MODEL``).
    - ``OPENAI_BASE_URL_MERGE_STAGE2`` — base URL (unset → same as merge base URL).
    - ``OPENAI_API_MERGE_STAGE2_KEY`` — API key (unset → ``OPENAI_API_MERGE_KEY`` / ``OPENAI_API_KEY``).
    """
    from llm_api import _load_dotenv_from_project

    _load_dotenv_from_project()
    s2_model = (os.getenv("LLM_MODEL_MERGE_STAGE2") or "").strip() or None
    s2_base = (os.getenv("OPENAI_BASE_URL_MERGE_STAGE2") or "").strip() or None
    s2_key = (os.getenv("OPENAI_API_MERGE_STAGE2_KEY") or "").strip() or None
    if not s2_model and not s2_base and not s2_key:
        return None
    return _merge_cluster_backend_from_env(
        model_override=s2_model,
        base_url_override=s2_base,
        api_key_override=s2_key,
    )


# Merge-phase LLM ``chat`` calls: exponential backoff on transient errors.
_MERGE_LLM_CHAT_MAX_RETRIES = max(1, int(os.getenv("MERGE_LLM_CHAT_MAX_RETRIES", "6")))
_MERGE_LLM_CHAT_BASE_DELAY_SEC = max(0.5, float(os.getenv("MERGE_LLM_CHAT_BASE_DELAY_SEC", "2.0")))
_MERGE_LLM_CHAT_MAX_DELAY_SEC = max(1.0, float(os.getenv("MERGE_LLM_CHAT_MAX_DELAY_SEC", "120.0")))


def _merge_llm_chat_transient_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    needles = (
        "429",
        "rate limit",
        "ratelimit",
        "too many requests",
        "connection",
        "timeout",
        "timed out",
        "temporarily",
        "overloaded",
        "503",
        "502",
        "500",
        "unavailable",
        "econnreset",
        "eof occurred",
        "broken pipe",
        "stream error",
    )
    return any(n in text for n in needles)


def _merge_llm_chat_with_exponential_backoff(
    backend: LLMBackend,
    messages: list[ChatMessage],
) -> str:
    """Call ``backend.chat`` with exponential backoff + jitter; re-raises last error if exhausted."""
    last: BaseException | None = None
    for attempt in range(_MERGE_LLM_CHAT_MAX_RETRIES):
        try:
            return backend.chat(messages)
        except Exception as e:
            last = e
            if not _merge_llm_chat_transient_error(e):
                raise
            delay = min(
                _MERGE_LLM_CHAT_BASE_DELAY_SEC * (2**attempt) + random.uniform(0, 1.5),
                _MERGE_LLM_CHAT_MAX_DELAY_SEC,
            )
            time.sleep(delay)
    assert last is not None
    raise last


def _parse_cluster_response(text: str, expected_names: list[str]) -> dict[str, str]:
    """Parse the LLM cluster response into a {raw_name: canonical_name} dict."""
    json_str = _extract_json_block(text)
    if not json_str:
        print("[warn] LLM cluster response contained no valid JSON, falling back to raw names.")
        return {n: normalize_llm_merge_label(n) or n for n in expected_names}

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        print("[warn] Failed to parse LLM cluster JSON, falling back to raw names.")
        return {n: normalize_llm_merge_label(n) or n for n in expected_names}

    clusters = data.get("clusters", [])
    if not isinstance(clusters, list):
        print("[warn] clusters is not a list, falling back to raw names.")
        return {n: normalize_llm_merge_label(n) or n for n in expected_names}

    by_norm: dict[str, list[str]] = defaultdict(list)
    for n in expected_names:
        key = normalize_llm_merge_label(n)
        if key:
            by_norm[key].append(n)

    result: dict[str, str] = {}
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        canonical_raw = cluster.get("canonical", "")
        members = cluster.get("members", [])
        if not isinstance(canonical_raw, str) or not canonical_raw.strip():
            continue
        if not isinstance(members, list):
            continue
        canonical = normalize_llm_merge_label(canonical_raw.strip())
        if not canonical:
            continue
        for member in members:
            if not isinstance(member, str) or not member.strip():
                continue
            key = normalize_llm_merge_label(member.strip())
            if not key:
                continue
            candidates = by_norm.get(key, [])
            if len(candidates) == 1:
                orig = candidates[0]
            elif len(candidates) > 1:
                ms = member.strip()
                orig = ms if ms in candidates else candidates[0]
            else:
                continue
            result[orig] = canonical

    for name in expected_names:
        if name not in result:
            result[name] = normalize_llm_merge_label(name) or name

    return result


def _merge_verbose_print() -> bool:
    """Full per-cluster console listing when env ``COGNITIVE_MERGE_VERBOSE`` is truthy."""
    return (os.getenv("COGNITIVE_MERGE_VERBOSE") or "").strip().lower() in ("1", "true", "yes")


def _batch_console_quiet() -> bool:
    """Set ``COGNITIVE_BATCH_QUIET=1`` (e.g. by ``batch_llm_merge``) to skip merge / chart console noise."""
    return (os.getenv("COGNITIVE_BATCH_QUIET") or "").strip().lower() in ("1", "true", "yes")


def _embed_quiet_print(*args, **kwargs) -> None:
    """Embedding / plot / [merge] progress lines; suppressed when ``COGNITIVE_BATCH_QUIET=1``."""
    if _batch_console_quiet():
        return
    print(*args, **kwargs)


@contextmanager
def _embed_plot_suppress_batch_noise():
    """Fewer matplotlib / UMAP warnings under parallel batch jobs."""
    if not _batch_console_quiet():
        yield
        return
    old_max = None
    try:
        import matplotlib.pyplot as plt

        old_max = plt.rcParams.get("figure.max_open_warning")
        plt.rcParams["figure.max_open_warning"] = 0
    except Exception:
        pass
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="More than", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Glyph", category=UserWarning)
        warnings.filterwarnings("ignore", message="n_jobs value", category=UserWarning)
        try:
            yield
        finally:
            try:
                import matplotlib.pyplot as plt

                if old_max is not None:
                    plt.rcParams["figure.max_open_warning"] = old_max
            except Exception:
                pass


def _print_merge_cluster_details(
    merge_map: dict[str, str],
    *,
    source: str,
    verbose: bool | None = None,
    summary_suffix: str = "",
) -> None:
    """Print merge outcome: one-line summary by default; full cluster list if ``verbose`` or env."""
    if not merge_map:
        return
    if _batch_console_quiet():
        return
    if verbose is None:
        verbose = _merge_verbose_print()

    buckets: dict[str, list[str]] = defaultdict(list)
    for raw, canonical in merge_map.items():
        buckets[canonical].append(raw)
    for canonical in buckets:
        buckets[canonical].sort(key=str.lower)
    n_raw = len(merge_map)
    n_canon = len(buckets)
    n_merged = sum(1 for ms in buckets.values() if len(ms) >= 2)

    if not verbose:
        msg = (
            f"[merge] {source}：{n_raw} 条原始标签 → {n_canon} 个 canonical"
            f"（{n_merged} 簇含 ≥2 别名）"
        )
        if summary_suffix:
            msg += f" · {summary_suffix}"
        print(msg)
        return

    items = sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0].lower()))
    print("\n" + "=" * 70)
    print(f"  聚类成员明细（{source}）")
    print("=" * 70)
    for i, (canonical, members) in enumerate(items, 1):
        if len(members) == 1:
            print(f"  [{i}] （单条）{members[0]}")
        else:
            print(f"  [{i}] 代表名: {canonical}  ·  {len(members)} 条")
            for m in members:
                tag = "  [代表]" if m == canonical else ""
                print(f"      · {m}{tag}")
    print("=" * 70 + "\n")


def merge_synonyms_via_llm(
    raw_counter: Counter,
    cache_path: Path | None = None,
    refresh: bool = False,
    *,
    top_n: int = MERGE_LLM_TOP_N,
    merge_backend: LLMBackend | None = None,
) -> dict[str, str]:
    """Use an LLM to cluster synonym names into canonical forms.

    Only the **top ``top_n`` distinct names by total occurrence count** are sent to the
    model in **one** clustering call (see ``MERGE_LLM_TOP_N``). All other raw names map
    to themselves (no LLM merge).

    The merge client is :class:`OpenAICompatibleBackend` using ``OPENAI_API_MERGE_KEY``,
    ``OPENAI_BASE_URL_MERGE``, and ``LLM_MODEL_MERGE`` in the environment / project ``.env``
    (with fallback to ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` / ``LLM_MODEL``), unless
    ``merge_backend`` is passed; independent of the probe-phase backend.

    Args:
        raw_counter: Per-raw-label occurrence counts from ``collect_raw_names``.
        cache_path: If given, save/load the full merge map here.
        refresh: If True, ignore cache and re-run the top-``top_n`` LLM clustering.
        merge_backend: Optional pre-built merge client (e.g. Stage 2 with a different model).

    Returns:
        A dict mapping every raw name to its canonical form.
    """
    unique_names = list(raw_counter.keys())
    counts_dict = dict(raw_counter)

    if cache_path and cache_path.exists() and not refresh:
        with open(cache_path, "r", encoding="utf-8") as f:
            cached: dict = json.load(f)
        merge_map = _normalize_llm_merge_map_values(dict(cached.get("merge_map", {})))
        missing = [n for n in unique_names if n not in merge_map]
        if not missing:
            final = {n: merge_map.get(n, n) for n in unique_names}
            _print_merge_cluster_details(
                final,
                source="LLM 同义归并（缓存）",
                summary_suffix=cache_path.name if cache_path else "",
            )
            return final
        for n in missing:
            merge_map[n] = normalize_llm_merge_label(n) or n
        merge_map = _normalize_llm_merge_map_values(merge_map)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"merge_map": merge_map}, f, ensure_ascii=False, indent=2)
        final = {n: merge_map.get(n, n) for n in unique_names}
        _print_merge_cluster_details(
            final,
            source="LLM 同义归并（缓存补全）",
            summary_suffix=(
                f"+{len(missing)} 新名恒等 · {cache_path.name} · --refresh-merge 可重算 Top-{top_n}"
            ),
        )
        return final

    merge_map = {}
    if not unique_names:
        return {}

    ranked = sorted(unique_names, key=lambda n: (-raw_counter[n], n.lower()))
    top_names = ranked[: min(top_n, len(ranked))]

    merge_backend = merge_backend or _merge_cluster_backend_from_env()
    if _merge_verbose_print() and not _batch_console_quiet():
        print(
            f"[merge] LLM 同义归并：模型 {getattr(merge_backend, 'model', '?')!r}；"
            f"Top-{len(top_names)} 名称，单次 API…"
        )

    prompt = _build_cluster_prompt(top_names, counts=counts_dict)
    messages = [ChatMessage(role="user", content=prompt)]
    reply = _merge_llm_chat_with_exponential_backoff(merge_backend, messages)
    batch_map = _parse_cluster_response(reply, top_names)
    merge_map.update(batch_map)

    n_clusters = len(set(batch_map.values()))
    if _merge_verbose_print() and not _batch_console_quiet():
        print(
            f"[merge] Top-{len(top_names)} 名称 → {n_clusters} 个 canonical 簇（其余标签恒等映射）"
        )

    for n in unique_names:
        if n not in merge_map:
            merge_map[n] = normalize_llm_merge_label(n) or n

    merge_map = _normalize_llm_merge_map_values(merge_map)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"merge_map": merge_map}, f, ensure_ascii=False, indent=2)
        if _merge_verbose_print() and not _batch_console_quiet():
            print(f"[merge] saved: {cache_path}")

    final = {n: merge_map.get(n, n) for n in unique_names}
    suf_parts = [
        f"Top-{len(top_names)}→{n_clusters} 簇",
        str(getattr(merge_backend, "model", "?")),
    ]
    if cache_path:
        suf_parts.append(cache_path.name)
    _print_merge_cluster_details(
        final,
        source="LLM 同义归并",
        summary_suffix=" · ".join(suf_parts),
    )
    return final


# ─── Embedding-based clustering (方案 B) ─────────────────────────────


def _resolve_embedding_device(preference: str | None) -> str:
    """Pick torch device for local SentenceTransformer: auto → cuda if available else cpu."""
    pref = (preference or os.getenv("EMBEDDING_DEVICE") or "auto").strip()
    if pref.lower() == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return pref


def merge_synonyms_via_embedding(
    unique_names: list[str],
    embedding_backend: str = "local",
    model: str = "google/gemma-embedding-ggml",
    distance_threshold: float = 0.3,
    cache_path: Path | None = None,
    refresh: bool = False,
    api_key: str | None = None,
    base_url: str | None = None,
    embedding_device: str | None = None,
    name_source: str | None = None,
) -> dict[str, str]:
    """Use embedding vectors + hierarchical clustering to merge synonyms.

    Args:
        unique_names: Deduplicated list of raw model/law names.
        embedding_backend: "local" (HuggingFace), "ollama", or "openai".
        model: Embedding model name/path.
            - local: HuggingFace model name, e.g. "BAAI/bge-small-zh-v1.5"
            - ollama: model name, e.g. "nomic-embed-text"
            - openai: e.g. "text-embedding-3-small"
        distance_threshold: Cosine distance threshold for clustering (0-1).
            Smaller = stricter (fewer merges), larger = looser (more merges).
            Recommended range: 0.2 - 0.4
        cache_path: Path to cache embeddings and merge map.
        refresh: If True, ignore cache and re-compute.
        api_key: API key (for openai backend).
        base_url: Base URL (for openai/ollama backend).
        embedding_device: For local backend only: "auto", "cuda", "cpu", "cuda:0", …
            Default / env EMBEDDING_DEVICE: "auto" uses GPU when torch sees CUDA.

    Returns:
        A dict mapping every raw name to its canonical form (cluster representative).
    """
    try:
        import numpy as np
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist
    except ImportError as e:
        print(f"[embed] Missing dependencies: {e}")
        print("[embed] Run: pip install numpy scipy")
        return {n: n for n in unique_names}

    if not unique_names:
        return {}

    resolved_device = (
        _resolve_embedding_device(embedding_device)
        if embedding_backend == "local"
        else None
    )

    embed_cache_path = cache_path.with_suffix(".embeddings.json") if cache_path else None
    
    if embed_cache_path and embed_cache_path.exists() and not refresh:
        _embed_quiet_print(f"[embed] Loading cached embeddings from {embed_cache_path}")
        with open(embed_cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_names = cached.get("names", [])
        cached_embeddings = cached.get("embeddings", [])
        
        if set(cached_names) == set(unique_names):
            name_to_idx = {n: i for i, n in enumerate(cached_names)}
            embeddings = np.array([cached_embeddings[name_to_idx[n]] for n in unique_names])
        else:
            _embed_quiet_print("[embed] Cache mismatch, will re-compute embeddings.")
            embeddings = _get_embeddings(
                unique_names,
                embedding_backend,
                model,
                api_key,
                base_url,
                embedding_device=resolved_device,
            )
            _save_embeddings_cache(embed_cache_path, unique_names, embeddings.tolist())
    else:
        embeddings = _get_embeddings(
            unique_names,
            embedding_backend,
            model,
            api_key,
            base_url,
            embedding_device=resolved_device,
        )
        if embed_cache_path:
            _save_embeddings_cache(embed_cache_path, unique_names, embeddings.tolist())

    _embed_quiet_print(
        f"[embed] Clustering {len(unique_names)} names with distance_threshold={distance_threshold}"
    )
    
    if len(unique_names) == 1:
        single = {unique_names[0]: unique_names[0]}
        _print_merge_cluster_details(single, source="向量聚类（embedding）")
        return single

    distances = pdist(embeddings, metric="cosine")
    linkage_matrix = linkage(distances, method="average")
    cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")

    clusters: dict[int, list[str]] = {}
    for name, label in zip(unique_names, cluster_labels):
        clusters.setdefault(label, []).append(name)

    merge_map: dict[str, str] = {}
    for members in clusters.values():
        canonical = _select_canonical_name(members)
        for name in members:
            merge_map[name] = canonical

    n_clusters = len(clusters)
    _embed_quiet_print(f"[embed] {len(unique_names)} names → {n_clusters} clusters")
    _print_merge_cluster_details(merge_map, source="向量聚类（embedding）")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            payload: dict = {
                "merge_map": merge_map,
                "method": "embedding",
                "backend": embedding_backend,
                "model": model,
                "distance_threshold": distance_threshold,
            }
            if resolved_device is not None:
                payload["embedding_device"] = resolved_device
            if name_source is not None:
                payload["name_source"] = name_source
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _embed_quiet_print(f"[embed] Saved merge map to {cache_path}")

        _embed_quiet_print(
            f"[plots] Generating dendrogram + UMAP scatter for {len(unique_names)} names …"
        )
        dendrogram_path = cache_path.parent / "cluster_dendrogram.png"
        _plot_dendrogram(
            linkage_matrix, unique_names, distance_threshold, dendrogram_path
        )

        scatter_path = cache_path.parent / "cluster_scatter.png"
        _plot_cluster_scatter(
            embeddings, unique_names, cluster_labels, merge_map, scatter_path
        )
        _embed_quiet_print(f"[plots] Done → {cache_path.parent.name}/")

    return merge_map


# ─── Hybrid: Embedding clustering + LLM naming ──────────────────────


def _parse_cluster_naming_response(text: str) -> tuple[str, str]:
    """Parse the JSON response from cluster naming LLM call.

    Returns ``(canonical, reason)``.  Falls back to empty strings on failure.
    """
    json_str = _extract_json_block(text)
    if not json_str:
        return "", ""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return "", ""
    if not isinstance(data, dict):
        return "", ""
    canonical = str(data.get("canonical", "")).strip()
    reason = str(data.get("reason", "")).strip()
    return canonical, reason


def _name_one_cluster_via_llm(
    members: list[str],
    cluster_id: int,
    backend: LLMBackend,
) -> tuple[int, str, str]:
    """Call LLM to produce a canonical name for *one* cluster.

    Returns ``(cluster_id, canonical_name, reason)``.
    Single-member clusters skip the LLM call and return the member itself.
    """
    if len(members) == 1:
        return cluster_id, members[0], ""

    member_list = "\n".join(f"{i}. {m}" for i, m in enumerate(members, 1))
    prompt = CLUSTER_NAMING_PROMPT_TEMPLATE.format(
        count=len(members), member_list=member_list,
    )
    messages = [ChatMessage(role="user", content=prompt)]
    try:
        reply = _merge_llm_chat_with_exponential_backoff(backend, messages)
    except Exception as e:
        canonical = _select_canonical_name(members)
        return (
            cluster_id,
            canonical,
            f"LLM failed after {_MERGE_LLM_CHAT_MAX_RETRIES} retries: {e}",
        )
    canonical, reason = _parse_cluster_naming_response(reply)
    if not canonical:
        canonical = _select_canonical_name(members)
        reason = "JSON parse fallback"
    return cluster_id, canonical, reason


def merge_synonyms_via_embedding_and_llm(
    unique_names: list[str],
    embedding_backend: str = "local",
    model: str = "BAAI/bge-small-zh-v1.5",
    distance_threshold: float = 0.3,
    cache_path: Path | None = None,
    refresh: bool = False,
    api_key: str | None = None,
    base_url: str | None = None,
    embedding_device: str | None = None,
    name_source: str | None = None,
    max_naming_workers: int = 8,
) -> dict[str, str]:
    """Hybrid merge: embedding-based clustering **+** LLM naming per cluster.

    Pipeline:
        1. Embed all attractor names via the configured embedding API.
        2. Hierarchical clustering (cosine distance, average linkage).
        3. For each cluster with ≥ 2 members, call the LLM (concurrently)
           to produce a single canonical academic name.

    Returns a ``{raw_name: canonical}`` merge map.
    """
    try:
        import numpy as np
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist
    except ImportError as e:
        print(f"[embed+llm] Missing dependencies: {e}")
        print("[embed+llm] Run: pip install numpy scipy")
        return {n: n for n in unique_names}

    if not unique_names:
        return {}

    # ── cache hit ────────────────────────────────────────────────────
    if cache_path and cache_path.exists() and not refresh:
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        merge_map = _normalize_llm_merge_map_values(dict(cached.get("merge_map", {})))
        missing = [n for n in unique_names if n not in merge_map]
        if not missing:
            final = {n: merge_map.get(n, n) for n in unique_names}
            _print_merge_cluster_details(
                final,
                source="embedding+LLM 归并（缓存）",
                summary_suffix=cache_path.name if cache_path else "",
            )
            return final
        for n in missing:
            merge_map[n] = normalize_llm_merge_label(n) or n
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"merge_map": merge_map, "method": "embedding_llm"}, f,
                       ensure_ascii=False, indent=2)
        final = {n: merge_map.get(n, n) for n in unique_names}
        _print_merge_cluster_details(
            final,
            source="embedding+LLM 归并（缓存补全）",
            summary_suffix=f"+{len(missing)} 新名恒等 · --refresh-merge 可重算",
        )
        return final

    # ── Step 1: Embedding ────────────────────────────────────────────
    resolved_device = (
        _resolve_embedding_device(embedding_device)
        if embedding_backend == "local"
        else None
    )

    embed_cache_path = cache_path.with_suffix(".embeddings.json") if cache_path else None

    if embed_cache_path and embed_cache_path.exists() and not refresh:
        _embed_quiet_print(f"[embed+llm] Loading cached embeddings from {embed_cache_path}")
        with open(embed_cache_path, "r", encoding="utf-8") as f:
            ec = json.load(f)
        cached_names = ec.get("names", [])
        cached_embeddings = ec.get("embeddings", [])
        if set(cached_names) == set(unique_names):
            name_to_idx = {n: i for i, n in enumerate(cached_names)}
            embeddings = np.array([cached_embeddings[name_to_idx[n]] for n in unique_names])
        else:
            _embed_quiet_print("[embed+llm] Cache mismatch, re-computing embeddings.")
            embeddings = _get_embeddings(
                unique_names, embedding_backend, model, api_key, base_url,
                embedding_device=resolved_device,
            )
            _save_embeddings_cache(embed_cache_path, unique_names, embeddings.tolist())
    else:
        embeddings = _get_embeddings(
            unique_names, embedding_backend, model, api_key, base_url,
            embedding_device=resolved_device,
        )
        if embed_cache_path:
            _save_embeddings_cache(embed_cache_path, unique_names, embeddings.tolist())

    # ── Step 2: Hierarchical clustering ──────────────────────────────
    _embed_quiet_print(
        f"[embed+llm] Clustering {len(unique_names)} names "
        f"(distance_threshold={distance_threshold})"
    )

    if len(unique_names) == 1:
        single = {unique_names[0]: unique_names[0]}
        _print_merge_cluster_details(single, source="embedding+LLM 归并")
        return single

    distances = pdist(embeddings, metric="cosine")
    linkage_matrix = linkage(distances, method="average")
    cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")

    clusters: dict[int, list[str]] = {}
    for name, label in zip(unique_names, cluster_labels):
        clusters.setdefault(label, []).append(name)

    n_clusters = len(clusters)
    n_multi = sum(1 for ms in clusters.values() if len(ms) >= 2)
    _embed_quiet_print(
        f"[embed+llm] {len(unique_names)} names → {n_clusters} clusters "
        f"({n_multi} with ≥2 members)"
    )

    # ── Step 3: LLM naming (concurrent) ─────────────────────────────
    merge_backend = _merge_cluster_backend_from_env()
    _embed_quiet_print(
        f"[embed+llm] LLM cluster naming: model={getattr(merge_backend, 'model', '?')!r}, "
        f"workers={max_naming_workers}, clusters_to_name={n_multi}"
    )

    merge_map: dict[str, str] = {}
    cluster_reasons: dict[int, str] = {}

    single_clusters = [(cid, ms) for cid, ms in clusters.items() if len(ms) == 1]
    multi_clusters = [(cid, ms) for cid, ms in clusters.items() if len(ms) >= 2]

    for _cid, members in single_clusters:
        canonical = normalize_llm_merge_label(members[0]) or members[0]
        merge_map[members[0]] = canonical

    if multi_clusters:
        from tqdm.auto import tqdm

        with ThreadPoolExecutor(max_workers=min(max_naming_workers, len(multi_clusters))) as executor:
            futures = {
                executor.submit(_name_one_cluster_via_llm, ms, cid, merge_backend): (cid, ms)
                for cid, ms in multi_clusters
            }
            pbar = tqdm(
                as_completed(futures),
                total=len(multi_clusters),
                desc="[LLM naming]",
                unit="cluster",
            )
            n_fallback = 0
            for fut in pbar:
                cid, members = futures[fut]
                try:
                    _, canonical_raw, reason = fut.result()
                    canonical = normalize_llm_merge_label(canonical_raw) or canonical_raw
                    for name in members:
                        merge_map[name] = canonical
                    cluster_reasons[cid] = reason
                    if reason and "retries" in reason.lower():
                        n_fallback += 1
                except Exception as e:
                    canonical = normalize_llm_merge_label(
                        _select_canonical_name(members)
                    ) or _select_canonical_name(members)
                    for name in members:
                        merge_map[name] = canonical
                    cluster_reasons[cid] = f"LLM error: {e}"
                    n_fallback += 1
                pbar.set_postfix(fallback=n_fallback, refresh=False)
            pbar.close()
            if n_fallback:
                tqdm.write(
                    f"[LLM naming] {n_fallback}/{len(multi_clusters)} clusters fell back to heuristic naming"
                )

    # ── Persist artefacts ────────────────────────────────────────────
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "merge_map": merge_map,
            "method": "embedding_llm",
            "embedding_backend": embedding_backend,
            "embedding_model": model,
            "distance_threshold": distance_threshold,
        }
        if resolved_device is not None:
            payload["embedding_device"] = resolved_device
        if name_source is not None:
            payload["name_source"] = name_source
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _embed_quiet_print(f"[embed+llm] Saved merge map to {cache_path}")

        _embed_quiet_print(
            f"[plots] Generating dendrogram + UMAP scatter for {len(unique_names)} names …"
        )
        dendrogram_path = cache_path.parent / "cluster_dendrogram.png"
        _plot_dendrogram(linkage_matrix, unique_names, distance_threshold, dendrogram_path)

        scatter_path = cache_path.parent / "cluster_scatter.png"
        _plot_cluster_scatter(embeddings, unique_names, cluster_labels, merge_map, scatter_path)
        _embed_quiet_print(f"[plots] Done → {cache_path.parent.name}/")

        _write_embedding_llm_cluster_log(
            output_dir=cache_path.parent,
            clusters=clusters,
            merge_map=merge_map,
            cluster_reasons=cluster_reasons,
            embedding_backend=embedding_backend,
            embedding_model=model,
            distance_threshold=distance_threshold,
            llm_model=getattr(merge_backend, "model", "?"),
            max_naming_workers=max_naming_workers,
            n_unique=len(unique_names),
        )

    _print_merge_cluster_details(
        merge_map,
        source="embedding+LLM 归并",
        summary_suffix=(
            f"{len(unique_names)}→{n_clusters} 簇 · "
            f"{getattr(merge_backend, 'model', '?')}"
        ),
    )
    return merge_map


def _write_embedding_llm_cluster_log(
    *,
    output_dir: Path,
    clusters: dict[int, list[str]],
    merge_map: dict[str, str],
    cluster_reasons: dict[int, str],
    embedding_backend: str,
    embedding_model: str,
    distance_threshold: float,
    llm_model: str,
    max_naming_workers: int,
    n_unique: int,
    log_filename: str = "embedding_llm_cluster.log",
) -> Path:
    """Write a detailed per-cluster log to the model/category output folder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_filename

    n_clusters = len(clusters)
    n_multi = sum(1 for ms in clusters.values() if len(ms) >= 2)
    n_single = n_clusters - n_multi

    canonicals: dict[str, list[str]] = defaultdict(list)
    for raw, can in merge_map.items():
        canonicals[can].append(raw)
    for can in canonicals:
        canonicals[can].sort(key=str.lower)

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("Embedding + LLM Cluster Naming — Detailed Cluster Log")
    lines.append("=" * 78)
    lines.append(f"Generated (local): {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    lines.append("[Parameters]")
    lines.append(f"  embedding_backend:   {embedding_backend}")
    lines.append(f"  embedding_model:     {embedding_model}")
    lines.append(f"  distance_threshold:  {distance_threshold}")
    lines.append(f"  llm_naming_model:    {llm_model}")
    lines.append(f"  llm_naming_workers:  {max_naming_workers}")
    lines.append("")

    lines.append("[Summary]")
    lines.append(f"  total_attractor_names:   {n_unique}")
    lines.append(f"  total_clusters:          {n_clusters}")
    lines.append(f"  multi_member_clusters:   {n_multi}  (LLM named)")
    lines.append(f"  singleton_clusters:      {n_single}")
    n_canonical = len(canonicals)
    lines.append(f"  distinct_canonical_names: {n_canonical}")
    lines.append("")

    sorted_clusters = sorted(
        clusters.items(),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )

    lines.append("-" * 78)
    lines.append("Per-cluster detail (sorted by size descending)")
    lines.append("-" * 78)

    for rank, (cid, members) in enumerate(sorted_clusters, 1):
        canonical = merge_map.get(members[0], members[0])
        is_llm = len(members) >= 2
        tag = "LLM named" if is_llm else "singleton"
        reason = cluster_reasons.get(cid, "")
        lines.append("")
        lines.append(
            f"[#{rank}]  canonical: {canonical}  "
            f"({len(members)} member{'s' if len(members) != 1 else ''}, {tag})"
        )
        if reason:
            lines.append(f"    reason: {reason}")
        for i, m in enumerate(members, 1):
            marker = "  ← canonical" if normalize_llm_merge_label(m) == canonical else ""
            lines.append(f"    {i:>3}. {m}{marker}")

    lines.append("")
    lines.append("=" * 78)

    text = "\n".join(lines) + "\n"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(text)
    _embed_quiet_print(f"[embed+llm] Cluster log written to {log_path}")
    return log_path


def _excerpt_for_scatter_label(text: str, *, max_chars: int) -> str:
    """Compress whitespace and truncate for plot labels (long step1/step2 text)."""
    if not text:
        return ""
    s = re.sub(r"\s+", " ", text.strip())
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def _collapse_whitespace(text: str) -> str:
    """Collapse all whitespace to single spaces (PIL wordcloud rejects multiline strings)."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


def _counter_with_single_line_keys(c: Counter) -> Counter:
    """Merge counts by whitespace-collapsed key for wordcloud / PIL."""
    out: Counter = Counter()
    for k, v in c.items():
        out[_collapse_whitespace(k)] += v
    return out


def _plot_dendrogram(
    linkage_matrix,
    labels: list[str],
    threshold: float,
    output_path: Path,
) -> None:
    """Plot and save a dendrogram visualization of the hierarchical clustering."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram
    except ImportError:
        _embed_quiet_print("[embed] matplotlib not installed, skipping dendrogram plot")
        return

    with _embed_plot_suppress_batch_noise():
        font_path = _find_chinese_font()
        plt.rcParams["font.sans-serif"] = [
            "SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "sans-serif",
        ]
        plt.rcParams["axes.unicode_minus"] = False

        truncated = len(labels) > 50
        if truncated:
            short_labels = [f"{l[:20]}..." if len(l) > 20 else l for l in labels]
        else:
            short_labels = [f"{l[:30]}..." if len(l) > 30 else l for l in labels]

        fig_height = max(8, len(labels) * 0.3)
        fig, ax = plt.subplots(figsize=(14, fig_height))

        dendrogram(
            linkage_matrix,
            labels=short_labels,
            orientation="right",
            leaf_font_size=9,
            ax=ax,
            color_threshold=threshold,
        )

        ax.axvline(x=threshold, color="r", linestyle="--", linewidth=1.5, label=f"阈值={threshold}")
        ax.legend(loc="upper right")

        title_fp = _chinese_title_fontproperties(font_path)
        ax.set_title(
            f"模型/定律名称聚类树状图 (阈值={threshold})",
            fontsize=14,
            fontweight="bold",
            fontproperties=title_fp,
        )
        ax.set_xlabel("余弦距离", fontsize=12)

        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    _embed_quiet_print(f"[embed] Dendrogram saved to {output_path}")


def _plot_cluster_scatter(
    embeddings,
    labels: list[str],
    cluster_labels,
    merge_map: dict[str, str],
    output_path: Path,
) -> None:
    """Plot a 2D scatter visualization using UMAP dimensionality reduction."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        _embed_quiet_print(f"[embed] Missing matplotlib: {e}")
        return

    try:
        import umap
    except ImportError:
        _embed_quiet_print("[embed] umap-learn not installed, skipping scatter plot")
        _embed_quiet_print("[embed] Run: pip install umap-learn")
        return

    if len(labels) < 5:
        _embed_quiet_print("[embed] Too few points for UMAP scatter plot, skipping")
        return

    _embed_quiet_print("[embed] Running UMAP for 2D visualization...")

    with _embed_plot_suppress_batch_noise():
        font_path = _find_chinese_font()
        plt.rcParams["font.sans-serif"] = [
            "SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "sans-serif",
        ]
        plt.rcParams["axes.unicode_minus"] = False

        n_neighbors = min(15, len(labels) - 1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        coords = reducer.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(14, 10))
        label_fp = _chinese_title_fontproperties(font_path)

        unique_clusters = list(set(cluster_labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        cluster_to_color = {c: colors[i] for i, c in enumerate(unique_clusters)}

        for i, (x, y) in enumerate(coords):
            cluster_id = cluster_labels[i]
            color = cluster_to_color[cluster_id]
            ax.scatter(x, y, c=[color], s=100, alpha=0.7, edgecolors="white", linewidth=0.5)

        point_max = 40
        legend_max = 48
        for i, (x, y) in enumerate(coords):
            label = labels[i]
            short_label = _excerpt_for_scatter_label(label, max_chars=point_max)
            ax.annotate(
                short_label,
                (x, y),
                fontsize=8,
                alpha=0.8,
                xytext=(5, 5),
                textcoords="offset points",
                fontproperties=label_fp,
            )

        canonical_names = list(set(merge_map.values()))
        handles = []
        for canonical in canonical_names[:20]:
            members = [k for k, v in merge_map.items() if v == canonical]
            if members:
                first_member_idx = labels.index(members[0])
                cluster_id = cluster_labels[first_member_idx]
                color = cluster_to_color[cluster_id]
                short_name = _excerpt_for_scatter_label(canonical, max_chars=legend_max)
                handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                  markersize=10, label=f"{short_name} ({len(members)})")
                handles.append(handle)

        if handles:
            leg = ax.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=8,
                title="聚类 (成员数)",
                prop=label_fp,
            )
            leg.get_title().set_fontproperties(label_fp)

        ax.set_title(
            "模型/定律名称聚类散点图 (UMAP降维)",
            fontsize=14,
            fontweight="bold",
            fontproperties=label_fp,
        )
        ax.set_xlabel("UMAP 维度 1", fontsize=11, fontproperties=label_fp)
        ax.set_ylabel("UMAP 维度 2", fontsize=11, fontproperties=label_fp)

        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    _embed_quiet_print(f"[embed] Scatter plot saved to {output_path}")


def _get_embeddings(
    texts: list[str],
    backend: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    embedding_device: str | None = None,
    *,
    quiet: bool = False,
):
    """Fetch embeddings from various backends."""
    import numpy as np

    if backend == "local":
        return _get_embeddings_local(texts, model, device=embedding_device or "cpu")
    elif backend == "ollama":
        return _get_embeddings_ollama(texts, model, base_url, quiet=quiet)
    elif backend == "openai":
        return _get_embeddings_openai(
            texts, model, api_key, base_url, quiet=quiet
        )
    else:
        raise ValueError(f"Unknown embedding backend: {backend}. Use 'local', 'ollama', or 'openai'.")


def _get_embeddings_local(texts: list[str], model: str, device: str = "cpu"):
    """Get embeddings using local HuggingFace model (e.g., BGE) on CPU or CUDA."""
    import numpy as np
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Local embedding needs `sentence-transformers` (and a working PyTorch stack).\n"
            "  pip install sentence-transformers\n"
            "Install torch + torchvision + torchaudio from https://pytorch.org/get-started/locally/\n"
            f"Underlying import error: {exc}"
        ) from exc

    _embed_quiet_print(f"[embed-local] Loading model: {model}  device={device}")
    _embed_quiet_print("[embed-local] (First run will download the model, may take a while)")

    # Transformers refuses loading pytorch_model.bin via torch.load when torch<2.6 (CVE-2025-32434).
    # Prefer safetensors when present so local embedding works without upgrading torch.
    load_kw: dict = {"trust_remote_code": True, "device": device}
    try:
        import torch

        pv = torch.__version__.split("+")[0].split(".")[:2]
        major, minor = int(pv[0]), int(pv[1])
        if (major, minor) < (2, 6):
            load_kw["model_kwargs"] = {"use_safetensors": True}
            _embed_quiet_print(
                "[embed-local] PyTorch < 2.6: using model_kwargs use_safetensors=True "
                "to load .safetensors instead of pytorch_model.bin."
            )
    except (ValueError, IndexError, ImportError):
        pass

    try:
        encoder = SentenceTransformer(model, **load_kw)
    except TypeError:
        # Older sentence-transformers may not accept model_kwargs
        load_kw.pop("model_kwargs", None)
        encoder = SentenceTransformer(model, trust_remote_code=True, device=device)
    
    _embed_quiet_print(f"[embed-local] Encoding {len(texts)} texts on {encoder.device}...")
    embeddings = encoder.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32,
    )
    
    return np.array(embeddings)


def _get_embeddings_ollama(
    texts: list[str],
    model: str,
    base_url: str | None = None,
    *,
    quiet: bool = False,
):
    """Get embeddings using Ollama API."""
    import numpy as np
    import requests

    from tqdm.auto import tqdm

    url = (base_url or "http://localhost:11434").rstrip("/")
    endpoint = f"{url}/api/embeddings"

    if not quiet:
        pbar = tqdm(
            range(len(texts)),
            desc=f"[embed-ollama] {model[:40]}{'…' if len(model) > 40 else ''}",
            total=len(texts),
            unit="text",
        )
    else:
        pbar = range(len(texts))

    all_embeddings = [None] * len(texts)
    for i in pbar:
        text = texts[i]
        response = requests.post(endpoint, json={"model": model, "prompt": text})
        response.raise_for_status()
        embedding = response.json()["embedding"]
        all_embeddings[i] = embedding
    if not quiet:
        tqdm.write(f"[embed-ollama] done: {len(texts)} texts @ {url}")
    return np.array(all_embeddings)


def _get_embeddings_openai(
    texts: list[str],
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    quiet: bool = False,
):
    """Get embeddings using OpenAI-compatible API."""
    import numpy as np
    import os

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai  to use OpenAI embedding API")

    from tqdm.auto import tqdm

    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=True)
        except ImportError:
            pass

    key = (api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY", "") or "").strip()
    url = (base_url or os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").strip() or None

    client = OpenAI(api_key=key, base_url=url)

    # OpenAI's Python SDK defaults encoding_format to base64; some OpenAI-compatible
    # gateways (e.g. OpenRouter free/cheap models) return bodies that parse to
    # empty `data` with that default, triggering "No embedding data received". Float
    # is widely supported and avoids the problematic base64 parser path.
    _batch_size = 32
    n_batches = (len(texts) + _batch_size - 1) // _batch_size
    if not n_batches:
        return np.array([], dtype=np.float64)

    batch_starts = list(range(0, len(texts), _batch_size))
    if not quiet:
        pbar = tqdm(
            batch_starts,
            desc=f"[embed-openai] {model[:48]}{'…' if len(model) > 48 else ''}",
            total=n_batches,
            unit="batch",
        )
    else:
        pbar = batch_starts

    all_embeddings = []
    for i in pbar:
        batch = texts[i : i + _batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch,
            encoding_format="float",
        )
        if not response.data:
            raise RuntimeError(
                f"Embeddings API returned an empty 'data' list. model={model!r}. "
                "Use a text-only embedding model on the gateway, or a smaller model id; "
                "multimodal (e.g. *-vl-*) IDs often do not return vectors for plain text. "
                "If the gateway is flaky, try again or reduce input length."
            )
        if len(response.data) != len(batch):
            raise RuntimeError(
                f"Embeddings count mismatch: sent {len(batch)} inputs, got {len(response.data)} vectors. "
                f"model={model!r}."
            )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    if not quiet:
        tqdm.write(
            f"[embed-openai] done: {len(texts)} texts, {n_batches} batch request(s)"
            + (f" · {url}" if url else "")
        )

    return np.array(all_embeddings)


def _save_embeddings_cache(path: Path, names: list[str], embeddings: list) -> None:
    """Save embeddings to cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"names": names, "embeddings": embeddings}, f)
    _embed_quiet_print(f"[embed] Cached embeddings to {path}")


def _select_canonical_name(members: list[str]) -> str:
    """Select the best canonical name from a cluster.
    
    Prefer: shorter names, Chinese names, names without parentheses.
    """
    if len(members) == 1:
        return members[0]
    
    def score(name: str) -> tuple:
        has_chinese = any("\u4e00" <= c <= "\u9fff" for c in name)
        has_parens = "(" in name or "（" in name
        length = len(name)
        return (not has_chinese, has_parens, length, name)
    
    return min(members, key=score)


# ─── Analysis entry point ────────────────────────────────────────────


def collect_raw_names(
    results_path: str | Path,
    name_source: str = "step3",
) -> Counter:
    """First pass: extract labels per record and count raw occurrences.

    ``name_source`` selects which JSONL field to read. For ``step1`` / ``step2``, each
    non-empty response is one raw label (full text); for ``step3`` / legacy ``step4``,
    names come from parsed JSON ``models[].name``.
    """
    field = _NAME_SOURCE_TO_FIELD.get(name_source, "step3_response")
    raw_counter: Counter = Counter()
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get(field, "")
            for name in extract_names_from_response(text, name_source=name_source):
                raw_counter[name] += 1
    return raw_counter


def analyse(
    results_path: str | Path,
    backend: LLMBackend | None = None,
    cache_path: Path | None = None,
    refresh_merge: bool = False,
    merge_method: str = "llm",
    embedding_threshold: float = 0.3,
    embedding_backend: str = "local",
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    embedding_device: str | None = None,
    name_source: str = "step3",
    llm_naming_workers: int = 8,
) -> tuple[Counter, Counter]:
    """Parse results, optionally merge synonyms, and return counts.

    Args:
        results_path: Path to probe_results.jsonl.
        backend: Unused for ``merge_method="llm"`` (归并读 ``OPENAI_API_MERGE_KEY`` 等，见
            :func:`_merge_cluster_backend_from_env`)。嵌入聚类也不使用此参数。
        cache_path: Path to cache merge_map.json.
        refresh_merge: If True, re-query even if cache exists.
        merge_method: "llm" for LLM-based clustering, "embedding" for vector clustering,
            "embedding_llm" for embedding clustering + LLM naming per cluster,
            "embedding_llm_llm" for embedding_llm plus a second LLM merge on canonicals,
            "embedding_llm_llm_stage2" to reload ``merge_map_llm_stage1.json`` and run only that
            second LLM merge (no embedding / no per-cluster naming),
            "embedding_llm_llm_stage1_final" to set ``merge_map.json`` = Stage 1 mapping only
            (read ``merge_map_llm_stage1.json``, no Stage-2 LLM call),
            "none" for no merging.
        embedding_threshold: Cosine distance threshold for embedding clustering (0-1).
        embedding_backend: "local" (HuggingFace), "ollama", or "openai".
        embedding_model: Model name for embedding.
        embedding_device: Local backend only: "auto", "cuda", "cpu", "cuda:0", … (None → env EMBEDDING_DEVICE or auto).
        name_source: ``step3`` / legacy ``step4`` parse JSON ``models[].name``. ``step1`` / ``step2``
            use each row's full response as one label (for embedding or ``none`` only; not supported with
            ``merge_method=llm`` or ``embedding_llm``).
        llm_naming_workers: Max concurrent LLM calls when ``merge_method="embedding_llm"``.
        For ``merge_method="embedding_llm_llm"`` or ``"embedding_llm_llm_stage2"``, Stage 2 LLM can be overridden via ``.env`` only:
        ``LLM_MODEL_MERGE_STAGE2``, ``OPENAI_BASE_URL_MERGE_STAGE2``, ``OPENAI_API_MERGE_STAGE2_KEY``
        (see :func:`_merge_stage2_backend_from_env_if_configured`); if unset, Stage 2 uses the same
        merge client as Stage-1 cluster naming / default ``merge_synonyms_via_llm``.

    Returns:
        (canonical_counts, raw_counts)
    """
    if name_source not in _NAME_SOURCE_TO_FIELD:
        raise ValueError(
            f"name_source must be one of {tuple(_NAME_SOURCE_TO_FIELD)}, got {name_source!r}"
        )
    if merge_method in (
        "llm",
        "embedding_llm",
        "embedding_llm_llm",
        "embedding_llm_llm_stage2",
        "embedding_llm_llm_stage1_final",
    ) and name_source in (
        "step1",
        "step2",
    ):
        raise ValueError(
            "merge_method 'llm' / 'embedding_llm' / 'embedding_llm_llm' / "
            "'embedding_llm_llm_stage2' / 'embedding_llm_llm_stage1_final' cannot be used "
            "with name_source 'step1' or 'step2' "
            "(full responses do not fit the LLM cluster prompt). "
            "Use --merge-method embedding or --merge-method none."
        )

    raw_counter = collect_raw_names(results_path, name_source=name_source)

    if merge_method == "none":
        return raw_counter, raw_counter

    unique_names = list(raw_counter.keys())

    if merge_method in ("embedding_llm_llm_stage2", "embedding_llm_llm_stage1_final"):
        if not cache_path:
            raise ValueError(
                f"merge_method {merge_method!r} requires cache_path "
                "(probe_results parent directory with merge_map_llm_stage1.json)."
            )
        stage1_path = cache_path.parent / "merge_map_llm_stage1.json"
        if not stage1_path.is_file():
            raise FileNotFoundError(
                f"merge_method {merge_method!r} requires {stage1_path}. "
                "Run full --merge-method embedding_llm_llm first to generate Stage 1."
            )
        with open(stage1_path, "r", encoding="utf-8") as f:
            stage1_payload: dict = json.load(f)
        merge_map = _normalize_llm_merge_map_values(
            dict(stage1_payload.get("merge_map", {}))
        )
        for n in unique_names:
            if n not in merge_map:
                merge_map[n] = normalize_llm_merge_label(n) or n
        merge_map = _normalize_llm_merge_map_values(merge_map)
        if merge_method == "embedding_llm_llm_stage2":
            _embed_quiet_print(
                f"[merge] Stage2-only: loaded Stage 1 from {stage1_path.name} "
                f"({len(merge_map)} raw → stage-1 canonical)"
            )
        else:
            _embed_quiet_print(
                f"[merge] Stage1-as-final: loaded {stage1_path.name} "
                f"({len(merge_map)} raw labels; no Stage-2 LLM)"
            )
    elif merge_method in ("embedding_llm", "embedding_llm_llm"):
        stage1_cache = (
            cache_path.parent / "merge_map_llm_stage1.json"
            if merge_method == "embedding_llm_llm" and cache_path
            else cache_path
        )
        merge_map = merge_synonyms_via_embedding_and_llm(
            unique_names,
            embedding_backend=embedding_backend,
            model=embedding_model,
            distance_threshold=embedding_threshold,
            cache_path=stage1_cache,
            refresh=refresh_merge,
            embedding_device=embedding_device,
            name_source=name_source,
            max_naming_workers=llm_naming_workers,
        )
    elif merge_method == "embedding":
        merge_map = merge_synonyms_via_embedding(
            unique_names,
            embedding_backend=embedding_backend,
            model=embedding_model,
            distance_threshold=embedding_threshold,
            cache_path=cache_path,
            refresh=refresh_merge,
            embedding_device=embedding_device,
            name_source=name_source,
        )
    else:
        merge_map = merge_synonyms_via_llm(
            raw_counter,
            cache_path=cache_path,
            refresh=refresh_merge,
        )

    # ── Stage 2 for embedding_llm_llm (full) or embedding_llm_llm_stage2 (reload stage1 from disk)
    if merge_method in ("embedding_llm_llm", "embedding_llm_llm_stage2"):
        stage1_counter: Counter = Counter()
        for raw_name, count in raw_counter.items():
            canonical = merge_map.get(raw_name, raw_name)
            c = normalize_llm_merge_label(str(canonical))
            canonical = c if c else (raw_name.strip() or raw_name)
            stage1_counter[canonical] += count

        _embed_quiet_print(
            f"[merge] Stage 2: LLM re-merge on {len(stage1_counter)} canonical names "
            f"from embedding+LLM clustering"
        )

        llm_cache_path = (
            cache_path.parent / "merge_map_llm_stage2.json" if cache_path else None
        )
        stage2_backend = _merge_stage2_backend_from_env_if_configured()
        if stage2_backend is not None:
            _embed_quiet_print(
                f"[merge] Stage 2 LLM (.env): model={getattr(stage2_backend, 'model', '?')!r}"
            )
        llm_merge_map = merge_synonyms_via_llm(
            stage1_counter,
            cache_path=llm_cache_path,
            refresh=refresh_merge,
            merge_backend=stage2_backend,
        )

        final_map: dict[str, str] = {}
        for raw_name in unique_names:
            stage1_canonical = merge_map.get(raw_name, raw_name)
            c = normalize_llm_merge_label(str(stage1_canonical))
            stage1_canonical = c if c else (raw_name.strip() or raw_name)
            final_canonical = llm_merge_map.get(stage1_canonical, stage1_canonical)
            final_map[raw_name] = final_canonical
        merge_map = final_map

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "merge_map": merge_map,
                        "method": merge_method,
                        "stage1_file": "merge_map_llm_stage1.json",
                        "stage2_file": "merge_map_llm_stage2.json",
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            _embed_quiet_print(
                f"[merge] Final combined map → {cache_path.name} "
                f"({len(set(merge_map.values()))} canonical)"
            )

    if merge_method == "embedding_llm_llm_stage1_final" and cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "merge_map": merge_map,
                    "method": merge_method,
                    "stage1_file": "merge_map_llm_stage1.json",
                    "stage2_llm": False,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        _embed_quiet_print(
            f"[merge] Final map (Stage 1 only) → {cache_path.name} "
            f"({len(set(merge_map.values()))} canonical)"
        )

    canonical_counter: Counter = Counter()
    for raw_name, count in raw_counter.items():
        canonical = merge_map.get(raw_name, raw_name)
        if merge_method in (
            "llm",
            "embedding_llm",
            "embedding_llm_llm",
            "embedding_llm_llm_stage2",
            "embedding_llm_llm_stage1_final",
        ):
            c = normalize_llm_merge_label(str(canonical))
            canonical = c if c else (raw_name.strip() or raw_name)
        canonical_counter[canonical] += count

    if cache_path:
        write_merge_stages_analysis_log(
            cache_path.parent,
            raw_counter=raw_counter,
            print_path=not _batch_console_quiet(),
        )

    return canonical_counter, raw_counter


def _read_merge_map_file(path: Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data: dict = json.load(f)
    return _normalize_llm_merge_map_values(dict(data.get("merge_map", {})))


def _stage2_target_for_c1(m2: dict[str, str], c1: str) -> str:
    """Match ``analyse`` composition: normalize c1 then look up in stage-2 map."""
    c = normalize_llm_merge_label(str(c1))
    c1k = c if c else c1.strip() or c1
    if c1k in m2:
        return m2[c1k]
    if c1 in m2:
        return m2[c1]
    for k, v in m2.items():
        if normalize_llm_merge_label(str(k)) == c1k:
            return v
    return c1k


def build_merge_stages_report_text(
    stage1_path: Path,
    stage2_path: Path,
    *,
    final_map_path: Path | None = None,
    raw_counter: Counter | None = None,
) -> str:
    """Build a text report comparing Stage-1 and Stage-2 LLM merge maps (``embedding_llm_llm``).

    Stage1: raw label → first canonical. Stage2: stage-1 canonical label → second canonical.
    """
    m1 = _read_merge_map_file(Path(stage1_path))
    m2 = _read_merge_map_file(Path(stage2_path))
    lines: list[str] = []
    gen = datetime.now().isoformat(timespec="seconds")
    lines.append("=" * 78)
    lines.append("Merge stages analysis — Stage1 (embedding+LLM) vs Stage2 (LLM re-merge)")
    lines.append("=" * 78)
    lines.append(f"Generated: {gen}")
    lines.append("")
    lines.append("[Input files]")
    lines.append(f"  stage1: {Path(stage1_path).resolve()}")
    lines.append(f"  stage2: {Path(stage2_path).resolve()}")
    if final_map_path and Path(final_map_path).is_file():
        lines.append(f"  final:  {Path(final_map_path).resolve()} (consistency check)")
    lines.append("")

    n_raw = len(m1)
    c1_set = set(m1.values())
    n_c1 = len(c1_set)
    n_m2_keys = len(m2)
    c2_set = set(m2.values())
    n_c2 = len(c2_set)

    inv2: dict[str, list[str]] = defaultdict(list)
    for k, v in m2.items():
        inv2[v].append(k)
    merge_groups = [(c2, keys) for c2, keys in inv2.items() if len(keys) > 1]
    merge_groups.sort(key=lambda t: (-len(t[1]), t[0].lower()))
    n_s2_merges = len(merge_groups)
    n_labels_collapsed = sum(len(keys) for _, keys in merge_groups)

    lines.append("[Summary]")
    lines.append(f"  raw labels in stage1 map:     {n_raw}")
    lines.append(f"  distinct stage-1 canonicals:  {n_c1}")
    lines.append(f"  stage2 map entries (keys):    {n_m2_keys}")
    lines.append(f"  distinct stage-2 canonicals:  {n_c2}")
    lines.append(
        f"  stage2 merge groups (|keys|>1 for same c2): {n_s2_merges} groups, "
        f"{n_labels_collapsed} stage-1 labels merged into shared c2"
    )
    lines.append("")

    if raw_counter:
        w_c1: Counter = Counter()
        w_c2: Counter = Counter()
        for raw_name, cnt in raw_counter.items():
            c1 = m1.get(raw_name, raw_name)
            c = normalize_llm_merge_label(str(c1))
            c1k = c if c else (raw_name.strip() or raw_name)
            c2 = _stage2_target_for_c1(m2, c1k)
            w_c1[c1k] += cnt
            w_c2[c2] += cnt
        lines.append("[Weighted by probe counts / 按探测频次加权]")
        lines.append(f"  total raw mentions: {int(sum(raw_counter.values()))}")
        lines.append(f"  distinct c1 (weighted rows): {len(w_c1)}")
        lines.append(f"  distinct c2 (weighted):      {len(w_c2)}")
        lines.append("")

    lines.append(f"[Largest stage-2 merge groups] (top 40, by number of stage-1 keys merged into one c2)")
    for i, (c2, keys) in enumerate(merge_groups[:40], 1):
        lines.append(f"  [{i:2d}] {c2!r}  ←  {len(keys)} stage-1 labels")
        keys_s = sorted(keys, key=str.lower)[:12]
        for k in keys_s:
            lines.append(f"        · {k}")
        if len(keys) > 12:
            lines.append(f"        · … +{len(keys) - 12} more")
    if not merge_groups:
        lines.append("  (none — every stage-2 key is its own group or identity)")
    lines.append("")

    reassign = [(k, m2[k]) for k in sorted(m2, key=str.lower) if m2[k] != k]
    reassign_n = [t for t in reassign if normalize_llm_merge_label(t[0]) != normalize_llm_merge_label(t[1])]
    lines.append(
        f"[Stage-2 renames] (key ≠ value, {len(reassign_n)} entries; list first 80)"
    )
    for k, v in reassign_n[:80]:
        lines.append(f"  {k!r}  →  {v!r}")
    if len(reassign_n) > 80:
        lines.append(f"  … {len(reassign_n) - 80} more")
    lines.append("")

    if final_map_path and Path(final_map_path).is_file():
        try:
            with open(Path(final_map_path), "r", encoding="utf-8") as f:
                _final_payload: dict = json.load(f)
        except (OSError, json.JSONDecodeError):
            _final_payload = {}
        if str(_final_payload.get("method", "")) == "embedding_llm_llm_stage1_final":
            lines.append(
                "[Consistency with merge_map.json]  (skipped: method=embedding_llm_llm_stage1_final, "
                "final equals stage1; on-disk stage2 may be from an earlier run.)"
            )
        else:
            mf = _read_merge_map_file(Path(final_map_path))
            mismatch = 0
            for r, fin in list(mf.items())[: min(5000, len(mf))]:
                c1 = m1.get(r, r)
                c = normalize_llm_merge_label(str(c1))
                c1k = c if c else (r.strip() or r)
                expect = _stage2_target_for_c1(m2, c1k)
                fn = normalize_llm_merge_label(str(fin)) or fin
                ex = normalize_llm_merge_label(str(expect)) or expect
                if fn != ex:
                    mismatch += 1
            lines.append("[Consistency with merge_map.json]")
            lines.append(
                f"  compared {min(5000, len(mf))} raw keys: "
                f"{mismatch} mismatch(es) vs compose(stage1, stage2) (0 = consistent)"
            )
        lines.append("")

    lines.append("=" * 78)
    return "\n".join(lines) + "\n"


def write_merge_stages_analysis_log(
    output_dir: str | Path,
    *,
    raw_counter: Counter | None = None,
    log_filename: str = "merge_stages_analysis.log",
    print_path: bool = True,
) -> Path | None:
    """Write ``merge_stages_analysis.log`` if both stage-1 and stage-2 JSON exist."""
    output_dir = Path(output_dir)
    p1 = output_dir / "merge_map_llm_stage1.json"
    p2 = output_dir / "merge_map_llm_stage2.json"
    if not p1.is_file() or not p2.is_file():
        return None
    final = output_dir / "merge_map.json"
    text = build_merge_stages_report_text(
        p1,
        p2,
        final_map_path=final if final.is_file() else None,
        raw_counter=raw_counter,
    )
    out = output_dir / log_filename
    out.write_text(text, encoding="utf-8")
    if print_path:
        _embed_quiet_print(f"[merge] Wrote {out.name} ({out.parent.name}/)")
    return out


def write_analysis_summary_log(
    *,
    results_path: str | Path,
    output_dir: str | Path,
    canonical_counter: Counter,
    raw_counter: Counter,
    merge_method: str,
    top_n: int,
    refresh_merge: bool,
    embedding_backend: str = "",
    embedding_model: str = "",
    embedding_threshold: float | None = None,
    embedding_device: str = "",
    name_source: str = "",
    log_filename: str = "analysis_summary.log",
    print_path: bool = True,
) -> Path:
    """Write a UTF-8 summary log after analysis (and optionally print its path to stdout).

    Returns the path to the written log file.
    """
    results_path = Path(results_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_filename

    n_records = 0
    if results_path.is_file():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n_records += 1

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("LLM Cognitive Probe — Analysis Summary")
    lines.append("=" * 78)
    lines.append(f"Generated (local): {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("[Input]")
    lines.append(f"  results_file: {results_path}")
    lines.append(f"  records (non-empty JSONL lines): {n_records}")
    lines.append("")
    lines.append("[Merge]")
    lines.append(f"  merge_method: {merge_method}")
    lines.append(f"  refresh_merge: {refresh_merge}")
    if name_source:
        lines.append(f"  name_source: {name_source}")
    if merge_method == "embedding":
        lines.append(f"  embedding_backend: {embedding_backend}")
        lines.append(f"  embedding_model: {embedding_model}")
        if embedding_threshold is not None:
            lines.append(f"  embedding_threshold: {embedding_threshold}")
        lines.append(f"  embedding_device: {embedding_device or '(default)'}")
    lines.append(f"  chart_top_n: {top_n}")
    lines.append("")
    lines.append("[Counts]")
    n_raw_u = len(raw_counter)
    n_raw_t = sum(raw_counter.values())
    n_can_u = len(canonical_counter)
    n_can_t = sum(canonical_counter.values())
    lines.append(f"  raw_unique_names: {n_raw_u}")
    lines.append(f"  raw_total_mentions: {n_raw_t}")
    lines.append(f"  canonical_unique_names: {n_can_u}")
    lines.append(f"  canonical_total_mentions: {n_can_t}")
    if merge_method != "none" and n_raw_u:
        lines.append(
            f"  distinct_label_reduction: {n_raw_u} raw → {n_can_u} canonical "
            f"({100.0 * (1 - n_can_u / n_raw_u):.1f}% fewer labels)"
        )
    lines.append("")

    lines.append("[Output artifacts]")
    merge_map_path = output_dir / "merge_map.json"
    artifacts = [
        merge_map_path,
        output_dir / "merge_map_llm_stage1.json",
        output_dir / "merge_map_llm_stage1.embeddings.json",
        output_dir / "merge_map_llm_stage2.json",
        output_dir / "model_law_frequency.png",
        output_dir / "model_law_wordcloud.png",
        output_dir / "cluster_dendrogram.png",
        output_dir / "cluster_scatter.png",
    ]
    for p in artifacts:
        lines.append(f"  {p.name}: {'yes' if p.exists() else 'no'}")
    lines.append("")

    top_k = min(50, max(1, top_n))

    def append_freq_block(title: str, ctr: Counter) -> None:
        lines.append(title)
        if not ctr:
            lines.append("  (no model/law names extracted)")
        else:
            lines.append(f"  {'rank':>4}  {'count':>6}  name")
            for rank, (name, count) in enumerate(ctr.most_common(top_k), 1):
                safe = name.replace("\n", " ").strip()
                lines.append(f"  {rank:>4}  {count:>6}  {safe}")
        lines.append("")

    if merge_method != "none":
        append_freq_block(
            f"[Top {top_k} raw names — before merge / 归并前（原始标签频次）]",
            raw_counter,
        )
        append_freq_block(
            f"[Top {top_k} canonical names — after merge / 归并后（代表名频次）]",
            canonical_counter,
        )
        if merge_map_path.is_file():
            try:
                with open(merge_map_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                mm = payload.get("merge_map", {})
                if isinstance(mm, dict):
                    buckets: dict[str, list[str]] = defaultdict(list)
                    for raw_n, can_n in mm.items():
                        r = str(raw_n).strip() if raw_n is not None else ""
                        c = str(can_n).strip() if can_n is not None else ""
                        if not r or not c:
                            continue
                        buckets[c].append(r)
                    for c in buckets:
                        buckets[c] = sorted(set(buckets[c]), key=str.lower)

                    actual_clusters = [
                        (c, members) for c, members in buckets.items() if len(members) >= 2
                    ]
                    actual_clusters.sort(key=lambda t: (-len(t[1]), t[0].lower()))

                    lines.append(
                        "[Clusters with actual merge (≥2 raw labels → one canonical / "
                        f"真正发生聚类的项): {len(actual_clusters)} clusters]"
                    )
                    if not actual_clusters:
                        lines.append("  (none — every mapped cluster is a singleton)")
                    else:
                        max_groups = 200
                        for gi, (canon, members) in enumerate(actual_clusters[:max_groups], 1):
                            c_safe = canon.replace("\n", " ").strip()
                            lines.append(
                                f"  [#{gi}] representative: {c_safe}  ·  {len(members)} members"
                            )
                            for m in members:
                                tag = "  [代表]" if m == canon else ""
                                ms = m.replace("\n", " ").strip()
                                lines.append(f"      · {ms}{tag}")
                        if len(actual_clusters) > max_groups:
                            lines.append(
                                f"  ... ({len(actual_clusters) - max_groups} more clusters omitted)"
                            )
                    lines.append("")

                    changed = [(str(r), str(c)) for r, c in mm.items() if str(r).strip() != str(c).strip()]
                    changed.sort(key=lambda t: (t[1].lower(), t[0].lower()))
                    lines.append(
                        f"[Merge map: raw → canonical where label changed "
                        f"({len(changed)} pairs)]"
                    )
                    max_pairs = 500
                    for raw_n, can_n in changed[:max_pairs]:
                        r_safe = raw_n.replace("\n", " ").strip()
                        c_safe = can_n.replace("\n", " ").strip()
                        lines.append(f"  {r_safe}  →  {c_safe}")
                    if len(changed) > max_pairs:
                        lines.append(f"  ... ({len(changed) - max_pairs} more pairs omitted)")
                    lines.append("")

                    n_b = len(buckets)
                    n_merged_b = sum(1 for ms in buckets.values() if len(ms) >= 2)
                    lines.append(
                        "[Merge cluster details / 归并簇完整明细"
                        f"（共 {n_b} 簇，其中 {n_merged_b} 簇含 ≥2 条原始标签；"
                        "按簇大小与代表名排序；含单条；"
                        "与开启 COGNITIVE_MERGE_VERBOSE 时控制台聚类明细一致，默认仅写入本日志）]"
                    )
                    if not buckets:
                        lines.append("  (empty merge_map)")
                    else:
                        items = sorted(
                            buckets.items(),
                            key=lambda kv: (-len(kv[1]), kv[0].lower()),
                        )
                        for i, (canonical, members) in enumerate(items, 1):
                            if len(members) == 1:
                                m0 = members[0].replace("\n", " ").strip()
                                lines.append(f"  [{i}] （单条）{m0}")
                            else:
                                c_safe = canonical.replace("\n", " ").strip()
                                lines.append(
                                    f"  [{i}] 代表名: {c_safe}  ·  {len(members)} 条"
                                )
                                for m in members:
                                    tag = "  [代表]" if m == canonical else ""
                                    ms = m.replace("\n", " ").strip()
                                    lines.append(f"      · {ms}{tag}")
                    lines.append("")
            except (json.JSONDecodeError, OSError) as e:
                lines.append(f"[Merge map file present but could not read: {e}]")
                lines.append("")
    else:
        append_freq_block(
            f"[Top {top_k} names — merge_method=none（无归并，与原始一致）]",
            raw_counter,
        )

    lines.append("=" * 78)

    text = "\n".join(lines) + "\n"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(text)

    if print_path:
        print(f"\n[summary] Analysis summary written to: {log_path}")
    return log_path


# ─── Visualisation ───────────────────────────────────────────────────


def visualise(
    canonical_counter: Counter,
    output_dir: str | Path = ".",
    top_n: int = 30,
    merged: bool = True,
    *,
    console: bool = True,
) -> None:
    """Generate a vertical bar chart, word cloud, and optionally print a table to stdout.
    
    Args:
        canonical_counter: Counter of model/law names (merged or raw).
        output_dir: Directory to save output images.
        top_n: Number of top items to display in chart/table.
        merged: If True, label as "synonym-merged"; if False, label as "raw (no merging)".
        console: If False, still write PNG files but skip stdout table and save-path lines.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not canonical_counter:
        if console:
            print("No model/law mentions found – nothing to visualise.")
        return

    most_common = canonical_counter.most_common(top_n)
    names = [n for n, _ in most_common]
    counts = [c for _, c in most_common]

    # ── Console table ──
    if console:
        label = "synonym-merged" if merged else "raw (no merging)"
        print(f"\n{'='*70}")
        print(f"  Top-{top_n} Model/Law Frequency ({label})")
        print(f"{'='*70}")
        for rank, (name, count) in enumerate(most_common, 1):
            bar = "█" * min(count, 50)
            disp = name if len(name) <= 76 else _excerpt_for_scatter_label(name, max_chars=76)
            print(f"  {rank:>3}. {disp:<76s}  {count:>4d}  {bar}")
        print(f"{'='*70}\n")

    # ── Word Cloud ──
    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        font_path = _find_chinese_font()
        title_fp = _chinese_title_fontproperties(font_path)
        plt.rcParams["font.sans-serif"] = [
            "SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "sans-serif",
        ]
        plt.rcParams["axes.unicode_minus"] = False

        wc = WordCloud(
            font_path=font_path,
            width=1600,
            height=900,
            background_color="white",
            max_words=200,
            max_font_size=150,
            min_font_size=12,
            colormap="viridis",
            prefer_horizontal=0.8,
            scale=2,
        )
        # Long step1/step2 labels contain newlines; Pillow raises on multiline text in WordCloud.
        wc_freq = _counter_with_single_line_keys(canonical_counter)
        wc.generate_from_frequencies(dict(wc_freq))

        fig, ax = plt.subplots(figsize=(16, 9))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        wc_title = "LLM 回复中模型/定律词云" + ("（同义归并）" if merged else "（原始结果）")
        ax.set_title(
            wc_title,
            fontsize=16,
            fontweight="bold",
            pad=10,
            fontproperties=title_fp,
        )

        plt.tight_layout()
        wordcloud_path = output_dir / "model_law_wordcloud.png"
        fig.savefig(wordcloud_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        if console:
            print(f"Word cloud saved to {wordcloud_path}")

    except ImportError:
        if console:
            print("[warn] wordcloud not installed – run: pip install wordcloud")

    # ── Bar Chart ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = [
            "SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "sans-serif",
        ]
        plt.rcParams["axes.unicode_minus"] = False

        fig_w = max(14.0, len(names) * 0.55)
        fig, ax = plt.subplots(figsize=(fig_w, 9))
        x_pos = range(len(names))
        bars = ax.bar(x_pos, counts, color="#4C72B0", edgecolor="white", width=0.7)
        ax.set_xticks(list(x_pos))
        tick_labels = [_excerpt_for_scatter_label(n, max_chars=56) for n in names]
        ax.set_xticklabels(tick_labels, fontsize=9, rotation=45, ha="right")
        ax.set_ylabel("出现次数", fontsize=12)
        title_suffix = "同义归并" if merged else "原始结果"
        ax.set_title(f"LLM 回复中模型/定律出现频次（{title_suffix}）", fontsize=14, fontweight="bold")

        y_pad = (max(counts) if counts else 1) * 0.02
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_pad,
                str(count),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.28)
        chart_path = output_dir / "model_law_frequency.png"
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        if console:
            print(f"Bar chart saved to {chart_path}")

    except ImportError:
        if console:
            print("[warn] matplotlib not installed – skipping chart generation.")


def _find_chinese_font() -> str | None:
    """Find a Chinese-capable font on the system."""
    import platform
    system = platform.system()
    
    if system == "Windows":
        candidates = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        ]
    elif system == "Darwin":
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ]
    
    for font_path in candidates:
        if Path(font_path).exists():
            return font_path
    return None


def _chinese_title_fontproperties(font_path: str | None):
    """Font for matplotlib titles/labels so Chinese does not fall back to DejaVu Sans."""
    import matplotlib.font_manager as fm

    if font_path and Path(font_path).exists():
        return fm.FontProperties(fname=font_path)
    return fm.FontProperties(
        family=["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "sans-serif"],
    )


def print_raw_names(raw_counter: Counter, top_n: int = 50, *, quiet: bool = False) -> None:
    """Print the raw (un-merged) model names for inspection."""
    if quiet or not raw_counter:
        return
    print(f"\n{'─'*70}")
    print("  Raw model/law names extracted (before synonym merging):")
    print(f"{'─'*70}")
    for name, count in raw_counter.most_common(top_n):
        print(f"    {count:>4d}  {name}")
    print()
