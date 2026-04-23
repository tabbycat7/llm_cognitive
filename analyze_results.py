"""Analyse step-3 responses: parse JSON, merge synonyms via LLM, visualise."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path

from llm_api import ChatMessage, LLMBackend, OpenAICompatibleBackend

# LLM 同义归并：只对原始频次最高的前 N 个**不同**名称调用一次聚类；其余名称保持自身（identity）。
MERGE_LLM_TOP_N = 100

CLUSTER_PROMPT_TEMPLATE = """你是一个学术概念归类专家。以下是从LLM对话中提取的模型/定律名称列表（已按在整个数据集中的出现频次从高到低排序；本批仅包含频次最高的前 {count} 个不同名称），将实质上同构的别名或子概念归并为一个标准的“介观层级”学术概念。

【输入名称列表】（共{count}个）：
{name_list}


【聚类规则】：
别名归并：将同一数理模型的不同称呼合并（如 "negative feedback control" 与 "feedback control"）。

我需要的是归并到的是"介观层级"的数理模型——
- 它必须比具体的操作技巧更抽象
  （错误示例："nonviolent communication", 
   "gray rock method" —— 这些太具体了）
- 它必须比整个学科或范式更具体
  （错误示例："game theory", "psychology", 
   "economics" —— 这些太宽泛了）  
- 正确的层级：一个有明确数学结构或因果机制的、可以用一句话定义的数理模型（如某个特定的博弈均衡、某个特定的认知偏误、某个特定的动力学模型、某个特定的优化原理）

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


def _merge_cluster_backend_from_env() -> LLMBackend:
    """OpenAI-compatible client for merge-only (与探测阶段解耦).

    优先使用归并专用：``OPENAI_API_MERGE_KEY``、``OPENAI_BASE_URL_MERGE``、``LLM_MODEL_MERGE``；
    未设置时回退 ``OPENAI_API_KEY``、``OPENAI_BASE_URL``、``LLM_MODEL``。
    同义聚类请求的 ``temperature`` 固定为 ``0.2``（与 ``MERGE_LLM_MAX_TOKENS`` 等无关）。
    """
    from llm_api import _load_dotenv_from_project

    _load_dotenv_from_project()
    key = (os.getenv("OPENAI_API_MERGE_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "merge_method=llm 需要配置 OPENAI_API_MERGE_KEY（或回退 OPENAI_API_KEY），"
            "可在项目根目录 .env 中设置。"
        )
    base = (
        (os.getenv("OPENAI_BASE_URL_MERGE") or os.getenv("OPENAI_BASE_URL") or "")
        .strip()
        or None
    )
    model = (os.getenv("LLM_MODEL_MERGE") or os.getenv("LLM_MODEL") or "gpt-4o").strip() or "gpt-4o"
    max_tok = int(os.getenv("MERGE_LLM_MAX_TOKENS", "8192"))
    return OpenAICompatibleBackend(
        api_key=key,
        base_url=base,
        model=model,
        temperature=0.2,
        max_tokens=max_tok,
    )


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
) -> dict[str, str]:
    """Use an LLM to cluster synonym names into canonical forms.

    Only the **top ``top_n`` distinct names by total occurrence count** are sent to the
    model in **one** clustering call (see ``MERGE_LLM_TOP_N``). All other raw names map
    to themselves (no LLM merge).

    The merge client is always :class:`OpenAICompatibleBackend` using ``OPENAI_API_MERGE_KEY``,
    ``OPENAI_BASE_URL_MERGE``, and ``LLM_MODEL_MERGE`` in the environment / project ``.env``
    (with fallback to ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` / ``LLM_MODEL``); independent of
    the probe-phase backend.

    Args:
        raw_counter: Per-raw-label occurrence counts from ``collect_raw_names``.
        cache_path: If given, save/load the full merge map here.
        refresh: If True, ignore cache and re-run the top-``top_n`` LLM clustering.

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

    merge_backend = _merge_cluster_backend_from_env()
    if _merge_verbose_print() and not _batch_console_quiet():
        print(
            f"[merge] LLM 同义归并：模型 {getattr(merge_backend, 'model', '?')!r}；"
            f"Top-{len(top_names)} 名称，单次 API…"
        )

    prompt = _build_cluster_prompt(top_names, counts=counts_dict)
    messages = [ChatMessage(role="user", content=prompt)]
    reply = merge_backend.chat(messages)
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
        print(f"[embed] Loading cached embeddings from {embed_cache_path}")
        with open(embed_cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_names = cached.get("names", [])
        cached_embeddings = cached.get("embeddings", [])
        
        if set(cached_names) == set(unique_names):
            name_to_idx = {n: i for i, n in enumerate(cached_names)}
            embeddings = np.array([cached_embeddings[name_to_idx[n]] for n in unique_names])
        else:
            print("[embed] Cache mismatch, will re-compute embeddings.")
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

    print(f"[embed] Clustering {len(unique_names)} names with distance_threshold={distance_threshold}")
    
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
    print(f"[embed] {len(unique_names)} names → {n_clusters} clusters")
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
        print(f"[embed] Saved merge map to {cache_path}")

        dendrogram_path = cache_path.parent / "cluster_dendrogram.png"
        _plot_dendrogram(
            linkage_matrix, unique_names, distance_threshold, dendrogram_path
        )

        scatter_path = cache_path.parent / "cluster_scatter.png"
        _plot_cluster_scatter(
            embeddings, unique_names, cluster_labels, merge_map, scatter_path
        )

    return merge_map


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
        print("[embed] matplotlib not installed, skipping dendrogram plot")
        return

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

    dendro = dendrogram(
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
    print(f"[embed] Dendrogram saved to {output_path}")


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
        print(f"[embed] Missing matplotlib: {e}")
        return

    try:
        import umap
    except ImportError:
        print("[embed] umap-learn not installed, skipping scatter plot")
        print("[embed] Run: pip install umap-learn")
        return

    font_path = _find_chinese_font()
    plt.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    if len(labels) < 5:
        print("[embed] Too few points for UMAP scatter plot, skipping")
        return

    print("[embed] Running UMAP for 2D visualization...")
    
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
    print(f"[embed] Scatter plot saved to {output_path}")


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

    print(f"[embed-local] Loading model: {model}  device={device}")
    print("[embed-local] (First run will download the model, may take a while)")

    # Transformers refuses loading pytorch_model.bin via torch.load when torch<2.6 (CVE-2025-32434).
    # Prefer safetensors when present so local embedding works without upgrading torch.
    load_kw: dict = {"trust_remote_code": True, "device": device}
    try:
        import torch

        pv = torch.__version__.split("+")[0].split(".")[:2]
        major, minor = int(pv[0]), int(pv[1])
        if (major, minor) < (2, 6):
            load_kw["model_kwargs"] = {"use_safetensors": True}
            print(
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
    
    print(f"[embed-local] Encoding {len(texts)} texts on {encoder.device}...")
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
    print(f"[embed] Cached embeddings to {path}")


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
) -> tuple[Counter, Counter]:
    """Parse results, optionally merge synonyms, and return counts.

    Args:
        results_path: Path to probe_results.jsonl.
        backend: Unused for ``merge_method="llm"`` (归并读 ``OPENAI_API_MERGE_KEY`` 等，见
            :func:`_merge_cluster_backend_from_env`)。嵌入聚类也不使用此参数。
        cache_path: Path to cache merge_map.json.
        refresh_merge: If True, re-query even if cache exists.
        merge_method: "llm" for LLM-based clustering, "embedding" for vector clustering, "none" for no merging.
        embedding_threshold: Cosine distance threshold for embedding clustering (0-1).
        embedding_backend: "local" (HuggingFace), "ollama", or "openai".
        embedding_model: Model name for embedding.
        embedding_device: Local backend only: "auto", "cuda", "cpu", "cuda:0", … (None → env EMBEDDING_DEVICE or auto).
        name_source: ``step3`` / legacy ``step4`` parse JSON ``models[].name``. ``step1`` / ``step2``
            use each row's full response as one label (for embedding or ``none`` only; not supported with
            ``merge_method=llm``).

    Returns:
        (canonical_counts, raw_counts)
    """
    if name_source not in _NAME_SOURCE_TO_FIELD:
        raise ValueError(
            f"name_source must be one of {tuple(_NAME_SOURCE_TO_FIELD)}, got {name_source!r}"
        )
    if merge_method == "llm" and name_source in ("step1", "step2"):
        raise ValueError(
            "merge_method 'llm' cannot be used with name_source 'step1' or 'step2' "
            "(full responses do not fit the LLM cluster prompt). "
            "Use --merge-method embedding or --merge-method none."
        )

    raw_counter = collect_raw_names(results_path, name_source=name_source)

    if merge_method == "none":
        return raw_counter, raw_counter

    unique_names = list(raw_counter.keys())

    if merge_method == "embedding":
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
        # LLM 归并使用 .env 中 OPENAI_*_MERGE / LLM_MODEL_MERGE（与 ``backend`` 探测阶段解耦）。
        merge_map = merge_synonyms_via_llm(
            raw_counter,
            cache_path=cache_path,
            refresh=refresh_merge,
        )

    canonical_counter: Counter = Counter()
    for raw_name, count in raw_counter.items():
        canonical = merge_map.get(raw_name, raw_name)
        if merge_method == "llm":
            c = normalize_llm_merge_label(str(canonical))
            canonical = c if c else (raw_name.strip() or raw_name)
        canonical_counter[canonical] += count

    return canonical_counter, raw_counter


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
        merge_map_path.with_suffix(".embeddings.json"),
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
