"""Three-step cognitive probing pipeline.

For each record in a JSONL dataset (Zhihu-KOL, Reddit exports, etc.),
conduct a progressive three-step conversation with an LLM and persist every turn.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import time
from typing import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock

from tqdm.auto import tqdm

from llm_api import ChatMessage, LLMBackend


def _tqdm_ncols() -> int:
    """Keep the bar on one physical line (wrapped tqdm breaks in-place refresh)."""
    try:
        w = shutil.get_terminal_size(fallback=(96, 24)).columns
    except OSError:
        return 88
    return max(52, min(w - 2, 100))


def _one_line_hint(text: str, max_chars: int) -> str:
    """Single-line, length-capped snippet for tqdm postfix (no newlines)."""
    s = " ".join((text or "").split())
    if len(s) <= max_chars:
        return s or "—"
    return s[: max_chars - 1] + "…"


# ─── Prompt templates ────────────────────────────────────────────────

STEP1_TEMPLATE_ZH = """{question}

请你给出你最真诚、最深入的分析和建议。"""

# Step 3 请求里不再带上用户原文，只保留第一步里的引导句，避免模型在第三步回到「分析用户问题」。
STEP1_USER_CONTENT_WITHOUT_QUESTION_ZH = "请你给出你最真诚、最深入的分析和建议。"

STEP2_PROMPT_ZH = """
请剥离所有情感安抚和表层修饰，将你的建议浓缩为一条严密的“因果干预链条”。
语义要求（四段因果，顺序不可打乱）：
1. current_state：当前状态变量（客观描述系统/情境的初始条件）
2. intervention：你的干预动作（你建议采取的具体行动或策略）
3. structural_change：系统发生的结构性改变（干预如何改变机制、约束或反馈）
4. expected_outcome：预期结果（可检验或可推断的后果）

请用最客观的语言填写各字段；不要输出 JSON 以外的任何文字。

请严格按以下 JSON 格式输出：

```json
{
  "current_state": "…",
  "intervention": "…",
  "structural_change": "…",
  "expected_outcome": "…"
}
```"""

# Must appear verbatim in STEP3 prompts; ``probe_one`` replaces it with Step-2 JSON text.
_STEP2_JSON_INJECTION_SLOT = "__STEP2_CAUSAL_CHAIN_JSON__"

#STEP3_PROMPT="""现在请你完成最后一步。将上述分析抽象为来自物理学、数学、信息论、生物学、经济学或其它学科的正式模型或定律。"""
STEP3_PROMPT_ZH = """现在，根据你给出的【建议/解决方案】，以及你给出的【因果干预链条】，反向推导出支撑这套策略运转的数理模型。不要告诉我这个问题属于什么分类，而是告诉我：你的解决方案实际上是在利用哪个特定的数理模型来改变现状？必须是数理模型。

【因果干预链条】

__STEP2_CAUSAL_CHAIN_JSON__


【抽象层级校准】：
我需要的是"介观层级"的数理模型——
- 它必须比具体的操作技巧更抽象
  （错误示例："nonviolent communication", 
   "gray rock method" —— 这些太具体了）
- 它必须比整个学科或范式更具体
  （错误示例："game theory", "psychology", 
   "economics" —— 这些太宽泛了）  
- 正确的层级：一个有明确数学结构或因果机制的、可以用一句话定义的数理模型（如某个特定的博弈均衡、某个特定的认知偏误、某个特定的动力学模型、某个特定的优化原理）

要求：
1. 只选择最核心的 1-3 个数理模型，必须是数理模型
2. 必须使用学术界公认、可检索的标准英文名称，绝不要包含括号或破折号
3. 必须使用英文名称!!!
4. 严格禁止创造新词，不要把多个数理模型拼接成一个名称

请严格按照以下JSON格式输出（不要输出任何JSON之外的内容）：

```json
{
  "models": [
    {
      "reasoning": "在此处简述你的思考过程...",
      "name": "数理模型的标准英文名称，绝不要包含括号或破折号",
      "explanation": "该数学结构或者因果机制如何精确解决用户的问题"
    }
  ]
}
```
其他要求：
- models数组包含1-3个最核心的数理模型（不要过多）
- 如果没有合适的数理模型，返回：{"models": [{"name": "无合适映射"}]}

请直接输出JSON。"""


# ── English versions (enable via --prompt-lang en / LLM_PROMPT_LANG=en) ──

STEP1_TEMPLATE_EN = """{question}

Please give your most sincere and in-depth analysis and advice."""

STEP1_USER_CONTENT_WITHOUT_QUESTION_EN = (
    "Please give your most sincere and in-depth analysis and advice."
)

STEP2_PROMPT_EN = """
Strip away emotional reassurance and surface rhetoric. Compress your advice into one tight
"causal intervention chain".

Semantic requirements (four causal segments, fixed order):
1. current_state: state variables now (an objective description of initial conditions of the system / situation)
2. intervention: your intervention (the concrete actions or strategy you recommend)
3. structural_change: structural change in the system (how the intervention alters mechanisms, constraints, or feedback)
4. expected_outcome: expected outcome (testable or inferable consequences)

Use maximally objective language for every field; output nothing outside JSON.

Use EXACTLY this JSON shape:

```json
{
  "current_state": "…",
  "intervention": "…",
  "structural_change": "…",
  "expected_outcome": "…"
}
```"""

STEP3_PROMPT_EN = """Now, based on the advice / solution you gave, and the [CAUSAL CHAIN JSON] you provided, reverse-engineer the underlying mathematical model that supports the operation of this strategy. Do NOT tell me what category this problem belongs to, but tell me: which specific mathematical model is your solution actually utilizing to change the situation? It MUST be a mathematical model.

[CAUSAL CHAIN JSON]

__STEP2_CAUSAL_CHAIN_JSON__


[ABSTRACTION-LEVEL CALIBRATION]:
I need "meso-level" mathematical models —
- It must be MORE abstract than concrete operational techniques
  (bad examples: "nonviolent communication", "gray rock method" — these are too specific)
- It must be MORE specific than an entire discipline or paradigm
  (bad examples: "game theory", "psychology", "economics" — these are too broad)
- The right level: a mathematical model with an explicit mathematical structure or a clear causal mechanism, definable in one sentence (e.g. a specific game-theoretic equilibrium, a specific cognitive bias, a specific dynamical model, a specific optimization principle).

Requirements:
1. Pick ONLY the 1-3 most central mathematical models, they MUST be mathematical models.
2. Use standard, academically recognizable English names; NEVER include parentheses or em dashes.
3. Names MUST be in English!!!
4. STRICTLY PROHIBITED to coin new terms; do NOT glue multiple mathematical models into one name.

Output STRICTLY in the following JSON format (do not output anything outside the JSON):

```json
{
  "models": [
    {
      "reasoning": "briefly explain your reasoning here...",
      "name": "standard English name of the mathematical model, NEVER include parentheses or dashes",
      "explanation": "how this mathematical structure or causal mechanism precisely solves the user's problem"
    }
  ]
}
```
Other requirements:
- The "models" array contains 1-3 most core mathematical models (do not list too many).
- If no suitable mathematical model applies, return: {"models": [{"name": "no suitable mapping"}]}

Output JSON only."""


# ── Prompt bundle + selector ──

PROMPT_BUNDLES: dict[str, dict[str, str]] = {
    "zh": {
        "step1_template": STEP1_TEMPLATE_ZH,
        "step1_without_question": STEP1_USER_CONTENT_WITHOUT_QUESTION_ZH,
        "step2": STEP2_PROMPT_ZH,
        "step3": STEP3_PROMPT_ZH,
    },
    "en": {
        "step1_template": STEP1_TEMPLATE_EN,
        "step1_without_question": STEP1_USER_CONTENT_WITHOUT_QUESTION_EN,
        "step2": STEP2_PROMPT_EN,
        "step3": STEP3_PROMPT_EN,
    },
}


def get_prompt_bundle(lang: str | None) -> dict[str, str]:
    """Pick the ``zh`` / ``en`` prompt bundle; unknown → zh."""
    key = (lang or "zh").strip().lower()
    if key in ("", "cn", "zh-cn", "chinese"):
        key = "zh"
    if key in ("english", "en-us"):
        key = "en"
    return PROMPT_BUNDLES.get(key, PROMPT_BUNDLES["zh"])


# Backwards-compatible module-level names (default = Chinese).
STEP1_TEMPLATE = STEP1_TEMPLATE_ZH
STEP1_USER_CONTENT_WITHOUT_QUESTION = STEP1_USER_CONTENT_WITHOUT_QUESTION_ZH
STEP2_PROMPT = STEP2_PROMPT_ZH
STEP3_PROMPT = STEP3_PROMPT_ZH


# ─── Data structures ─────────────────────────────────────────────────


@dataclass
class ProbeResult:
    question_index: int
    question: str
    step1_response: str = ""
    step2_response: str = ""
    step3_response: str = ""
    metadata: dict = field(default_factory=dict)


# ─── Core logic ──────────────────────────────────────────────────────

# JSONL question field: Zhihu-KOL uses INSTRUCTION; Reddit-style dumps use text, etc.
_QUESTION_FIELD_ORDER = ("INSTRUCTION", "text", "body", "content", "title", "question")

_LABEL_KEYS = ("label", "communityName", "community", "subreddit")

# Same canonical names as classify_reddit_taxonomy (for --filter-category).
TAXONOMY_CANONICAL_CATEGORIES: tuple[str, ...] = (
    "[Relational & Intimate]",
    "[Professional & Economic]",
    "[Societal & Ethical]",
    "[Personal & Existential]",
    "[Invalid / Noise]",
)
_TAXONOMY_NAKED_TO_CANONICAL: dict[str, str] = {c[1:-1]: c for c in TAXONOMY_CANONICAL_CATEGORIES}

# Subdirectory names under ``<model_dir>/`` (e.g. ``deepseek/Relational&Intimate/``).
CANONICAL_CATEGORY_OUTPUT_SUBDIR: dict[str, str] = {
    "[Relational & Intimate]": "Relational&Intimate",
    "[Professional & Economic]": "Professional&Economic",
    "[Societal & Ethical]": "Societal&Ethical",
    "[Personal & Existential]": "Personal&Existential",
    "[Invalid / Noise]": "Invalid-Noise",
}


def normalize_taxonomy_category(raw: str | None) -> tuple[str | None, bool]:
    """Map a category string to canonical bracket form (same rules as taxonomy classifier)."""
    if not raw or not isinstance(raw, str):
        return None, False
    s = raw.strip()
    if s in TAXONOMY_CANONICAL_CATEGORIES:
        return s, True
    inner = s[1:-1].strip() if s.startswith("[") and s.endswith("]") else s
    if inner in _TAXONOMY_NAKED_TO_CANONICAL:
        return _TAXONOMY_NAKED_TO_CANONICAL[inner], True
    return s, False


def category_to_probe_output_subdir(filter_category: str) -> str:
    """Map ``--filter-category`` string to a filesystem-safe subdir (see ``deepseek/`` layout)."""
    canon, ok = normalize_taxonomy_category(filter_category)
    if ok and canon in CANONICAL_CATEGORY_OUTPUT_SUBDIR:
        return CANONICAL_CATEGORY_OUTPUT_SUBDIR[canon]
    s = filter_category.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    s = s.replace("/", "-").replace(" & ", "&").replace(" ", "-")
    out = "".join(ch if ch not in '<>:"\\|?*' else "-" for ch in s)
    while "--" in out:
        out = out.replace("--", "-")
    return (out.strip("-")[:150] or "unknown-category")


def model_id_to_probe_output_dirname(model: str) -> str:
    """Top-level directory under project root (e.g. ``deepseek`` for DeepSeek models)."""
    m = (model or "unknown").strip() or "unknown"
    if "deepseek" in m.lower():
        return "deepseek"
    slug = "".join(ch if ch not in '<>:"/\\|?*' else "-" for ch in m)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return (slug.strip("-")[:120] or "model")


def load_taxonomy_category_by_index(labels_path: str | Path) -> dict[int, str]:
    """Build ``question_index`` → canonical ``category`` from a taxonomy_labels JSONL.

    Only rows with ``category_valid`` true and a recognized canonical category are kept.
    """
    path = Path(labels_path)
    out: dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                idx = int(obj["question_index"])
            except (KeyError, TypeError, ValueError):
                continue
            meta = obj.get("metadata")
            if isinstance(meta, dict) and "category" in meta:
                valid = meta.get("category_valid")
                cat_raw = meta.get("category")
            else:
                valid = obj.get("category_valid")
                cat_raw = obj.get("category")
            if not valid:
                continue
            if not isinstance(cat_raw, str):
                continue
            canon, ok = normalize_taxonomy_category(cat_raw)
            if ok and canon:
                out[idx] = canon
    return out


def default_taxonomy_labels_path() -> Path:
    """Default ``taxonomy_labels.jsonl`` beside this package."""
    return Path(__file__).resolve().parent / "taxonomy_labels.jsonl"


def _metadata_as_dict(rec: dict) -> dict | None:
    """Return METADATA object if present as dict or JSON string."""
    raw = rec.get("METADATA")
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def iter_label_values(rec: dict) -> Iterator[str]:
    """Yield label-like strings from a JSONL row (top-level and METADATA)."""
    for key in _LABEL_KEYS:
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            yield val.strip()
    md = _metadata_as_dict(rec)
    if md:
        for key in _LABEL_KEYS:
            val = md.get(key)
            if isinstance(val, str) and val.strip():
                yield val.strip()


def record_matches_label_filter(rec: dict, filter_label: str) -> bool:
    """True if any label-like field equals ``filter_label`` (case-insensitive, stripped)."""
    want = filter_label.strip().casefold()
    if not want:
        return True
    for cand in iter_label_values(rec):
        if cand.strip().casefold() == want:
            return True
    return False


def _question_and_meta_from_record(rec: dict) -> tuple[str, dict]:
    """Pick question text and metadata from a single JSON object.

    Supports Zhihu-KOL (``INSTRUCTION`` / ``METADATA``) and common exports
    where the prompt lives in ``text`` or similar keys.

    Webis-TLDR-17 style rows (``normalizedBody`` + ``summary_len``): use ``content``
    only so we skip the huge ``body`` (TL;DR duplication). If ``content`` is empty,
    fall through to the generic field order below.
    """
    question = ""
    question_key: str | None = None
    if isinstance(rec.get("normalizedBody"), str) and "summary_len" in rec:
        c = rec.get("content")
        if isinstance(c, str) and c.strip():
            question = c.strip()
            question_key = "content"
    if not question:
        for key in _QUESTION_FIELD_ORDER:
            val = rec.get(key)
            if isinstance(val, str) and val.strip():
                question = val.strip()
                question_key = key
                break

    if "METADATA" in rec:
        meta = rec["METADATA"]
        if isinstance(meta, str):
            if not meta.strip():
                meta = {}
            else:
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {"raw": meta}
        elif isinstance(meta, dict):
            meta = dict(meta)
        else:
            meta = {"METADATA": meta}
    else:
        meta = {k: v for k, v in rec.items() if k != question_key}

    return question, meta


# ─── Step3: normalize model names before persist / downstream stats ───


def extract_json_from_markdown(text: str) -> str | None:
    """Strip `` ```json ... ``` `` fences if present and return the inner JSON string."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    text = text.strip()
    if text.startswith("{"):
        return text
    return None


def format_step2_json_for_step3_prompt(step2_response: str) -> str:
    """Pretty-print Step-2 causal-chain JSON for injection into the Step-3 user message."""
    raw = (step2_response or "").strip()
    inner = extract_json_from_markdown(raw) or (raw if raw.startswith("{") else None)
    if inner:
        try:
            data = json.loads(inner)
            if isinstance(data, dict):
                return json.dumps(data, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return inner
        return inner
    return raw if raw else "{}"


def normalize_model_name(raw_name: str) -> str:
    """Normalize a model/theory ``name`` for consistent storage and frequency counts."""
    name = raw_name.lower()
    name = re.sub(r"[-_/\\]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def clean_step3_response(text: str) -> str:
    """Parse step-3 JSON, normalize each model ``name``, re-serialize.

    If parsing fails, returns ``text`` unchanged.
    """
    json_str = extract_json_from_markdown(text)
    if not json_str:
        return text
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return text
    models = data.get("models")
    if not isinstance(models, list):
        return text
    for m in models:
        if not isinstance(m, dict):
            continue
        raw = m.get("name")
        if isinstance(raw, str) and raw.strip():
            m["name"] = normalize_model_name(raw)
    new_inner = json.dumps(data, ensure_ascii=False, indent=2)
    if re.search(r"```", text):
        return f"```json\n{new_inner}\n```"
    return new_inner


def load_questions(
    jsonl_path: str | Path,
    max_count: int | None = None,
    start: int = 0,
    filter_label: str | None = None,
    taxonomy_labels_path: str | Path | None = None,
    filter_category: str | None = None,
    *,
    random_sample: bool = False,
    random_seed: int | None = None,
    exclude_question_indices: set[int] | None = None,
) -> list[tuple[int, dict]]:
    """Load JSON objects from a JSONL file (Zhihu-KOL, Reddit, etc.).

    Each non-empty line is assigned a **global line index** (0-based among
    non-empty lines in the file). Skips indices ``< start``, then keeps at
    most ``max_count`` rows that pass filters (if any).

    Returns ``(line_index, record)`` tuples so :func:`run_pipeline` can set
    ``question_index`` to the true file order for resume/checkpoints.

    When ``filter_label`` is set, only rows whose ``label`` / ``communityName``
    / ``community`` / ``subreddit`` (top-level or inside ``METADATA``) match
    the string (case-insensitive) are kept.

    When ``filter_category`` is set, ``taxonomy_labels_path`` must point to a
    JSONL like ``taxonomy_labels.jsonl``. Rows are kept when the join key's
    category matches: use ``int(record["question_index"])`` when that field
    exists (same key as in the labels file), otherwise the file's 0-based line
    index among non-empty lines (for inputs without ``question_index``).

    When ``random_sample`` is True, ``max_count`` must be set. Among all lines
    after ``start`` that pass filters and are not excluded, up to ``max_count``
    rows are drawn uniformly at random without replacement (reservoir sampling;
    one pass). If fewer than ``max_count`` remain, a shorter list is returned.

    When ``exclude_question_indices`` is set, rows whose join key is in that
    set are skipped (e.g. resume: already written ``question_index`` values).
    """
    if start < 0:
        raise ValueError("start must be >= 0")
    if random_sample and max_count is None:
        raise ValueError("random_sample=True requires max_count (e.g. -n on CLI)")

    category_by_index: dict[int, str] | None = None
    want_category: str | None = None
    if filter_category is not None:
        want_category, cat_ok = normalize_taxonomy_category(filter_category.strip())
        if not cat_ok or not want_category:
            raise ValueError(
                f"Unknown filter_category {filter_category!r}; "
                f"use a canonical name such as {TAXONOMY_CANONICAL_CATEGORIES[0]!r}"
            )
        tpath = Path(taxonomy_labels_path) if taxonomy_labels_path else default_taxonomy_labels_path()
        if not tpath.is_file():
            raise FileNotFoundError(f"taxonomy labels file not found: {tpath}")
        category_by_index = load_taxonomy_category_by_index(tpath)

    items: list[tuple[int, dict]] = []
    line_idx = -1
    rng = random.Random(random_seed)
    eligible_i = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_idx += 1
            if line_idx < start:
                continue
            record = json.loads(line)
            if filter_label is not None and not record_matches_label_filter(record, filter_label):
                continue
            try:
                join_key = (
                    int(record["question_index"])
                    if "question_index" in record
                    else line_idx
                )
            except (TypeError, ValueError):
                join_key = line_idx
            if want_category is not None and category_by_index is not None:
                if category_by_index.get(join_key) != want_category:
                    continue
            if exclude_question_indices is not None and join_key in exclude_question_indices:
                continue
            if random_sample:
                eligible_i += 1
                if len(items) < max_count:
                    items.append((join_key, record))
                else:
                    j = rng.randrange(eligible_i)
                    if j < max_count:
                        items[j] = (join_key, record)
            else:
                items.append((join_key, record))
                if max_count is not None and len(items) >= max_count:
                    break
    return items


def probe_one(
    backend: LLMBackend,
    question: str,
    question_index: int,
    metadata: dict | None = None,
    *,
    prompt_lang: str | None = None,
) -> ProbeResult:
    """Run the 3-step probe for a single question and return the result.

    ``prompt_lang``: ``"zh"`` (default) or ``"en"``; picks a :data:`PROMPT_BUNDLES` entry.
    """

    bundle = get_prompt_bundle(prompt_lang)
    result = ProbeResult(
        question_index=question_index,
        question=question,
        metadata=metadata or {},
    )
    messages: list[ChatMessage] = []

    # Step 1
    user_msg1 = bundle["step1_template"].format(question=question)
    messages.append(ChatMessage(role="user", content=user_msg1))
    reply1 = backend.chat(messages)
    result.step1_response = reply1
    messages.append(ChatMessage(role="assistant", content=reply1))

    # Step 2
    messages.append(ChatMessage(role="user", content=bundle["step2"]))
    reply2 = backend.chat(messages)
    result.step2_response = reply2
    messages.append(ChatMessage(role="assistant", content=reply2))

    # Step 3：从发给模型的上下文中去掉用户具体问题，仅依赖前两轮助手内容做抽象。
    messages[0].content = bundle["step1_without_question"]
    step3_user = bundle["step3"].replace(
        _STEP2_JSON_INJECTION_SLOT,
        format_step2_json_for_step3_prompt(reply2),
    )
    messages.append(ChatMessage(role="user", content=step3_user))
    reply3 = backend.chat(messages)
    result.step3_response = clean_step3_response(reply3)
    messages.append(ChatMessage(role="assistant", content=reply3))

    return result


def run_pipeline(
    backend: LLMBackend,
    jsonl_path: str | Path,
    output_path: str | Path,
    max_count: int | None = None,
    resume: bool = True,
    concurrency: int = 1,
    start: int = 0,
    filter_label: str | None = None,
    taxonomy_labels_path: str | Path | None = None,
    filter_category: str | None = None,
    prompt_lang: str | None = None,
) -> list[ProbeResult]:
    """Run the full pipeline: load → probe → save (with checkpoint resume).

    Args:
        backend: LLM backend instance.
        jsonl_path: Path to a JSONL file (``INSTRUCTION`` or ``text``-style rows).
        output_path: Where to write results (JSONL, one line per question).
        max_count: Limit number of questions to process.
        resume: If True, skip questions already present in output_path.
        concurrency: Number of concurrent API calls (default: 1 = sequential).
        start: 0-based index of the first **non-empty line** in the JSONL to
            include (earlier lines are skipped). ``question_index`` in probe
            output uses ``record["question_index"]`` when present, else that
            line's global line index (same rule as taxonomy join key).
        filter_label: If set, only rows whose label-like fields match this
            string (case-insensitive) are sent to the LLM; see
            :func:`record_matches_label_filter`.
        taxonomy_labels_path: JSONL with ``question_index`` / ``category`` (from
            taxonomy classification). Used when ``filter_category`` is set.
        filter_category: If set, only rows whose index appears in that file with
            this canonical category (e.g. ``[Relational & Intimate]``).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_indices: set[int] = set()
    if resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                done_indices.add(obj["question_index"])
        print(f"[resume] Found {len(done_indices)} completed entries, skipping.")

    records = load_questions(
        jsonl_path,
        max_count,
        start=start,
        filter_label=filter_label,
        taxonomy_labels_path=taxonomy_labels_path,
        filter_category=filter_category,
        exclude_question_indices=done_indices if resume and done_indices else None,
    )

    tasks: list[tuple[int, str, dict]] = []
    for global_idx, rec in records:
        if global_idx in done_indices:
            continue
        question, meta = _question_and_meta_from_record(rec)
        tasks.append((global_idx, question, meta))

    if not tasks:
        print("No new questions to process.")
        return []

    total = len(records)
    total_tasks = len(tasks)

    if concurrency <= 1:
        return _run_sequential(
            backend, tasks, output_path, start=start, loaded=total, prompt_lang=prompt_lang
        )
    else:
        return _run_concurrent(
            backend,
            tasks,
            output_path,
            concurrency,
            start=start,
            loaded=total,
            prompt_lang=prompt_lang,
        )


def _run_sequential(
    backend: LLMBackend,
    tasks: list[tuple[int, str, dict]],
    output_path: Path,
    *,
    start: int = 0,
    loaded: int | None = None,
    prompt_lang: str | None = None,
) -> list[ProbeResult]:
    """Run tasks sequentially with a single in-place progress bar."""
    results: list[ProbeResult] = []
    n = len(tasks)
    loaded_n = loaded if loaded is not None else n
    ncols = _tqdm_ncols()
    bar_desc = f"探测 c1 s{start} n{loaded_n}"

    with open(output_path, "a", encoding="utf-8") as out_f:
        with tqdm(
            tasks,
            total=n,
            desc=bar_desc,
            unit="题",
            ncols=ncols,
            dynamic_ncols=False,
            smoothing=0.05,
            mininterval=0.2,
        ) as pbar:
            for global_idx, question, meta in pbar:
                hint = _one_line_hint(question, max_chars=16)
                pbar.set_postfix_str(f"#{global_idx} | {hint}", refresh=False)
                t0 = time.time()
                result = probe_one(
                    backend, question, global_idx, meta, prompt_lang=prompt_lang
                )
                elapsed = time.time() - t0
                pbar.set_postfix_str(f"#{global_idx} {elapsed:.1f}s {hint}", refresh=True)

                out_f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                out_f.flush()
                results.append(result)

    tqdm.write(f"Pipeline complete. {len(results)} new results → {output_path}")
    return results


def _run_concurrent(
    backend: LLMBackend,
    tasks: list[tuple[int, str, dict]],
    output_path: Path,
    concurrency: int,
    *,
    start: int = 0,
    loaded: int | None = None,
    prompt_lang: str | None = None,
) -> list[ProbeResult]:
    """Run tasks concurrently using ThreadPoolExecutor and one progress bar."""
    results: list[ProbeResult] = []
    write_lock = Lock()
    total_tasks = len(tasks)
    loaded_n = loaded if loaded is not None else total_tasks
    ncols = _tqdm_ncols()
    bar_desc = f"探测 c{concurrency} s{start} n{loaded_n}"

    def process_task(task: tuple[int, str, dict]) -> tuple[ProbeResult, float, int, str]:
        idx, question, meta = task
        t0 = time.time()
        result = probe_one(backend, question, idx, meta, prompt_lang=prompt_lang)
        elapsed = time.time() - t0
        return result, elapsed, idx, question

    with open(output_path, "a", encoding="utf-8") as out_f:
        with tqdm(
            total=total_tasks,
            desc=bar_desc,
            unit="题",
            ncols=ncols,
            dynamic_ncols=False,
            smoothing=0.05,
            mininterval=0.15,
        ) as pbar:
            tqdm.write(
                "每题 3 轮 API（step1→2→3）；并发时每完成一题进度 +1；首条可能需数分钟。"
            )
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_task = {
                    executor.submit(process_task, task): task for task in tasks
                }

                for future in as_completed(future_to_task):
                    try:
                        result, elapsed, idx, question = future.result()
                        hint = _one_line_hint(question, max_chars=14)

                        with write_lock:
                            out_f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                            out_f.flush()
                            results.append(result)

                        pbar.set_postfix_str(f"ok #{idx} {elapsed:.0f}s {hint}")
                        pbar.update(1)

                    except Exception as e:
                        task = future_to_task[future]
                        tqdm.write(f"[error] Task {task[0]} failed: {e}")

    tqdm.write(f"Pipeline complete. {len(results)} new results → {output_path}")
    return results
