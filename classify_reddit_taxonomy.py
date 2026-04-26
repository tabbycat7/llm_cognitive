"""Use an OpenAI-compatible LLM to classify each line of a JSONL file (e.g. reddit_r_advice.jsonl).

Reads credentials from .env (OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL) like run.py.

Input resolution (first match wins): ``-i`` / ``--input`` > ``--source`` preset >
env ``CLASSIFY_TAXONOMY_INPUT`` > default ``reddit_r_advice.jsonl`` in project root.

Example:
    python classify_reddit_taxonomy.py --list-sources
    python classify_reddit_taxonomy.py --source reddit_data -o taxonomy_labels.jsonl
    python classify_reddit_taxonomy.py -i reddit_r_advice.jsonl -o taxonomy_labels.jsonl
    python classify_reddit_taxonomy.py -i data/reddit_r_advice.jsonl -n 50 --concurrency 4
    python classify_reddit_taxonomy.py --source selected_t3_submissions -o taxonomy_t3.jsonl -n 100
    python classify_reddit_taxonomy.py -i subreddits.jsonl -n 5000 --random-sample --random-seed 42 -o out.jsonl
    python classify_reddit_taxonomy.py --migrate-compact-to-slim reddit_taxonomy_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from string import Template
from threading import Lock

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
_ENV = ROOT / ".env"
if _ENV.is_file():
    load_dotenv(_ENV, override=True)

# Optional default input when neither -i nor --source is given (relative paths = cwd).
_ENV_INPUT_KEY = "CLASSIFY_TAXONOMY_INPUT"

# Named JSONL locations — add keys here for quick switching with --source NAME.
TAXONOMY_INPUT_PRESETS: dict[str, Path] = {
    "reddit_advice": ROOT / "reddit_r_advice.jsonl",
    "reddit_data": ROOT / "data" / "reddit_r_advice.jsonl",
    "zhihu_kol_train": ROOT / "zhihu_kol_train.jsonl",
    "probe_results": ROOT / "model" / "probe_results.jsonl",
    "selected_t3_submissions": ROOT / "selected_subreddits_t3_submissions.jsonl",
    "subreddits": ROOT / "subreddits.jsonl",
}

from cognitive_probe import (  # noqa: E402  (after dotenv)
    _question_and_meta_from_record,
    _tqdm_ncols,
    default_taxonomy_labels_path,
    extract_json_from_markdown,
    load_questions,
)
from llm_api import ChatMessage, create_backend  # noqa: E402

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]

# ── Taxonomy (must match model output validation) ─────────────────────────

CANONICAL_CATEGORIES: tuple[str, ...] = (
    "[Relational & Intimate]",
    "[Professional & Economic]",
    "[Societal & Ethical]",
    "[Personal & Existential]",
    "[Invalid / Noise]",
)

_NAKED_TO_CANONICAL: dict[str, str] = {c[1:-1]: c for c in CANONICAL_CATEGORIES}

CLASSIFICATION_USER_TEMPLATE = Template(
    r"""# Role
你是一个文本分类专家。你的任务是对收集到的真实人类求助/讨论帖进行分类。请严格按照下面的分类体系判断，只有在文本确实无法归入前 4 类时才归入第 5 类 [Invalid / Noise]。

# Taxonomy (分类体系)
请仔细阅读以下 5 个类别的定义与判定标准，并仅从中选择一个最匹配的。禁止创造新类别，禁止不分类。

1. [Relational & Intimate] （人际与情感网络）

定义：困境的核心围绕人与人之间的情感纽带、信任关系或社交互动展开。

典型主题：亲情矛盾、友情冲突、恋爱/婚姻问题、背叛与出轨、信任危机、社交边界、家庭关系修复。

判定标准：
（1）困境中存在明确的人际关系对象（家人、朋友、伴侣、同学等）
（2）核心痛点在于关系本身的维系、破裂或重建，而非经济利益或制度规则
（3）情感因素（爱、恨、信任、失望、依赖）是驱动困境的主要力量

2. [Professional & Economic] （职场与资源竞争）

定义：困境的核心围绕工作环境、职业发展或经济利益的获取与分配展开。

典型主题：晋升竞争、职场内卷、跳槽决策、薪资谈判、团队管理冲突、创业困境、财务压力、求职就业。

判定标准：
（1）困境发生在工作场景或经济活动中，涉及可量化的利益（薪资、职位、市场份额等）
（2）核心痛点在于资源竞争、职业路径选择或经济收益最大化
（3）即使涉及人际关系，关系只是职业/经济问题的载体，而非困境本身

3. [Societal & Ethical] （社会规则与道德）

定义：困境的核心涉及社会公共规则、法律边界、道德准则或群体责任之间的张力。

典型主题：公共资源使用争议、法律灰色地带、弱势群体权益保障、集体责任与个人自由的冲突、社会公平问题、文化/宗教规范碰撞。

判定标准：
（1）困境超越了纯个人层面，涉及社会规范、法律、伦理原则或公共利益
（2）核心痛点在于"应该怎样才是对的/公正的"，而非个人情感或经济得失
（3）存在至少两个各自有理的立场（个人 vs 集体、自由 vs 秩序、传统 vs 变革）

4. [Personal & Existential] （自我演化与存在）

定义：困境的核心发生在个体内部，围绕自我认知、生活习惯、心理状态或人生意义展开。

典型主题：习惯养成与自律、成瘾戒断（游戏/烟酒/社交媒体）、人生意义迷茫、身份认同危机、中年危机、心理健康问题（焦虑/抑郁/倦怠）、学业/考试压力中的自我怀疑。

判定标准：
（1）困境的核心对象是"我自己"——即使有外部触发，根本矛盾在自我内部
（2）痛点集中在自我控制、意义感、身份认同、心理状态中的一项或多项
（3）即使外部条件完全改善，困境仍然存在（区别于职场/人际/社会类）

5. [Invalid / Noise]

定义：纯粹的情绪宣泄、无意义内容、钓鱼帖、广告，或提供的信息极度不足，无法判断属于上述任何类别。

# Output Constraints
1. 强制单选：很多求助是复合型的，请识别导致该困境的最根本驱动力，选择最贴切的一类。
2. confidence：请给出你对本次分类的置信度，取值 0.0 到 1.0。
   - 0.85–1.0：非常确信，文本高度典型
   - 0.65–0.84：比较确信，但存在少量边界模糊
   - 0.45–0.64：不太确定，文本可能同时符合两个类别
   - 0.0–0.44：很不确定，信息不足或极度模糊
3. JSON 输出：请严格按照以下 JSON 格式输出，不要包含任何额外的 Markdown 语法或解释性文字。

{
  "reasoning": "用 1-2 句话简述分类逻辑：先指出识别到的关键特征，再给出结论。",
  "category": "必须完全复制上述某一个类别标签，包括英文方括号，例如：[Relational & Intimate]",
  "confidence": 0.85
}

# 输入文本
$input_text"""
)


def _build_user_prompt(input_text: str) -> str:
    """Escape / wrap input as a single JSON string literal for clarity."""
    quoted = json.dumps(input_text, ensure_ascii=False)
    return CLASSIFICATION_USER_TEMPLATE.substitute(input_text=quoted)


def _resolve_category(raw: str | None) -> tuple[str | None, bool]:
    """Map model output to a canonical label; returns (category_or_none, matched)."""
    if not raw or not isinstance(raw, str):
        return None, False
    s = raw.strip()
    if s in CANONICAL_CATEGORIES:
        return s, True
    inner = s[1:-1].strip() if s.startswith("[") and s.endswith("]") else s
    if inner in _NAKED_TO_CANONICAL:
        return _NAKED_TO_CANONICAL[inner], True
    return s, False


def _parse_classification_reply(text: str) -> tuple[dict | None, str | None]:
    """Return (parsed_dict, error_message)."""
    inner = extract_json_from_markdown(text) or text.strip()
    try:
        data = json.loads(inner)
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"
    if not isinstance(data, dict):
        return None, "Top-level JSON is not an object"
    return data, None


def classify_one(backend, question: str, temperature: float, max_tokens: int) -> tuple[dict, str]:
    """Call LLM once; returns (result_fields, raw_assistant_text)."""
    user_content = _build_user_prompt(question)
    raw = backend.chat(
        [ChatMessage(role="user", content=user_content)],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parsed, err = _parse_classification_reply(raw)
    out: dict = {
        "reasoning": None,
        "category": None,
        "category_valid": False,
        "confidence": None,
        "parse_error": err,
        "raw_response": raw,
    }
    if parsed:
        out["reasoning"] = parsed.get("reasoning")
        cat_raw = parsed.get("category")
        if isinstance(cat_raw, str):
            canon, ok = _resolve_category(cat_raw)
            out["category"] = canon
            out["category_valid"] = ok
        conf = parsed.get("confidence")
        if isinstance(conf, (int, float)):
            out["confidence"] = round(float(conf), 4)
    return out, raw


def pack_taxonomy_label_row(
    question_index: int,
    question: str,
    source_meta: dict | None,
    *,
    category: str | None,
    category_valid: bool,
    confidence: float | None = None,
    parse_error: str | None,
    reasoning: str | None,
) -> dict:
    """Slim JSONL row: only ``question_index``, ``question``, and ``metadata``."""
    src = source_meta if isinstance(source_meta, dict) else {}
    return {
        "question_index": question_index,
        "question": question,
        "metadata": {
            "category": category,
            "category_valid": bool(category_valid),
            "confidence": confidence,
            "parse_error": parse_error,
            "content_len": src.get("content_len"),
            "subreddit": src.get("subreddit"),
            "reasoning": reasoning,
        },
    }


def compact_taxonomy_label_row(obj: dict) -> dict:
    """Convert legacy (category at top + fat metadata) or slim rows to the slim schema."""
    idx = int(obj["question_index"])
    q = obj.get("question") if isinstance(obj.get("question"), str) else ""
    m = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    if isinstance(m, dict) and "category" in m:
        cat = m.get("category")
        cv = m.get("category_valid")
        conf = m.get("confidence")
        pe = m.get("parse_error")
        rs = m.get("reasoning")
    else:
        cat = obj.get("category")
        cv = obj.get("category_valid")
        conf = obj.get("confidence")
        pe = obj.get("parse_error")
        rs = obj.get("reasoning")
    if cv is None:
        cv = False
    cl = m.get("content_len") if "content_len" in m else obj.get("content_len")
    sub = m.get("subreddit") if "subreddit" in m else obj.get("subreddit")
    slim_meta = {
        "category": cat,
        "category_valid": bool(cv),
        "confidence": conf,
        "parse_error": pe,
        "content_len": cl,
        "subreddit": sub,
        "reasoning": rs,
    }
    return {"question_index": idx, "question": q, "metadata": slim_meta}


def migrate_labels_file_to_slim(path: Path) -> None:
    """Rewrite JSONL in place to slim rows; copies original to ``path`` + ``.bak``."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"error: file not found: {path}")
    bak = path.with_name(path.name + ".bak")
    shutil.copy2(path, bak)
    print(f"[backup] {bak}", flush=True)
    out_rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out_rows.append(compact_taxonomy_label_row(json.loads(line)))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[migrate] wrote {len(out_rows)} slim rows -> {path}", flush=True)


def _short_cat_label(cat: object, max_len: int = 20) -> str:
    """Compact category for tqdm postfix (avoid terminal wrap)."""
    if not isinstance(cat, str) or not cat.strip():
        return "?"
    s = cat.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _load_done_indices(output_path: Path) -> set[int]:
    done: set[int] = set()
    if not output_path.is_file():
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(int(obj["question_index"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return done


def run(
    input_path: Path,
    output_path: Path,
    *,
    max_count: int | None,
    start: int,
    resume: bool,
    concurrency: int,
    backend_type: str,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
    temperature: float,
    max_tokens: int,
    max_question_chars: int | None,
    strip_raw_on_success: bool,
    filter_label: str | None = None,
    taxonomy_labels_path: str | Path | None = None,
    filter_category: str | None = None,
    random_sample: bool = False,
    random_seed: int | None = None,
) -> None:
    print(f"[input] {input_path}", flush=True)
    if filter_label:
        print(f"[filter] metadata label must match (case-insensitive): {filter_label!r}", flush=True)
    if filter_category:
        tpath = Path(taxonomy_labels_path) if taxonomy_labels_path else default_taxonomy_labels_path()
        print(f"[filter] taxonomy category={filter_category!r} using index file {tpath}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = _load_done_indices(output_path) if resume else set()
    if resume and done:
        print(f"[resume] {len(done)} rows already in {output_path}, will skip.")

    records = load_questions(
        input_path,
        max_count,
        start=start,
        filter_label=filter_label,
        taxonomy_labels_path=taxonomy_labels_path,
        filter_category=filter_category,
        random_sample=random_sample,
        random_seed=random_seed,
        exclude_question_indices=done if resume and done else None,
    )
    if random_sample:
        msg = f"[random-sample] drew {len(records)} rows (reservoir, seed={random_seed!r})"
        if resume and done:
            msg += f", excluding {len(done)} completed question_index values from the pool"
        print(msg, flush=True)

    model_name = (model or os.getenv("LLM_MODEL") or "gpt-4o").strip()
    backend = create_backend(
        backend_type,
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    tasks: list[tuple[int, str, dict]] = []
    for global_idx, rec in records:
        if global_idx in done:
            continue
        question, meta = _question_and_meta_from_record(rec)
        if max_question_chars and len(question) > max_question_chars:
            question = question[:max_question_chars] + "\n…[truncated]"
        tasks.append((global_idx, question, meta))

    if not tasks:
        print("Nothing to do (all skipped or empty input range).")
        return

    use_tqdm = tqdm is not None
    print(f"Model={model_name!r}, tasks={len(tasks)}, concurrency={concurrency}", flush=True)

    def one(task: tuple[int, str, dict]) -> dict:
        idx, question, meta = task
        if not question.strip():
            return pack_taxonomy_label_row(
                idx,
                question,
                meta,
                category=CANONICAL_CATEGORIES[-1],
                category_valid=True,
                parse_error="empty question text",
                reasoning=None,
            )
        fields, _raw = classify_one(backend, question, temperature, max_tokens)
        if strip_raw_on_success and fields.get("parse_error") is None and fields.get("category_valid"):
            fields.pop("raw_response", None)
        return pack_taxonomy_label_row(
            idx,
            question,
            meta,
            category=fields.get("category"),
            category_valid=bool(fields.get("category_valid")),
            confidence=fields.get("confidence"),
            parse_error=fields.get("parse_error"),
            reasoning=fields.get("reasoning"),
        )

    write_lock = Lock()

    ncols = _tqdm_ncols()

    if concurrency <= 1:
        with open(output_path, "a", encoding="utf-8") as out_f:
            loop = (
                tqdm(
                    tasks,
                    total=len(tasks),
                    desc=f"分类 c1 n{len(tasks)}",
                    unit="条",
                    ncols=ncols,
                    dynamic_ncols=False,
                    mininterval=0.2,
                )
                if use_tqdm
                else tasks
            )
            for task in loop:
                idx = task[0]
                t0 = time.time()
                row = one(task)
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                cat = row.get("metadata", {}).get("category")
                if use_tqdm:
                    loop.set_postfix_str(
                        f"#{idx} {_short_cat_label(cat)} {time.time() - t0:.1f}s",
                        refresh=True,
                    )
                else:
                    print(f"[{idx}] -> {_short_cat_label(cat)!r} ({time.time() - t0:.1f}s)", flush=True)
    else:
        with open(output_path, "a", encoding="utf-8") as out_f:
            pbar = (
                tqdm(
                    total=len(tasks),
                    desc=f"分类 c{concurrency} n{len(tasks)}",
                    unit="条",
                    ncols=ncols,
                    dynamic_ncols=False,
                    mininterval=0.15,
                )
                if use_tqdm
                else None
            )
            if use_tqdm:
                tqdm.write("并发时每完成一条进度 +1；首条可能较慢。")
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = {ex.submit(one, t): t for t in tasks}
                for fut in as_completed(futs):
                    task = futs[fut]
                    try:
                        row = fut.result()
                    except Exception as e:
                        idx = task[0]
                        if use_tqdm:
                            tqdm.write(f"[error] index={idx}: {e}")
                        else:
                            print(f"[error] index={idx}: {e}", flush=True)
                        row = pack_taxonomy_label_row(
                            idx,
                            task[1],
                            task[2],
                            category=None,
                            category_valid=False,
                            parse_error=str(e),
                            reasoning=None,
                        )
                    with write_lock:
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_f.flush()
                        cat = row.get("metadata", {}).get("category")
                        if pbar is not None:
                            pbar.set_postfix_str(
                                f"ok #{row['question_index']} {_short_cat_label(cat)}"
                            )
                            pbar.update(1)
                        else:
                            print(
                                f"index={row['question_index']} category={cat!r}",
                                flush=True,
                            )
            if pbar is not None:
                pbar.close()

    if use_tqdm:
        tqdm.write(f"Done. Appended results to {output_path}")
    else:
        print(f"Done. Appended results to {output_path}", flush=True)


def resolve_input_path(args: argparse.Namespace) -> Path:
    """Resolve input JSONL: -i > --source > env > reddit_r_advice.jsonl under project root."""
    if getattr(args, "input", None):
        return Path(args.input).expanduser().resolve()
    if getattr(args, "source", None):
        p = TAXONOMY_INPUT_PRESETS[args.source]
        return p.expanduser().resolve()
    env = os.getenv(_ENV_INPUT_KEY, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (ROOT / "reddit_r_advice.jsonl").resolve()


def build_parser() -> argparse.ArgumentParser:
    preset_keys = sorted(TAXONOMY_INPUT_PRESETS)
    p = argparse.ArgumentParser(description="LLM taxonomy classification for JSONL (Reddit / Zhihu-style rows).")
    p.add_argument(
        "-i",
        "--input",
        default=None,
        metavar="PATH",
        help="Input JSONL path (overrides --source and %s if set)" % _ENV_INPUT_KEY,
    )
    p.add_argument(
        "-S",
        "--source",
        choices=preset_keys,
        default=None,
        metavar="NAME",
        help="Use a built-in preset path (--list-sources). Ignored if -i is set.",
    )
    p.add_argument(
        "--list-sources",
        action="store_true",
        help="Print preset names and paths, then exit",
    )
    p.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "reddit_taxonomy_labels.jsonl"),
        help="Output JSONL path (default: %(default)s)",
    )
    p.add_argument("-n", "--num", type=int, default=None, help="Max records to load after --start")
    p.add_argument(
        "--random-sample",
        action="store_true",
        help="Uniformly sample -n rows without replacement from all lines after --start that pass filters "
        "(reservoir; requires -n)",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=None,
        metavar="INT",
        help="Reproducible RNG seed for --random-sample (default: non-deterministic)",
    )
    p.add_argument("--start", type=int, default=0, help="Skip first N non-empty JSONL lines (0-based)")
    p.add_argument("--no-resume", action="store_true", help="Do not skip rows already present in output")
    p.add_argument("--concurrency", type=int, default=1, help="Parallel API calls (default: 1)")
    p.add_argument("--backend", default="openai", choices=("openai", "dummy"), help="LLM backend")
    p.add_argument("--api-key", default=None, help="Override OPENAI_API_KEY")
    p.add_argument("--base-url", default=None, help="Override OPENAI_BASE_URL")
    p.add_argument("--model", default=None, help="Override LLM_MODEL / default model name")
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2)")
    p.add_argument("--max-tokens", type=int, default=512, help="Max completion tokens (default: 512)")
    p.add_argument(
        "--max-question-chars",
        type=int,
        default=None,
        help="Truncate input text to this many characters (after field selection)",
    )
    p.add_argument(
        "--keep-raw",
        action="store_true",
        help="Always keep raw_response even when parse succeeds",
    )
    p.add_argument(
        "--filter-label",
        default=None,
        metavar="TEXT",
        help="Reddit metadata: label/communityName/subreddit (case-insensitive). Not taxonomy category.",
    )
    p.add_argument(
        "--taxonomy-labels",
        default=None,
        metavar="PATH",
        help="taxonomy_labels-style JSONL for --filter-category (default: taxonomy_labels.jsonl in project root)",
    )
    p.add_argument(
        "--filter-category",
        default=None,
        metavar="CATEGORY",
        help="Only rows whose question_index has this category in taxonomy labels (e.g. '[Relational & Intimate]')",
    )
    p.add_argument(
        "--migrate-compact-to-slim",
        metavar="PATH",
        default=None,
        dest="migrate_slim",
        help="Rewrite labels JSONL to slim schema; original saved as PATH.bak then exit",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.migrate_slim:
        migrate_labels_file_to_slim(Path(args.migrate_slim))
        return
    if args.list_sources:
        print("Built-in --source presets (paths relative to project unless absolute):\n")
        for name in sorted(TAXONOMY_INPUT_PRESETS):
            print(f"  {name:22}  {TAXONOMY_INPUT_PRESETS[name]}")
        print(f"\nOptional env default (when -i and --source omitted): {_ENV_INPUT_KEY}")
        print(f"Fallback file: {ROOT / 'reddit_r_advice.jsonl'}")
        return

    input_path = resolve_input_path(args)
    if args.random_sample and args.num is None:
        raise SystemExit("error: --random-sample requires -n / --num (how many rows to draw)")
    taxonomy_path = None
    if args.filter_category:
        taxonomy_path = Path(args.taxonomy_labels or default_taxonomy_labels_path()).expanduser().resolve()
        if not taxonomy_path.is_file():
            raise SystemExit(f"error: taxonomy labels file not found: {taxonomy_path}")
        taxonomy_path = str(taxonomy_path)

    run(
        input_path,
        Path(args.output).expanduser().resolve(),
        max_count=args.num,
        start=args.start,
        resume=not args.no_resume,
        concurrency=max(1, args.concurrency),
        backend_type=args.backend,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_question_chars=args.max_question_chars,
        strip_raw_on_success=not args.keep_raw,
        filter_label=args.filter_label,
        taxonomy_labels_path=taxonomy_path,
        filter_category=args.filter_category,
        random_sample=args.random_sample,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
