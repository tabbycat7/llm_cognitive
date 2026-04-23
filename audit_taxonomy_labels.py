"""Use an LLM to review whether existing taxonomy labels in a JSONL file are reasonable.

Reads rows like ``reddit_taxonomy_labels.jsonl`` (slim schema from ``classify_reddit_taxonomy``):
``question_index``, ``question``, ``metadata.{category, category_valid, reasoning, ...}``.

For each row, the model sees the 7 canonical categories, the post text, the assigned category,
and the classifier's original ``reasoning``; it returns a structured verdict (agree / disagree /
uncertain) and an optional alternative category.

Examples
--------
    python audit_taxonomy_labels.py -i reddit_taxonomy_labels.jsonl -o taxonomy_audit.jsonl -n 20
    python audit_taxonomy_labels.py -i reddit_taxonomy_labels.jsonl --filter-category "[Relational & Intimate]" -n 50 -c 4
    python audit_taxonomy_labels.py --backend dummy -i reddit_taxonomy_labels.jsonl -n 3 -o /tmp/audit.jsonl --no-resume

统计展示（读审计结果 JSONL）::

    python summarize_taxonomy_audit.py -i reddit_taxonomy_audit.jsonl
    python audit_taxonomy_labels.py ... -o out.jsonl --print-summary   # 跑完后顺带打印摘要
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

from classify_reddit_taxonomy import (  # noqa: E402
    CANONICAL_CATEGORIES,
    _resolve_category,
    extract_json_from_markdown,
)
from cognitive_probe import normalize_taxonomy_category  # noqa: E402
from llm_api import ChatMessage, create_backend  # noqa: E402

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


def _tqdm_ncols() -> int:
    try:
        w = shutil.get_terminal_size(fallback=(96, 24)).columns
    except OSError:
        return 88
    return max(52, min(w - 2, 100))


AUDIT_PROMPT_MARKER = "[TASK: audit_taxonomy_labels_v1]"

AUDIT_USER_TEMPLATE = Template(
    """$marker

# 角色
你是分类质量审计员。已有程序按固定 7 类体系给 Reddit 求助帖打了标签；你要判断该标签在体系内是否合理，**不要**重新写故事摘要，只根据「帖子原文 + 原分类理由」做判断。

# 允许的 7 个类别（必须逐字一致才算有效类别名）
$category_block

# 输入
## 帖子原文（question）
$question_quoted

## 程序已打标签（assigned_category）
$assigned_quoted

## 标签是否被解析为合法类（category_valid）
$valid_json

## 原分类模型的理由（reasoning，可为空）
$reasoning_quoted

# 输出要求
1. 只输出一个 JSON 对象，不要 Markdown 围栏、不要前后解释文字。
2. ``verdict`` 取值只能是: ``"agree"`` | ``"disagree"`` | ``"uncertain"``  
   - agree: 在体系内合理，或归为 [Invalid / Noise] 也合理  
   - disagree: 明显应归入另一类（或明显应是 Invalid 却标成核心类，反之亦然）  
   - uncertain: 复合/边界，难以在单选体系下断言
3. ``suggested_category``: 若 agree 或 uncertain 且无更强替代，填 null；若 disagree，填你认为更合适的**完整**类别字符串（必须从上面 7 类中逐字复制其一）。
4. ``confidence``: 0 到 1 的小数，表示你对 verdict 的信心。0.8-1.0是非常有信心，0.6-0.8是比较有信心，0.4-0.6是可能有点信心，0.2-0.4是可能没信心，0.0-0.2是没信心。
5. ``rationale``: 1–4 句中文，说明最关键的判定依据（可点名原 reasoning 是否合理）。

JSON 形状示例：
{"verdict": "agree", "rationale": "…", "suggested_category": null, "confidence": 0.82}
"""
)


def _build_audit_prompt(
    question: str,
    assigned: str | None,
    category_valid: bool,
    reasoning: str | None,
) -> str:
    cat_lines = "\n".join(f"- {c}" for c in CANONICAL_CATEGORIES)
    return AUDIT_USER_TEMPLATE.substitute(
        marker=AUDIT_PROMPT_MARKER,
        category_block=cat_lines,
        question_quoted=json.dumps(question, ensure_ascii=False),
        assigned_quoted=json.dumps(assigned, ensure_ascii=False),
        valid_json=json.dumps(bool(category_valid), ensure_ascii=False),
        reasoning_quoted=json.dumps(reasoning, ensure_ascii=False),
    )


def _parse_audit_reply(text: str) -> tuple[dict | None, str | None]:
    inner = extract_json_from_markdown(text) or text.strip()
    try:
        data = json.loads(inner)
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"
    if not isinstance(data, dict):
        return None, "Top-level JSON is not an object"
    return data, None


def audit_one(
    backend,
    question: str,
    assigned: str | None,
    category_valid: bool,
    reasoning: str | None,
    *,
    temperature: float,
    max_tokens: int,
) -> tuple[dict, str]:
    """Returns (audit_fields, raw_assistant_text)."""
    user_content = _build_audit_prompt(question, assigned, category_valid, reasoning)
    raw = backend.chat(
        [ChatMessage(role="user", content=user_content)],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parsed, err = _parse_audit_reply(raw)
    out: dict = {
        "verdict": None,
        "rationale": None,
        "suggested_category": None,
        "suggested_category_valid": False,
        "confidence": None,
        "parse_error": err,
        "raw_response": raw,
    }
    if not parsed:
        return out, raw
    verdict = parsed.get("verdict")
    if isinstance(verdict, str):
        v = verdict.strip().lower()
        if v in ("agree", "disagree", "uncertain"):
            out["verdict"] = v
    out["rationale"] = parsed.get("rationale") if isinstance(parsed.get("rationale"), str) else None
    conf = parsed.get("confidence")
    if isinstance(conf, (int, float)):
        out["confidence"] = max(0.0, min(1.0, float(conf)))
    sug = parsed.get("suggested_category")
    if sug is None or (isinstance(sug, str) and not sug.strip()):
        out["suggested_category"] = None
        out["suggested_category_valid"] = True
    elif isinstance(sug, str):
        canon, ok = _resolve_category(sug)
        out["suggested_category"] = canon
        out["suggested_category_valid"] = ok
    return out, raw


def _load_label_row(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    return json.loads(line)


def _row_original_fields(obj: dict) -> dict:
    """Normalize slim / legacy row into original{} for output."""
    meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    if meta.get("category") is not None or "category_valid" in meta:
        return {
            "category": meta.get("category"),
            "category_valid": bool(meta.get("category_valid")),
            "reasoning": meta.get("reasoning"),
            "subreddit": meta.get("subreddit"),
            "parse_error": meta.get("parse_error"),
            "content_len": meta.get("content_len"),
        }
    return {
        "category": obj.get("category"),
        "category_valid": bool(obj.get("category_valid")),
        "reasoning": obj.get("reasoning"),
        "subreddit": obj.get("subreddit"),
        "parse_error": obj.get("parse_error"),
        "content_len": obj.get("content_len"),
    }


def _load_done_indices(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.is_file():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["question_index"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return done


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"error: input not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_indices(output_path) if args.resume else set()
    if done:
        print(f"[resume] {len(done)} rows already in {output_path}", flush=True)

    want_filter_cat: str | None = None
    if args.filter_category:
        want_filter_cat, ok = normalize_taxonomy_category(args.filter_category.strip())
        if not ok or not want_filter_cat:
            raise SystemExit(
                f"error: unknown --filter-category {args.filter_category!r}; "
                f"use a canonical name like {CANONICAL_CATEGORIES[0]!r}"
            )

    rows: list[dict] = []
    line_idx = -1
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = _load_label_row(line)
            if obj is None:
                continue
            line_idx += 1
            if line_idx < args.start:
                continue
            try:
                qidx = int(obj["question_index"])
            except (KeyError, TypeError, ValueError):
                continue
            if qidx in done:
                continue
            orig = _row_original_fields(obj)
            if args.only_valid and not orig["category_valid"]:
                continue
            if want_filter_cat is not None and orig.get("category") != want_filter_cat:
                continue
            rows.append(obj)
            if args.num is not None and len(rows) >= args.num:
                break

    if not rows:
        print("Nothing to audit (empty range, all done, or filters excluded everything).", flush=True)
        return

    model_name = (args.model or os.getenv("LLM_MODEL") or "gpt-4o").strip()
    backend_kwargs: dict = {}
    if args.backend == "openai":
        backend_kwargs.update(
            api_key=args.api_key,
            base_url=args.base_url,
            model=model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    backend = create_backend(args.backend, **backend_kwargs)

    tasks: list[tuple[dict, str, dict]] = []
    for obj in rows:
        q = obj.get("question") if isinstance(obj.get("question"), str) else ""
        if args.max_question_chars and len(q) > args.max_question_chars:
            q = q[: args.max_question_chars] + "\n…[truncated]"
        orig = _row_original_fields(obj)
        tasks.append((obj, q, orig))

    print(
        f"[audit] model={model_name!r} tasks={len(tasks)} concurrency={args.concurrency} -> {output_path}",
        flush=True,
    )

    def one(item: tuple[dict, str, dict]) -> dict:
        obj, question, orig = item
        qidx = int(obj["question_index"])
        audit_fields, raw = audit_one(
            backend,
            question,
            orig.get("category"),
            bool(orig.get("category_valid")),
            orig.get("reasoning"),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        if args.strip_raw_on_success and audit_fields.get("parse_error") is None:
            audit_fields.pop("raw_response", None)
        full_q = obj.get("question") if isinstance(obj.get("question"), str) else question
        return {
            "question_index": qidx,
            "question": full_q,
            "original": orig,
            "audit": audit_fields,
        }

    write_lock = Lock()
    use_tqdm = tqdm is not None

    if args.concurrency <= 1:
        with open(output_path, "a", encoding="utf-8") as out_f:
            loop = tqdm(
                tasks,
                total=len(tasks),
                desc=f"审计 c1 n{len(tasks)}",
                unit="条",
                ncols=_tqdm_ncols(),
                dynamic_ncols=False,
                mininterval=0.2,
            ) if use_tqdm else tasks
            for (obj, q, orig) in loop:
                t0 = time.time()
                row = one((obj, q, orig))
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                if use_tqdm:
                    loop.set_postfix_str(
                        f"#{row['question_index']} {row['audit'].get('verdict') or '?'} {time.time() - t0:.1f}s"
                    )
    else:
        with open(output_path, "a", encoding="utf-8") as out_f:
            pbar = (
                tqdm(
                    total=len(tasks),
                    desc=f"审计 c{args.concurrency} n{len(tasks)}",
                    unit="条",
                    ncols=_tqdm_ncols(),
                    dynamic_ncols=False,
                    mininterval=0.15,
                )
                if use_tqdm
                else None
            )
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = {ex.submit(one, t): t for t in tasks}
                for fut in as_completed(futs):
                    try:
                        row = fut.result()
                    except Exception as e:
                        task = futs[fut]
                        obj, q, orig = task
                        qidx = int(obj["question_index"])
                        row = {
                            "question_index": qidx,
                            "question": obj.get("question", q),
                            "original": orig,
                            "audit": {
                                "verdict": None,
                                "rationale": None,
                                "suggested_category": None,
                                "suggested_category_valid": False,
                                "confidence": None,
                                "parse_error": str(e),
                                "raw_response": None,
                            },
                        }
                        if tqdm is not None:
                            tqdm.write(f"[error] #{qidx}: {e}")
                    with write_lock:
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_f.flush()
                    if pbar is not None:
                        v = row["audit"].get("verdict") or "?"
                        pbar.set_postfix_str(f"ok #{row['question_index']} {v}")
                        pbar.update(1)
            if pbar is not None:
                pbar.close()

    if tqdm is not None:
        tqdm.write(f"Done. Appended {len(tasks)} rows to {output_path}")
    else:
        print(f"Done. Appended {len(tasks)} rows to {output_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLM audit of taxonomy labels (reddit_taxonomy_labels.jsonl style).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "-i",
        "--input",
        default=str(ROOT / "reddit_taxonomy_labels.jsonl"),
        help="Input labels JSONL (default: %(default)s)",
    )
    p.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "reddit_taxonomy_audit.jsonl"),
        help="Output JSONL path (default: %(default)s)",
    )
    p.add_argument("-n", "--num", type=int, default=None, help="Max rows to audit after filters")
    p.add_argument("--start", type=int, default=0, help="Skip first N non-empty JSONL lines (0-based)")
    p.add_argument("--no-resume", action="store_true", help="Do not skip question_index already in -o")
    p.add_argument("--concurrency", "-c", type=int, default=1, help="Parallel API calls (default: 1)")
    p.add_argument("--backend", default="openai", choices=("openai", "dummy"))
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument(
        "--max-question-chars",
        type=int,
        default=None,
        help="Truncate post text sent to the auditor to this length (full question still stored in output row)",
    )
    p.add_argument(
        "--only-valid",
        action="store_true",
        help="Only audit rows where original category_valid is true",
    )
    p.add_argument(
        "--filter-category",
        default=None,
        metavar="CATEGORY",
        help="Only rows whose original metadata.category equals this (e.g. '[Relational & Intimate]')",
    )
    p.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw_response in audit object even when JSON parses OK",
    )
    p.add_argument(
        "--print-summary",
        action="store_true",
        help="After run, print human-readable stats for -o (same as summarize_taxonomy_audit.py)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.start < 0:
        raise SystemExit("error: --start must be >= 0")
    args.resume = not args.no_resume
    args.strip_raw_on_success = not args.keep_raw
    run(args)
    if args.print_summary:
        from summarize_taxonomy_audit import summarize_path

        out_p = Path(args.output).expanduser().resolve()
        if out_p.is_file():
            summarize_path(out_p, json_mode=False)


if __name__ == "__main__":
    main()
