"""Filter taxonomy label JSONL by metadata.confidence; print per-category details.

Rows without a numeric ``confidence`` are skipped (legacy files).

Among rows with ``confidence > threshold``, optionally take the top ``K`` rows per
canonical category by confidence (ties broken by ``question_index``).

Examples:
    python filter_taxonomy_by_confidence.py -i reddit_taxonomy_labels.jsonl
    python filter_taxonomy_by_confidence.py -i reddit_taxonomy_labels.jsonl -t 0.85 -o high_conf.jsonl
    python filter_taxonomy_by_confidence.py -i reddit_taxonomy_labels.jsonl -t 0.85 \\
        --top-per-category 500 --output-top top500_per_cat.jsonl
    python filter_taxonomy_by_confidence.py -i reddit_taxonomy_labels.jsonl --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from classify_reddit_taxonomy import CANONICAL_CATEGORIES

ROOT = Path(__file__).resolve().parent


def _row_fields(obj: dict) -> tuple[object, bool, object, object, object, object]:
    """category, category_valid, confidence, parse_error, subreddit, content_len."""
    meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    if "category" in meta:
        return (
            meta.get("category"),
            bool(meta.get("category_valid")),
            meta.get("confidence"),
            meta.get("parse_error"),
            meta.get("subreddit"),
            meta.get("content_len"),
        )
    return (
        obj.get("category"),
        bool(obj.get("category_valid")),
        obj.get("confidence"),
        obj.get("parse_error"),
        obj.get("subreddit"),
        obj.get("content_len"),
    )


def _confidence_value(raw: object) -> float | None:
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _confidence_sort_key(obj: dict) -> tuple[float, int]:
    """Higher confidence first; tie-breaker: lower question_index."""
    _, _, cr, _, _, _ = _row_fields(obj)
    conf = _confidence_value(cr) or 0.0
    q = obj.get("question_index")
    try:
        qi = int(q) if q is not None else 0
    except (TypeError, ValueError):
        qi = 0
    return (-conf, qi)


def _take_top_per_category(rows: list[dict], k: int) -> list[dict]:
    if k <= 0 or len(rows) <= k:
        return sorted(rows, key=_confidence_sort_key)
    return sorted(rows, key=_confidence_sort_key)[:k]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Keep rows with confidence > threshold; report stats per category."
    )
    p.add_argument(
        "-i",
        "--input",
        default=str(ROOT / "reddit_taxonomy_labels.jsonl"),
        help="Taxonomy labels JSONL (default: %(default)s)",
    )
    p.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.85,
        help="Keep rows where confidence > this value (default: %(default)s)",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="PATH",
        help="If set, write filtered rows as JSONL to this path",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable report JSON to stdout",
    )
    p.add_argument(
        "--top-subreddits",
        type=int,
        default=8,
        metavar="N",
        help="Per category, show top N subreddits by count (default: %(default)s)",
    )
    p.add_argument(
        "--top-per-category",
        type=int,
        default=500,
        metavar="K",
        help="After threshold filter, keep at most K rows per canonical category "
        "with highest confidence (default: %(default)s). Use 0 for no cap.",
    )
    p.add_argument(
        "--output-top",
        default=None,
        metavar="PATH",
        help="Write JSONL: union of per-category top-K rows (see --top-per-category).",
    )
    args = p.parse_args()

    path = Path(args.input).expanduser().resolve()
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    total_lines = 0
    skipped_no_conf = 0
    skipped_low = 0
    kept: list[dict] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            _cat, _cv, conf_raw, _pe, _sub, _cl = _row_fields(obj)
            conf = _confidence_value(conf_raw)
            if conf is None:
                skipped_no_conf += 1
                continue
            if conf <= args.threshold:
                skipped_low += 1
                continue
            kept.append(obj)

    by_cat: dict[str, list[dict]] = {c: [] for c in CANONICAL_CATEGORIES}
    unknown_cat: list[dict] = []

    for obj in kept:
        cat, _, _, _, _, _ = _row_fields(obj)
        key = cat if isinstance(cat, str) and cat.strip() else "(null / empty)"
        if key in by_cat:
            by_cat[key].append(obj)
        else:
            unknown_cat.append(obj)

    def category_detail(rows: list[dict]) -> dict:
        confs: list[float] = []
        valid_true = 0
        parse_err = 0
        lens: list[int] = []
        subs = Counter()
        for obj in rows:
            cat, cv, cr, pe, sub, cl = _row_fields(obj)
            c = _confidence_value(cr)
            if c is not None:
                confs.append(c)
            if cv:
                valid_true += 1
            if pe:
                parse_err += 1
            if isinstance(cl, int):
                lens.append(cl)
            if isinstance(sub, str) and sub.strip():
                subs[sub.strip()] += 1
        n = len(rows)
        out: dict = {
            "count": n,
            "category_valid_true": valid_true,
            "rows_with_parse_error": parse_err,
        }
        if confs:
            out["confidence_mean"] = round(statistics.mean(confs), 4)
            out["confidence_min"] = round(min(confs), 4)
            out["confidence_max"] = round(max(confs), 4)
            out["confidence_median"] = round(statistics.median(confs), 4)
        if lens:
            out["content_len_mean"] = round(statistics.mean(lens), 1)
            out["content_len_min"] = min(lens)
            out["content_len_max"] = max(lens)
        out["top_subreddits"] = dict(subs.most_common(max(1, args.top_subreddits)))
        return out

    retained_count_by_category: dict[str, int] = {c: len(by_cat[c]) for c in CANONICAL_CATEGORIES}

    top_k = max(0, int(args.top_per_category))
    top_by_cat: dict[str, list[dict]] = {}
    selected_rows: list[dict] = []
    for c in CANONICAL_CATEGORIES:
        picked = _take_top_per_category(by_cat[c], top_k)
        top_by_cat[c] = picked
        selected_rows.extend(picked)

    selected_count_by_category: dict[str, int] = {c: len(top_by_cat[c]) for c in CANONICAL_CATEGORIES}

    report = {
        "input": str(path),
        "threshold_exclusive_gt": args.threshold,
        "total_jsonl_rows": total_lines,
        "kept_count": len(kept),
        "skipped_missing_confidence": skipped_no_conf,
        "skipped_confidence_not_greater_than_threshold": skipped_low,
        "retained_count_by_category": retained_count_by_category,
        "top_per_category_limit": top_k if top_k > 0 else None,
        "selected_count_by_category": selected_count_by_category,
        "selected_total": len(selected_rows),
        "by_category": {c: category_detail(by_cat[c]) for c in CANONICAL_CATEGORIES},
        "by_category_top_pick": {c: category_detail(top_by_cat[c]) for c in CANONICAL_CATEGORIES},
    }
    if unknown_cat:
        report["non_canonical_category_bucket"] = {
            "note": "category string not in CANONICAL_CATEGORIES",
            "count": len(unknown_cat),
            "detail": category_detail(unknown_cat),
            "distinct_labels": sorted(
                {(_row_fields(o)[0] or "(null)") for o in unknown_cat if isinstance(_row_fields(o)[0], str)}
            ),
        }

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as wf:
            for obj in kept:
                wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
        report["filtered_output_written"] = str(out_path)

    if args.output_top:
        top_path = Path(args.output_top).expanduser().resolve()
        top_path.parent.mkdir(parents=True, exist_ok=True)
        with open(top_path, "w", encoding="utf-8", newline="\n") as wf:
            for c in CANONICAL_CATEGORIES:
                for obj in top_by_cat[c]:
                    wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
        report["top_per_category_output_written"] = str(top_path)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(f"Input: {path}")
    print(f"Threshold: confidence > {args.threshold}")
    print(f"Total rows: {total_lines}")
    print(f"Kept: {len(kept)}")
    print(f"Skipped (no numeric confidence): {skipped_no_conf}")
    print(f"Skipped (confidence <= {args.threshold}): {skipped_low}")
    if args.output:
        print(f"Wrote filtered JSONL → {Path(args.output).expanduser().resolve()}")
    if args.output_top:
        label = f"top-{top_k}" if top_k > 0 else "all (no K cap)"
        print(f"Wrote {label} per category JSONL → {Path(args.output_top).expanduser().resolve()}")
    print()

    kept_n = len(kept)
    print("=== 各类别保留条数（满足置信度阈值后的筛选结果）===")
    for c in CANONICAL_CATEGORIES:
        n = retained_count_by_category[c]
        pct = (100.0 * n / kept_n) if kept_n else 0.0
        print(f"  {n:5d}  ({pct:5.1f}% of retained)  {c}")
    canon_sum = sum(retained_count_by_category.values())
    if unknown_cat:
        u = len(unknown_cat)
        pct_u = (100.0 * u / kept_n) if kept_n else 0.0
        print(f"  {u:5d}  ({pct_u:5.1f}% of retained)  (non-canonical / other label)")
    print(f"  -----")
    checksum = canon_sum + (len(unknown_cat) if unknown_cat else 0)
    print(f"  {kept_n:5d}  total retained  (sum of per-category rows: {checksum})")
    print()

    sel_n = len(selected_rows)
    cap_desc = f"top {top_k} by confidence per category" if top_k > 0 else "no per-category cap (all retained per class)"
    print(f"=== 各类别条数（在阈值之后，再按置信度取 {cap_desc}）===")
    for c in CANONICAL_CATEGORIES:
        n = selected_count_by_category[c]
        pct = (100.0 * n / sel_n) if sel_n else 0.0
        print(f"  {n:5d}  ({pct:5.1f}% of selected)  {c}")
    print(f"  -----")
    sel_tail = f"max {top_k} per category when pool allows" if top_k > 0 else "no per-category cap"
    print(f"  {sel_n:5d}  total selected ({sel_tail})")
    print()
    print(
        "Note: detailed stats below use the per-category top-K subset only. "
        "JSON field ``by_category`` still describes the full >threshold pool."
    )
    print()

    print("=== Per category — detailed stats on SELECTED top pool (canonical order) ===")
    for c in CANONICAL_CATEGORIES:
        d = report["by_category_top_pick"][c]
        n = d["count"]
        print(f"\n{c}")
        print(f"  selected: {n}")
        if n == 0:
            continue
        print(f"  category_valid=True: {d['category_valid_true']}")
        print(f"  rows with parse_error set: {d['rows_with_parse_error']}")
        if "confidence_mean" in d:
            print(
                f"  confidence: mean={d['confidence_mean']} "
                f"median={d['confidence_median']} "
                f"min={d['confidence_min']} max={d['confidence_max']}"
            )
        if "content_len_mean" in d:
            print(
                f"  content_len: mean={d['content_len_mean']} "
                f"min={d['content_len_min']} max={d['content_len_max']}"
            )
        tops = d.get("top_subreddits") or {}
        if tops:
            print(f"  top subreddits (up to {args.top_subreddits}):")
            for sub, k in tops.items():
                print(f"    {k:5d}  {sub}")

    if "non_canonical_category_bucket" in report:
        nb = report["non_canonical_category_bucket"]
        print(f"\n(Non-canonical labels — retained: {nb['count']} rows)")
        dd = nb["detail"]
        print(f"  retained: {dd['count']}")
        if dd["count"]:
            if "confidence_mean" in dd:
                print(
                    f"  confidence: mean={dd['confidence_mean']} "
                    f"median={dd['confidence_median']}"
                )
            print(f"  distinct labels: {nb.get('distinct_labels', [])}")


if __name__ == "__main__":
    main()
