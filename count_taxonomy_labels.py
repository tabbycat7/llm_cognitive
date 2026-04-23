"""Count records per `category` in a taxonomy JSONL (output of classify_reddit_taxonomy.py).

Examples:
    python count_taxonomy_labels.py
    python count_taxonomy_labels.py -i data/my_labels.jsonl
    python count_taxonomy_labels.py --json > summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from classify_reddit_taxonomy import CANONICAL_CATEGORIES

ROOT = Path(__file__).resolve().parent


def _taxonomy_label_fields(obj: dict) -> tuple[object, bool, object]:
    """Support slim rows (category inside metadata) and legacy top-level fields."""
    meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    if "category" in meta:
        return meta.get("category"), bool(meta.get("category_valid")), meta.get("parse_error")
    return obj.get("category"), bool(obj.get("category_valid")), obj.get("parse_error")


def _json_objects_on_line(line: str) -> list[object]:
    """Decode one physical line that may contain multiple JSON values concatenated (no newline)."""
    dec = json.JSONDecoder()
    i = 0
    n = len(line)
    out: list[object] = []
    while i < n:
        while i < n and line[i].isspace():
            i += 1
        if i >= n:
            break
        obj, end = dec.raw_decode(line, i)
        out.append(obj)
        i = end
    while i < n and line[i].isspace():
        i += 1
    if i < n:
        raise json.JSONDecodeError("trailing garbage after JSON value(s)", line, i)
    return out


def load_counts(path: Path) -> tuple[Counter[str], int, int, int, int]:
    """Returns (category_counter, rows_invalid_flag, rows_parse_error, malformed_lines, multi_json_lines)."""
    cat: Counter[str] = Counter()
    invalid_valid_flag = 0
    parse_errors = 0
    malformed_lines = 0
    multi_json_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                objs = _json_objects_on_line(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue
            if len(objs) > 1:
                multi_json_lines += 1
            for obj in objs:
                if not isinstance(obj, dict):
                    malformed_lines += 1
                    continue
                c, cv, pe = _taxonomy_label_fields(obj)
                if not cv:
                    invalid_valid_flag += 1
                if pe:
                    parse_errors += 1
                key = c if isinstance(c, str) and c.strip() else "(null / empty)"
                cat[key] += 1
    return cat, invalid_valid_flag, parse_errors, malformed_lines, multi_json_lines


def build_report(
    cat: Counter[str],
    invalid_valid_flag: int,
    parse_errors: int,
    total: int,
    *,
    malformed_lines: int = 0,
    multi_json_lines: int = 0,
) -> dict:
    """Structured report for printing or --json."""
    by_canonical: dict[str, int] = {c: int(cat.get(c, 0)) for c in CANONICAL_CATEGORIES}
    extras = {k: v for k, v in cat.items() if k not in CANONICAL_CATEGORIES}
    return {
        "input_total_lines": total,
        "rows_category_valid_false": invalid_valid_flag,
        "rows_with_parse_error_field": parse_errors,
        "malformed_physical_lines_skipped": malformed_lines,
        "physical_lines_with_multiple_json_values": multi_json_lines,
        "by_category_canonical_order": by_canonical,
        "extra_labels_not_in_taxonomy": extras if extras else {},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Count taxonomy categories in a labels JSONL file.")
    p.add_argument(
        "-i",
        "--input",
        default=str(ROOT / "top500_per_category.jsonl"),
        help="Path to JSONL (default: %(default)s)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print full report as JSON to stdout",
    )
    args = p.parse_args()
    path = Path(args.input).expanduser().resolve()
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    cat, invalid_ct, parse_err_ct, bad_lines, multi_json = load_counts(path)
    total = sum(cat.values())
    report = build_report(
        cat,
        invalid_ct,
        parse_err_ct,
        total,
        malformed_lines=bad_lines,
        multi_json_lines=multi_json,
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(f"File: {path}")
    print(f"Total records: {total}")
    print(f"category_valid=False: {invalid_ct}")
    print(f"Rows with non-empty parse_error: {parse_err_ct}")
    if bad_lines:
        print(f"Malformed physical lines skipped (JSON decode): {bad_lines}")
    if multi_json:
        print(f"Physical lines containing 2+ JSON objects (glued): {multi_json}")
    print()
    print("By category (canonical order):")
    for label in CANONICAL_CATEGORIES:
        n = report["by_category_canonical_order"][label]
        print(f"  {n:5d}  {label}")
    extras = report["extra_labels_not_in_taxonomy"]
    if extras:
        print()
        print("Other labels (not in built-in taxonomy list):")
        for label, n in sorted(extras.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {n:5d}  {label}")


if __name__ == "__main__":
    main()
