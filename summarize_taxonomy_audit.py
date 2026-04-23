"""Aggregate statistics from ``audit_taxonomy_labels.py`` output JSONL.

Examples
--------
    python summarize_taxonomy_audit.py -i reddit_taxonomy_audit.jsonl
    python summarize_taxonomy_audit.py -i reddit_taxonomy_audit.jsonl --json > audit_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from classify_reddit_taxonomy import CANONICAL_CATEGORIES

ROOT = Path(__file__).resolve().parent


def _pct(n: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * n / total, 2)


def build_report(path: Path) -> dict:
    """Stream JSONL and return a structured summary dict."""
    total = 0
    verdicts: Counter[str] = Counter()
    audit_parse_errors = 0
    conf_vals: list[float] = []
    disagree_invalid_suggestion = 0
    per_original: dict[str, Counter[str]] = defaultdict(Counter)
    relabel_moves: Counter[tuple[str, str]] = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            audit = obj.get("audit") if isinstance(obj.get("audit"), dict) else {}
            orig = obj.get("original") if isinstance(obj.get("original"), dict) else {}

            if audit.get("parse_error"):
                audit_parse_errors += 1

            v = audit.get("verdict")
            if isinstance(v, str) and v.strip():
                vk = v.strip().lower()
                if vk in ("agree", "disagree", "uncertain"):
                    verdicts[vk] += 1
                else:
                    verdicts[f"(other:{vk[:24]})"] += 1
            else:
                verdicts["(no_verdict)"] += 1

            c = audit.get("confidence")
            if isinstance(c, (int, float)):
                conf_vals.append(float(c))

            oc = orig.get("category")
            orig_key = oc if isinstance(oc, str) and oc.strip() else "(no_original_category)"
            vkey = (
                v.strip().lower()
                if isinstance(v, str) and v.strip().lower() in ("agree", "disagree", "uncertain")
                else "(no_verdict)"
            )
            per_original[orig_key][vkey] += 1

            if vkey == "disagree":
                sug = audit.get("suggested_category")
                sug_ok = bool(audit.get("suggested_category_valid"))
                if sug and not sug_ok:
                    disagree_invalid_suggestion += 1
                if sug_ok and isinstance(sug, str) and sug.strip() and sug != oc:
                    relabel_moves[(str(oc), sug)] += 1

    conf_n = len(conf_vals)
    conf_mean = round(sum(conf_vals) / conf_n, 4) if conf_n else None
    conf_min = round(min(conf_vals), 4) if conf_n else None
    conf_max = round(max(conf_vals), 4) if conf_n else None

    by_canonical_orig = {c: dict(per_original.get(c, {})) for c in CANONICAL_CATEGORIES}
    extras_orig = {k: dict(v) for k, v in per_original.items() if k not in CANONICAL_CATEGORIES}

    return {
        "file": str(path),
        "total_rows": total,
        "verdict_counts": dict(verdicts),
        "audit_parse_error_rows": audit_parse_errors,
        "confidence": {
            "count": conf_n,
            "mean": conf_mean,
            "min": conf_min,
            "max": conf_max,
        },
        "disagree_rows_with_invalid_suggested_category": disagree_invalid_suggestion,
        "by_original_category": by_canonical_orig,
        "by_original_category_other_keys": extras_orig,
        "top_relabel_moves": [
            {"from": a, "to": b, "count": n}
            for (a, b), n in relabel_moves.most_common(50)
        ],
    }


def _bar(n: int, width: int, max_n: int) -> str:
    if max_n <= 0:
        return ""
    w = int(round(width * n / max_n))
    return "█" * w


def print_human(report: dict, *, bar_width: int = 28) -> None:
    path = report["file"]
    total = report["total_rows"]
    print("=" * 72)
    print("  taxonomy audit — 检验统计摘要")
    print("=" * 72)
    print(f"文件: {path}")
    print(f"总行数: {total}")
    if total == 0:
        print("(无数据)")
        return

    print()
    print("審計 verdict 分布:")
    vc = report["verdict_counts"]
    order = ["agree", "disagree", "uncertain", "(no_verdict)"]
    keys = [k for k in order if k in vc] + [k for k in sorted(vc.keys()) if k not in order]
    max_c = max(vc.values()) if vc else 0
    for k in keys:
        n = int(vc.get(k, 0))
        if n == 0 and k not in ("agree", "disagree", "uncertain", "(no_verdict)"):
            continue
        pct = _pct(n, total)
        bar = _bar(n, bar_width, max_c)
        print(f"  {n:6d}  ({pct:6.2f}%)  {bar}  {k}")

    print()
    print(f"audit.parse_error 非空行数: {report['audit_parse_error_rows']}")
    ce = report["confidence"]
    if ce["count"]:
        print(
            f"confidence: 有值 {ce['count']} 条, "
            f"均值 {ce['mean']}, 最小 {ce['min']}, 最大 {ce['max']}"
        )
    else:
        print("confidence: (无数值)")

    print()
    print(
        "disagree 且 suggested_category 无效: "
        f"{report['disagree_rows_with_invalid_suggested_category']}"
    )

    print()
    print("按「原始类别」的審計 verdict（仅显示计数>0 的格）:")
    for cat in CANONICAL_CATEGORIES:
        sub = report["by_original_category"].get(cat) or {}
        if not sub:
            continue
        parts = [f"{vk}={sub[vk]}" for vk in ("agree", "disagree", "uncertain", "(no_verdict)") if sub.get(vk)]
        extra_v = {k: v for k, v in sub.items() if k not in ("agree", "disagree", "uncertain", "(no_verdict)")}
        if extra_v:
            parts.extend(f"{k}={v}" for k, v in sorted(extra_v.items()))
        line = "  " + cat + "\n    " + ", ".join(parts)
        print(line)

    other = report.get("by_original_category_other_keys") or {}
    if other:
        print()
        print("原始类别不在内置 7 类中的键（或其它）:")
        for k, sub in sorted(other.items(), key=lambda x: (-sum(x[1].values()), x[0])):
            tot = sum(sub.values())
            parts = [f"{vk}={sub[vk]}" for vk in sorted(sub.keys()) if sub[vk]]
            print(f"  [{k}] 共 {tot}: " + ", ".join(parts))

    moves = report.get("top_relabel_moves") or []
    if moves:
        print()
        print("disagree 且建议类名有效、并与原类不同的 Top 迁移 (from → to):")
        for item in moves[:20]:
            print(f"  {item['count']:4d}  {item['from']!r} → {item['to']!r}")


def summarize_path(path: Path, *, json_mode: bool) -> None:
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    report = build_report(path)
    if json_mode:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize audit_taxonomy_labels JSONL output.")
    p.add_argument(
        "-i",
        "--input",
        default=str(ROOT / "reddit_taxonomy_audit.jsonl"),
        help="Audit JSONL path (default: %(default)s)",
    )
    p.add_argument("--json", action="store_true", help="Print full report as JSON")
    return p


def main() -> None:
    args = build_parser().parse_args()
    summarize_path(Path(args.input).expanduser().resolve(), json_mode=args.json)


if __name__ == "__main__":
    main()
