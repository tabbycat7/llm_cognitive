#!/usr/bin/env python3
"""CLI: analyze Stage1 vs Stage2 merge maps (embedding_llm_llm) and write merge_stages_analysis.log.

Per directory, reads:
  - merge_map_llm_stage1.json
  - merge_map_llm_stage2.json
Optionally uses merge_map.json for consistency check.

The same report is generated at the end of ``analyze_results.analyse()`` when both
stage files exist. Use ``--all`` to (re)generate ``merge_stages_analysis.log`` under
every model×category folder under ``model/`` that has both stage files.

Examples
--------
  python analyze_merge_stages.py "model/anthropic-claude-haiku-4.5/Personal&Existential"
  python analyze_merge_stages.py --all
  python analyze_merge_stages.py --all --root model --use-probe
  python analyze_merge_stages.py -d "model/kimi-k2.6/Relational&Intimate" --print-only
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batch_llm_merge import (  # noqa: E402
    DEFAULT_EXCLUDE_DIR_NAMES,
    MODEL_ROOT,
    _path_touches_excluded,
)

from analyze_results import (  # noqa: E402
    build_merge_stages_report_text,
    collect_raw_names,
)


def _default_dir() -> Path:
    p = Path("model/anthropic-claude-haiku-4.5/Personal&Existential").resolve()
    return p if p.is_dir() else ROOT / "model"


def discover_category_dirs_with_merge_stages(
    root: Path,
    *,
    extra_exclude: list[str],
) -> list[Path]:
    """Directories that contain both ``merge_map_llm_stage1.json`` and stage2 (under ``root``)."""
    root = root.expanduser().resolve()
    exclude: frozenset[str] = frozenset(DEFAULT_EXCLUDE_DIR_NAMES) | frozenset(
        x.strip() for x in extra_exclude if x and x.strip()
    )
    seen: set[Path] = set()
    for s1 in root.rglob("merge_map_llm_stage1.json"):
        if _path_touches_excluded(s1, exclude):
            continue
        d = s1.parent
        if (d / "merge_map_llm_stage2.json").is_file():
            seen.add(d)
    return sorted(seen, key=lambda p: p.as_posix().lower())


def _process_one_dir(
    d: Path,
    *,
    print_only: bool,
    probe_path: Path | None,
    use_probe: bool,
) -> bool:
    """Process one category directory. Return True on success."""
    d = d.resolve()
    s1 = d / "merge_map_llm_stage1.json"
    s2 = d / "merge_map_llm_stage2.json"
    if not s1.is_file() or not s2.is_file():
        return False
    final = d / "merge_map.json"
    raw_counter: Counter | None = None
    if probe_path is not None and Path(probe_path).is_file():
        raw_counter = collect_raw_names(probe_path, name_source="step3")
    elif use_probe:
        pr = d / "probe_results.jsonl"
        if pr.is_file():
            raw_counter = collect_raw_names(pr, name_source="step3")

    text = build_merge_stages_report_text(
        s1,
        s2,
        final_map_path=final if final.is_file() else None,
        raw_counter=raw_counter,
    )
    if print_only:
        print(text, end="")
        return True
    out = d / "merge_stages_analysis.log"
    out.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "-d",
        "--dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Single directory with both stage JSONs (if not using --all).",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Scan --root (default: model/) for all folders with stage1+stage2 merge maps.",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=MODEL_ROOT,
        help="Root to scan with --all (default: model/ next to this script).",
    )
    ap.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        metavar="NAME",
        help="Skip paths with this path segment (repeatable). Merged with batch defaults.",
    )
    ap.add_argument(
        "--print-only",
        action="store_true",
        help="Print report to stdout only, no file (single -d only; not used with --all).",
    )
    ap.add_argument(
        "--probe",
        type=Path,
        default=None,
        metavar="PATH",
        help="Single-dir only: use this probe_results.jsonl for weighted counts.",
    )
    ap.add_argument(
        "--use-probe",
        action="store_true",
        help="For each directory, load probe_results.jsonl in the same folder if present "
        "(--all or single -d).",
    )
    args = ap.parse_args()

    if args.all:
        scan_root = args.root.expanduser().resolve()
        if not scan_root.is_dir():
            print(f"[error] --root is not a directory: {scan_root}", file=sys.stderr)
            return 2
        dirs = discover_category_dirs_with_merge_stages(
            scan_root, extra_exclude=list(args.exclude_dir or [])
        )
        if not dirs:
            print(
                f"[warn] no folders with merge_map_llm_stage1+stage2 under {scan_root}",
                file=sys.stderr,
            )
            return 1
        n_ok = 0
        n_err = 0
        for d in tqdm(dirs, desc="merge_stages", unit="dir"):
            try:
                if _process_one_dir(
                    d,
                    print_only=False,
                    probe_path=None,
                    use_probe=bool(args.use_probe),
                ):
                    n_ok += 1
                else:
                    n_err += 1
            except OSError as e:
                n_err += 1
                tqdm.write(f"[error] {d}: {e}", file=sys.stderr)
        print(
            f"[done] {n_ok} wrote merge_stages_analysis.log, {n_err} skipped/failed, "
            f"root={scan_root} (total {len(dirs)} dirs)"
        )
        return 0 if n_err == 0 else 3

    d = (args.dir or _default_dir()).expanduser().resolve()
    if not d.is_dir():
        print(f"[error] not a directory: {d}", file=sys.stderr)
        return 2
    if not (d / "merge_map_llm_stage1.json").is_file() or not (
        d / "merge_map_llm_stage2.json"
    ).is_file():
        print(
            f"[error] need merge_map_llm_stage1.json and merge_map_llm_stage2.json under {d}",
            file=sys.stderr,
        )
        return 2
    try:
        ok = _process_one_dir(
            d,
            print_only=bool(args.print_only),
            probe_path=args.probe,
            use_probe=bool(args.use_probe),
        )
    except OSError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    if not ok:
        return 2
    if not args.print_only:
        print(f"Wrote {d / 'merge_stages_analysis.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
