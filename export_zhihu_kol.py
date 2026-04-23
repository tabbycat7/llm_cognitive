"""Export local wangrui6/Zhihu-KOL (save_to_disk folder) to CSV and/or JSONL.

Default input: ./data/zhihu_kol
Full train split is ~1M rows; use --max-rows to test or cap size.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import DatasetDict, load_from_disk

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "zhihu_kol"


def resolve_input_dir(cli_path: str | None) -> Path:
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    env = os.environ.get("DATASET_SAVE_ROOT")
    if env:
        return Path(env).expanduser().resolve() / "zhihu_kol"
    return DEFAULT_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Export zhihu_kol Arrow dataset to CSV / JSONL")
    parser.add_argument(
        "--input",
        "-i",
        metavar="DIR",
        default=None,
        help="load_from_disk path (default: ./data/zhihu_kol or DATASET_SAVE_ROOT/zhihu_kol)",
    )
    parser.add_argument(
        "--output-dir",
        "-O",
        metavar="DIR",
        default=None,
        help="where to write files (default: same directory as this script)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="split name (default: train)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=("csv", "jsonl", "both"),
        default="both",
        help="output format (default: both)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="export at most first N rows (recommended for a quick test)",
    )
    parser.add_argument(
        "--basename",
        default="zhihu_kol_train",
        help="output filename without extension (default: zhihu_kol_train)",
    )
    args = parser.parse_args()

    data_dir = resolve_input_dir(args.input)
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(str(data_dir))
    if isinstance(ds, DatasetDict):
        split = ds[args.split]
    else:
        split = ds

    if args.max_rows is not None:
        n = min(args.max_rows, len(split))
        split = split.select(range(n))

    stem = args.basename
    if args.format in ("csv", "both"):
        csv_path = out_dir / f"{stem}.csv"
        split.to_csv(str(csv_path))
        print(f"Wrote {csv_path} ({len(split)} rows)")
    if args.format in ("jsonl", "both"):
        jsonl_path = out_dir / f"{stem}.jsonl"
        split.to_json(str(jsonl_path), lines=True)
        print(f"Wrote {jsonl_path} ({len(split)} rows)")


if __name__ == "__main__":
    main()
