"""Download wangrui6/Zhihu-KOL from Hugging Face to <data-root>/zhihu_kol/

缓存走 Hugging Face 默认目录；若项目所在盘空间不够，用 -o 或 DATASET_SAVE_ROOT 把 save_to_disk 指到大盘。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
DATASET_ID = "wangrui6/Zhihu-KOL"
OUT_SUBDIR = "zhihu_kol"


def resolve_data_root(cli_path: str | None) -> Path:
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    env = os.environ.get("DATASET_SAVE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Download {DATASET_ID}")
    parser.add_argument(
        "--data-root",
        "-o",
        metavar="DIR",
        default=None,
        help="save_to_disk base directory (default: ./data or DATASET_SAVE_ROOT)",
    )
    args = parser.parse_args()
    data_root = resolve_data_root(args.data_root)
    out = data_root / OUT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_ID)
    ds.save_to_disk(str(out))
    print(f"{DATASET_ID} -> {out}")


if __name__ == "__main__":
    main()
