"""Download Reddit datasets from Hugging Face into <data-root>/<subdir>/

gk4u/reddit_dataset_218 需要先登录（终端执行一次）:
    huggingface-cli login

磁盘：save_to_disk 用 --data-root 或 REDDIT_DS_ROOT 指到大盘。
load_dataset 的缓存默认多在用户目录；C 盘不够时可设 HF_HOME 到另一块盘。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent

# id -> 保存目录名
DATASETS: dict[str, tuple[str, str]] = {
    "44": ("quanglt/reddit_dataset_44", "reddit_dataset_44"),
    "218": ("gk4u/reddit_dataset_218", "reddit_dataset_218"),
}


def resolve_data_root(cli_path: str | None) -> Path:
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    env = os.environ.get("REDDIT_DS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return ROOT / "data"


def download_one(key: str, data_root: Path) -> None:
    dataset_id, subdir = DATASETS[key]
    out = data_root / subdir
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset_id)
    ds.save_to_disk(str(out))
    print(f"[{key}] {dataset_id} -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download HF Reddit datasets (default ./data; override with -o / REDDIT_DS_ROOT).",
    )
    parser.add_argument(
        "which",
        nargs="*",
        choices=list(DATASETS.keys()),
        metavar="KEY",
        help="44 and/or 218; omit to download both",
    )
    parser.add_argument(
        "--data-root",
        "-o",
        metavar="DIR",
        default=None,
        help="save_to_disk base directory",
    )
    args = parser.parse_args()
    data_root = resolve_data_root(args.data_root)
    keys = args.which if args.which else list(DATASETS.keys())
    for k in keys:
        download_one(k, data_root)


if __name__ == "__main__":
    main()
