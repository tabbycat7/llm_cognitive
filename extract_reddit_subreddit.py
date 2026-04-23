"""从本地 save_to_disk 的 reddit_dataset_44 中按子版块提取行（流式，省内存）。

默认提取 ``r/Advice``（与 ``communityName`` 字段精确匹配，不会误含 ``r/AdviceAnimals``）。

示例::

    python extract_reddit_subreddit.py -o data/reddit_r_advice.jsonl
    python extract_reddit_subreddit.py --subreddit r/relationship_advice -o data/rel.jsonl
    python extract_reddit_subreddit.py -o data/sample.jsonl --max-rows 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "data" / "reddit_dataset_44"


def iter_arrow_batches(arrow_path: Path):
    with pa.memory_map(str(arrow_path), "r") as src:
        reader = ipc.open_stream(src)
        while True:
            try:
                yield reader.read_next_batch()
            except StopIteration:
                break


def extract_subreddit(
    train_dir: Path,
    subreddit: str,
    output_path: Path,
    max_rows: int | None = None,
) -> int:
    """Stream all ``data-*.arrow`` shards, keep rows where ``communityName == subreddit``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    arrow_files = sorted(train_dir.glob("data-*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"no data-*.arrow under {train_dir}")

    with open(output_path, "w", encoding="utf-8") as fout:
        for ap in arrow_files:
            for batch in iter_arrow_batches(ap):
                names = batch.column("communityName")
                mask = pc.equal(names, subreddit)
                if pc.sum(mask).as_py() == 0:
                    continue
                filtered = batch.filter(mask)
                n = filtered.num_rows
                col_names = filtered.schema.names
                arrays = [filtered.column(c) for c in col_names]
                for row_idx in range(n):
                    obj = {c: arrays[i][row_idx].as_py() for i, c in enumerate(col_names)}
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    written += 1
                    if max_rows is not None and written >= max_rows:
                        print(f"[cap] wrote {written} rows (max-rows={max_rows})")
                        return written
            print(f"[shard] {ap.name} … total written so far: {written}", flush=True)

    print(f"[done] wrote {written} rows -> {output_path}")
    return written


def main() -> None:
    p = argparse.ArgumentParser(
        description="Filter reddit_dataset_44 (local Arrow shards) by communityName."
    )
    p.add_argument(
        "--dataset",
        "-d",
        type=Path,
        default=DEFAULT_DATASET,
        help="save_to_disk root (default: ./data/reddit_dataset_44)",
    )
    p.add_argument(
        "--split",
        default="train",
        help="split subfolder (default: train)",
    )
    p.add_argument(
        "--subreddit",
        "-s",
        default="r/Advice",
        help="exact communityName to keep (default: r/Advice)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        metavar="FILE",
        help="output JSONL path",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="stop after writing N rows (for testing)",
    )
    args = p.parse_args()

    train_dir = (args.dataset / args.split).resolve()
    if not train_dir.is_dir():
        raise SystemExit(f"split directory not found: {train_dir}")

    extract_subreddit(train_dir, args.subreddit, args.output.expanduser().resolve(), args.max_rows)


if __name__ == "__main__":
    main()
