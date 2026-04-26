#!/usr/bin/env python3
"""
批量替换 model 下所有 merge_map.json 里 merge_map 条目的「值」（不改动键）。

在下方 REPLACEMENTS 中填写 (旧词, 新词) 列表；会按顺序依次对每条 value 做 str.replace。
默认仅改值；若也要在键上替换，将 REPLACE_IN_KEYS 设为 True。

用法:
  python replace_merge_map_values.py              # 直接写回文件
  python replace_merge_map_values.py --dry-run  # 只打印会改动的文件与条数
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 在这里填写要替换的内容：按顺序依次对每条「值」执行 str.replace(old, new)
# 示例:
#   REPLACEMENTS = [
#       ("old phrase", "new phrase"),
#       ("foo", "bar"),
#   ]
# ---------------------------------------------------------------------------
REPLACEMENTS: list[tuple[str, str]] = [
    # ("这里写要被替换掉的词或短语", "这里写替换后的内容"),
    ("principal agent problem", "principal agent theory"),
    ("bayesian belief updating","bayesian inference"),
    ("bayesian updating","bayesian inference"),
]

# 是否在「键」上也做同样的替换（一般保持 False，只改值）
REPLACE_IN_KEYS = False

# model 根目录（相对本脚本所在目录）
MODEL_ROOT = Path(__file__).resolve().parent / "model"

MERGE_MAP_FILENAME = "merge_map.json"
TOP_LEVEL_KEY = "merge_map"


def apply_replacements(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        if old == "":
            continue
        text = text.replace(old, new)
    return text


def process_file(path: Path, dry_run: bool) -> int:
    """返回改动的条目数（值或键与原来不同的条目数）。"""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)

    if TOP_LEVEL_KEY not in data or not isinstance(data[TOP_LEVEL_KEY], dict):
        raise ValueError(f"{path}: 缺少顶层键 {TOP_LEVEL_KEY!r} 或其类型不是对象")

    mm: dict[str, str] = data[TOP_LEVEL_KEY]
    changed_keys: list[str] = []
    new_mm: dict[str, str] = {}

    for k, v in mm.items():
        nk = apply_replacements(k, REPLACEMENTS) if REPLACE_IN_KEYS else k
        nv = apply_replacements(v, REPLACEMENTS) if isinstance(v, str) else v
        if not isinstance(nv, str):
            nv = v
        new_mm[nk] = nv
        if nk != k or nv != v:
            changed_keys.append(k)

    n_changed = len(changed_keys)

    if n_changed and not dry_run:
        # 写回时保持原有顶层其它键（若有）
        out = dict(data)
        out[TOP_LEVEL_KEY] = new_mm
        path.write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return n_changed


def main() -> int:
    parser = argparse.ArgumentParser(description="替换所有 merge_map.json 中 merge_map 的值")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不写入，只统计并列出会修改的文件",
    )
    args = parser.parse_args()

    reps = [(a, b) for a, b in REPLACEMENTS if a != ""]
    if not reps:
        print("错误: REPLACEMENTS 为空或只有空字符串占位，请至少填写一对 (旧词, 新词)。", file=sys.stderr)
        return 1

    if not MODEL_ROOT.is_dir():
        print(f"错误: 找不到 model 目录: {MODEL_ROOT}", file=sys.stderr)
        return 1

    paths = sorted(MODEL_ROOT.rglob(MERGE_MAP_FILENAME))
    if not paths:
        print(f"未找到任何 {MERGE_MAP_FILENAME}（根: {MODEL_ROOT}）")
        return 0

    total_files = 0
    total_entries = 0
    for p in paths:
        try:
            n = process_file(p, args.dry_run)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"跳过（解析失败）: {p}\n  {e}", file=sys.stderr)
            continue
        if n:
            total_files += 1
            total_entries += n
            rel = p.relative_to(MODEL_ROOT.parent)
            print(f"{'[dry-run] ' if args.dry_run else ''}{rel}: {n} 条")

    print(
        f"\n完成: {len(paths)} 个文件扫描, "
        f"{total_files} 个文件有改动, 共 {total_entries} 条条目值（或键）被替换。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
