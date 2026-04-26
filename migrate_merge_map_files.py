"""Migrate merge_map files to the new three-file naming convention.

Before (old layout):
  merge_map.json              ← Stage 1 (embedding + LLM naming)
  merge_map.embeddings.json   ← Embeddings cache for Stage 1
  merge_map_llm_stage2.json   ← Stage 2 (LLM re-merge)

After (new layout):
  merge_map_llm_stage1.json              ← Stage 1
  merge_map_llm_stage1.embeddings.json   ← Embeddings cache for Stage 1
  merge_map_llm_stage2.json              ← Stage 2 (unchanged)
  merge_map.json                         ← Final combined (raw → final canonical)

Usage:
  python migrate_merge_map_files.py               # preview (dry-run)
  python migrate_merge_map_files.py --execute      # actually do it
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_results import normalize_llm_merge_label

MODEL_ROOT = Path(__file__).resolve().parent / "model"


def _compose_final_map(
    stage1_map: dict[str, str],
    stage2_map: dict[str, str],
) -> dict[str, str]:
    """Replicate the composition logic from analyse() in analyze_results.py."""
    final: dict[str, str] = {}
    for raw_name, s1_canonical in stage1_map.items():
        c = normalize_llm_merge_label(str(s1_canonical))
        s1_canonical = c if c else (raw_name.strip() or raw_name)
        final[raw_name] = stage2_map.get(s1_canonical, s1_canonical)
    return final


def migrate_folder(folder: Path, dry_run: bool) -> bool:
    stage1_old = folder / "merge_map.json"
    stage1_new = folder / "merge_map_llm_stage1.json"
    embed_old = folder / "merge_map.embeddings.json"
    embed_new = folder / "merge_map_llm_stage1.embeddings.json"
    stage2 = folder / "merge_map_llm_stage2.json"
    final_path = folder / "merge_map.json"

    if not stage1_old.exists() or not stage2.exists():
        return False

    if stage1_new.exists():
        print(f"  [skip] {folder.relative_to(MODEL_ROOT)}: merge_map_llm_stage1.json already exists")
        return False

    with open(stage1_old, "r", encoding="utf-8") as f:
        stage1_data = json.load(f)
    stage1_map: dict = stage1_data.get("merge_map", {})

    with open(stage2, "r", encoding="utf-8") as f:
        stage2_data = json.load(f)
    stage2_map: dict = stage2_data.get("merge_map", {})

    final_map = _compose_final_map(stage1_map, stage2_map)
    n_canonical = len(set(final_map.values()))
    rel = folder.relative_to(MODEL_ROOT)

    if dry_run:
        print(f"  [dry-run] {rel}:")
        print(f"    rename  merge_map.json → merge_map_llm_stage1.json")
        if embed_old.exists():
            print(f"    rename  merge_map.embeddings.json → merge_map_llm_stage1.embeddings.json")
        print(f"    create  merge_map.json  (final combined: {len(final_map)} raw → {n_canonical} canonical)")
        return True

    stage1_old.rename(stage1_new)
    print(f"  [rename] {rel}/merge_map.json → merge_map_llm_stage1.json")

    if embed_old.exists():
        embed_old.rename(embed_new)
        print(f"  [rename] {rel}/merge_map.embeddings.json → merge_map_llm_stage1.embeddings.json")

    final_payload = {
        "merge_map": final_map,
        "method": "embedding_llm_llm",
        "stage1_file": "merge_map_llm_stage1.json",
        "stage2_file": "merge_map_llm_stage2.json",
    }
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)
    print(f"  [create] {rel}/merge_map.json  ({len(final_map)} raw → {n_canonical} canonical)")
    return True


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", type=Path, default=MODEL_ROOT,
                    help="Root model directory (default: model/ next to this script).")
    p.add_argument("--execute", action="store_true",
                    help="Actually perform the migration (default: dry-run preview).")
    args = p.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"Root not found: {root}", file=sys.stderr)
        return 1

    dry_run = not args.execute
    if dry_run:
        print("=== DRY RUN (pass --execute to apply changes) ===\n")

    stage2_files = sorted(root.rglob("merge_map_llm_stage2.json"))
    print(f"Found {len(stage2_files)} folders with merge_map_llm_stage2.json under {root}\n")

    n_migrated = 0
    for s2 in stage2_files:
        if migrate_folder(s2.parent, dry_run=dry_run):
            n_migrated += 1

    verb = "Would migrate" if dry_run else "Migrated"
    print(f"\n{verb}: {n_migrated} folder(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
