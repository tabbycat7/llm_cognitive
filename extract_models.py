"""Extract model names from step3_response JSONL and count frequencies.

Names are normalized in :func:`cognitive_probe.clean_step3_response` when the
pipeline runs; this script uses the same :func:`cognitive_probe.normalize_model_name`
so older files without that step still count consistently.

Usage:
    python extract_models.py probe_results.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from cognitive_probe import extract_json_from_markdown, normalize_model_name


def extract_model_names(step3_text: str) -> list[str]:
    """Parse step3_response and return list of cleaned model names."""
    json_str = extract_json_from_markdown(step3_text)
    if not json_str:
        return []
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return []

    names: list[str] = []
    models = data.get("models", [])
    for m in models:
        raw_name = m.get("name", "")
        if raw_name:
            names.append(normalize_model_name(raw_name))
    return names


def main(jsonl_path: str) -> None:
    path = Path(jsonl_path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    counter: Counter[str] = Counter()
    total_records = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            step3 = record.get("step3_response", "")
            names = extract_model_names(step3)
            counter.update(names)

    print(f"Processed {total_records} records, extracted {sum(counter.values())} model mentions.\n")
    print("Model/Theory Name Frequencies (descending):\n")
    for name, count in counter.most_common():
        print(f"  {count:3d}  {name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    main(sys.argv[1])
