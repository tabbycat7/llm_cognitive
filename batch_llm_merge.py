#!/usr/bin/env python3
"""Batch LLM synonym merge + charts + analysis_summary.log across model/category folders.

Scans the project (or ``--root``) for ``**/probe_results.jsonl``, then for each file runs
the same analysis block as ``run.py --analyse-only`` (``analyse`` → ``visualise`` →
``write_analysis_summary_log``).

LLM 归并使用 ``.env`` 中的 ``OPENAI_API_MERGE_KEY`` / ``OPENAI_BASE_URL_MERGE`` /
``LLM_MODEL_MERGE``（与探测阶段解耦），见 ``analyze_results.merge_synonyms_via_llm``。

``--merge-method embedding_llm_llm`` 或 ``embedding_llm_llm_stage2`` 时，第二阶段归并可单独用 ``.env`` 指定（不设则与上面相同）：
``LLM_MODEL_MERGE_STAGE2``、``OPENAI_BASE_URL_MERGE_STAGE2``、``OPENAI_API_MERGE_STAGE2_KEY``。

Examples
--------
  # 默认：项目根下所有符合条件的 probe_results.jsonl，顺序执行 LLM 归并
  python batch_llm_merge.py

  # 只处理若干模型目录
  python batch_llm_merge.py --models deepseek kimi-k2.6

  # 只处理若干分类（可写 taxonomy 方括号形式，或磁盘上的子目录名）
  python batch_llm_merge.py --categories "[Personal & Existential]" Relational&Intimate

  # 强制忽略缓存重新聚类
  python batch_llm_merge.py --refresh-merge

  # 只列出将要处理的文件
  python batch_llm_merge.py --dry-run

  # 批处理结束后，额外写一份汇总 JSON（每个模型×分类下前 50 个名称及频次）
  python batch_llm_merge.py --refresh-merge -j 28 --export-models-summary-json merged_top50.json --export-models-summary-phase after_merge 

python batch_llm_merge.py  --export-models-summary-json merged_top50.json --export-models-summary-phase after_merge

  # 汇总「归并前」原始标签频次
  python batch_llm_merge.py --export-models-summary-json raw_top50.json \\
      --export-models-summary-phase before_merge

  # 不归并，只写汇总 JSON（不画图、不写 analysis_summary.log）
  python batch_llm_merge.py --merge-method none \\
      --export-models-summary-json merged_top50.json \\
      --export-models-summary-only

  # 默认安静：每任务一行；需要与 run.py 相同的表格/归并行可加 --verbose
  python batch_llm_merge.py --verbose

  # 新模型目录下出现新的 probe_results.jsonl 时：再次运行即可被自动扫描到。
  # 若只想处理新模型目录、省时间：--models 新模型文件夹名
  # 若汇总 JSON 要在旧文件上「追加/覆盖本次任务」而不是整文件重写：
  python batch_llm_merge.py --models 新模型名 --export-models-summary-json merged_top50.json \\
      --export-models-summary-merge

  # 混合归并：先 embedding 聚类，再并发调用 LLM 为每个簇命名
  python batch_llm_merge.py --merge-method embedding_llm --refresh-merge \\
      --llm-naming-workers 12

  python batch_llm_merge.py --embedding-threshold 0.35 --merge-method embedding_llm --refresh-merge -j 28 --llm-naming-workers 12

  # 三阶段归并：embedding 聚类 + LLM 簇命名 + LLM 二次归并（进一步合并 canonical 名称）
  python batch_llm_merge.py --merge-method embedding_llm_llm --refresh-merge \\
      --llm-naming-workers 12

  # 只重跑 Stage 2（读已有 merge_map_llm_stage1.json，调用 LLM 二次归并，写 merge_map_llm_stage2.json + merge_map.json）
  python batch_llm_merge.py --merge-method embedding_llm_llm_stage2 --refresh-merge -j 4

  # 只用 Stage 1 作为最终归并（读 merge_map_llm_stage1.json 写入 merge_map.json，不调 Stage-2 LLM）
  python batch_llm_merge.py --merge-method embedding_llm_llm_stage1_final -j 4
  # 同上，短名：
  python batch_llm_merge.py --merge-method embedding_llm_stage1_final -j 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "model"
_ENV_FILE = ROOT / ".env"
if _ENV_FILE.is_file():
    load_dotenv(_ENV_FILE, override=True)

from analyze_results import analyse, print_raw_names, visualise, write_analysis_summary_log
from cognitive_probe import category_to_probe_output_subdir


# Long name in ``analyze_results.analyse``; CLI may accept a shorter alias (see ``main``).
MERGE_METHOD_ALIASES: dict[str, str] = {
    # 与 ``embedding_llm_llm_stage1_final`` 相同：仅用 merge_map_llm_stage1.json 写最终 merge_map.json
    "embedding_llm_stage1_final": "embedding_llm_llm_stage1_final",
}

DEFAULT_EXCLUDE_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".history",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "corpus-webis-tldr-17",
        "reasoning_embedding_2d",
        "reasoning_embedding_2d_batch",
        "figures_top_attractors_powerlaw",
        "data",
    }
)


def _path_touches_excluded(path: Path, exclude: frozenset[str]) -> bool:
    return any(part in exclude for part in path.parts)


def _normalize_category_filters(tokens: list[str]) -> frozenset[str]:
    """Map CLI tokens to possible on-disk category directory names."""
    out: set[str] = set()
    for raw in tokens:
        s = raw.strip()
        if not s:
            continue
        out.add(s)
        mapped = category_to_probe_output_subdir(s)
        out.add(mapped)
    return frozenset(out)


def discover_probe_jsonl(
    root: Path,
    *,
    models: frozenset[str] | None,
    categories: frozenset[str] | None,
    include_root_file: bool,
    exclude_dir_names: frozenset[str],
) -> list[Path]:
    """Sorted list of probe_results.jsonl paths under root."""
    root = root.resolve()
    hits: list[Path] = []
    for p in root.rglob("probe_results.jsonl"):
        if not p.is_file():
            continue
        if _path_touches_excluded(p, exclude_dir_names):
            continue
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) == 1:
            if not include_root_file:
                continue
            # treat as anonymous single-level job; no model/category filter
            if models or categories:
                continue
        else:
            model_dir = parts[0]
            if models is not None and model_dir not in models:
                continue
            if categories is not None:
                # Expect <model>/<category>/probe_results.jsonl
                if len(parts) < 3:
                    continue
                if parts[1] not in categories:
                    continue
        hits.append(p)
    hits.sort(key=lambda x: (x.as_posix().lower()))
    return hits


def _probe_rel(probe_jsonl: Path, root: Path) -> str:
    try:
        return str(probe_jsonl.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(probe_jsonl.resolve())


def _parse_model_category(probe_jsonl: Path, root: Path) -> tuple[str, str]:
    """First path segment = model dir; second = category dir when layout is nested."""
    root = root.resolve()
    probe_jsonl = probe_jsonl.resolve()
    rel = probe_jsonl.relative_to(root)
    parts = rel.parts
    if len(parts) >= 3:
        return parts[0], parts[1]
    if len(parts) == 2:
        return parts[0], ""
    return "", ""


def _counter_top_k(counter: Counter, k: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for rank, (name, count) in enumerate(counter.most_common(k), 1):
        out.append({"rank": rank, "name": name, "count": int(count)})
    return out


def _merge_export_items(
    previous_items: list[object],
    new_items: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Merge by ``probe_results`` path: this run overwrites the same key; others kept."""
    by_key: dict[str, dict[str, object]] = {}
    for r in previous_items:
        if not isinstance(r, dict):
            continue
        k = str(r.get("probe_results", "")).strip()
        if k:
            by_key[k] = dict(r)  # type: ignore[arg-type]
    for r in new_items:
        k = str(r.get("probe_results", "")).strip()
        if k:
            by_key[k] = r
    return sorted(
        by_key.values(),
        key=lambda x: (
            str(x.get("model", "")).lower(),
            str(x.get("category", "")).lower(),
            str(x.get("probe_results", "")),
        ),
    )


def write_models_summary_json(
    *,
    out_path: Path,
    root: Path,
    phase: str,
    top_k: int,
    merge_method: str,
    name_source: str,
    refresh_merge: bool,
    rows: list[dict[str, object]],
    partial: bool,
    failed_paths: list[str],
    merged_with_existing_file: bool = False,
) -> Path:
    """Write aggregated top-K name frequencies per model×category (UTF-8 JSON)."""
    payload: dict[str, object] = {
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "root": str(root.resolve()),
        "export_phase": phase,
        "phase_description_zh": (
            "归并后（canonical 代表名频次）"
            if phase == "after_merge"
            else "归并前（原始标签频次，未应用 merge_map）"
        ),
        "top_k": top_k,
        "merge_method": merge_method,
        "name_source": name_source,
        "refresh_merge": refresh_merge,
        "partial": partial,
        "failed_probe_paths": failed_paths,
        "items": rows,
        "merged_with_existing_file": merged_with_existing_file,
    }
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out_path


def _run_one(
    output_path: Path,
    *,
    merge_method: str,
    refresh_merge: bool,
    top_n: int,
    name_source: str,
    embedding_backend: str,
    embedding_model: str,
    embedding_threshold: float,
    embedding_device: str,
    llm_naming_workers: int,
    quiet: bool,
) -> tuple[Counter, Counter]:
    cache_path = output_path.parent / "merge_map.json"
    merge_backend = None  # LLM merge uses env merge client; embedding ignores this too
    canonical_counts, raw_counts = analyse(
        output_path,
        backend=merge_backend,
        cache_path=cache_path,
        refresh_merge=refresh_merge,
        merge_method=merge_method,
        embedding_threshold=embedding_threshold,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        name_source=name_source,
        llm_naming_workers=llm_naming_workers,
    )
    print_raw_names(raw_counts, quiet=quiet)
    is_merged = merge_method != "none"
    visualise(
        canonical_counts,
        output_dir=output_path.parent,
        top_n=top_n,
        merged=is_merged,
        console=not quiet,
    )
    write_analysis_summary_log(
        results_path=output_path,
        output_dir=output_path.parent,
        canonical_counter=canonical_counts,
        raw_counter=raw_counts,
        merge_method=merge_method,
        top_n=top_n,
        refresh_merge=refresh_merge,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_threshold=embedding_threshold,
        embedding_device=embedding_device,
        name_source=name_source,
        print_path=not quiet,
    )
    return canonical_counts, raw_counts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch synonym merge + visualisation for many probe_results.jsonl trees.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--root",
        type=Path,
        default=MODEL_ROOT,
        help="Root directory to scan for probe_results.jsonl (default: model/ next to this script).",
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        metavar="DIR",
        help="Only first path segment under --root (e.g. deepseek). Default: all models.",
    )
    p.add_argument(
        "--categories",
        nargs="*",
        default=None,
        metavar="CAT",
        help="Category subdir names or taxonomy strings like '[Societal & Ethical]'. "
        "Default: all categories.",
    )
    p.add_argument(
        "--include-root-file",
        action="store_true",
        help="Also process <root>/probe_results.jsonl if present (ignored when --models/--categories set).",
    )
    p.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        metavar="NAME",
        help=f"Skip paths containing this directory name (repeatable). "
        f"Built-in defaults: {', '.join(sorted(DEFAULT_EXCLUDE_DIR_NAMES))}.",
    )
    p.add_argument(
        "--merge-method",
        choices=[
            "llm",
            "embedding",
            "embedding_llm",
            "embedding_llm_llm",
            "embedding_llm_llm_stage2",
            "embedding_llm_llm_stage1_final",
            "embedding_llm_stage1_final",
            "none",
        ],
        default="llm",
        help="Same as run.py (default: llm). "
        "embedding_llm = embedding clustering + concurrent LLM naming per cluster. "
        "embedding_llm_llm = embedding_llm then a second LLM pass to further merge canonicals. "
        "embedding_llm_llm_stage2 = skip embedding/Stage1; load merge_map_llm_stage1.json "
        "and run Stage 2 + write merge_map.json only (needs existing Stage 1 file). "
        "embedding_llm_llm_stage1_final (alias: embedding_llm_stage1_final) = write merge_map.json "
        "from merge_map_llm_stage1.json only (no Stage-2 LLM, no merge API).",
    )
    p.add_argument("--refresh-merge", action="store_true", help="Ignore merge_map.json cache.")
    p.add_argument("--top-n", type=int, default=30, help="Chart / log top-N (default: 30).")
    p.add_argument(
        "--name-source",
        choices=["step1", "step2", "step3", "step4"],
        default="step3",
        help="Same as run.py (default: step3).",
    )
    p.add_argument(
        "--embedding-backend",
        choices=["local", "ollama", "openai"],
        default=os.getenv("EMBEDDING_BACKEND", "local"),
        help="Embedding backend (default: env EMBEDDING_BACKEND or 'local').",
    )
    p.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        help="Embedding model (default: env EMBEDDING_MODEL or 'BAAI/bge-small-zh-v1.5').",
    )
    p.add_argument(
        "--embedding-device",
        default=os.getenv("EMBEDDING_DEVICE", "auto"),
    )
    p.add_argument("--embedding-threshold", type=float, default=0.3)
    p.add_argument(
        "--llm-naming-workers",
        type=int,
        default=8,
        metavar="N",
        help="Max concurrent LLM calls for cluster naming in embedding_llm mode (default: 8).",
    )
    p.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Concurrent jobs (default: 1). LLM merge may rate-limit; try 2–3 with care.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths only, do not call APIs or write outputs.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose console: per-job banners, raw-name table, freq chart text, merge line, "
        "summary path; disables batch quiet + tqdm main progress bar (default: quiet with tqdm).",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Log errors and continue (default: true).",
    )
    p.add_argument(
        "--no-continue-on-error",
        action="store_false",
        dest="continue_on_error",
        help="Stop on first failure.",
    )
    p.add_argument(
        "--export-models-summary-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="After the batch, write one UTF-8 JSON file: per model×category, top-K "
        "数理模型名称及频次（由 --export-models-summary-phase 选择归并后或归并前）。",
    )
    p.add_argument(
        "--export-models-summary-phase",
        choices=["after_merge", "before_merge"],
        default="after_merge",
        help="after_merge=归并后 canonical；before_merge=归并前 raw（默认: after_merge）。",
    )
    p.add_argument(
        "--export-models-summary-top-k",
        type=int,
        default=50,
        metavar="K",
        help="Each model×category keeps top K names by count (default: 50).",
    )
    p.add_argument(
        "--export-models-summary-merge",
        action="store_true",
        help="With --export-models-summary-json: if the file already exists, merge items: "
        "keep rows for other probe_results paths, add/update rows for this run (key=probe_results).",
    )
    p.add_argument(
        "--export-models-summary-only",
        action="store_true",
        help="With --export-models-summary-json: only run analyse() for frequency counters and write "
        "the summary JSON; skip bar chart, word cloud, and analysis_summary.log (faster). "
        "Use --merge-method none for raw labels only; other merge methods still apply merge_map.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.merge_method in MERGE_METHOD_ALIASES:
        args.merge_method = MERGE_METHOD_ALIASES[str(args.merge_method)]
    root: Path = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"[error] --root is not a directory: {root}", file=sys.stderr)
        return 2

    exclude = frozenset(DEFAULT_EXCLUDE_DIR_NAMES) | frozenset(
        x.strip() for x in (args.exclude_dir or []) if x and x.strip()
    )
    models = frozenset(m.strip() for m in args.models) if args.models else None
    categories = (
        _normalize_category_filters(list(args.categories)) if args.categories else None
    )

    jobs = discover_probe_jsonl(
        root,
        models=models,
        categories=categories,
        include_root_file=args.include_root_file,
        exclude_dir_names=exclude,
    )

    if not jobs:
        print(f"[warn] No probe_results.jsonl found under {root} with current filters.")
        return 1

    print(f"[batch] root={root}")
    print(f"[batch] jobs={len(jobs)}  merge_method={args.merge_method}  refresh_merge={args.refresh_merge}")
    export_path: Path | None = args.export_models_summary_json
    if args.dry_run:
        if export_path is not None:
            print(
                "[warn] --export-models-summary-json is ignored with --dry-run.",
                file=sys.stderr,
            )
        for p in jobs:
            print(p.as_posix())
        return 0

    if args.merge_method in (
        "llm",
        "embedding_llm",
        "embedding_llm_llm",
        "embedding_llm_llm_stage2",
        "embedding_llm_llm_stage1_final",
    ) and args.name_source in ("step1", "step2"):
        print(
            f"[error] merge_method={args.merge_method} does not support name_source step1/step2. "
            "Use embedding or none, or name_source step3.",
            file=sys.stderr,
        )
        return 2

    quiet = not args.verbose
    _batch_quiet_env_set = False
    if quiet:
        os.environ["COGNITIVE_BATCH_QUIET"] = "1"
        _batch_quiet_env_set = True

    failed: list[tuple[Path, str]] = []
    summary_rows: list[dict[str, object]] = []
    export_phase = args.export_models_summary_phase
    top_k_export = max(1, int(args.export_models_summary_top_k))

    def build_summary_row(
        probe_path: Path, canonical: Counter, raw: Counter
    ) -> dict[str, object]:
        ctr = canonical if export_phase == "after_merge" else raw
        model, cat = _parse_model_category(probe_path, root)
        return {
            "model": model,
            "category": cat,
            "probe_results": str(probe_path.resolve()),
            "unique_names": len(ctr),
            "total_mentions": int(sum(ctr.values())),
            "top_names": _counter_top_k(ctr, top_k_export),
        }

    def work(p: Path) -> tuple[Path, str, tuple[Counter, Counter] | None]:
        t0 = time.perf_counter()
        try:
            canonical_counts, raw_counts = _run_one(
                p,
                merge_method=args.merge_method,
                refresh_merge=args.refresh_merge,
                top_n=args.top_n,
                name_source=args.name_source,
                embedding_backend=args.embedding_backend,
                embedding_model=args.embedding_model,
                embedding_threshold=args.embedding_threshold,
                embedding_device=args.embedding_device,
                llm_naming_workers=args.llm_naming_workers,
                quiet=quiet,
            )
            dt = time.perf_counter() - t0
            return p, f"ok ({dt:.1f}s)", (canonical_counts, raw_counts)
        except Exception as e:  # noqa: BLE001 — batch driver surfaces per-file errors
            return p, f"error: {e}", None

    exit_code = 0
    try:
        if args.jobs <= 1:
            job_iter = jobs
            pbar: tqdm | None
            if quiet:
                pbar = tqdm(
                    job_iter,
                    total=len(jobs),
                    desc="[batch]",
                    unit="file",
                    smoothing=0.05,
                )
            else:
                pbar = None
            for i, path in enumerate(pbar or job_iter, 1):
                if pbar is not None:
                    _rel = str(_probe_rel(path, root))
                    pbar.set_postfix_str(
                        _rel[:52] + ("…" if len(_rel) > 52 else ""),
                        refresh=False,
                    )
                if not quiet:
                    print(f"\n{'=' * 70}\n[{i}/{len(jobs)}] {path}\n{'=' * 70}", flush=True)
                rpath, msg, counts = work(path)
                if not quiet:
                    print(msg, flush=True)
                if counts is not None and export_path is not None:
                    summary_rows.append(
                        build_summary_row(rpath, counts[0], counts[1])
                    )
                if not str(msg).startswith("ok"):
                    failed.append((rpath, msg))
                    if quiet and pbar is not None:
                        tqdm.write(f"[FAIL] {_probe_rel(rpath, root)}  {msg}", file=sys.stderr)
                    if not args.continue_on_error:
                        if quiet and pbar is not None:
                            tqdm.write(
                                "[batch] stopped (--no-continue-on-error).",
                                file=sys.stderr,
                            )
                        else:
                            print(
                                "[batch] stopped (--no-continue-on-error).",
                                file=sys.stderr,
                            )
                        exit_code = 3
                        break
        else:
            print(f"[batch] parallel workers: {args.jobs}", flush=True)
            with ThreadPoolExecutor(max_workers=args.jobs) as ex:
                futs = {ex.submit(work, p): p for p in jobs}
                pbar2: tqdm | None
                if quiet:
                    pbar2 = tqdm(
                        as_completed(futs),
                        total=len(futs),
                        desc="[batch]",
                        unit="job",
                        smoothing=0.05,
                    )
                else:
                    pbar2 = None
                done_iter = pbar2 if pbar2 is not None else as_completed(futs)
                for fut in done_iter:
                    path, msg, counts = fut.result()
                    ok = str(msg).startswith("ok")
                    rel = _probe_rel(path, root)
                    if pbar2 is not None:
                        pbar2.set_postfix_str(
                            f"{rel[:40]}{'…' if len(rel) > 40 else ''} | {msg[:24]}",
                            refresh=False,
                        )
                    else:
                        flag = "OK" if ok else "FAIL"
                        print(f"[{flag}] {path}: {msg}", flush=True)
                    if counts is not None and export_path is not None:
                        summary_rows.append(
                            build_summary_row(path, counts[0], counts[1])
                        )
                    if not ok:
                        failed.append((path, msg))
                        if quiet and pbar2 is not None:
                            tqdm.write(f"[FAIL] {path}: {msg}", file=sys.stderr)
                        if not args.continue_on_error:
                            ex.shutdown(wait=False, cancel_futures=True)
                            if quiet and pbar2 is not None:
                                tqdm.write(
                                    "[batch] stopped (--no-continue-on-error).",
                                    file=sys.stderr,
                                )
                            else:
                                print(
                                    "[batch] stopped (--no-continue-on-error).",
                                    file=sys.stderr,
                                )
                            exit_code = 3
                            break
    finally:
        if _batch_quiet_env_set:
            os.environ.pop("COGNITIVE_BATCH_QUIET", None)

    if export_path is not None:
        merged_from_disk = False
        final_rows = list(summary_rows)
        if args.export_models_summary_merge and export_path.is_file():
            try:
                with open(export_path, encoding="utf-8") as f:
                    old = json.load(f)
                old_items = old.get("items", [])
                old_phase = old.get("export_phase")
                if old_phase is not None and str(old_phase) != export_phase:
                    print(
                        f"[warn] existing JSON export_phase={old_phase!r} != this run {export_phase!r}; "
                        f"merging item lists anyway — verify results.",
                        file=sys.stderr,
                    )
                if isinstance(old_items, list):
                    before_n = len(old_items)
                    final_rows = _merge_export_items(old_items, summary_rows)
                    merged_from_disk = True
                    print(
                        f"[batch] export merge: {before_n} existing + {len(summary_rows)} this run "
                        f"→ {len(final_rows)} total items",
                        flush=True,
                    )
            except (json.JSONDecodeError, OSError) as e:
                print(
                    f"[warn] could not read existing export for merge, writing this run only: {e}",
                    file=sys.stderr,
                )
        final_rows.sort(
            key=lambda r: (
                str(r["model"]).lower(),
                str(r["category"]).lower(),
                str(r["probe_results"]),
            )
        )
        written = write_models_summary_json(
            out_path=export_path,
            root=root,
            phase=export_phase,
            top_k=top_k_export,
            merge_method=args.merge_method,
            name_source=args.name_source,
            refresh_merge=args.refresh_merge,
            rows=final_rows,
            partial=bool(failed),
            failed_paths=[str(p.resolve()) for p, _ in failed],
            merged_with_existing_file=merged_from_disk,
        )
        print(f"\n[batch] models summary JSON written to: {written}", flush=True)

    if failed:
        note = "（已提前停止）" if exit_code == 3 else ""
        print(
            f"\n[batch] completed with {len(failed)} failure(s){note}:",
            file=sys.stderr,
        )
        for path, msg in failed:
            print(f"  - {path}: {msg}", file=sys.stderr)
        return 3
    print("\n[batch] all jobs finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
