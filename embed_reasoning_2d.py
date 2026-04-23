"""Embed all reasoning fields in a JSONL file and project to 2D for plotting.

Supports reasoning fields from:
1) Native JSON keys named "reasoning" in each JSONL record.
2) Nested JSON blocks embedded in step response strings (for example step3_response).
3) The full text of step2_response (--source-scope step2_full), without parsing JSON.
4) String values of key "explanation" inside step3_response JSON (--source-scope step3_explanation).
5) Use ``--color-by merge`` to color points by canonical names in ``merge_map.json`` (same
   as analysis; default path: directory of ``--input`` / merge_map.json).
6) ``--batch``: scan ``--root`` for ``probe_results.jsonl``. By default each job writes
   ``reasoning_embeddings_2d*.csv/.png`` **next to that JSONL** (same folder as analysis).
   The combined overview image (if ``--mode`` includes combined) goes under ``--output-root``.
   Use ``--batch-output-subdirs`` to instead mirror the old layout ``--output-root/<model>/<category>/``.
   ``-j / --batch-workers N`` parallelizes **openai** and **ollama** batch jobs
   (thread pool); **local** keeps a single shared encoder (N ignored).
7) ``--mode combined|separate|both`` in batch mode:
    combined = one overview figure, separate = one figure per job, both = export both.

python embed_reasoning_2d.py --input x/probe_results.jsonl --from-embeddings prev_run/
reasoning_embeddings_2d_points.csv

python embed_reasoning_2d.py --batch -j 7 --mode combined

Many Hub models (e.g. BAAI/bge-base-zh-v1.5) ship only pytorch_model.bin. Recent
transformers releases require PyTorch 2.6+ to load those weights; upgrade torch or
use a model that provides model.safetensors.

``--embedding-backend openai|ollama`` reuses the same clients as `analyze_results`
(OpenAI ``embeddings.create`` or Ollama ``/api/embeddings``; keys/URLs from
``--embedding-api-key`` / ``--embedding-base-url`` or `EMBEDDING_*` / `OPENAI_*` env).

The output CSV (``reasoning_embeddings_2d_points.csv``) saves ``x``/``y`` 2D coordinates
plus metadata per row. High-dimensional embedding vectors are **not** stored in the CSV.

``--plot-only-2d`` rereads a prior 2D CSV, matches rows to ``--input``/``--source-scope``,
and saves **PNG + 2D-only CSV (no high-dim column)** in the same folder as ``probe_results.jsonl``.

In `.env` you can set `EMBEDDING_MODEL` (e.g. `BAAI/bge-small-zh-v1.5` or
`text-embedding-3-small`); it is used when you omit ``--model``. Optional:
`EMBEDDING_BACKEND` = `local` | `openai` | `ollama` (used when you omit
``--embedding-backend``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from textwrap import fill

import matplotlib
from tqdm.auto import tqdm

from batch_llm_merge import (
    DEFAULT_EXCLUDE_DIR_NAMES,
    _parse_model_category,
    _normalize_category_filters,
    discover_probe_jsonl,
)

from analyze_results import _get_embeddings, normalize_llm_merge_label

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


STEP_TEXT_FIELDS = ("step1_response", "step2_response", "step3_response", "step4_response")
_EMBED_BACKENDS = frozenset({"local", "openai", "ollama"})


def _tqdm_ncols() -> int:
    try:
        w = shutil.get_terminal_size(fallback=(96, 24)).columns
    except OSError:
        return 88
    return max(52, min(w - 2, 100))


def _tqdm_path_hint(path_s: str | Path, max_chars: int = 64) -> str:
    s = " ".join(str(path_s).split())
    if len(s) <= max_chars:
        return s or "—"
    return s[: max_chars - 1] + "…"


def _load_project_dotenv() -> None:
    p = Path(__file__).resolve().parent / ".env"
    if not p.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(p, override=True)


def _apply_embedding_env_defaults(args: argparse.Namespace) -> None:
    """If ``--model`` / ``--embedding-backend`` not on the CLI, allow .env to override arg defaults."""
    if "--model" not in sys.argv:
        m = (os.getenv("EMBEDDING_MODEL") or "").strip()
        if m:
            args.model = m
    if "--embedding-backend" not in sys.argv:
        be = (os.getenv("EMBEDDING_BACKEND") or "").strip().lower()
        if be in _EMBED_BACKENDS:
            args.embedding_backend = be


def l2_normalize_rows(emb: np.ndarray) -> np.ndarray:
    """Row-wise L2 norm to unit length (match SentenceTransformer normalize_embeddings for cosine)."""
    out = np.asarray(emb, dtype=np.float64)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return out / norms


def extract_json_block(text: str) -> str | None:
    """Extract JSON body from plain text or markdown fenced text."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    rough = re.search(r"\{.*\}", text, re.DOTALL)
    if rough:
        return rough.group(0).strip()
    return None


def collect_string_field(obj: object, field: str, prefix: str) -> list[tuple[str, str]]:
    """Recursively collect all key == `field` string values with path info."""
    rows: list[tuple[str, str]] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            if key == field and isinstance(value, str):
                text = value.strip()
                if text:
                    rows.append((next_prefix, text))
            rows.extend(collect_string_field(value, field, next_prefix))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]"
            rows.extend(collect_string_field(value, field, next_prefix))

    return rows


def collect_reasonings(obj: object, prefix: str) -> list[tuple[str, str]]:
    """Recursively collect all key == 'reasoning' text fields with path info."""
    return collect_string_field(obj, "reasoning", prefix)


def raw_model_name_from_parsed_path(parsed: object, source_path: str) -> str:
    """If source_path includes models[i], return that entry's 'name' from parsed JSON, else ""."""
    if not isinstance(parsed, dict):
        return ""
    m = re.search(r"models\[(\d+)\]", source_path)
    if not m:
        return ""
    idx = int(m.group(1))
    models = parsed.get("models")
    if not isinstance(models, list) or not 0 <= idx < len(models):
        return ""
    ent = models[idx]
    if not isinstance(ent, dict):
        return ""
    n = ent.get("name", "")
    if isinstance(n, str) and n.strip():
        return n.strip()
    return ""


def load_merge_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    mm = data.get("merge_map", {})
    return {str(k): str(v) for k, v in mm.items() if k is not None} if isinstance(mm, dict) else {}


def canonical_merged_name(raw: str, merge_map: dict[str, str]) -> str:
    """Map raw model `name` to final standard name using merge_map (values normalized)."""
    if not (raw and raw.strip()):
        return "(n/a)"
    r = raw.strip()
    key = normalize_llm_merge_label(r)
    canon = merge_map.get(r) or merge_map.get(key) or merge_map.get(r.lower()) or key or r
    c = normalize_llm_merge_label(str(canon)) if canon else key
    return c if c else (key or r)


def parse_jsonl_reasonings(input_path: Path, source_scope: str) -> list[dict]:
    """Load JSONL and gather all reasoning text sources."""
    records: list[dict] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            qid = row.get("question_index", line_no)

            if source_scope == "step2_full":
                raw2 = row.get("step2_response")
                if isinstance(raw2, str) and raw2.strip():
                    records.append(
                        {
                            "question_index": qid,
                            "line_no": line_no,
                            "source": "step2_response.full",
                            "reasoning": raw2.strip(),
                            "raw_model_name": "",
                        }
                    )
                continue

            if source_scope == "step3_explanation":
                raw3 = row.get("step3_response")
                if not isinstance(raw3, str) or not raw3.strip():
                    continue
                block = extract_json_block(raw3)
                if not block:
                    continue
                try:
                    parsed = json.loads(block)
                except json.JSONDecodeError:
                    continue
                for path, text in collect_string_field(
                    parsed, "explanation", prefix="step3_response.$parsed"
                ):
                    records.append(
                        {
                            "question_index": qid,
                            "line_no": line_no,
                            "source": path,
                            "reasoning": text,
                            "raw_model_name": raw_model_name_from_parsed_path(parsed, path),
                        }
                    )
                continue

            if source_scope == "all":
                for path, text in collect_reasonings(row, prefix="record"):
                    records.append(
                        {
                            "question_index": qid,
                            "line_no": line_no,
                            "source": path,
                            "reasoning": text,
                            "raw_model_name": raw_model_name_from_parsed_path(row, path),
                        }
                    )

            fields = ("step3_response",) if source_scope == "step3" else STEP_TEXT_FIELDS
            for field in fields:
                raw_text = row.get(field)
                if not isinstance(raw_text, str) or not raw_text.strip():
                    continue
                block = extract_json_block(raw_text)
                if not block:
                    continue

                try:
                    parsed = json.loads(block)
                except json.JSONDecodeError:
                    continue

                for path, text in collect_reasonings(parsed, prefix=f"{field}.$parsed"):
                    records.append(
                        {
                            "question_index": qid,
                            "line_no": line_no,
                            "source": path,
                            "reasoning": text,
                            "raw_model_name": raw_model_name_from_parsed_path(parsed, path),
                        }
                    )

    return records


def source_group(source: str) -> str:
    s = source.lower()
    if s == "step2_response.full" or s.startswith("step2_response.full"):
        return "step2_response_full"
    if s.startswith("step3_response.$parsed"):
        last_seg = s.rsplit(".", 1)[-1]
        if last_seg == "explanation":
            return "step3_explanation"
        return "step3_models_reasoning"
    if "metadata" in s:
        return "metadata_reasoning"
    if s.startswith("step"):
        return "other_step_reasoning"
    return "other_reasoning"


def project_2d(embeddings: np.ndarray, method: str) -> np.ndarray:
    n = embeddings.shape[0]
    if n == 1:
        return np.array([[0.0, 0.0]], dtype=float)
    if n == 2:
        return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    if method == "umap":
        try:
            import umap

            n_neighbors = min(15, n - 1)
            reducer = umap.UMAP(
                n_components=2,
                metric="cosine",
                n_neighbors=80,
                min_dist=0.001,
                random_state=42,
                n_jobs=1,
            )
            return reducer.fit_transform(embeddings)
        except Exception as err:
            print(f"[warn] UMAP failed, fallback to t-SNE. reason={err}")

    perplexity = min(30, max(5, (n - 1) // 3))
    if perplexity >= n:
        perplexity = max(2, n - 1)
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
        return tsne.fit_transform(embeddings)
    except Exception as err:
        print(f"[warn] t-SNE failed, fallback to PCA. reason={err}")
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(embeddings)


def _qualitative_color_series(n: int) -> list:
    if n == 0:
        return []
    t20 = [plt.get_cmap("tab20")(i / 19.0) for i in range(20)]
    t20b = [plt.get_cmap("tab20b")(i / 19.0) for i in range(20)]
    s3 = [plt.get_cmap("Set3")(i / 11.0) for i in range(12)]
    pool = t20 + t20b + s3
    return [pool[i % len(pool)] for i in range(n)]


_MERGE_TOP_ATTRACTORS = 10
_MERGE_OTHER_FACE = "#d9d9d9"#其余吸引子的颜色
_MERGE_OTHER_EDGE = "#bfbfbf"


def plot_points_on_ax(
    ax: plt.Axes,
    points: np.ndarray,
    rows: list[dict],
    *,
    color_by: str = "source",
    merge_map: dict[str, str] | None = None,
    title: str | None = None,
    legend_fontsize: float = 7.5,
) -> None:
    merge_map = merge_map or {}

    palette = {
        "step2_response_full": "#d62728",
        "step3_models_reasoning": "#1f77b4",
        "step3_explanation": "#17becf",
        "metadata_reasoning": "#ff7f0e",
        "other_step_reasoning": "#2ca02c",
        "other_reasoning": "#9467bd",
    }

    if color_by == "merge":
        groups = [canonical_merged_name((x.get("raw_model_name") or ""), merge_map) for x in rows]
        use_palette = None
    else:
        groups = [source_group(x["source"]) for x in rows]
        use_palette = palette

    ax.set_facecolor("#f8f8f8")
    unique_groups = sorted(set(groups))
    count_by: Counter[str] = Counter(groups)

    if color_by == "merge":
        top_ordered = [n for n, _ in count_by.most_common(_MERGE_TOP_ATTRACTORS)]
        top_set = frozenset(top_ordered)
        color_series = _qualitative_color_series(max(_MERGE_TOP_ATTRACTORS, 1))
        others = sorted(g for g in unique_groups if g not in top_set)
        plot_order = list(others) + [g for g in top_ordered if g in unique_groups]
        for grp in plot_order:
            idx = [i for i, g in enumerate(groups) if g == grp]
            xy = points[idx]
            if grp not in top_set:
                ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    s=45,
                    alpha=0.75,
                    color=_MERGE_OTHER_FACE,
                    label=None,
                    edgecolors=_MERGE_OTHER_EDGE,
                    linewidths=0.45,
                )
            else:
                ci = top_ordered.index(grp)
                color = color_series[ci % len(color_series)]
                ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    s=45,
                    alpha=0.82,
                    color=color,
                    label=f"{grp} (n={len(idx)})",
                    edgecolors="white",
                    linewidths=0.6,
                )
    else:
        color_series = _qualitative_color_series(len(unique_groups)) if use_palette is None else None
        for gi, grp in enumerate(unique_groups):
            idx = [i for i, g in enumerate(groups) if g == grp]
            xy = points[idx]
            if use_palette is not None:
                color = use_palette.get(grp, "#333333")
            else:
                color = color_series[gi] if color_series is not None else "#333333"
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                s=45,
                alpha=0.82,
                color=color,
                label=f"{grp} (n={len(idx)})",
                edgecolors="white",
                linewidths=0.6,
            )

    if color_by == "merge":
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=legend_fontsize,
            title="Attractors name",
        )
    else:
        ax.legend(loc="best", frameon=True, fontsize=legend_fontsize)

    if title:
        ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, linestyle="--", alpha=0.35)


def plot_points(
    points: np.ndarray,
    rows: list[dict],
    output_png: Path,
    *,
    color_by: str = "source",
    merge_map: dict[str, str] | None = None,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "figure.dpi": 140,
            "savefig.dpi": 320,
        }
    )

    fig, ax = plt.subplots(figsize=(14, 9.0) if color_by == "merge" else (13.5, 9.0))
    plot_points_on_ax(
        ax,
        points,
        rows,
        color_by=color_by,
        merge_map=merge_map,
    )
    if color_by == "merge":
        fig.tight_layout(rect=(0, 0, 0.76, 1))
    else:
        fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def write_points_csv(
    points: np.ndarray,
    rows: list[dict],
    output_csv: Path,
    merge_map: dict[str, str],
    embedding_model: str,
) -> None:
    n = len(rows)
    if points.shape[0] != n:
        raise ValueError(f"row count mismatch: rows={n}, points={points.shape[0]}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "question_index",
            "line_no",
            "source",
            "source_group",
            "merged_name",
            "model_name",
            "embedding_model",
            "x",
            "y",
            "reasoning",
        ])
        for iid, (xy, row) in enumerate(zip(points, rows), start=1):
            rmn = (row.get("raw_model_name") or "").strip()
            merged = canonical_merged_name(rmn, merge_map)
            writer.writerow([
                iid,
                row["question_index"],
                row["line_no"],
                row["source"],
                source_group(row["source"]),
                merged,
                rmn,
                embedding_model,
                float(xy[0]),
                float(xy[1]),
                row["reasoning"],
            ])


def read_xy_from_2d_points_csv(path: Path, n_expected: int) -> np.ndarray:
    """Load ``x``/``y`` from a prior ``reasoning_embeddings_2d_points.csv``-style file."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"2D CSV not found: {path}")
    xs: list[float] = []
    ys: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or "x" not in r.fieldnames or "y" not in r.fieldnames:
            raise ValueError("2D CSV must have header row with columns `x` and `y`")
        for _row in r:
            try:
                xs.append(float(str(_row.get("x") or "").strip()))
                ys.append(float(str(_row.get("y") or "").strip()))
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid x/y in 2D CSV {path!s}: {e}") from e
    n = len(xs)
    if n != n_expected:
        raise ValueError(
            f"2D CSV has {n} data rows, but JSONL+source-scope produced {n_expected} rows. "
            "Regenerate the CSV with the same --source-scope, or use matching inputs."
        )
    return np.column_stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)])


def run_plot_only_2d_job(
    *,
    input_path: Path,
    source_scope: str,
    from_2d_csv: Path,
    color_by: str,
    merge_map_path: Path | None,
    out_dir: Path,
    save_separate_figure: bool,
    collect_plot_payload: bool,
    fig_lock: threading.Lock | None = None,
    console_quiet: bool = False,
    embedding_model_label: str = "2d_replot",
    log_prefix: str = "",
) -> dict | None:
    """Read prior 2D table, parse ``input_path`` for labels, write PNG+2D-only CSV in ``out_dir``."""
    rows = parse_jsonl_reasonings(input_path, source_scope=source_scope)
    if not rows:
        raise RuntimeError(f"No reasoning fields found: {input_path}")
    points = read_xy_from_2d_points_csv(from_2d_csv, n_expected=len(rows))
    lp = f"{log_prefix} " if log_prefix else ""
    if not console_quiet:
        print(f"{lp}[info] --plot-only-2d: input={input_path}")
        print(f"{lp}[info] 2D table={from_2d_csv}  records={len(rows)}")

    mpath = merge_map_path if merge_map_path is not None else (input_path.parent / "merge_map.json")
    merge_map: dict[str, str] = {}
    if mpath.is_file():
        merge_map = load_merge_map(mpath)
        if not console_quiet:
            print(f"{lp}[info] merge_map: {len(merge_map)} rules from {mpath}")
    elif color_by == "merge" and not console_quiet:
        print(
            f"{lp}[warn] merge map not found: {mpath} — using normalized raw model names as colors"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_png = out_dir / "reasoning_embeddings_2d.png"
    output_csv = out_dir / "reasoning_embeddings_2d_points.csv"
    if save_separate_figure:
        _cm = fig_lock if fig_lock is not None else nullcontext()
        with _cm:
            plot_points(
                points,
                rows,
                output_png,
                color_by=color_by,
                merge_map=merge_map,
            )
    write_points_csv(
        points,
        rows,
        output_csv,
        merge_map=merge_map,
        embedding_model=embedding_model_label,
    )
    if not console_quiet:
        if save_separate_figure:
            print(f"{lp}[saved] figure={output_png}")
        print(f"{lp}[saved] 2D-only table={output_csv} (no high-dim embedding column)")
    if collect_plot_payload:
        return {
            "points": points,
            "rows": rows,
            "merge_map": merge_map,
        }
    return None


def save_combined_overview(
    records: list[dict],
    output_root: Path,
    *,
    color_by: str,
    ncols: int,
) -> Path:
    n = len(records)
    if n <= 0:
        raise ValueError("No records for combined overview.")

    ncols = max(1, int(ncols))
    nrows = math.ceil(n / ncols)
    fig_w = ncols * (8.6 if color_by == "merge" else 7.8)
    fig_h = nrows * 6.4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = axes.flatten()

    for i, rec in enumerate(records):
        title = f"{fill(str(rec['model']), width=28)}\n{fill(str(rec['category']), width=28)}"
        plot_points_on_ax(
            axes_flat[i],
            rec["points"],
            rec["rows"],
            color_by=color_by,
            merge_map=rec.get("merge_map"),
            title=title,
            legend_fontsize=6.5,
        )

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Reasoning Embedding 2D Overview", y=0.997)
    fig.tight_layout(rect=(0, 0.0, 1, 0.985))
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / "combined_reasoning_embeddings_2d.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_embedding_job(
    *,
    input_path: Path,
    output_dir: Path,
    source_scope: str,
    color_by: str,
    reduce: str,
    embedding_backend: str,
    embedding_model: str,
    batch_size: int,
    embedding_api_key: str | None,
    embedding_base_url: str | None,
    merge_map_path: Path | None,
    st_model: object | None = None,
    show_progress_bar: bool = True,
    log_prefix: str = "",
    save_separate_figure: bool = True,
    collect_plot_payload: bool = False,
    fig_lock: threading.Lock | None = None,
    console_quiet: bool = False,
) -> dict | None:
    """Run embed → reduce → plot → CSV. ``merge_map_path`` = explicit file, or use input parent (pass None for default)."""
    rows = parse_jsonl_reasonings(input_path, source_scope=source_scope)
    if not rows:
        raise RuntimeError(f"No reasoning fields found: {input_path}")

    texts = [x["reasoning"] for x in rows]
    lp = f"{log_prefix} " if log_prefix else ""
    if not console_quiet:
        print(f"{lp}[info] input={input_path}")
        print(f"{lp}[info] records={len(rows)}")
        print(f"{lp}[info] embedding_backend={embedding_backend}")
        print(f"{lp}[info] model={embedding_model}")

    if embedding_backend == "local":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not console_quiet:
            print(f"{lp}[info] device={device}")
        encoder = st_model
        if encoder is None:
            try:
                encoder = SentenceTransformer(embedding_model, device=device)
            except ValueError as err:
                err_text = str(err).lower()
                if "torch" in err_text and "2.6" in str(err):
                    raise RuntimeError(
                        "Loading this model failed: HuggingFace weights are pytorch_model.bin, and "
                        "transformers now requires torch>=2.6 to load them (CVE-2025-32434). "
                        'Fix:  pip install "torch>=2.6"  in this environment, or use a model that '
                        "ships model.safetensors on the Hub.\n"
                        f"Original: {err}"
                    ) from err
                raise
        embeddings = encoder.encode(  # type: ignore[union-attr]
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float64)
    else:
        embeddings = _get_embeddings(
            texts,
            embedding_backend,
            embedding_model,
            embedding_api_key,
            embedding_base_url,
            None,
            quiet=console_quiet,
        )
        embeddings = l2_normalize_rows(embeddings)

    points = project_2d(embeddings, method=reduce)

    mpath = merge_map_path if merge_map_path is not None else (input_path.parent / "merge_map.json")
    merge_map: dict[str, str] = {}
    if mpath.is_file():
        merge_map = load_merge_map(mpath)
        if not console_quiet:
            print(f"{lp}[info] merge_map: {len(merge_map)} rules from {mpath}")
    elif color_by == "merge":
        if not console_quiet:
            print(
                f"{lp}[warn] merge map not found: {mpath} — using normalized raw model names as colors"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / "reasoning_embeddings_2d.png"
    output_csv = output_dir / "reasoning_embeddings_2d_points.csv"
    if save_separate_figure:
        _cm = fig_lock if fig_lock is not None else nullcontext()
        with _cm:
            plot_points(
                points,
                rows,
                output_png,
                color_by=color_by,
                merge_map=merge_map,
            )
    write_points_csv(
        points,
        rows,
        output_csv,
        merge_map=merge_map,
        embedding_model=embedding_model,
    )
    if not console_quiet:
        if save_separate_figure:
            print(f"{lp}[saved] figure={output_png}")
        print(f"{lp}[saved] points={output_csv}")
    if collect_plot_payload:
        return {
            "points": points,
            "rows": rows,
            "merge_map": merge_map,
        }
    return None


def _execute_one_batch_job(
    args: argparse.Namespace,
    root: Path,
    n: int,
    idx: int,
    input_path: Path,
    st_model: object | None,
    export_separate: bool,
    export_combined: bool,
    fig_lock: threading.Lock | None,
) -> dict:
    model_dir, cat_dir = _parse_model_category(input_path, root)
    if getattr(args, "batch_output_subdirs", False):
        out_dir = (args.output_root / model_dir / cat_dir).resolve()
    else:
        out_dir = input_path.parent.resolve()
    log_prefix = f"[batch {idx}/{n}]"
    try:
        if getattr(args, "plot_only_2d", False):
            probe_dir = input_path.parent
            t2d = (probe_dir / str(getattr(args, "d2_csv_basename", "reasoning_embeddings_2d_points.csv"))).resolve()
            payload = run_plot_only_2d_job(
                input_path=input_path,
                source_scope=args.source_scope,
                from_2d_csv=t2d,
                color_by=args.color_by,
                merge_map_path=None,
                out_dir=probe_dir,
                save_separate_figure=export_separate,
                collect_plot_payload=export_combined,
                fig_lock=fig_lock,
                console_quiet=True,
                embedding_model_label="2d_replot",
                log_prefix=log_prefix,
            )
        else:
            payload = run_embedding_job(
                input_path=input_path,
                output_dir=out_dir,
                source_scope=args.source_scope,
                color_by=args.color_by,
                reduce=args.reduce,
                embedding_backend=args.embedding_backend,
                embedding_model=args.model,
                batch_size=args.batch_size,
                embedding_api_key=args.embedding_api_key,
                embedding_base_url=args.embedding_base_url,
                merge_map_path=None,
                st_model=st_model,
                show_progress_bar=st_model is not None,
                log_prefix=log_prefix,
                save_separate_figure=export_separate,
                collect_plot_payload=export_combined,
                fig_lock=fig_lock,
                console_quiet=True,
            )
        return {
            "ok": True,
            "input_path": input_path,
            "model_dir": model_dir,
            "cat_dir": cat_dir,
            "payload": payload,
        }
    except Exception as e:
        return {
            "ok": False,
            "input_path": input_path,
            "error": e,
            "log_prefix": log_prefix,
        }


def run_batch_from_args(args: argparse.Namespace) -> int:
    """Process every discovered probe_results.jsonl. Returns 0 if all ok, 1 if any job failed."""
    root = args.root.resolve()
    models = frozenset(args.models) if args.models else None
    categories = _normalize_category_filters(args.categories) if args.categories else None
    paths = discover_probe_jsonl(
        root,
        models=models,
        categories=categories,
        include_root_file=False,
        exclude_dir_names=DEFAULT_EXCLUDE_DIR_NAMES,
    )
    if not paths:
        print("[batch] No probe_results.jsonl matched (check --root, --models, --categories).")
        return 1
    n = len(paths)
    if args.dry_run:
        for p in paths:
            line = f"[dry-run] {p.relative_to(root)}"
            if getattr(args, "plot_only_2d", False):
                bn = str(getattr(args, "d2_csv_basename", "reasoning_embeddings_2d_points.csv"))
                line += f"  [2D table: {p.parent / bn}]"
            print(line)
        print(f"[batch] {n} file(s) would be processed")
        return 0

    export_separate = args.mode in ("separate", "both")
    export_combined = args.mode in ("combined", "both")

    plot_only = getattr(args, "plot_only_2d", False)
    batch_workers = max(1, int(getattr(args, "batch_workers", 1)))
    use_parallel = (
        batch_workers > 1
        and n > 1
        and (args.embedding_backend in ("openai", "ollama") or plot_only)
    )
    if batch_workers > 1 and args.embedding_backend == "local" and not plot_only:
        tqdm.write(
            "[batch] --batch-workers>1 ignored for embedding-backend=local "
            "(one shared SentenceTransformer). Use openai/ollama to parallelize API I/O."
        )
    w = min(batch_workers, n) if use_parallel else 1

    st_model = None
    if args.embedding_backend == "local" and not plot_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tqdm.write(f"[batch] load SentenceTransformer once: {args.model!r}  device={device}")
        try:
            st_model = SentenceTransformer(args.model, device=device)
        except ValueError as err:
            err_text = str(err).lower()
            if "torch" in err_text and "2.6" in str(err):
                raise RuntimeError(
                    "Batch local embedding: model load failed (pytorch_model.bin + torch<2.6). "
                    'Install torch>=2.6 or use a .safetensors model.\n'
                    f"Original: {err}"
                ) from err
            raise

    failed: list[tuple[Path, Exception]] = []
    combined_records: list[dict] = []
    fig_lock: threading.Lock | None = threading.Lock() if use_parallel and w > 1 else None

    ncols = _tqdm_ncols()
    if plot_only:
        tqdm.write(
            "[batch] --plot-only-2d: reads <probe_dir>/BASENAME for x,y, writes figure+2D-only CSV there "
            f"(basename={getattr(args, 'd2_csv_basename', 'reasoning_embeddings_2d_points.csv')!r})"
        )
    if use_parallel and w > 1:
        tqdm.write(f"[batch] parallel: {w} worker(s) · {n} file(s) total")

    def _consume(
        r: dict,
        pbar,
        done_rel: str,
    ) -> None:
        if r.get("ok"):
            pl = r.get("payload")
            if pl is not None and export_combined:
                combined_records.append(
                    {
                        "model": (r.get("model_dir") or "(root)"),
                        "category": (r.get("cat_dir") or "(root)"),
                        "points": pl["points"],
                        "rows": pl["rows"],
                        "merge_map": pl["merge_map"],
                    }
                )
        else:
            ex = r.get("error")
            lp = r.get("log_prefix", "[batch]")
            tqdm.write(f"{lp} [error] {ex}")
            failed.append((r["input_path"], ex))  # type: ignore[misc]
        tag = "ok" if r.get("ok") else "err"
        pbar.set_postfix_str(f"{tag} ← {_tqdm_path_hint(done_rel, 56)}")
        pbar.update(1)

    with tqdm(
        total=n,
        desc="[batch] jobs",
        ncols=ncols,
        unit="file",
        dynamic_ncols=False,
    ) as pbar:
        pbar.set_postfix_str("starting", refresh=False)
        if not use_parallel or w <= 1:
            for idx, input_path in enumerate(paths, start=1):
                pbar.set_postfix_str(
                    f"… {_tqdm_path_hint(input_path.relative_to(root), 56)}",
                    refresh=True,
                )
                r = _execute_one_batch_job(
                    args,
                    root,
                    n,
                    idx,
                    input_path,
                    st_model,
                    export_separate,
                    export_combined,
                    None,
                )
                _consume(r, pbar, str(input_path.relative_to(root)))
        else:
            with ThreadPoolExecutor(max_workers=w) as ex:
                fmap = {
                    ex.submit(
                        _execute_one_batch_job,
                        args,
                        root,
                        n,
                        idx,
                        input_path,
                        None,
                        export_separate,
                        export_combined,
                        fig_lock,
                    ): input_path
                    for idx, input_path in enumerate(paths, start=1)
                }
                pbar.set_postfix_str("workers", refresh=True)
                for fut in as_completed(fmap):
                    path_key = fmap[fut]
                    r = fut.result()
                    _consume(r, pbar, str(path_key.relative_to(root)))

    combined_records.sort(
        key=lambda e: (str(e.get("model", "")), str(e.get("category", "")))
    )

    if st_model is not None:
        del st_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if export_combined:
        if combined_records:
            combined_path = save_combined_overview(
                combined_records,
                args.output_root,
                color_by=args.color_by,
                ncols=args.ncols,
            )
            tqdm.write(f"[saved] combined figure={combined_path}")
        else:
            tqdm.write("[batch] no successful jobs for combined overview")

    if failed:
        tqdm.write(f"[batch] done with {len(failed)}/{n} error(s):")
        for p, e in failed:
            tqdm.write(f"  - {p}: {e}")
        return 1
    if getattr(args, "batch_output_subdirs", False):
        tqdm.write(
            f"[batch] all {n} job(s) ok. Per-job files under: {args.output_root.resolve()}"
        )
    else:
        tqdm.write(
            f"[batch] all {n} job(s) ok. Per-job CSV/figs next to each probe; "
            f"combined (if any): {args.output_root.resolve()}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed all reasoning fields from a JSONL file and project to 2D.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch: discover probe_results.jsonl. Default: per-job CSV/figs go next to each JSONL; "
        "use --batch-output-subdirs to use --output-root/<model>/<category>/.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("model"),
        help="Root directory to scan in --batch (default: model/).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reasoning_embedding_2d_batch"),
        help="In --batch: where to write the combined overview PNG (and subdirs if --batch-output-subdirs). "
        "Default: %(default)s.",
    )
    parser.add_argument(
        "--batch-output-subdirs",
        action="store_true",
        help="In --batch: write each job's CSV/PNG under --output-root/<model>/<category>/ instead of next to probe.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="In --batch: only these top-level model folder names (space-separated), e.g. deepseek openai-gpt-5.4-mini",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="In --batch: only these category subfolder names; accepts taxonomy [Brackets] or on-disk names.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --batch: list matching files and exit (no API / no encode).",
    )
    parser.add_argument(
        "-j",
        "--batch-workers",
        type=int,
        default=1,
        metavar="N",
        help="With --batch: parallel jobs for openai/ollama, or for --plot-only-2d. "
        "Ignored for local embedding. Default: %(default)s",
    )
    parser.add_argument(
        "--mode",
        choices=["combined", "separate", "both"],
        default="both",
        help="In --batch: figure export mode. combined=single overview, separate=one figure per model/category, both=export both (default: %(default)s).",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=4,
        help="In --batch and --mode combined/both: columns of combined overview layout (default: %(default)s).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("model/deepseek/Personal&Existential/probe_results.jsonl"),
        help="Input JSONL file path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-small-zh-v1.5",
        help="Embedding model: local=HF id; openai/ollama per provider. If omitted on CLI, uses EMBEDDING_MODEL from .env when set (else default: %(default)s).",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["local", "openai", "ollama"],
        default="local",
        help="If omitted on CLI, can use EMBEDDING_BACKEND from .env (local: SentenceTransformers; openai/ollama: API). (default: %(default)s).",
    )
    parser.add_argument(
        "--embedding-api-key",
        type=str,
        default=None,
        help="For openai backend; else EMBEDDING_API_KEY or OPENAI_API_KEY from .env",
    )
    parser.add_argument(
        "--embedding-base-url",
        type=str,
        default=None,
        help="For openai/ollama; else EMBEDDING_BASE_URL or OPENAI_BASE_URL; ollama default http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (local SentenceTransformers only).",
    )
    parser.add_argument(
        "--reduce",
        choices=["umap", "tsne"],
        default="umap",
        help="2D projection method.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reasoning_embedding_2d"),
        help="Output directory in single-file mode. Ignored when using --batch (use --output-root).",
    )
    parser.add_argument(
        "--source-scope",
        choices=["step3", "all", "step2_full", "step3_explanation"],
        default="step3_explanation",
        help="Text source: step3 reasonings, all reasonings, full step2_response, or step3 explanation fields (default: %(default)s).",
    )
    parser.add_argument(
        "--color-by",
        choices=["source", "merge"],
        default="merge",
        help="Scatter color: by source field group, or by merge_map canonical name (default: %(default)s).",
    )
    parser.add_argument(
        "--merge-map",
        type=Path,
        default=None,
        help="merge_map.json from analysis. Default when --color-by merge: <input parent>/merge_map.json",
    )
    parser.add_argument(
        "--plot-only-2d",
        action="store_true",
        help="Skip embedding and UMAP/t-SNE. Read x,y from a prior 2D CSV, re-parse --input for "
        "labels, and write PNG + 2D-only CSV next to probe_results (same folder as model analysis).",
    )
    parser.add_argument(
        "--from-2d-csv",
        type=Path,
        default=None,
        help="With --plot-only-2d (single file): path to 2D CSV. Default: <input parent>/--2d-csv-basename.",
    )
    parser.add_argument(
        "--2d-csv-basename",
        type=str,
        default="reasoning_embeddings_2d_points.csv",
        dest="d2_csv_basename",
        help="Default 2D table filename per probe directory (single: under --input parent; batch: each model/category folder).",
    )
    return parser


def main() -> None:
    _load_project_dotenv()
    args = build_parser().parse_args()
    _apply_embedding_env_defaults(args)
    if args.batch:
        raise SystemExit(run_batch_from_args(args))
    if args.plot_only_2d:
        t2d = (args.from_2d_csv or (args.input.parent / args.d2_csv_basename)).resolve()
        if not t2d.is_file():
            print(f"ERROR: 2D CSV not found: {t2d}")
            raise SystemExit(1)
        run_plot_only_2d_job(
            input_path=args.input,
            source_scope=args.source_scope,
            from_2d_csv=t2d,
            color_by=args.color_by,
            merge_map_path=args.merge_map,
            out_dir=args.input.parent,
            save_separate_figure=True,
            collect_plot_payload=False,
        )
        return
    if args.mode != "both":
        print("[info] --mode is primarily for --batch; single-file mode always writes one figure.")
    run_embedding_job(
        input_path=args.input,
        output_dir=args.output_dir,
        source_scope=args.source_scope,
        color_by=args.color_by,
        reduce=args.reduce,
        embedding_backend=args.embedding_backend,
        embedding_model=args.model,
        batch_size=args.batch_size,
        embedding_api_key=args.embedding_api_key,
        embedding_base_url=args.embedding_base_url,
        merge_map_path=args.merge_map,
        st_model=None,
        show_progress_bar=True,
        log_prefix="",
        save_separate_figure=True,
        collect_plot_payload=False,
    )


if __name__ == "__main__":
    main()
