"""Plot top attractor distributions with power-law fitting.

This script reads merged results from ``merged_top50.json`` and generates:
1) A bar chart for top-N attractors of each model-category item.
2) A power-law fit curve overlaid in the same chart.

It supports two output layouts:
- combined: all charts in one large multi-subplot figure
- separate: one figure per model-category item
- both: generate both outputs

只导出一张总览大图
python plot_top_attractors_powerlaw.py --mode combined --show-attractor-names

只导出一批单图
python plot_top_attractors_powerlaw.py --mode separate

两种都导出（默认）
python plot_top_attractors_powerlaw.py --mode both
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STYLE_PAIRS = [
    ("#4E79A7", "#E15759"),
    ("#59A14F", "#F28E2B"),
    ("#76B7B2", "#B07AA1"),
    ("#9C755F", "#EDC948"),
    ("#3A86B9", "#D1495B"),
    ("#2A9D8F", "#E76F51"),
]


MODEL_DISPLAY = {
    "deepseek": "DeepSeek",
    "anthropic-claude-haiku-4.5": "Claude 4.5",
    "google-gemini-2.5-flash": "Gemini 2.5",
    "openai-gpt-5.4-mini": "GPT-5.4",
    "kimi-k2.6": "Kimi K2.6",
    "MiniMax-M2.5": "MiniMax M2.5",
    "doubao-seed-2-0-lite-260215": "Doubao Seed",
}


def configure_plot_style() -> None:
    """Set publication-ready style with large Times New Roman fonts."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "figure.titlesize": 24,
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "axes.linewidth": 1.2,
            "axes.edgecolor": "#222222",
            "axes.facecolor": "#F8F6F1",
            "figure.facecolor": "#F4F1EA",
            "savefig.facecolor": "#F4F1EA",
            "grid.color": "#C2CAD3",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.unicode_minus": False,
        }
    )


def load_items(input_path: Path) -> list[dict]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", [])
    if not isinstance(items, list):
        raise ValueError("Input JSON does not contain a valid 'items' list.")
    return items


def sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    cleaned = cleaned.strip("_")
    return cleaned or "untitled"


def power_law_fit(counts: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """Fit y = c * x^(-alpha) on log-log scale.

    Returns (y_hat, alpha, r2_log) if successful.
    """
    x = np.arange(1, len(counts) + 1, dtype=float)
    y = counts.astype(float)
    mask = y > 0
    if np.sum(mask) < 2:
        return None

    x_m = x[mask]
    y_m = y[mask]
    logx = np.log(x_m)
    logy = np.log(y_m)

    slope, intercept = np.polyfit(logx, logy, 1)
    alpha = -float(slope)
    c = float(np.exp(intercept))
    y_hat = c * np.power(x, -alpha)

    pred_logy = intercept + slope * logx
    ss_res = float(np.sum((logy - pred_logy) ** 2))
    ss_tot = float(np.sum((logy - np.mean(logy)) ** 2))
    r2_log = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return y_hat, alpha, r2_log


def _format_attractor_labels(names: list[str], max_len: int = 24, wrap_width: int = 12) -> list[str]:
    labels: list[str] = []
    for name in names:
        cleaned = re.sub(r"\s+", " ", str(name).strip())
        if len(cleaned) > max_len:
            cleaned = cleaned[: max_len - 3].rstrip() + "..."
        labels.append(fill(cleaned, width=wrap_width))
    return labels


def _auto_tick_step(n: int) -> int:
    """Choose a sparse x-tick step to reduce label overlap."""
    if n <= 10:
        return 1
    if n <= 20:
        return 2
    return 3


def display_model_name(model: str) -> str:
    """Return a publication-friendly model display name."""
    return MODEL_DISPLAY.get(model, model)


def plot_one_distribution(
    ax: plt.Axes,
    model: str,
    category: str,
    counts: np.ndarray,
    attractor_names: list[str],
    show_attractor_names: bool,
    show_x_tick_labels: bool,
    color_index: int,
) -> None:
    bar_color, line_color = STYLE_PAIRS[color_index % len(STYLE_PAIRS)]
    x = np.arange(1, len(counts) + 1)

    ax.bar(
        x,
        counts,
        color=bar_color,
        edgecolor="#1F2933",
        linewidth=0.8,
        alpha=0.9,
        label="Top attractor counts",
        zorder=3,
    )

    fit = power_law_fit(counts)
    if fit is not None:
        y_hat, alpha, r2_log = fit
        legend_label = f"Power-law fit: y=c*x^(-a), a={alpha:.2f}, log R^2={r2_log:.2f}"
        ax.plot(x, y_hat, color=line_color, linewidth=3.0, label=legend_label, zorder=4)

    model_title = display_model_name(model)
    title = f"{fill(model_title, width=36)}\n{fill(category, width=36)}"
    ax.set_title(title, pad=10)
    ax.set_xlabel("Attractor name" if show_attractor_names else "Attractor rank")
    ax.set_ylabel("Mention count")
    if not show_x_tick_labels:
        ax.set_xticks([])
        ax.tick_params(axis="x", which="both", length=0)
        ax.set_xlabel("")
    elif show_attractor_names:
        step = _auto_tick_step(len(counts))
        tick_idx = np.arange(1, len(counts) + 1, step)
        labels_all = _format_attractor_labels(attractor_names)
        labels = [labels_all[i - 1] for i in tick_idx]
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(labels, rotation=58, ha="right", fontsize=10)
    else:
        step = _auto_tick_step(len(counts))
        ax.set_xticks(np.arange(1, len(counts) + 1, step))
    ax.set_xlim(0.2, len(counts) + 0.8)
    if show_x_tick_labels and not show_attractor_names:
        ax.tick_params(axis="x", rotation=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)


def extract_counts_and_names(item: dict, top_n: int) -> tuple[np.ndarray, list[str]] | None:
    top_names = item.get("top_names", [])
    if not isinstance(top_names, list) or not top_names:
        return None
    selected = top_names[:top_n]
    counts: list[int] = []
    names: list[str] = []
    for row in selected:
        if not isinstance(row, dict):
            continue
        value = row.get("count")
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        name = str(row.get("name", "")).strip()
        if not name:
            name = "unknown"
        counts.append(count)
        names.append(name)
    if not counts:
        return None
    return np.asarray(counts, dtype=float), names


def build_records(items: list[dict], top_n: int) -> list[dict]:
    records: list[dict] = []
    for item in items:
        model = str(item.get("model", "unknown_model"))
        category = str(item.get("category", "unknown_category"))
        extracted = extract_counts_and_names(item, top_n)
        if extracted is None:
            continue
        counts, names = extracted
        records.append({"model": model, "category": category, "counts": counts, "names": names})
    return records


def save_separate_figures(
    records: list[dict],
    output_dir: Path,
    ext: str,
    show_attractor_names: bool,
    show_x_tick_labels: bool,
) -> int:
    separate_dir = output_dir / "separate"
    separate_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx, rec in enumerate(records, start=1):
        fig_h = 9.5 if (show_attractor_names and show_x_tick_labels) else 8.0
        fig, ax = plt.subplots(figsize=(12.8, fig_h))
        plot_one_distribution(
            ax,
            rec["model"],
            rec["category"],
            rec["counts"],
            rec["names"],
            show_attractor_names,
            show_x_tick_labels,
            idx - 1,
        )
        if show_attractor_names and show_x_tick_labels:
            fig.tight_layout(rect=[0, 0.08, 1, 1])
        else:
            fig.tight_layout()

        fname = f"{idx:03d}_{sanitize_filename(rec['model'])}__{sanitize_filename(rec['category'])}.{ext}"
        path = separate_dir / fname
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        saved += 1

    return saved


def save_combined_figure(
    records: list[dict],
    output_dir: Path,
    ext: str,
    ncols: int,
    show_attractor_names: bool,
    show_x_tick_labels: bool,
) -> Path:
    n = len(records)
    ncols = max(1, ncols)
    nrows = math.ceil(n / ncols)

    fig_w = ncols * 8.0
    fig_h = nrows * (8.3 if (show_attractor_names and show_x_tick_labels) else 5.7)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = axes.flatten()

    for i, rec in enumerate(records):
        plot_one_distribution(
            axes_flat[i],
            rec["model"],
            rec["category"],
            rec["counts"],
            rec["names"],
            show_attractor_names,
            show_x_tick_labels,
            i,
        )

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Top Attractor Distribution with Power-law Fitting", y=0.995)
    bottom_pad = 0.02 if not (show_attractor_names and show_x_tick_labels) else 0.04
    fig.tight_layout(rect=[0, bottom_pad, 1, 0.98])

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"combined_top_attractors_powerlaw.{ext}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot top attractor distributions and overlay power-law fits.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("merged_top50.json"),
        help="Path to merged_top50.json (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures_top_attractors_powerlaw"),
        help="Directory to save generated figures (default: %(default)s)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top attractors per model-category chart (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        choices=["combined", "separate", "both"],
        default="both",
        help="Figure export mode (default: %(default)s)",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=4,
        help="Number of columns in combined layout (default: %(default)s)",
    )
    parser.add_argument(
        "--ext",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output figure format (default: %(default)s)",
    )
    parser.add_argument(
        "--show-attractor-names",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to show attractor names on x-axis (default: %(default)s)",
    )
    parser.add_argument(
        "--show-x-tick-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to show x-axis tick labels (default: %(default)s)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.top_n <= 0:
        raise ValueError("--top-n must be > 0")

    configure_plot_style()

    items = load_items(args.input)
    records = build_records(items, args.top_n)
    if not records:
        raise RuntimeError("No valid plotting records found in input data.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("combined", "both"):
        combined_path = save_combined_figure(
            records,
            args.output_dir,
            args.ext,
            args.ncols,
            args.show_attractor_names,
            args.show_x_tick_labels,
        )
        print(f"[saved] combined figure: {combined_path}")

    if args.mode in ("separate", "both"):
        count = save_separate_figures(
            records,
            args.output_dir,
            args.ext,
            args.show_attractor_names,
            args.show_x_tick_labels,
        )
        print(f"[saved] separate figures: {count} files in {args.output_dir / 'separate'}")


if __name__ == "__main__":
    main()
