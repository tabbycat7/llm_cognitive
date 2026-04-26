"""桑基图：测试数据四分类 → 归并后吸引子（仅全局 Top-N，不汇总其余）.

从 ``model/<model_id>/<category>/probe_results.jsonl`` 读取 step3 中的 ``models[].name``，
用同目录 ``merge_map.json`` 映射为规范吸引子名，与 ``analyze_results`` 管线一致。

导出模式（与 ``plot_top_attractors_powerlaw`` 类似）：

- ``combined``：多模型排布在一张图（子图）或一个 HTML 中
- ``separate``：每个模型单独文件
- ``both``：两种都写

依赖：``pip install plotly kaleido``（PNG 需要 kaleido）。

用法示例::

    python sankey_category_attractors.py --mode separate --format both
    python sankey_category_attractors.py --mode combined -o figures/sankey
    python sankey_category_attractors.py --models deepseek anthropic-claude-haiku-4.5
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# --- 与 analyze_results 一致的标签规范化（避免 import 整包） ---
_HYPHEN_LIKE = re.compile(
    r"[\u002D\u2010\u2011\u2012\u2013\u2014\u2212\uFE58\uFE63\uFF0D]"
)


def normalize_llm_merge_label(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return ""
    s = _HYPHEN_LIKE.sub("-", s)
    s = s.lower()
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\s*-\s*", "-", s)
        s = re.sub(r"-{2,}", "-", s)
        s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_merge_map(merge_map: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw, can in merge_map.items():
        c = normalize_llm_merge_label(str(can))
        out[raw] = c if c else raw
    return out


def extract_json_from_text(text: str) -> str | None:
    if not text or not text.strip():
        return None
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0).strip()
    return None


def extract_model_names_from_step3(text: str) -> list[str]:
    json_str = extract_json_from_text(text)
    if not json_str:
        return []
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return []
    names: list[str] = []
    for item in data.get("models", []):
        if isinstance(item, dict):
            name = (item.get("name") or "").strip()
            if name:
                names.append(name)
    return names


def load_merge_map(merge_path: Path) -> dict[str, str]:
    if not merge_path.is_file():
        return {}
    with merge_path.open(encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("merge_map", {})
    if not isinstance(raw, dict):
        return {}
    return _normalize_merge_map({str(k): str(v) for k, v in raw.items()})


def resolve_canonical(raw: str, merge_map: dict[str, str]) -> str:
    if raw in merge_map:
        return merge_map[raw]
    c = normalize_llm_merge_label(raw)
    return c if c else raw


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = ROOT / "model"

# 与 cognitive_probe / bipartite 一致
CATEGORY_ORDER: tuple[str, ...] = (
    "Personal&Existential",
    "Professional&Economic",
    "Relational&Intimate",
    "Societal&Ethical",
)

CATEGORY_SHORT: dict[str, str] = {
    "Personal&Existential": "Personal & Existential",
    "Professional&Economic": "Professional & Economic",
    "Relational&Intimate": "Relational & Intimate",
    "Societal&Ethical": "Societal & Ethical",
}

MODEL_DISPLAY: dict[str, str] = {
    "deepseek": "DeepSeek",
    "anthropic-claude-haiku-4.5": "Claude 4.5",
    "google-gemini-2.5-flash": "Gemini 2.5",
    "openai-gpt-5.4-mini": "GPT-5.4",
    "kimi-k2.6": "Kimi K2.6",
    "MiniMax-M2.5": "MiniMax M2.5",
    "doubao-seed-2-0-lite-260215": "Doubao Seed",
}

# 左列四分类：柔和、可区分（类似 Tableau 10 变体）
CATEGORY_NODE_COLORS: tuple[str, ...] = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
)

# 右列吸引子：暖冷交错的科研风配色
ATTRACTOR_PALETTE: tuple[str, ...] = (
    "#2CA02C",
    "#9467BD",
    "#D62728",
    "#17BECF",
    "#8C564B",
    "#E377C2",
    "#BCBD22",
    "#7F7F7F",
    "#1F77B4",
    "#FF7F0E",
)


def discover_models(model_root: Path) -> list[str]:
    found: set[str] = set()
    for p in model_root.rglob("probe_results.jsonl"):
        try:
            rel = p.relative_to(model_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) >= 3 and parts[1] in CATEGORY_ORDER:
            found.add(parts[0])
    return sorted(found, key=str.lower)


def collect_flows_for_model(
    model_id: str,
    model_root: Path,
) -> tuple[Counter[tuple[str, str]], int]:
    """Return ( (category, canonical) -> count ), n_lines processed."""
    pair_counts: Counter[tuple[str, str]] = Counter()
    n_lines = 0
    for cat in CATEGORY_ORDER:
        jsonl = model_root / model_id / cat / "probe_results.jsonl"
        if not jsonl.is_file():
            continue
        merge_path = model_root / model_id / cat / "merge_map.json"
        mm = load_merge_map(merge_path)
        with jsonl.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step3 = rec.get("step3_response", "")
                for raw in extract_model_names_from_step3(step3):
                    can = resolve_canonical(raw, mm)
                    if not can:
                        continue
                    pair_counts[(cat, can)] += 1
                n_lines += 1
    return pair_counts, n_lines


def top_n_attractors(
    pair_counts: Counter[tuple[str, str]], n: int
) -> tuple[list[str], Counter[tuple[str, str]]]:
    """按全局出现次数取 Top-N 规范吸引子；只保留射向这 N 个的边，不统计也不展示「其余」."""
    total_by_attr: Counter[str] = Counter()
    for (cat, attr), c in pair_counts.items():
        total_by_attr[attr] += c
    top = [a for a, _ in total_by_attr.most_common(n)]
    top_set = set(top)
    out: Counter[tuple[str, str]] = Counter()
    for (cat, attr), c in pair_counts.items():
        if attr in top_set:
            out[(cat, attr)] += c
    return top, out


def _truncate(s: str, max_len: int = 32) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def build_sankey_traces(
    remapped: Counter[tuple[str, str]],
    top_names: list[str],
) -> tuple[list[str], list[str], list[int], list[int], list[float], list[str]]:
    """Node labels, colors, and link arrays for go.Sankey."""
    left_labels = [CATEGORY_SHORT[c] for c in CATEGORY_ORDER]
    right_order = [n for n in top_names]
    right_display = [_truncate(n) for n in top_names]

    n_left = len(left_labels)
    labels = left_labels + right_display
    # 左半颜色
    node_color = [CATEGORY_NODE_COLORS[i % len(CATEGORY_NODE_COLORS)] for i in range(n_left)]
    # 右半：与吸引子一一对应
    for i in range(len(right_order)):
        node_color.append(ATTRACTOR_PALETTE[i % len(ATTRACTOR_PALETTE)])

    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    link_colors: list[str] = []

    # index: right j -> n_left + j
    r_index = {name: n_left + j for j, name in enumerate(right_order)}

    for i, cat in enumerate(CATEGORY_ORDER):
        for name in right_order:
            v = float(remapped.get((cat, name), 0))
            if v <= 0:
                continue
            sources.append(i)
            targets.append(r_index[name])
            values.append(v)
            # 与源分类同系、半透明
            base = CATEGORY_NODE_COLORS[i % len(CATEGORY_NODE_COLORS)]
            link_colors.append(_hex_to_rgba(base, 0.35))

    return labels, node_color, sources, targets, values, link_colors


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def make_sankey_figure(
    model_id: str,
    remapped: Counter[tuple[str, str]],
    top_names: list[str],
    *,
    width: int = 900,
    height: int = 520,
) -> object:
    import plotly.graph_objects as go

    labels, node_color, sources, targets, values, link_colors = build_sankey_traces(
        remapped, top_names
    )
    if not values:
        # 空数据占位
        labels = ["(no data)"]
        node_color = ["#CCCCCC"]
        sources, targets, values, link_colors = [0], [0], [0.0], ["rgba(128,128,128,0.3)"]

    title = MODEL_DISPLAY.get(model_id, model_id)
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    label=labels,
                    color=node_color,
                    pad=18,
                    thickness=18,
                    line=dict(color="rgba(0,0,0,0.18)", width=0.5),
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                ),
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family="Georgia, Times New Roman, serif")),
        font=dict(family="Georgia, Times New Roman, serif", size=12, color="#2A2A2A"),
        paper_bgcolor="#FAF8F3",
        plot_bgcolor="#FAF8F3",
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def write_figure(
    fig: object,
    out_path: Path,
    fmt: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "html":
        fig.write_html(out_path, include_plotlyjs="cdn", config={"displayModeBar": True})
    else:
        try:
            fig.write_image(str(out_path), scale=2)
        except Exception as e:
            raise RuntimeError(
                f"PNG export failed (install kaleido: pip install kaleido): {e}"
            ) from e


def run(
    model_root: Path,
    out_dir: Path,
    mode: str,
    export_format: str,
    top_n: int,
    models_filter: set[str] | None,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    models = discover_models(model_root)
    if models_filter is not None:
        models = [m for m in models if m in models_filter]
    if not models:
        print("No probe_results.jsonl found under <root>/*/; check --root.")
        return

    out_dir = out_dir.resolve()
    # (model_id, top_names, remapped, n_lines) for reuse in combined
    prepared: list[tuple[str, list[str], Counter[tuple[str, str]], int]] = []
    single_figures: list[tuple[str, object]] = []

    for model_id in models:
        pair_counts, n_lines = collect_flows_for_model(model_id, model_root)
        if not pair_counts:
            print(f"[skip] {model_id}: no step3 model names")
            continue
        top_names, remapped = top_n_attractors(pair_counts, top_n)
        print(
            f"{model_id}: {n_lines} probe lines, {len(pair_counts)} distinct (cat×name) "
            f"edges → top {top_n} only (no other bucket)"
        )
        prepared.append((model_id, top_names, remapped, n_lines))
        fig = make_sankey_figure(model_id, remapped, top_names)
        single_figures.append((model_id, fig))

    if not prepared:
        return

    # separate
    if mode in ("separate", "both"):
        for model_id, fig in single_figures:
            base = out_dir / f"sankey_{_safe_filename(model_id)}"
            if export_format in ("html", "both"):
                write_figure(fig, base.with_suffix(".html"), "html")
            if export_format in ("png", "both"):
                write_figure(fig, base.with_suffix(".png"), "png")
        w = []
        if export_format in ("html", "both"):
            w.append("html")
        if export_format in ("png", "both"):
            w.append("png")
        ext = " + ".join(f"*.{e}" for e in w)
        print(f"Wrote per-model sankey_ ({ext}) in {out_dir}")

    # combined: one HTML with multiple figures, and/or subplot PNG
    if mode in ("combined", "both"):
        n = len(prepared)
        cols = 2
        rows = (n + cols - 1) // cols
        cfig = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{"type": "domain"} for _ in range(cols)] for _ in range(rows)],
            subplot_titles=[MODEL_DISPLAY.get(mid, mid) for mid, _, _, _ in prepared]
            + [""] * (rows * cols - n),
            vertical_spacing=0.10,
            horizontal_spacing=0.06,
        )
        for idx, (model_id, top_names, remapped, _) in enumerate(prepared):
            r = idx // cols + 1
            c = idx % cols + 1
            labels, node_color, sources, targets, values, link_colors = build_sankey_traces(
                remapped, top_names
            )
            cfig.add_trace(
                go.Sankey(
                    arrangement="snap",
                    node=dict(
                        label=labels,
                        color=node_color,
                        pad=10,
                        thickness=14,
                        line=dict(color="rgba(0,0,0,0.12)", width=0.4),
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=link_colors,
                    ),
                ),
                row=r,
                col=c,
            )
        cfig.update_layout(
            font=dict(family="Georgia, Times New Roman, serif", size=11, color="#2A2A2A"),
            paper_bgcolor="#F5F2EB",
            plot_bgcolor="#F5F2EB",
            width=min(2000, 480 * cols),
            height=max(400, 380 * rows),
            margin=dict(l=24, r=24, t=80, b=24),
            title=dict(
                text="Category → attractor (Top-N) flow",
                font=dict(size=20, family="Georgia, Times New Roman, serif"),
            ),
        )
        comb_base = out_dir / "sankey_combined"
        if export_format in ("html", "both"):
            # 组合 HTML：用子图单文件可能很大，写独立组合页
            cfig.write_html(
                comb_base.with_suffix(".html"),
                include_plotlyjs="cdn",
                config={"displayModeBar": True},
            )
        if export_format in ("png", "both"):
            write_figure(cfig, comb_base.with_suffix(".png"), "png")
        parts2 = [str(comb_base.with_suffix(".html"))] if export_format in ("html", "both") else []
        if export_format in ("png", "both"):
            parts2.append(str(comb_base.with_suffix(".png")))
        print("Combined:", "  ".join(parts2) if parts2 else str(comb_base))


def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "model"


def main() -> None:
    p = argparse.ArgumentParser(description="四分类 → Top-N 吸引子 桑基图")
    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help="model 根目录（默认: ./model）",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=ROOT / "figures_sankey",
        help="输出目录",
    )
    p.add_argument(
        "--mode",
        choices=("combined", "separate", "both"),
        default="both",
        help="组合图 / 单图 / 两者",
    )
    p.add_argument(
        "--format",
        dest="export_format",
        choices=("html", "png", "both"),
        default="html",
        help="导出 HTML、静态 PNG，或两者（PNG 需 kaleido）",
    )
    p.add_argument("--top-n", type=int, default=10, help="右侧吸引子显示数量（默认 10）")
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="只处理这些 model 子目录名；默认全部",
    )
    args = p.parse_args()
    mf = {str(m) for m in args.models} if args.models else None
    run(
        args.root.resolve(),
        args.out_dir,
        args.mode,
        args.export_format,
        args.top_n,
        mf,
    )


if __name__ == "__main__":
    main()
