"""
Bipartite Network Graph: 模型 × 数理概念 认知图谱
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent

MODEL_DIRS = [
    "deepseek",
    "anthropic-claude-haiku-4.5",
    "google-gemini-2.5-flash",
    "openai-gpt-5.4-mini",
    "kimi-k2.6",
    "MiniMax-M2.5",
    "doubao-seed-2-0-lite-260215",
]

MODEL_DISPLAY = {
    "deepseek":                     "DeepSeek",
    "anthropic-claude-haiku-4.5":   "Claude 4.5",
    "google-gemini-2.5-flash":      "Gemini 2.5",
    "openai-gpt-5.4-mini":         "GPT-5.4",
    "kimi-k2.6":                    "Kimi K2.6",
    "MiniMax-M2.5":                 "MiniMax M2.5",
    "doubao-seed-2-0-lite-260215":  "Doubao Seed",
}

MODEL_COLORS = {
    "deepseek":                     "#7B68EE",
    "anthropic-claude-haiku-4.5":   "#E8A838",
    "google-gemini-2.5-flash":      "#FF6B8A",
    "openai-gpt-5.4-mini":         "#4ECDC4",
    "kimi-k2.6":                    "#45B7D1",
    "MiniMax-M2.5":                 "#96CEB4",
    "doubao-seed-2-0-lite-260215":  "#FF8C42",
}

TAXONOMY_CATS = [
    "Personal&Existential",
    "Professional&Economic",
    "Relational&Intimate",
    "Societal&Ethical",
]

MIN_MODEL_COUNT = 3
MAX_CONCEPTS = 18


MERGED_JSON = ROOT / "merged_top50.json"


def load_all_data() -> dict[str, Counter]:
    """Read merged_top50.json — each item already contains canonical
    (post-clustering) concept names and their counts per model×category."""
    model_concepts: dict[str, Counter] = {m: Counter() for m in MODEL_DIRS}

    with open(MERGED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data.get("items", []):
        model = item.get("model", "")
        if model not in model_concepts:
            continue
        for entry in item.get("top_names", []):
            name = entry.get("name", "").strip().lower()
            count = entry.get("count", 0)
            if name and count > 0:
                model_concepts[model][name] += count

    return model_concepts


def build_bipartite_graph(model_concepts, min_model_count=MIN_MODEL_COUNT,
                          max_concepts=MAX_CONCEPTS):
    concept_model_sets: dict[str, set] = defaultdict(set)
    concept_total_weight: Counter = Counter()
    for model, concepts in model_concepts.items():
        for concept, count in concepts.items():
            concept_model_sets[concept].add(model)
            concept_total_weight[concept] += count

    shared = {c for c, ms in concept_model_sets.items() if len(ms) >= min_model_count}
    ranked = sorted(shared, key=lambda c: -concept_total_weight[c])
    top_concepts = ranked[:max_concepts]
    top_set = set(top_concepts)

    G = nx.Graph()
    for model in MODEL_DIRS:
        G.add_node(model, bipartite=0)
    for concept in top_concepts:
        G.add_node(concept, bipartite=1)
    for model, concepts in model_concepts.items():
        for concept, count in concepts.items():
            if concept in top_set:
                G.add_edge(model, concept, weight=count)
    return G, top_concepts


def _repel_nodes(pos, nodes, min_dist=1.8, iterations=600):
    """Push nodes apart until every pair is at least min_dist away."""
    cpos = {c: list(pos[c]) for c in nodes}
    for _ in range(iterations):
        moved = False
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                dx = cpos[a][0] - cpos[b][0]
                dy = cpos[a][1] - cpos[b][1]
                d = math.sqrt(dx * dx + dy * dy)
                if d < min_dist and d > 1e-6:
                    force = (min_dist - d) * 0.4
                    ux, uy = dx / d, dy / d
                    cpos[a][0] += ux * force
                    cpos[a][1] += uy * force
                    cpos[b][0] -= ux * force
                    cpos[b][1] -= uy * force
                    moved = True
        if not moved:
            break
    for c in nodes:
        pos[c] = tuple(cpos[c])
    return pos


def bipartite_force_layout(G, model_nodes, concept_nodes, seed=42):
    n_models = len(model_nodes)
    n_concepts = len(concept_nodes)

    pos = {}
    for i, m in enumerate(model_nodes):
        angle = 2 * math.pi * i / n_models - math.pi / 2
        pos[m] = (6.0 * math.cos(angle), 6.0 * math.sin(angle))

    rng = np.random.RandomState(seed)
    for i, c in enumerate(concept_nodes):
        angle = 2 * math.pi * i / n_concepts + 0.3
        r = 1.5 + 2.5 * rng.random()
        pos[c] = (r * math.cos(angle), r * math.sin(angle))

    pos = nx.spring_layout(
        G, pos=pos, fixed=model_nodes,
        k=6.0 / math.sqrt(max(len(G), 1)),
        iterations=400, seed=seed,
    )

    pos = _repel_nodes(pos, concept_nodes, min_dist=1.6, iterations=600)
    return pos


def draw_bipartite_network(G, pos, model_nodes, concept_nodes, output_path):
    fp = FontProperties(family="Times New Roman")

    fig, ax = plt.subplots(figsize=(32, 28))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # ── Edges ──
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    min_w = min(edge_weights) if edge_weights else 1

    for (u, v), w in zip(G.edges(), edge_weights):
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        norm_w = (w - min_w) / (max_w - min_w + 1e-9)
        alpha = 0.12 + 0.40 * norm_w
        lw = 3.2 + 7.5 * norm_w
        if u in MODEL_DIRS:
            color = MODEL_COLORS[u]
        elif v in MODEL_DIRS:
            color = MODEL_COLORS[v]
        else:
            color = "#888888"
        ax.plot(x, y, color=color, alpha=alpha, linewidth=lw, zorder=1)

    # ── Concept nodes + labels (label right below node) ──
    concept_total_w = {}
    for c in concept_nodes:
        total = sum(G[c][nbr].get("weight", 1) for nbr in G.neighbors(c))
        concept_total_w[c] = total
    max_tw = max(concept_total_w.values()) if concept_total_w else 1

    for c in concept_nodes:
        cx, cy = pos[c]
        norm_tw = concept_total_w[c] / max_tw
        r = 0.20 + 0.36 * norm_tw
        base_alpha = 0.55 + 0.35 * norm_tw

        circle = plt.Circle(
            (cx, cy), radius=r,
            facecolor="#b0b0c0", edgecolor="#8888a0",
            alpha=base_alpha, linewidth=1.5, zorder=3,
        )
        ax.add_patch(circle)

        glow = plt.Circle(
            (cx, cy), radius=r + 0.07,
            facecolor="none", edgecolor="#9999aa",
            alpha=0.10 + 0.15 * norm_tw, linewidth=3.0, zorder=2,
        )
        ax.add_patch(glow)

        dot_r = r * 0.22
        dots = [
            (cx, cy + dot_r * 1.2),
            (cx - dot_r * 1.1, cy - dot_r * 0.6),
            (cx + dot_r * 1.1, cy - dot_r * 0.6),
        ]
        for dx, dy in dots:
            ax.add_patch(plt.Circle(
                (dx, dy), radius=dot_r * 0.45,
                facecolor="#ffffff", edgecolor="none",
                alpha=base_alpha * 0.9, zorder=4,
            ))
        for i in range(3):
            for j in range(i + 1, 3):
                ax.plot(
                    [dots[i][0], dots[j][0]], [dots[i][1], dots[j][1]],
                    color="#ffffff", alpha=base_alpha * 0.6,
                    linewidth=0.6, zorder=4,
                )

        label = c.title() if len(c) < 26 else c[:23].title() + "..."
        font_size = 21.0 + 8.0 * norm_tw

        ax.text(
            cx, cy - r - 0.06, label,
            fontsize=font_size, fontweight="bold", color="#111111",
            ha="center", va="top", fontproperties=fp, zorder=5,
        )

    # ── Model nodes ──
    for m in model_nodes:
        mx, my = pos[m]
        color = MODEL_COLORS[m]
        display = MODEL_DISPLAY[m]

        ax.add_patch(plt.Circle(
            (mx, my), radius=0.55,
            facecolor=color, edgecolor="none", alpha=0.10, zorder=8,
        ))
        ax.add_patch(plt.Circle(
            (mx, my), radius=0.42,
            facecolor=color, edgecolor="#ffffff",
            linewidth=3.5, alpha=0.92, zorder=10,
        ))
        ax.add_patch(plt.Circle(
            (mx, my), radius=0.22,
            facecolor="#ffffff", edgecolor=color,
            linewidth=2.5, alpha=0.9, zorder=11,
        ))
        ax.add_patch(plt.Circle(
            (mx, my + 0.04), radius=0.07,
            facecolor=color, edgecolor="none", alpha=0.9, zorder=12,
        ))
        ax.add_patch(matplotlib.patches.Arc(
            (mx, my - 0.07), 0.18, 0.11,
            angle=0, theta1=0, theta2=180,
            edgecolor=color, linewidth=3, zorder=12,
        ))

        ax.text(
            mx, my - 0.62, display,
            fontsize=32, fontweight="bold", color="#111111",
            ha="center", va="top", fontproperties=fp, zorder=12,
        )

    # ── Legend (large, upper-right) ──
    legend_handles = []
    for m in MODEL_DIRS:
        legend_handles.append(
            mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_DISPLAY[m])
        )
    legend_handles.append(
        mpatches.Patch(color="#aaaaaa", label="Attractors")
    )
    leg = ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=28,
        facecolor="#f5f5f8",
        edgecolor="#cccccc",
        labelcolor="#111111",
        prop=FontProperties(family="Times New Roman", size=28),
        framealpha=0.95,
        borderpad=1.2,
        handlelength=2.2,
        handleheight=1.9,
    )
    leg.get_frame().set_linewidth(1.5)

    ax.set_xlim(-8.0, 8.0)
    ax.set_ylim(-8.0, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[bipartite] Saved to {output_path}")


def main():
    print("[bipartite] Loading data ...")
    model_concepts = load_all_data()
    for m in MODEL_DIRS:
        n = len(model_concepts[m])
        total = sum(model_concepts[m].values())
        print(f"  {MODEL_DISPLAY[m]:20s}: {n:>4d} unique, {total:>5d} total")

    print(f"\n[bipartite] Building graph (min_models={MIN_MODEL_COUNT}, top={MAX_CONCEPTS}) ...")
    G, concept_nodes = build_bipartite_graph(model_concepts)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("[bipartite] Layout ...")
    pos = bipartite_force_layout(G, MODEL_DIRS, concept_nodes)

    output_path = ROOT / "bipartite_network.png"
    print("[bipartite] Drawing ...")
    draw_bipartite_network(G, pos, MODEL_DIRS, concept_nodes, output_path)
    print("[bipartite] Done!")


if __name__ == "__main__":
    main()
