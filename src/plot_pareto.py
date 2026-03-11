#!/usr/bin/env python3
"""
Pareto frontier: parameter count vs AIME 2026 I accuracy.

Usage:
    python src/plot_pareto.py [--log-dir results/AIME2026] [--out results/pareto.png]
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import CORRECT

# ── Known parameter counts (billions) ─────────────────────────────────────────
PARAM_COUNTS = {
    "lfm2.5-thinking:1.2b":           1.2,
    "deepseek-r1:1.5b":               1.5,
    "OpenReasoning-Nemotron-1.5B":    1.5,
    "POLARIS-4B-Preview":             4.0,
    "OpenThinker3-7B":                7.0,
    "OpenReasoning-Nemotron-7B":      7.0,
    "AceReason-Nemotron-1.1-7B":      7.0,
    "rnj-1:8b":                       8.0,
    "Klear-Reasoner-8B":              8.0,
    "DeepSeek-R1-0528-8B":            8.0,
}

MODEL_DISPLAY = {
    "ollama/lucifers/Polaris-4B-Preview.Q8_0:latest": "POLARIS-4B-Preview",
    "ollama/nvidia/AceReason-Nemotron-1.1-7B":        "AceReason-Nemotron-1.1-7B",
    "ollama/nvidia/OpenReasoning-Nemotron-1.5B":      "OpenReasoning-Nemotron-1.5B",
    "ollama/open-thoughts/OpenThinker3-7B":           "OpenThinker3-7B",
    "ollama/deepseek-r1:1.5b":                        "deepseek-r1:1.5b",
    "ollama/rnj-1:8b":                                "rnj-1:8b",
    "ollama/lfm2.5-thinking:1.2b":                    "lfm2.5-thinking:1.2b",
}

MODEL_COLOURS = {
    "POLARIS-4B-Preview":             "#A3D97A",
    "AceReason-Nemotron-1.1-7B":      "#6AC8EB",
    "OpenReasoning-Nemotron-1.5B":    "#F4B94A",
    "OpenThinker3-7B":                "#C8A87A",
    "deepseek-r1:1.5b":               "#F4896A",
    "rnj-1:8b":                       "#9B59B6",
    "lfm2.5-thinking:1.2b":           "#58D68D",
}


def load_2026(log_dir: Path) -> dict[str, float]:
    runs: dict[str, list[float]] = defaultdict(list)
    for path in sorted(log_dir.glob("*.eval")):
        try:
            log = read_eval_log(str(path), header_only=False)
            if log.status != "success":
                continue
            display = MODEL_DISPLAY.get(log.eval.model)
            if display is None:
                continue
            samples = log.samples or []
            if not samples:
                continue
            correct = sum(
                1 for s in samples
                if s.scores and list(s.scores.values())[0].value == CORRECT
            )
            runs[display].append(correct / len(samples) * 100)
        except Exception:
            continue
    return {name: sum(scores) / len(scores) for name, scores in runs.items()}


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Return the Pareto-efficient subset (maximise accuracy, minimise params).
    A point is on the frontier if no other point has fewer-or-equal params
    AND higher-or-equal accuracy (with at least one strictly better).
    Returns points sorted by params ascending.
    """
    pts = sorted(points, key=lambda p: (p[0], -p[1]))  # sort by params asc, accuracy desc
    frontier = []
    best_acc = -1.0
    for params, acc in pts:
        if acc > best_acc:
            frontier.append((params, acc))
            best_acc = acc
    return frontier


def plot(log_dir: Path, out: Path) -> None:
    scores = load_2026(log_dir)

    # Build list of (name, params, accuracy)
    models = []
    for name, acc in scores.items():
        params = PARAM_COUNTS.get(name)
        if params is None:
            print(f"  WARNING: no param count for '{name}' — skipping")
            continue
        models.append((name, params, acc))

    if not models:
        print("No data to plot.")
        return

    # Pareto frontier
    frontier_pts = pareto_frontier([(p, a) for _, p, a in models])
    # Extend the step-line to the right edge
    frontier_pts_extended = frontier_pts + [(frontier_pts[-1][0] * 1.5, frontier_pts[-1][1])]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#F5F5F5")
    ax.set_facecolor("#F5F5F5")

    # Draw Pareto step-line
    fx = [p[0] for p in frontier_pts_extended]
    fy = [p[1] for p in frontier_pts_extended]
    ax.step(fx, fy, where="post", color="#E74C3C", linewidth=2,
            linestyle="--", zorder=2, label="Pareto frontier")
    ax.fill_between(fx, fy, step="post", alpha=0.08, color="#E74C3C", zorder=1)

    # Scatter all models
    frontier_set = set(frontier_pts)
    for name, params, acc in sorted(models, key=lambda x: x[1]):
        colour = MODEL_COLOURS.get(name, "#AAAAAA")
        on_frontier = (params, acc) in frontier_set
        ax.scatter(params, acc,
                   s=120 if on_frontier else 80,
                   color=colour,
                   edgecolors="#333" if on_frontier else "white",
                   linewidths=1.5 if on_frontier else 0.5,
                   zorder=5)

        # Label — nudge to avoid overlap
        x_off = params * 0.04
        y_off = 1.2
        ax.annotate(
            name,
            (params, acc),
            xytext=(params + x_off, acc + y_off),
            fontsize=8,
            color="#222",
            va="bottom",
            path_effects=[pe.withStroke(linewidth=2, foreground="#F5F5F5")],
        )

    ax.set_xlabel("Parameter count (B)", fontsize=12)
    ax.set_ylabel("AIME 2026 I accuracy (%)", fontsize=12)
    ax.set_title("Pareto frontier — size vs. AIME 2026 I performance", fontsize=13, fontweight="bold")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}B"))
    ax.set_ylim(0, 85)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.7)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")

    print(f"\n{'Model':<40} {'Params':>8}  {'Acc':>7}  Pareto")
    print("-" * 65)
    for name, params, acc in sorted(models, key=lambda x: -x[2]):
        on = "  ★" if (params, acc) in frontier_set else ""
        print(f"{name:<40} {params:>6.1f}B  {acc:>6.1f}%{on}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="results/AIME2026", type=Path)
    parser.add_argument("--out", default="results/pareto.png", type=Path)
    args = parser.parse_args()
    plot(args.log_dir, args.out)
