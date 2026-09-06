#!/usr/bin/env python3
"""
AIME 2025 vs AIME 2026 I comparison chart.

AIME 2025 values read from graphs/AIMEresults.png (Avg@64).
AIME 2026 I values loaded from results/AIME2026/*.eval (mean across runs).

Usage:
    python src/plot_comparison.py [--out results/aime_comparison.png]
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import CORRECT

# ── AIME 2025 data (read from AIMEresults.png, Avg@64 column, Qwen3-8B removed) ──
AIME_2025 = {
    "Klear-Reasoner-8B":          83.2,
    "POLARIS-4B-Preview":         79.4,
    "OpenReasoning-Nemotron-7B":  78.2,
    "DeepSeek-R1-0528-8B":        76.3,
    "AceReason-Nemotron-1.1-7B":  64.8,
}

# ── Model name mapping: eval log model string → display name ──────────────────
MODEL_DISPLAY = {
    "ollama/Kwai-Klear/Klear-Reasoner-8B":            "Klear-Reasoner-8B",
    "ollama/deepseek-r1:8b-0528-qwen3-q4_K_M":       "DeepSeek-R1-0528-8B",
    "ollama/lucifers/Polaris-4B-Preview.Q8_0:latest": "POLARIS-4B-Preview",
    "ollama/nvidia/AceReason-Nemotron-1.1-7B":        "AceReason-Nemotron-1.1-7B",
    "ollama/nvidia/OpenReasoning-Nemotron-1.5B":      "OpenReasoning-Nemotron-1.5B",
    "ollama/nvidia/OpenReasoning-Nemotron-7B":        "OpenReasoning-Nemotron-7B",
    "ollama/open-thoughts/OpenThinker3-7B":           "OpenThinker3-7B",
    "ollama/deepseek-r1:1.5b":                        "DeepSeek-R1:1.5b",
    "ollama/rnj-1:8b":                                "rnj-1:8b",
    "ollama/lfm2.5-thinking:1.2b":                    "lfm2.5-thinking:1.2b",
}

# ── Colours: consistent with AIMEresults.png where models overlap ──────────────
MODEL_COLOURS = {
    "Klear-Reasoner-8B":              "#B0A0E8",   # lavender (hatch in original)
    "POLARIS-4B-Preview":             "#A3D97A",   # light green
    "OpenReasoning-Nemotron-7B":      "#F4B94A",   # orange
    "OpenReasoning-Nemotron-1.5B":    "#F4B94A",   # same family → same colour
    "DeepSeek-R1-0528-8B":            "#F4896A",   # salmon
    "DeepSeek-R1:1.5b":               "#F4896A",   # same family → same colour
    "AceReason-Nemotron-1.1-7B":      "#6AC8EB",   # sky blue
    "OpenThinker3-7B":                "#C8A87A",   # warm brown
    "rnj-1:8b":                       "#9B59B6",   # purple
    "lfm2.5-thinking:1.2b":           "#58D68D",   # mint green
}

# No hatching on any bars — hatching is reserved for estimates
HATCH: dict[str, str] = {}

# Models whose 2026 I bar is an initial estimate (striped), not yet measured
ESTIMATE_2026 = {
    "Klear-Reasoner-8B":         70.0,
    "OpenReasoning-Nemotron-7B": 70.0,
    "DeepSeek-R1-0528-8B":       70.0,
}
ESTIMATE_HATCH = "////"


def load_2026(log_dir: Path) -> tuple[dict[str, float], dict[str, int]]:
    """Return ({display_name: mean_score_%}, {display_name: n_runs}) from successful eval logs."""
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
    means = {name: sum(scores) / len(scores) for name, scores in runs.items()}
    n_runs = {name: len(scores) for name, scores in runs.items()}
    return means, n_runs


def plot(log_dir: Path, out: Path) -> None:
    aime_2026, run_counts = load_2026(log_dir)

    # Merge measured 2026 results with estimates for unmeasured models
    aime_2026_full = {**ESTIMATE_2026, **aime_2026}  # measured values override estimates

    # All unique models across both years, ordered by 2025 score desc then 2026 desc
    all_models = list(dict.fromkeys(
        list(AIME_2025.keys()) +
        sorted(aime_2026_full.keys(), key=lambda m: -aime_2026_full[m])
    ))

    groups = {
        "AIME 2025\n(Avg@64)": AIME_2025,
        "AIME 2026 I\n(Our eval)": aime_2026_full,
    }
    group_labels = list(groups.keys())
    n_groups = len(group_labels)
    n_models = len(all_models)

    bar_width = 0.13
    group_span = n_models * bar_width
    group_gap = 0.55
    group_centers = np.arange(n_groups) * (group_span + group_gap)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#F5F5F5")
    ax.set_facecolor("#F5F5F5")

    for mi, model in enumerate(all_models):
        colour = MODEL_COLOURS.get(model, "#AAAAAA")
        offset = (mi - (n_models - 1) / 2) * bar_width

        for gi, (group_label, data) in enumerate(groups.items()):
            if model not in data:
                continue
            val = data[model]
            # Use hatch only for estimate bars in 2026 group
            is_estimate = (gi == 1 and model in ESTIMATE_2026 and model not in aime_2026)
            hatch = ESTIMATE_HATCH if is_estimate else ""
            x = group_centers[gi] + offset
            ax.bar(
                x, val,
                width=bar_width * 0.9,
                color=colour,
                hatch=hatch,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.text(
                x, val + 0.8,
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold",
                color="#333333",
            )
            # Show run count at the bottom of 2026 bars (measured runs only).
            # Rotated 90° so adjacent labels don't collide at this bar width.
            if gi == 1 and model in run_counts:
                n = run_counts[model]
                ax.text(
                    x, 1.5,
                    f"n={n}",
                    ha="center", va="bottom",
                    rotation=90,
                    fontsize=8, fontweight="bold", color="white",
                    zorder=4,
                )

    # X-axis group labels — no ticks
    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=12)
    ax.tick_params(axis="x", length=0)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.grid(axis="y", color="white", linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False)

    # Top legend: model colours
    model_handles = [
        mpatches.Patch(
            facecolor=MODEL_COLOURS.get(m, "#AAAAAA"),
            edgecolor="white",
            label=m,
        )
        for m in all_models
    ]
    top_legend = fig.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=8.5,
        frameon=False,
        handlelength=1.5,
        handleheight=1.0,
    )

    # Bottom legend: solid vs striped explanation
    solid_patch  = mpatches.Patch(facecolor="#AAAAAA", edgecolor="white", label="Solid bars — final data")
    stripe_patch = mpatches.Patch(facecolor="#AAAAAA", hatch=ESTIMATE_HATCH, edgecolor="white", label="Striped bars — initial estimates")
    ax.legend(
        handles=[solid_patch, stripe_patch],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        fontsize=9,
        frameon=False,
        handlelength=2.0,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")

    # Console summary
    print(f"\n{'Model':<35} {'2025':>8}  {'2026 I':>8}")
    print("-" * 56)
    for m in all_models:
        v25 = f"{AIME_2025[m]:.1f}%" if m in AIME_2025 else "—"
        v26 = f"{aime_2026[m]:.1f}%" if m in aime_2026 else "—"
        print(f"{m:<35} {v25:>8}  {v26:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="results/AIME2026", type=Path)
    parser.add_argument("--out", default="results/aime_comparison.png", type=Path)
    args = parser.parse_args()
    plot(args.log_dir, args.out)
