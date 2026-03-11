#!/usr/bin/env python3
"""
Plot AIME 2026 I results by model with error bars.

Usage:
    python src/plot.py [--log-dir results/AIME2026] [--out results/aime2026_scores.png]

Error bars:
  - Multiple runs for a model → std error of the mean across runs
  - Single run              → binomial 95% CI half-width  sqrt(p*(1-p)/n)
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import CORRECT

N_PROBLEMS = 15


def short_name(model: str) -> str:
    """Strip backend prefix and shorten for display."""
    name = model.removeprefix("ollama/").removeprefix("anthropic/").removeprefix("openai/").removeprefix("google/")
    # Collapse common long prefixes
    replacements = {
        "lucifers/": "",
        "nvidia/": "",
        "open-thoughts/": "",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name


def load_results(log_dir: Path) -> dict[str, list[float]]:
    """Return {model: [score_run1, score_run2, ...]} using only successful logs."""
    runs: dict[str, list[float]] = defaultdict(list)
    for path in sorted(log_dir.glob("*.eval")):
        try:
            log = read_eval_log(str(path), header_only=False)
        except Exception as e:
            print(f"  SKIP {path.name}: {e}")
            continue
        if log.status != "success":
            print(f"  SKIP {path.name}: status={log.status}")
            continue
        samples = log.samples or []
        total = len(samples)
        if total == 0:
            continue
        correct = sum(
            1 for s in samples
            if s.scores and list(s.scores.values())[0].value == CORRECT
        )
        runs[log.eval.model].append(correct / total)
    return dict(runs)


def compute_stats(scores: list[float]) -> tuple[float, float]:
    """Return (mean, error_half_width) where error is 95% CI half-width."""
    n_runs = len(scores)
    mean = sum(scores) / n_runs
    if n_runs >= 2:
        variance = sum((s - mean) ** 2 for s in scores) / (n_runs - 1)
        # Standard error of the mean, ×1.96 for 95% CI
        err = 1.96 * math.sqrt(variance / n_runs)
    else:
        # Binomial 95% CI half-width
        err = 1.96 * math.sqrt(mean * (1 - mean) / N_PROBLEMS)
    return mean, err


def plot(log_dir: Path, out: Path) -> None:
    print(f"Reading logs from {log_dir} …")
    runs = load_results(log_dir)

    if not runs:
        print("No successful eval logs found.")
        return

    # Build sorted rows
    rows = []
    for model, scores in runs.items():
        mean, err = compute_stats(scores)
        n_runs = len(scores)
        rows.append((short_name(model), mean, err, n_runs, scores))
    rows.sort(key=lambda r: r[1])  # ascending so highest is at top of hbar

    labels   = [r[0] for r in rows]
    means    = [r[1] * 100 for r in rows]          # convert to percentage
    errors   = [r[2] * 100 for r in rows]
    n_runs   = [r[3] for r in rows]
    all_runs = [r[4] for r in rows]

    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.7 + 1.5)))

    colours = ["#4e9af1" if n == 1 else "#27ae60" for n in n_runs]
    bars = ax.barh(labels, means, xerr=errors, color=colours,
                   error_kw=dict(elinewidth=1.5, capsize=4, ecolor="#333"),
                   height=0.6, zorder=3)

    # Annotate each bar: "X/15" or "avg X/15 (k runs)"
    for i, (bar, row) in enumerate(zip(bars, rows)):
        _, mean, _, n, scores = row
        score_15 = mean * N_PROBLEMS
        if n == 1:
            label = f"{scores[0]*N_PROBLEMS:.0f}/{N_PROBLEMS}"
        else:
            label = f"avg {score_15:.1f}/{N_PROBLEMS}  ({n} runs)"
        x = bar.get_width() + errors[i] + 0.5
        ax.text(x, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=9)

    ax.set_xlabel("Score (%)", fontsize=11)
    ax.set_title("AIME 2026 I — Model Performance", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 115)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.axvline(100 / 3, color="#aaa", linestyle="--", linewidth=0.8, label="Random (1/3)")
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend for colour coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#27ae60", label="Multiple runs (SEM error bars)"),
        Patch(facecolor="#4e9af1", label="Single run (binomial 95% CI)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    # Print summary table
    print(f"\n{'Model':<40} {'Runs':>4}  {'Mean':>6}  {'±95%CI':>7}  Individual scores")
    print("-" * 90)
    for label, mean, err, n, scores in reversed(rows):
        score_strs = "  ".join(f"{s*N_PROBLEMS:.0f}/{N_PROBLEMS}" for s in scores)
        print(f"{label:<40} {n:>4}  {mean*100:>5.1f}%  ±{err*100:>5.1f}%  {score_strs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="results/AIME2026", type=Path)
    parser.add_argument("--out", default="results/aime2026_scores.png", type=Path)
    args = parser.parse_args()
    plot(args.log_dir, args.out)
