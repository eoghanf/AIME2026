#!/usr/bin/env python3
"""
Phase 3: Aggregate Inspect eval logs into results/summary.md

Reads all .eval files in results/ and produces a ranked summary table.
"""

from pathlib import Path

from inspect_ai.log import read_eval_log
from inspect_ai.scorer import CORRECT

RESULTS_DIR = Path(__file__).parent.parent / "results"
SUMMARY_FILE = RESULTS_DIR / "summary.md"


def collect_results() -> list[dict]:
    rows = []
    for f in sorted(RESULTS_DIR.glob("*.eval")):
        try:
            ilog = read_eval_log(str(f))
        except Exception as e:
            print(f"  Warning: could not read {f.name}: {e}")
            continue

        if ilog.status != "success":
            print(f"  Skipping {f.name} — status={ilog.status}")
            continue

        model_str = ilog.eval.model          # e.g. "ollama/qwen3.5:0.8b"
        backend, _, model_name = model_str.partition("/")

        samples = ilog.samples or []
        total = len(samples)
        correct = sum(
            1 for s in samples
            if s.scores and list(s.scores.values())[0].value == CORRECT
        )
        accuracy = correct / total if total > 0 else 0.0

        rows.append({
            "model": model_name,
            "backend": backend.capitalize(),
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        })

    return sorted(rows, key=lambda r: r["correct"], reverse=True)


def main():
    rows = collect_results()
    if not rows:
        print(f"No completed Inspect evaluation logs found in {RESULTS_DIR}/")
        return

    # Write summary.md
    lines = [
        "# AIME 2026 I — Results",
        "",
        "| Model | Backend | Score | Accuracy |",
        "|-------|---------|-------|----------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['backend']} | {r['correct']}/{r['total']} | {r['accuracy']*100:.1f}% |"
        )
    lines.append("")
    SUMMARY_FILE.write_text("\n".join(lines))
    print(f"Summary written: {SUMMARY_FILE}")

    # Console table
    print(f"\n{'Model':<32} {'Backend':<12} {'Score':<8} {'Accuracy'}")
    print("-" * 65)
    for r in rows:
        print(
            f"  {r['model']:<30} {r['backend']:<12} "
            f"{r['correct']}/{r['total']:<6} {r['accuracy']*100:.1f}%"
        )


if __name__ == "__main__":
    main()
