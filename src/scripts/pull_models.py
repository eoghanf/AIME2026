#!/usr/bin/env python3
"""
Pull all non-skipped Ollama models listed in models.yaml.

Usage:
    python src/scripts/pull_models.py
    python src/scripts/pull_models.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

MODELS_FILE = Path(__file__).parent.parent.parent / "models.yaml"


def get_ollama_models() -> list[str]:
    data = yaml.safe_load(MODELS_FILE.read_text())
    models = []
    for family in data.get("models", []):
        if family.get("backend") != "ollama":
            continue
        for name in family.get("models", []):
            if name.startswith("?"):
                print(f"  skip  {name[1:]}")
                continue
            models.append(name)
    return models


def pull(model: str) -> bool:
    print(f"\n{'='*60}")
    print(f"Pulling: {model}")
    print(f"{'='*60}")
    result = subprocess.run(["ollama", "pull", model])
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Pull Ollama models from models.yaml")
    parser.add_argument("--dry-run", action="store_true", help="List models without pulling")
    args = parser.parse_args()

    models = get_ollama_models()

    if not models:
        print("No Ollama models to pull.")
        return

    print(f"Models to pull ({len(models)}):")
    for m in models:
        print(f"  {m}")

    if args.dry_run:
        return

    failed = []
    for model in models:
        if not pull(model):
            failed.append(model)

    print(f"\n{'='*60}")
    print(f"Done. {len(models) - len(failed)}/{len(models)} pulled successfully.")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
