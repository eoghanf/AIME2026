#!/usr/bin/env python3
"""
Download GGUF models from HuggingFace and register them with Ollama.

These are the `?`-prefixed models in models.yaml that aren't in the Ollama
library. For each one this script will:
  1. Download the Q4_K_M GGUF file from HuggingFace
  2. Create an Ollama Modelfile pointing at it
  3. Run `ollama create <tag>`
  4. Delete the GGUF file (Ollama has imported it; no need to keep it)
  5. Remove the `?` prefix from models.yaml so main.py picks it up

Usage:
    python src/scripts/pull_gguf_models.py
    python src/scripts/pull_gguf_models.py --dry-run
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
MODELS_FILE = ROOT / "models.yaml"

# Ollama model tag → HuggingFace repo (Q4_K_M GGUF)
GGUF_MODELS = {
    "klear-reasoner:8b":         "mradermacher/Klear-Reasoner-8B-GGUF",
    "openreasoning-nemotron:7b": "bartowski/nvidia_OpenReasoning-Nemotron-7B-GGUF",
    "acereason-nemotron:7b":     "bartowski/nvidia_AceReason-Nemotron-1.1-7B-GGUF",
    "openthinker3:7b":           "bartowski/open-thoughts_OpenThinker3-7B-GGUF",
}


def hf_download(repo: str, dest_dir: Path) -> Path | None:
    print(f"  Downloading {repo} (Q4_K_M) …")
    result = subprocess.run([
        "hf", "download", repo,
        "--include", "*Q4_K_M*",
        "--local-dir", str(dest_dir),
    ])
    if result.returncode != 0:
        print(f"  ERROR: download failed for {repo}", file=sys.stderr)
        return None
    gguf_files = list(dest_dir.rglob("*Q4_K_M*.gguf"))
    if not gguf_files:
        print(f"  ERROR: no Q4_K_M GGUF found in {dest_dir}", file=sys.stderr)
        return None
    return gguf_files[0]


def ollama_create(tag: str, gguf_path: Path) -> bool:
    modelfile_path = gguf_path.parent / "Modelfile"
    modelfile_path.write_text(f"FROM {gguf_path}\n")
    print(f"  Running: ollama create {tag} …")
    result = subprocess.run(["ollama", "create", tag, "-f", str(modelfile_path)])
    modelfile_path.unlink(missing_ok=True)
    return result.returncode == 0


def remove_question_mark(tag: str) -> None:
    text = MODELS_FILE.read_text()
    updated = text.replace(f"?{tag}", tag, 1)
    if updated != text:
        MODELS_FILE.write_text(updated)
        print(f"  models.yaml updated: removed `?` from {tag}")
    else:
        print(f"  Warning: could not find `?{tag}` in models.yaml", file=sys.stderr)


def process(tag: str, hf_repo: str) -> bool:
    print(f"\n{'='*60}")
    print(f"Model : {tag}")
    print(f"HF    : {hf_repo}")
    print(f"{'='*60}")
    with tempfile.TemporaryDirectory(prefix="gguf_") as tmpdir:
        gguf_path = hf_download(hf_repo, Path(tmpdir))
        if gguf_path is None:
            return False
        print(f"  GGUF : {gguf_path.name} ({gguf_path.stat().st_size / 1e9:.1f} GB)")
        if not ollama_create(tag, gguf_path):
            print(f"  ERROR: ollama create failed for {tag}", file=sys.stderr)
            return False
    print(f"  Done: {tag} registered in Ollama")
    remove_question_mark(tag)
    return True


def main():
    parser = argparse.ArgumentParser(description="Create GGUF-based Ollama models from HuggingFace")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    args = parser.parse_args()

    print(f"GGUF models to create ({len(GGUF_MODELS)}):")
    for tag, repo in GGUF_MODELS.items():
        print(f"  {tag}  ←  {repo}")

    if args.dry_run:
        print("\n(dry run — nothing downloaded)")
        return

    if subprocess.run(["which", "hf"], capture_output=True).returncode != 0:
        print("ERROR: hf not found. Install with: uv pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    failed = []
    for tag, repo in GGUF_MODELS.items():
        if not process(tag, repo):
            failed.append(tag)

    print(f"\n{'='*60}")
    print(f"Done. {len(GGUF_MODELS) - len(failed)}/{len(GGUF_MODELS)} models created.")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
