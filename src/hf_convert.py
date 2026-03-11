#!/usr/bin/env python3
"""
HuggingFace → GGUF → Ollama conversion pipeline.

Pipeline:
  1. Clone/update llama.cpp
  2. Build llama-quantize (needed for Q4_K_M)
  3. Download model weights from HuggingFace
  4. Convert to F16 GGUF  (convert_hf_to_gguf.py)
  5. Quantize F16 → Q4_K_M (llama-quantize)
  6. Register with Ollama via `ollama create`

Q4_K_M (~4–5 GB for a 7-8B model) fits comfortably in 8 GB VRAM;
Q8_0 (~8.7 GB) does not.
"""

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger("aime.hf_convert")

TOOLS_DIR = Path(__file__).parent.parent / "tools"
LLAMA_CPP_DIR = TOOLS_DIR / "llama.cpp"
GGUF_DIR = TOOLS_DIR / "gguf"
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
QUANTISATION = "Q4_K_M"


# ── llama.cpp setup ───────────────────────────────────────────────────────────

def ensure_llama_cpp() -> bool:
    """Clone llama.cpp if not present and install Python conversion deps."""
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    if not LLAMA_CPP_DIR.exists():
        log.info("[HF] Cloning llama.cpp into %s …", LLAMA_CPP_DIR)
        result = subprocess.run(
            ["git", "clone", "--depth=1", LLAMA_CPP_REPO, str(LLAMA_CPP_DIR)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.error("[HF] git clone failed:\n%s", result.stderr.strip())
            return False
        log.info("[HF] llama.cpp cloned.")
    else:
        log.info("[HF] llama.cpp already present at %s", LLAMA_CPP_DIR)

    req_file = LLAMA_CPP_DIR / "requirements" / "requirements-convert_hf_to_gguf.txt"
    if req_file.exists():
        log.info("[HF] Installing conversion requirements …")
        subprocess.run(
            ["uv", "pip", "install", "-q", "--index-strategy", "unsafe-best-match",
             "-r", str(req_file)],
            check=False,
        )
    else:
        subprocess.run(
            ["uv", "pip", "install", "-q", "--index-strategy", "unsafe-best-match",
             "numpy", "sentencepiece", "transformers", "gguf", "protobuf"],
            check=False,
        )
    return True


def ensure_llama_quantize() -> Path | None:
    """Build the llama-quantize binary from llama.cpp if not already built."""
    candidates = [
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize",
        LLAMA_CPP_DIR / "build" / "llama-quantize",
    ]
    for p in candidates:
        if p.exists():
            return p

    log.info("[HF] Building llama-quantize (this takes a minute) …")
    build_dir = LLAMA_CPP_DIR / "build"

    result = subprocess.run(
        ["cmake", "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF"],
        cwd=str(LLAMA_CPP_DIR), capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error("[HF] cmake configure failed:\n%s", result.stderr[-1000:].strip())
        return None

    result = subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", "Release",
         "--target", "llama-quantize", "-j4"],
        cwd=str(LLAMA_CPP_DIR), capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error("[HF] cmake build failed:\n%s", result.stderr[-1000:].strip())
        return None

    for p in candidates:
        if p.exists():
            log.info("[HF] llama-quantize built: %s", p)
            return p

    log.error("[HF] llama-quantize binary not found after build")
    return None


# ── Download ──────────────────────────────────────────────────────────────────

def hf_download(hf_model_id: str, local_dir: Path) -> bool:
    """Download model files from HuggingFace Hub (safetensors preferred)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        subprocess.run(["uv", "pip", "install", "-q", "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download

    log.info("[HF] Downloading %s → %s …", hf_model_id, local_dir)
    for ignore in (
        ["*.gguf", "*.bin", "*.pt", "*.pth", "original/*", "flax_model*", "tf_model*"],
        ["*.gguf", "original/*", "flax_model*", "tf_model*"],  # fallback: include .bin
    ):
        try:
            snapshot_download(repo_id=hf_model_id, local_dir=str(local_dir),
                              ignore_patterns=ignore)
            log.info("[HF] Download complete: %s", hf_model_id)
            return True
        except Exception as e:
            log.warning("[HF] Download attempt failed (%s), retrying …", e)

    log.error("[HF] All download attempts failed for %s", hf_model_id)
    return False


# ── Conversion ────────────────────────────────────────────────────────────────

def hf_convert_to_f16(model_dir: Path, f16_path: Path) -> bool:
    """Convert HuggingFace weights to F16 GGUF using llama.cpp's script."""
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        log.error("[HF] convert_hf_to_gguf.py not found at %s", convert_script)
        return False

    log.info("[HF] Converting to F16 GGUF: %s → %s …", model_dir, f16_path)
    result = subprocess.run(
        [sys.executable, str(convert_script),
         str(model_dir), "--outtype", "f16", "--outfile", str(f16_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error("[HF] F16 conversion failed:\n%s", result.stderr[-3000:].strip())
        return False
    log.info("[HF] F16 GGUF written: %s (%.1f GB)", f16_path,
             f16_path.stat().st_size / 1e9)
    return True


def quantize_gguf(src: Path, dst: Path, quant_type: str = QUANTISATION) -> bool:
    """Quantize a GGUF file using llama-quantize."""
    quantize_bin = ensure_llama_quantize()
    if quantize_bin is None:
        return False

    log.info("[HF] Quantizing %s → %s (%s) …", src.name, dst.name, quant_type)
    result = subprocess.run(
        [str(quantize_bin), str(src), str(dst), quant_type],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error("[HF] Quantization failed:\n%s", result.stderr[-2000:].strip())
        return False
    log.info("[HF] Quantized GGUF written: %s (%.1f GB)", dst,
             dst.stat().st_size / 1e9)
    return True


# ── Ollama registration ───────────────────────────────────────────────────────

def ollama_create_from_gguf(ollama_name: str, gguf_path: Path, num_ctx: int) -> bool:
    """Register a GGUF file as an Ollama model."""
    modelfile_content = f"FROM {gguf_path}\nPARAMETER num_ctx {num_ctx}\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix="_Modelfile", delete=False) as f:
        f.write(modelfile_content)
        modelfile_path = f.name

    log.info("[HF] Creating Ollama model '%s' from %s …", ollama_name, gguf_path)
    result = subprocess.run(
        ["ollama", "create", ollama_name, "-f", modelfile_path],
        capture_output=True, text=True,
    )
    Path(modelfile_path).unlink(missing_ok=True)

    if result.returncode != 0:
        log.error("[HF] ollama create failed:\n%s", result.stderr.strip())
        return False
    log.info("[HF] Ollama model ready: %s", ollama_name)
    return True


# ── Top-level entry point ─────────────────────────────────────────────────────

def hf_to_ollama(hf_model_id: str, num_ctx: int) -> bool:
    """
    Full pipeline: HuggingFace model → Q4_K_M GGUF → Ollama.

    Caches intermediate files under tools/gguf/ to avoid re-work on re-runs.
    If a Q8_0 GGUF already exists from a previous run it is reused as the
    quantization source rather than re-downloading/re-converting.
    """
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = hf_model_id.replace("/", "_")
    local_dir = GGUF_DIR / safe_name
    f16_path = GGUF_DIR / f"{safe_name}_F16.gguf"
    q8_path = GGUF_DIR / f"{safe_name}_Q8_0.gguf"    # legacy path from earlier runs
    final_path = GGUF_DIR / f"{safe_name}_{QUANTISATION}.gguf"

    if final_path.exists():
        log.info("[HF] %s already exists — skipping conversion.", final_path.name)
    else:
        if not ensure_llama_cpp():
            return False

        # Determine the best source for quantization
        if q8_path.exists():
            log.info("[HF] Reusing existing Q8_0 as quantization source: %s", q8_path)
            src = q8_path
        elif f16_path.exists():
            log.info("[HF] Reusing existing F16 as quantization source: %s", f16_path)
            src = f16_path
        else:
            if not hf_download(hf_model_id, local_dir):
                return False
            if not hf_convert_to_f16(local_dir, f16_path):
                return False
            src = f16_path

        if not quantize_gguf(src, final_path):
            return False

        # Clean up large intermediate F16 file (Q8_0 is kept as it may be useful)
        if f16_path.exists() and src != f16_path:
            pass  # src was q8, leave f16 untouched (wasn't created this run)
        elif f16_path.exists() and src == f16_path:
            log.info("[HF] Removing F16 intermediate to free disk space …")
            f16_path.unlink()

    # Rename old Q8_0 → descriptive name so future runs find it correctly
    old_unnamed = GGUF_DIR / f"{safe_name}.gguf"
    if old_unnamed.exists() and not q8_path.exists():
        old_unnamed.rename(q8_path)
        log.info("[HF] Renamed legacy GGUF to %s", q8_path.name)

    return ollama_create_from_gguf(hf_model_id, final_path, num_ctx)
