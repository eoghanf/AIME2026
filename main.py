#!/usr/bin/env python3
"""
AIME 2026 Evaluation Harness — Main Entry Point

Orchestrates the full pipeline using the Inspect AI framework:
  1. Verify (or scrape) problem data in data/aime_2026_i.json
  2. For each model in models.yaml:
     - Skip if a result already exists in results/
     - Pull the model (Ollama), run inspect eval, write .eval log
     - Remove the model from the Ollama cache to free disk space
  3. Generate results/summary.md

Usage:
    python main.py
"""

import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Logging — must be configured before any library imports ───────────────────
ROOT = Path(__file__).parent
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
_run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = LOGS_DIR / f"run_{_run_ts}.log"

_fmt = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s")
_fh = logging.FileHandler(_log_file, encoding="utf-8")
_fh.setFormatter(_fmt)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_fmt)
logging.root.setLevel(logging.WARNING)
log = logging.getLogger("aime")
log.setLevel(logging.INFO)
log.addHandler(_fh)
log.addHandler(_ch)
log.propagate = False  # don't let inspect_ai's root-level setLevel suppress us

import yaml
from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.log._log import EvalLog
from inspect_ai.scorer import CORRECT

load_dotenv()

OLLAMA_NUM_PARALLEL = 1

# Per-run sampling seed: seed = SEED_BASE + run_number (1-based), so every run
# slot has a distinct, reproducible seed (e.g. runs 1-4 → 1235-1238). Recorded
# in each eval log's metadata; rerunning a run slot reuses the same seed.
SEED_BASE = 1234
RUN_TEMPERATURE = 1.0
RUN_TOP_P = 0.95

DATA_FILE = ROOT / "data" / "aime_2026_i.json"
RESULTS_DIR = ROOT / "results"
LOG_DIR = RESULTS_DIR / "AIME2026"
MODELS_FILE = ROOT / "models.yaml"

# src/ must be on path for task.py import
sys.path.insert(0, str(ROOT))
from src.task import MAX_TOKENS, aime_2026_i  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Problem data
# ══════════════════════════════════════════════════════════════════════════════

def ensure_data() -> None:
    import json
    if DATA_FILE.exists():
        try:
            probs = json.loads(DATA_FILE.read_text()).get("problems", [])
            if len(probs) == 15:
                log.info("Data: %s present with 15 problems — skipping scrape.", DATA_FILE.name)
                return
            log.warning("Data file has %d problems (expected 15) — re-scraping.", len(probs))
        except Exception as e:
            log.warning("Data file unreadable (%s) — re-scraping.", e)

    log.info("Running src/scrape.py …")
    proc = subprocess.run([sys.executable, str(ROOT / "src" / "scrape.py")], text=True)
    if proc.returncode != 0:
        log.error("scrape.py failed. Populate %s manually and retry.", DATA_FILE)
        sys.exit(1)
    log.info("Scrape complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Model queue
# ══════════════════════════════════════════════════════════════════════════════

def load_model_queue() -> list[dict]:
    with open(MODELS_FILE) as f:
        config = yaml.safe_load(f)
    queue = []
    for family in config.get("models", []):
        backend = family["backend"]
        for name in family.get("models", []):
            if "?" in name:
                log.warning("Skipping unresolved model: %s", name)
                continue
            queue.append({
                "model": name,
                "backend": backend,
                "runs": family.get("runs", 1),
                "base_url": family.get("base_url"),
            })
    log.info("Model queue: %d models to consider.", len(queue))
    return queue


def count_results(inspect_model: str) -> int:
    """Return the number of successfully completed Inspect logs for this model.

    Scans results/ (where stray logs may land) and results/AIME2026/ (the
    canonical per-task directory the plot scripts read).
    """
    count = 0
    seen: set[Path] = set()
    for d in (RESULTS_DIR, LOG_DIR):
        if not d.exists():
            continue
        for f in d.glob("*.eval"):
            if f in seen:
                continue
            seen.add(f)
            try:
                ilog = read_eval_log(str(f), header_only=True)
                if ilog.eval.model == inspect_model and ilog.status == "success":
                    count += 1
            except Exception:
                continue
    return count


# ── Interrupted-run resume ────────────────────────────────────────────────────

def find_incomplete_run(log_dir: Path, inspect_model: str) -> Path | None:
    """
    Return the newest non-success .eval log for this model that has at least
    one flushed sample, so an interrupted run can resume instead of restarting.
    """
    newest: Path | None = None
    for f in log_dir.glob("*.eval"):
        try:
            ilog = read_eval_log(str(f), header_only=False)
        except Exception:
            continue
        if ilog.eval.model != inspect_model:
            continue
        if ilog.status == "success" or not ilog.samples:
            continue
        if newest is None or f.name > newest.name:
            newest = f
    return newest


def resume_run(
    partial_path: Path, inspect_model: str, log_dir: Path,
    run_seed: int, run_number: int,
) -> EvalLog | None:
    """
    Continue an interrupted run: evaluate only the problems missing from the
    partial log, merge the results into it, and re-write it as a successful
    full-run log. Returns the merged EvalLog, or None if the completion run
    failed (in which case the partial log is kept for a later retry).
    """
    partial = read_eval_log(str(partial_path))
    done = {
        int(s.id) for s in (partial.samples or [])
        if s.id and s.id.isdigit() and 1 <= int(s.id) <= 15
    }
    remaining = [i for i in range(1, 16) if i not in done]
    log.info(
        "RESUME %s: %d/15 problems already done — running problems %s",
        inspect_model, len(done), remaining,
    )

    comp_logs = inspect_eval(
        aime_2026_i(problem_ids=remaining),
        model=inspect_model,
        log_dir=str(log_dir),
        seed=run_seed,
        temperature=RUN_TEMPERATURE,
        top_p=RUN_TOP_P,
        metadata={"seed": run_seed, "run_number": run_number, "resumed": True},
        log_buffer=1,
        display="log",
    )
    comp = comp_logs[0]
    if comp.status != "success":
        log.error(
            "Resume eval failed for %s (status=%s) — partial log kept; "
            "re-run main.py to retry the remaining problems.",
            inspect_model, comp.status,
        )
        return None

    merged_samples = sorted(
        list(partial.samples or []) + list(comp.samples or []),
        key=lambda s: int(s.id) if s.id and s.id.isdigit() else 0,
    )
    partial.samples = merged_samples
    partial.status = "success"
    partial.error = None
    write_eval_log(partial, location=str(partial_path))

    # The subset completion log must not linger — count_results would
    # otherwise treat it as an additional run.
    if comp.location:
        Path(comp.location).unlink(missing_ok=True)

    log.info(
        "RESUME complete: %s merged to %d/15 samples in %s",
        inspect_model, len(merged_samples), partial_path.name,
    )
    return partial


def log_run_result(inspect_model: str, ilog: EvalLog) -> None:
    samples = ilog.samples or []
    total = len(samples)
    correct = sum(
        1 for s in samples
        if s.scores and list(s.scores.values())[0].value == CORRECT
    )
    pct = 100 * correct / total if total else 0
    log.info("RESULT: %s  %d/%d (%.1f%%)", inspect_model, correct, total, pct)


# ── Ollama cache management ───────────────────────────────────────────────────

# Background pull processes: model_name → Popen
_bg_pulls: dict[str, subprocess.Popen] = {}


def ollama_start_pull(model: str) -> None:
    """Fire off a background `ollama pull` (non-blocking)."""
    if model in _bg_pulls:
        return
    log.info("[OLLAMA] Background pull started: %s", model)
    _bg_pulls[model] = subprocess.Popen(
        ["ollama", "pull", model],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _ollama_pull_with_hf_fallback(model: str, stderr: str) -> bool:
    """
    Called after a plain `ollama pull` fails.
    Tries:
      1. ollama pull hf.co/<model>  (works if the HF repo has pre-built GGUFs)
      2. Full HF download → GGUF conversion → ollama create
    """
    log.info("[OLLAMA] Trying hf.co/%s …", model)
    result = subprocess.run(
        ["ollama", "pull", f"hf.co/{model}"], capture_output=True, text=True
    )
    if result.returncode == 0:
        log.info("[OLLAMA] Pull via hf.co succeeded: %s", model)
        return True

    log.info("[OLLAMA] hf.co pull also failed — attempting HF download + GGUF conversion …")
    from src.hf_convert import hf_to_ollama
    return hf_to_ollama(model, MAX_TOKENS)


def ollama_finish_pull(model: str) -> bool:
    """
    Block until the model is ready to use.
    If a background pull is in flight, wait for it; otherwise pull synchronously.
    Falls back to HF download + GGUF conversion if the Ollama registry pull fails.
    Returns True on success.
    """
    if model in _bg_pulls:
        proc = _bg_pulls.pop(model)
        log.info("[OLLAMA] Waiting for background pull: %s …", model)
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            log.warning("[OLLAMA] Pull failed for %s — trying HF fallback …", model)
            return _ollama_pull_with_hf_fallback(model, stderr)
        log.info("[OLLAMA] Pull complete: %s", model)
        return True

    # Not pre-pulled — do it synchronously now
    log.info("[OLLAMA] Pulling %s …", model)
    proc = subprocess.run(["ollama", "pull", model], capture_output=True, text=True)
    if proc.returncode != 0:
        log.warning("[OLLAMA] Pull failed for %s — trying HF fallback …", model)
        return _ollama_pull_with_hf_fallback(model, proc.stderr)
    log.info("[OLLAMA] Pull complete: %s", model)
    return True


def ollama_set_ctx(model: str) -> bool:
    """
    Inject PARAMETER num_ctx into the model's Modelfile so that every call —
    including those from Inspect's framework — uses MAX_TOKENS context.

    Ollama 0.17.x ignores per-request num_ctx options; the only reliable way
    to set context is at the model level via ollama create.
    """
    import tempfile

    show = subprocess.run(
        ["ollama", "show", "--modelfile", model],
        capture_output=True, text=True,
    )
    if show.returncode != 0:
        log.warning("[OLLAMA] Could not read modelfile for %s: %s", model, show.stderr.strip())
        return False

    # Strip any existing num_ctx line, then append our value
    lines = [l for l in show.stdout.splitlines()
             if not l.strip().lower().startswith("parameter num_ctx")]
    lines.append(f"PARAMETER num_ctx {MAX_TOKENS}")

    with tempfile.NamedTemporaryFile(mode="w", suffix="_Modelfile", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        modelfile_path = f.name

    result = subprocess.run(
        ["ollama", "create", model, "-f", modelfile_path],
        capture_output=True, text=True,
    )
    Path(modelfile_path).unlink(missing_ok=True)

    if result.returncode != 0:
        log.error("[OLLAMA] ollama create failed for %s: %s", model, result.stderr.strip())
        return False

    log.info("[OLLAMA] Set num_ctx=%d in modelfile: %s", MAX_TOKENS, model)
    return True


def ollama_verify_context(model: str) -> bool:
    """
    Prime the model with a 1-token request at the correct num_ctx, then check
    /api/ps to confirm Ollama loaded it at that context size.

    This also warms the model before Inspect initialises it, preventing the
    transient 4096-token load that would otherwise force an unnecessary reload.
    """
    import requests

    # Warm the model at the correct context size
    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_predict": 1, "num_ctx": MAX_TOKENS},
            },
            timeout=120,
        )
        resp.raise_for_status()
    except Exception as e:
        log.error("[OLLAMA] Context warm-up request failed for %s: %s", model, e)
        return False

    # Verify via /api/ps
    try:
        ps = requests.get("http://localhost:11434/api/ps", timeout=10).json()
        for m in ps.get("models", []):
            if m.get("model") == model or m.get("name") == model:
                ctx = m.get("context_length")
                if ctx is None:
                    log.warning("[OLLAMA] /api/ps has no context_length field — skipping check")
                    return True
                if ctx != MAX_TOKENS:
                    log.error(
                        "[OLLAMA] Context mismatch for %s: loaded=%d expected=%d",
                        model, ctx, MAX_TOKENS,
                    )
                    return False
                log.info("[OLLAMA] Context verified: %s @ %d tokens", model, ctx)
                return True
        log.warning("[OLLAMA] %s not found in /api/ps after warm-up", model)
    except Exception as e:
        log.warning("[OLLAMA] Could not read /api/ps: %s — skipping context check", e)

    return True  # Don't block evaluation if the check itself fails


_ollama_server_proc: subprocess.Popen | None = None


def ollama_restart_server() -> bool:
    """Kill any running Ollama server and restart with OLLAMA_NUM_PARALLEL set."""
    global _ollama_server_proc
    import requests

    log.info("[OLLAMA] Stopping existing server …")
    subprocess.run(["pkill", "-x", "ollama"], capture_output=True)
    time.sleep(3)

    env = os.environ.copy()
    env["OLLAMA_NUM_PARALLEL"] = str(OLLAMA_NUM_PARALLEL)
    log.info("[OLLAMA] Starting server with OLLAMA_NUM_PARALLEL=%d …", OLLAMA_NUM_PARALLEL)
    _ollama_server_proc = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(30):
        try:
            if requests.get("http://localhost:11434/api/tags", timeout=2).ok:
                log.info("[OLLAMA] Server ready (PID %d)", _ollama_server_proc.pid)
                return True
        except Exception:
            pass
        time.sleep(1)

    log.error("[OLLAMA] Server failed to start within 30s")
    return False


def ollama_remove(model: str) -> None:
    log.info("[OLLAMA] Freeing cache: %s", model)
    proc = subprocess.run(["ollama", "rm", model], capture_output=True, text=True)
    if proc.returncode == 0:
        log.info("[OLLAMA] Removed: %s", model)
    else:
        log.warning("[OLLAMA] Remove failed for %s: %s", model, proc.stderr.strip())


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Report
# ══════════════════════════════════════════════════════════════════════════════

def run_report() -> None:
    log.info("Generating summary report …")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "src" / "report.py")], text=True
    )
    if proc.returncode != 0:
        log.warning("report.py exited with code %d", proc.returncode)
    else:
        log.info("Summary written: results/summary.md")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("═" * 60)
    log.info("AIME 2026 Evaluation Harness (Inspect AI %s)", _inspect_version())
    log.info("Log: %s", _log_file)
    log.info("═" * 60)

    ensure_data()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not ollama_restart_server():
        log.error("Cannot start Ollama — aborting.")
        sys.exit(1)

    queue = load_model_queue()
    evaluated = skipped = 0

    # Kick off a background pull for the first Ollama model we'll need,
    # so the download overlaps with any startup work.
    for entry in queue:
        if entry["backend"] == "ollama" and count_results(f"ollama/{entry['model']}") < entry["runs"]:
            ollama_start_pull(entry["model"])
            break

    for i, entry in enumerate(queue):
        model = entry["model"]
        backend = entry["backend"]
        base_url = entry.get("base_url")
        inspect_model = f"{backend}/{model}"
        target_runs = entry["runs"]

        # Skip if API keys are missing (local OpenAI-compatible servers exempt)
        if backend == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            log.info("SKIP %s — ANTHROPIC_API_KEY not set", inspect_model)
            skipped += 1
            continue
        if backend == "openai" and not base_url and not os.getenv("OPENAI_API_KEY"):
            log.info("SKIP %s — OPENAI_API_KEY not set", inspect_model)
            skipped += 1
            continue
        if backend == "google" and not os.getenv("GOOGLE_API_KEY"):
            log.info("SKIP %s — GOOGLE_API_KEY not set", inspect_model)
            skipped += 1
            continue

        # Local OpenAI-compatible server (e.g. llama-server) must be reachable
        if base_url:
            import requests
            try:
                # Any HTTP response counts (the server root may 404); only
                # connection errors mean it's actually down.
                requests.get(base_url, timeout=5)
            except Exception as e:
                log.error("SKIP %s — local server %s not reachable: %s", inspect_model, base_url, e)
                skipped += 1
                continue

        # Skip if target run count already reached
        existing_runs = count_results(inspect_model)
        if existing_runs >= target_runs:
            log.info("SKIP %s — %d/%d runs complete", inspect_model, existing_runs, target_runs)
            skipped += 1
            continue

        runs_to_do = target_runs - existing_runs
        log.info("%s — %d/%d runs complete, running %d more", inspect_model, existing_runs, target_runs, runs_to_do)

        # Ensure this Ollama model is ready (wait for bg pull or pull now)
        if backend == "ollama":
            if not ollama_finish_pull(model):
                continue
            ollama_set_ctx(model)
            if not ollama_verify_context(model):
                log.error("SKIP %s — context size mismatch, see above", inspect_model)
                skipped += 1
                continue

        # While evaluating this model, start pulling the next one in the background
        for j in range(i + 1, len(queue)):
            ne = queue[j]
            if (ne["backend"] == "ollama"
                    and ne["model"] not in _bg_pulls
                    and count_results(f"ollama/{ne['model']}") < ne["runs"]):
                ollama_start_pull(ne["model"])
                break

        for run_i in range(runs_to_do):
            # Run slot number and its reproducible seed
            run_number = existing_runs + run_i + 1
            run_seed = SEED_BASE + run_number

            # Route this entry to its own OpenAI-compatible endpoint (e.g.
            # local llama-server) without disturbing the cloud OPENAI_BASE_URL,
            # and pass sampling/seed settings to the task's native Ollama path.
            prev_base_url = os.environ.get("OPENAI_BASE_URL")
            prev_seed_env = os.environ.get("AIME_RUN_SEED")
            prev_temp_env = os.environ.get("AIME_TEMP")
            prev_topp_env = os.environ.get("AIME_TOP_P")
            if base_url:
                os.environ["OPENAI_BASE_URL"] = base_url
            os.environ["AIME_RUN_SEED"] = str(run_seed)
            os.environ["AIME_TEMP"] = str(RUN_TEMPERATURE)
            os.environ["AIME_TOP_P"] = str(RUN_TOP_P)
            try:
                log.info("━" * 60)
                log.info("EVALUATING: %s (run %d/%d)", inspect_model, run_number, target_runs)
                log.info("━" * 60)

                # Resume an interrupted run if one exists: evaluate only the
                # missing problems and merge into the partial log.
                partial_path = find_incomplete_run(LOG_DIR, inspect_model)
                resumed = False
                if partial_path is not None:
                    try:
                        ilog = resume_run(
                            partial_path, inspect_model, LOG_DIR,
                            run_seed, run_number,
                        )
                        if ilog is not None:
                            log_run_result(inspect_model, ilog)
                            evaluated += 1
                            resumed = True
                    except Exception as e:
                        log.error(
                            "Resume failed for %s: %s — partial log corrupted, "
                            "deleting it and starting a fresh run.",
                            inspect_model, e,
                        )
                        partial_path.unlink(missing_ok=True)

                if not resumed:
                    eval_logs = inspect_eval(
                        aime_2026_i(),
                        model=inspect_model,
                        log_dir=str(LOG_DIR),
                        seed=run_seed,
                        temperature=RUN_TEMPERATURE,
                        top_p=RUN_TOP_P,
                        metadata={"seed": run_seed, "run_number": run_number},
                        log_buffer=1,
                        display="log",
                    )

                    ilog = eval_logs[0]
                    log_run_result(inspect_model, ilog)
                    evaluated += 1

            except Exception as e:
                log.error("Evaluation failed for %s: %s", inspect_model, e)
            finally:
                if prev_seed_env is None:
                    os.environ.pop("AIME_RUN_SEED", None)
                else:
                    os.environ["AIME_RUN_SEED"] = prev_seed_env
                if prev_temp_env is None:
                    os.environ.pop("AIME_TEMP", None)
                else:
                    os.environ["AIME_TEMP"] = prev_temp_env
                if prev_topp_env is None:
                    os.environ.pop("AIME_TOP_P", None)
                else:
                    os.environ["AIME_TOP_P"] = prev_topp_env
                if base_url:
                    if prev_base_url is None:
                        os.environ.pop("OPENAI_BASE_URL", None)
                    else:
                        os.environ["OPENAI_BASE_URL"] = prev_base_url

    log.info("═" * 60)
    log.info("Evaluation complete — ran: %d  skipped: %d", evaluated, skipped)
    log.info("═" * 60)

    run_report()
    log.info("Done. Results in: %s/", RESULTS_DIR)


def _inspect_version() -> str:
    try:
        import inspect_ai
        return inspect_ai.__version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
