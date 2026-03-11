#!/usr/bin/env python3
"""
AIME 2026 I — Inspect AI task definition.

Direct CLI usage (per-model):
  inspect eval src/task.py -m ollama/qwen3.5:0.8b --log-dir results/
  inspect eval src/task.py -m anthropic/claude-sonnet-4-6 --log-dir results/
  inspect eval src/task.py -m openai/gpt-4o --log-dir results/

Or run the full pipeline via main.py.
"""

import json
import logging
import re
import time
from pathlib import Path

log = logging.getLogger("aime.task")

import httpx
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, GenerateConfig, ModelOutput, get_model
from inspect_ai.model._providers.ollama import OllamaAPI
from inspect_ai.scorer import CORRECT, INCORRECT, Score, accuracy, scorer, stderr
from inspect_ai.solver import Generate, TaskState, generate, solver

DATA_FILE = Path(__file__).parent.parent / "data" / "aime_2026_i.json"

PROMPT_TEMPLATE = """\
You are a mathematics expert. Solve the following competition mathematics problem.
Your final answer must be an integer between 0 and 999 inclusive.

Problem: {problem_text}

Think step by step. At the very end of your response, write your final answer on a new line in this exact format:
ANSWER: <integer>"""

# Ollama native API endpoint (bypasses OpenAI-compat layer)
OLLAMA_NATIVE_URL = "http://localhost:11434/api/chat"
MAX_TOKENS = 32768


# ── Dataset ───────────────────────────────────────────────────────────────────

def _load_dataset() -> MemoryDataset:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run 'python src/scrape.py' first."
        )
    data = json.loads(DATA_FILE.read_text())
    samples = [
        Sample(
            id=str(p["id"]),
            input=PROMPT_TEMPLATE.format(problem_text=p["problem"]),
            target=str(p["answer"]),
            metadata={"problem_id": p["id"]},
        )
        for p in data["problems"]
    ]
    return MemoryDataset(samples, name="2026_AIME_I")


# ── Solver ────────────────────────────────────────────────────────────────────

@solver
def generate_with_thinking():
    """
    For Ollama models: calls the native /api/chat endpoint which exposes
    both `thinking` and `content` fields separately.  Combines them as
    <think>{thinking}</think>\\n{content} so the scorer sees the full output.

    Inspect's Ollama provider uses the OpenAI-compat endpoint which only
    reads `message.content` and ignores `message.thinking`, causing empty
    completions when thinking models exhaust their token budget inside
    <think> chains before producing any response text.

    For non-Ollama models: delegates to the standard generate() solver.
    """
    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        model = get_model()
        model_name = str(model)  # e.g. "ollama/qwen3.5:0.8b"
        is_ollama = isinstance(model.api, OllamaAPI)
        log.info("[SOLVER] sample=%s  model=%s  path=%s",
                 state.sample_id, model_name, "native" if is_ollama else "generate_fn")

        if not is_ollama:
            t0 = time.monotonic()
            state = await generate_fn(state)
            elapsed = time.monotonic() - t0
            usage = state.output.usage
            if usage and usage.output_tokens:
                tps = usage.output_tokens / elapsed
                log.info(
                    "[SPEED] %s  sample=%s  %d tok / %.1fs = %.1f tok/s",
                    model_name, state.sample_id, usage.output_tokens, elapsed, tps,
                )
            return state

        ollama_model = model.api.model_name  # already stripped of "ollama/" prefix

        # Build message list from current state
        messages = [
            {"role": m.role, "content": m.text}
            for m in state.messages
            if hasattr(m, "text")
        ]

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(
                OLLAMA_NATIVE_URL,
                json={
                    "model": ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_predict": MAX_TOKENS, "num_ctx": MAX_TOKENS},
                },
            )
            resp.raise_for_status()
            data = resp.json()
        wall_elapsed = time.monotonic() - t0

        eval_count = data.get("eval_count", 0)
        eval_ns = data.get("eval_duration", 0)
        if eval_count and eval_ns:
            elapsed = eval_ns / 1e9
            tps = eval_count / elapsed
        elif eval_count:
            elapsed = wall_elapsed
            tps = eval_count / elapsed
        else:
            elapsed = wall_elapsed
            tps = 0.0
        state.metadata["_tokens"] = eval_count
        state.metadata["_elapsed"] = elapsed
        state.metadata["_tps"] = tps

        msg = data.get("message", {})
        thinking = msg.get("thinking", "")
        content = msg.get("content", "")

        # Reconstruct full text so the scorer can see both reasoning and answer.
        # An unclosed <think> tag signals that thinking was truncated (no answer).
        if thinking and content:
            full_text = f"<think>{thinking}</think>\n{content}"
        elif thinking:
            full_text = f"<think>{thinking}"   # truncated — no closing tag
        else:
            full_text = content

        # Update state so Inspect records the output and the scorer can read it
        state.messages.append(
            ChatMessageAssistant(content=full_text, model=model_name, source="generate")
        )
        state.output = ModelOutput.from_content(model=model_name, content=full_text)

        return state

    return solve


# ── Scorer ────────────────────────────────────────────────────────────────────

@scorer(metrics=[accuracy(), stderr()])
def aime_integer_scorer():
    """
    Extract an integer from the model's response and compare to the target.

    Primary extraction: ANSWER: <integer> line (looks outside <think> block).
    Fallback: last integer in [0, 999] found anywhere in the response.
    """
    async def score(state, target) -> Score:
        text = state.output.completion or ""

        # Strip thinking block for primary extraction so we only look in the
        # response text, not in the middle of a reasoning chain.
        post_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        search_text = post_think if post_think else text

        extracted: int | None = None

        # Primary: ANSWER: <integer> in the response (outside think block)
        m = re.search(r"ANSWER:\s*(\d+)", search_text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 999:
                extracted = val

        # Fallback: last integer in [0, 999] in response text
        if extracted is None:
            for candidate in reversed(re.findall(r"\b(\d{1,3})\b", search_text)):
                val = int(candidate)
                if 0 <= val <= 999:
                    extracted = val
                    break

        # Last resort: search entire text including thinking
        if extracted is None:
            for candidate in reversed(re.findall(r"\b(\d{1,3})\b", text)):
                val = int(candidate)
                if 0 <= val <= 999:
                    extracted = val
                    break

        correct_answer = int(target.text)
        is_correct = extracted is not None and extracted == correct_answer

        verdict = "Correct  ✓" if is_correct else "Incorrect"
        tokens = state.metadata.get("_tokens", 0)
        elapsed = state.metadata.get("_elapsed", 0.0)
        tps = state.metadata.get("_tps", 0.0)
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        log.info(
            "[RESULT] Problem %2s — %-11s  got=%-4s  expected=%-4s  %s  %d tok  %.0f tok/s",
            state.sample_id, verdict, extracted, correct_answer, time_str, tokens, tps,
        )

        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=str(extracted) if extracted is not None else None,
            explanation=f"extracted={extracted}  expected={correct_answer}",
        )

    return score


# ── Task ──────────────────────────────────────────────────────────────────────

@task
def aime_2026_i():
    """Evaluate a model on all 15 problems from the 2026 AIME I."""
    return Task(
        dataset=_load_dataset(),
        solver=generate_with_thinking(),
        scorer=aime_integer_scorer(),
        config=GenerateConfig(max_tokens=MAX_TOKENS, max_connections=1),
    )
