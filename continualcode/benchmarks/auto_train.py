"""Automated SDPO training harness for LiveCodeBench.

Replaces the human-in-the-loop TUI with automated sandbox grading + LLM feedback.
The model solves coding problems, sandbox execution provides pass/fail, and on failure
GPT-5.2-Codex analyzes the error to produce rich feedback for SDPO training.

Usage:
    python -m continualcode.benchmarks.auto_train \
        model_name=Qwen/Qwen3-4B-Instruct-2507

    # Disable LLM feedback (stderr-only, cheaper)
    python -m continualcode.benchmarks.auto_train \
        model_name=Qwen/Qwen3-4B-Instruct-2507 \
        llm_feedback=false

    # With wandb tracking
    python -m continualcode.benchmarks.auto_train \
        model_name=Qwen/Qwen3-4B-Instruct-2507 \
        wandb_project=sdpo-lcb
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import chz
import tinker
from tinker import types

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer
from continualcode.benchmarks.lcb_eval import (
    _load_deepcoder_split,
    _normalize_tests,
    _build_question,
    _ensure_dict,
)
from tinker_cookbook.recipes.code_rl.code_grading import (
    extract_code_from_model,
    sandbox_check_correctness,
)
from tinker_cookbook.sandbox import SandboxBackend

from continualcode.train import SampledCompletion, SDPOConfig, sdpo_train_step

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM feedback via OpenAI
# ---------------------------------------------------------------------------

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        try:
            from openai import AsyncOpenAI
            _openai_client = AsyncOpenAI()  # uses OPENAI_API_KEY env var
        except ImportError:
            raise ImportError("pip install openai — required for LLM feedback")
    return _openai_client


async def generate_llm_feedback(
    question: str,
    code: str,
    stderr: str,
    stdout: str,
    *,
    model: str = "gpt-5.2-codex",
    max_tokens: int = 1024,
) -> str:
    """Call an external LLM to produce rich analysis of why the code failed.

    Returns a natural language explanation that the SDPO self-teacher conditions on
    to produce dense per-token advantages.
    """
    client = _get_openai_client()

    # Truncate from tail — Python tracebacks put the error at the bottom
    stderr_trunc = stderr[-3000:] if len(stderr) > 3000 else (stderr or "(no stderr)")
    stdout_trunc = stdout[-2000:] if len(stdout) > 2000 else (stdout or "(no stdout)")
    code_trunc = code[:4000]

    prompt = (
        "You are a competitive programming expert analyzing a failed solution.\n\n"
        f"## Problem\n{question[:3000]}\n\n"
        f"## Submitted Code\n```python\n{code_trunc}\n```\n\n"
        f"## Stderr\n```\n{stderr_trunc}\n```\n\n"
        f"## Stdout\n```\n{stdout_trunc}\n```\n\n"
        "Analyze the failure concisely. Identify:\n"
        "1. The root cause (bug, wrong algorithm, edge case, TLE, etc.)\n"
        "2. Which specific lines/logic are wrong\n"
        "3. What the correct approach should be\n\n"
        "Be specific and actionable. Do NOT write corrected code — just explain what's wrong and how to fix it."
    )

    try:
        resp = await client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_tokens,
            reasoning={"effort": "low"},
        )
        return resp.output_text.strip()
    except Exception as e:
        # Fallback to raw error if LLM call fails
        logger.warning(f"LLM feedback failed, falling back to stderr: {e}")
        return f"Code failed with error:\n{stderr_trunc}"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@chz.chz
class AutoTrainConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    checkpoint_path: str | None = None          # resume from LoRA checkpoint
    teacher_model: str | None = None            # optional stronger teacher model (sampling only)
    enforce_teacher_tokenizer_compat: bool = True
    lora_rank: int = 32
    learning_rate: float = 1e-6                 # lower LR for multi-rollout GRPO (canonical LCB config)
    kl_coef: float = 1.0
    train_signal: Literal["pure_sdpo", "hybrid"] = "pure_sdpo"
    max_tokens: int = 16384
    temperature: float = 0.7
    num_rollouts: int = 4                       # rollouts per problem for GRPO contrastive signal
    batch_size: int = 16                        # problems sampled per batch (smaller = finer-grained curves)
    min_sdpo_examples: int = 8                  # accumulate before SDPO step (more with multi-rollout)
    sandbox_concurrency: int = 64
    sandbox_timeout: int = 6
    sandbox_backend: SandboxBackend = SandboxBackend.MODAL
    split: Literal["train", "test"] = "train"
    max_problems: int = -1                      # -1 = all
    shuffle_seed: int = 42
    log_path: str = "./checkpoints/auto-sdpo"
    save_every: int = 10                        # checkpoint every N SDPO steps
    # LLM feedback (on by default — the whole point of SDPO is rich feedback)
    llm_feedback: bool = True
    llm_feedback_model: str = "gpt-5.2-codex"
    llm_feedback_concurrency: int = 32
    # Fallback template when llm_feedback=False
    feedback_template: str = "Code failed with error:\n{error}\n\nFix the code."
    # Curriculum: skip problems the model has already solved
    skip_solved: bool = False
    # Held-out eval
    eval_every: int = 1                         # eval every N SDPO steps (0 = disabled)
    eval_size: int = 50                         # number of eval problems
    eval_split: Literal["train", "test"] = "test"
    # Wandb (optional — set project name to enable)
    wandb_project: str | None = None
    wandb_run_name: str | None = None


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    """Tokenizer-agnostic helper for compatibility checks."""
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def _check_tokenizer_compatibility(student_model: str, teacher_model: str) -> tuple[bool, str]:
    """Return (is_compatible, reason) for teacher scoring on student token IDs."""
    try:
        student_tok = get_tokenizer(student_model)
        teacher_tok = get_tokenizer(teacher_model)
    except Exception as e:
        return False, f"failed to load tokenizers: {e}"

    if len(student_tok) != len(teacher_tok):
        return False, f"vocab size mismatch (student={len(student_tok)}, teacher={len(teacher_tok)})"

    if type(student_tok).__name__ != type(teacher_tok).__name__:
        return False, (
            "tokenizer class mismatch "
            f"(student={type(student_tok).__name__}, teacher={type(teacher_tok).__name__})"
        )

    probes = [
        "def solve():\n    pass",
        "a += 1",
        "tool_call(path='foo.py')",
        "print('ok')",
    ]
    for probe in probes:
        s_ids = _encode_without_special_tokens(student_tok, probe)
        t_ids = _encode_without_special_tokens(teacher_tok, probe)
        if s_ids != t_ids:
            return False, f"tokenization mismatch on probe text: {probe!r}"

    return True, "compatible"


# ---------------------------------------------------------------------------
# Sample + Grade
# ---------------------------------------------------------------------------


@dataclass
class GradeResult:
    idx: int
    passed: bool
    completion: SampledCompletion | None
    code: str | None
    error: str           # stderr + stdout from sandbox
    question: str        # original problem text (for LLM feedback)
    response_tokens: int = 0  # token count of model response
    response_text: str = ""   # full model response (for solution demonstrations)


async def sample_and_grade_group(
    idx: int,
    row: dict[str, Any],
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    cfg: AutoTrainConfig,
    sandbox_sem: asyncio.Semaphore,
) -> list[GradeResult]:
    """Sample N rollouts for one problem and grade all completions.

    sandbox_sem throttles concurrent sandbox_check_correctness calls (not API sampling).
    """
    metadata = _ensure_dict(row.get("metadata", {}))
    tests = _normalize_tests(row.get("tests") or row.get("ground_truth"), metadata)
    question = _build_question(row)

    if not tests or not question:
        return [GradeResult(idx=idx, passed=True, completion=None, code=None, error="", question="")]

    # Build prompt once
    messages = renderer.create_conversation_prefix_with_tools([], "")
    messages.append(renderers.Message(role="user", content=question))
    model_input = renderer.build_generation_prompt(messages)
    prompt_len = model_input.length

    # Sample N rollouts in a single API call.
    # No semaphore here — Tinker API handles its own concurrency.
    # The sem only throttles sandbox_check_correctness calls below.
    response: types.SampleResponse = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=cfg.num_rollouts,
        sampling_params=types.SamplingParams(
            stop=renderer.get_stop_sequences(),
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        ),
    )

    # Grade each rollout in parallel
    async def _grade_seq(seq: Any) -> GradeResult:
        response_tokens = len(seq.tokens)
        parsed_msg, _ok = renderer.parse_response(seq.tokens)
        content = get_text_content(parsed_msg)
        code = extract_code_from_model(content)

        completion = None
        if seq.logprobs is not None:
            completion = SampledCompletion(
                prompt_input=model_input,
                prompt_messages=list(messages),
                tokens=seq.tokens,
                logprobs=seq.logprobs,
                prompt_len=prompt_len,
            )

        if code is None:
            return GradeResult(
                idx=idx, passed=False, completion=None, code=None,
                error="No code block found in model output", question=question,
                response_tokens=response_tokens, response_text=content or "",
            )

        try:
            async with sandbox_sem:
                passed, details = await sandbox_check_correctness(
                    tests, code, timeout=cfg.sandbox_timeout, backend=cfg.sandbox_backend,
                )
        except Exception as e:
            logger.warning(f"Sandbox error for problem {idx}: {e}")
            return GradeResult(
                idx=idx, passed=False, completion=completion, code=code,
                error=f"Sandbox execution error: {e}", question=question,
                response_tokens=response_tokens, response_text=content or "",
            )

        stderr = details.get("stderr", "")
        stdout = details.get("stdout", "")
        error_text = f"{stderr}\n{stdout}".strip()

        return GradeResult(
            idx=idx, passed=passed, completion=completion, code=code,
            error=error_text, question=question,
            response_tokens=response_tokens, response_text=content or "",
        )

    results = await asyncio.gather(*[_grade_seq(seq) for seq in response.sequences])

    return list(results)


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------


async def _eval_single(
    row: dict[str, Any],
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    cfg: AutoTrainConfig,
    sem: asyncio.Semaphore,
) -> bool:
    """Evaluate a single problem: sample (no logprobs needed), extract code, sandbox grade."""
    async with sem:
        metadata = _ensure_dict(row.get("metadata", {}))
        tests = _normalize_tests(row.get("tests") or row.get("ground_truth"), metadata)
        question = _build_question(row)
        if not tests or not question:
            return True  # skip non-gradeable

        messages = renderer.create_conversation_prefix_with_tools([], "")
        messages.append(renderers.Message(role="user", content=question))
        model_input = renderer.build_generation_prompt(messages)

        response = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=types.SamplingParams(
                stop=renderer.get_stop_sequences(),
                max_tokens=cfg.max_tokens,
                temperature=0.0,  # greedy for eval determinism
            ),
        )

        seq = response.sequences[0]
        parsed_msg, _ok = renderer.parse_response(seq.tokens)
        content = get_text_content(parsed_msg)
        code = extract_code_from_model(content)
        if code is None:
            return False

        try:
            passed, _ = await sandbox_check_correctness(
                tests, code, timeout=cfg.sandbox_timeout, backend=cfg.sandbox_backend,
            )
            return passed
        except Exception:
            return False


async def run_eval_pass(
    eval_set: list[dict[str, Any]],
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    cfg: AutoTrainConfig,
) -> float:
    """Run eval on held-out set. Returns pass@1."""
    sem = asyncio.Semaphore(32)  # higher concurrency — no training overhead
    tasks = [_eval_single(row, sampling_client, renderer, cfg, sem) for row in eval_set]
    results = await asyncio.gather(*tasks)
    n_passed = sum(1 for r in results if r)
    return n_passed / max(1, len(results))


# ---------------------------------------------------------------------------
# Wandb helpers
# ---------------------------------------------------------------------------

_wandb_run = None


def _init_wandb(cfg: AutoTrainConfig) -> Any:
    """Initialize wandb run if project is set. Returns run or None."""
    global _wandb_run
    if cfg.wandb_project is None:
        return None
    try:
        import wandb
        if cfg.wandb_run_name:
            run_name = cfg.wandb_run_name
        else:
            import time
            ts = time.strftime("%m%d-%H%M")
            model_short = cfg.model_name.split("/")[-1]
            run_name = f"sdpo-{model_short}-r{cfg.num_rollouts}b{cfg.batch_size}-{cfg.train_signal}-{ts}"
        _wandb_run = wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            config={
                "model_name": cfg.model_name,
                "teacher_model": cfg.teacher_model,
                "enforce_teacher_tokenizer_compat": cfg.enforce_teacher_tokenizer_compat,
                "lora_rank": cfg.lora_rank,
                "learning_rate": cfg.learning_rate,
                "kl_coef": cfg.kl_coef,
                "train_signal": cfg.train_signal,
                "max_tokens": cfg.max_tokens,
                "temperature": cfg.temperature,
                "num_rollouts": cfg.num_rollouts,
                "batch_size": cfg.batch_size,
                "min_sdpo_examples": cfg.min_sdpo_examples,
                "llm_feedback": cfg.llm_feedback,
                "llm_feedback_model": cfg.llm_feedback_model,
                "split": cfg.split,
                "max_problems": cfg.max_problems,
                "feedback_template": cfg.feedback_template,
                "skip_solved": cfg.skip_solved,
                "eval_every": cfg.eval_every,
                "eval_size": cfg.eval_size,
                "eval_split": cfg.eval_split,
            },
        )
        return _wandb_run
    except ImportError:
        logger.warning("wandb not installed — pip install wandb. Continuing without it.")
        return None


def _log_wandb(metrics: dict[str, Any], step: int) -> None:
    """Log metrics to wandb if initialized."""
    if _wandb_run is not None:
        _wandb_run.log(metrics, step=step)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


async def run_auto_train(cfg: AutoTrainConfig) -> None:
    Path(cfg.log_path).mkdir(parents=True, exist_ok=True)

    # Setup dataset
    logger.info(f"Loading LCB {cfg.split} split...")
    ds = _load_deepcoder_split(cfg.split)
    indices = list(range(len(ds)))
    if cfg.shuffle_seed >= 0:
        random.Random(cfg.shuffle_seed).shuffle(indices)
    if cfg.max_problems > 0:
        indices = indices[:cfg.max_problems]
    logger.info(f"  {len(indices)} problems, batch_size={cfg.batch_size}")

    # Setup model
    logger.info(f"Connecting to Tinker — model: {cfg.model_name}")
    service_client = tinker.ServiceClient()

    # Check for resumable checkpoint
    sdpo_steps = 0
    start_batch = 0
    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        loop_state = resume_info.get("loop_state", {})
        sdpo_steps = loop_state.get("sdpo_steps", 0)
        start_batch = loop_state.get("batch_idx", 0)
        logger.info(f"Resumed from checkpoint: sdpo_steps={sdpo_steps}, batch_idx={start_batch}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=cfg.model_name, rank=cfg.lora_rank,
        )
        if cfg.checkpoint_path:
            await training_client.load_state_async(cfg.checkpoint_path)
            logger.info(f"  Loaded checkpoint: {cfg.checkpoint_path}")

    # Sampling client from current weights (student)
    sampling_client = await training_client.save_weights_and_get_sampling_client_async("current")

    # Optional separate teacher sampling client (e.g., stronger model)
    teacher_sampling_client = sampling_client
    teacher_is_student = True
    if cfg.teacher_model:
        if cfg.teacher_model != cfg.model_name:
            is_compatible, reason = _check_tokenizer_compatibility(
                student_model=cfg.model_name,
                teacher_model=cfg.teacher_model,
            )
            if not is_compatible:
                if cfg.enforce_teacher_tokenizer_compat:
                    raise ValueError(
                        "Incompatible teacher_model for token-ID scoring: "
                        f"student={cfg.model_name}, teacher={cfg.teacher_model}, reason={reason}. "
                        "Set enforce_teacher_tokenizer_compat=false to bypass (unsafe)."
                    )
                logger.warning(
                    "Teacher tokenizer compatibility check failed but enforcement is disabled: "
                    f"{reason}"
                )
        try:
            teacher_sampling_client = await service_client.create_sampling_client_async(
                base_model=cfg.teacher_model
            )
            teacher_is_student = False
            logger.info(f"  Teacher model: {cfg.teacher_model}")
            if cfg.teacher_model != cfg.model_name:
                logger.warning(
                    "Teacher model differs from student. Ensure tokenizer/vocab are compatible, "
                    "since teacher logprobs are computed on student token IDs."
                )
        except Exception as e:
            logger.warning(
                f"Failed to create teacher sampling client for {cfg.teacher_model}: {e}. "
                "Falling back to student as teacher."
            )
            teacher_sampling_client = sampling_client
            teacher_is_student = True

    # Renderer
    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
    tokenizer = get_tokenizer(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    # SDPO config (paper-aligned 3-slot template)
    sdpo_config = SDPOConfig(
        kl_coef=cfg.kl_coef,
        include_environment_feedback=True,
    )

    # Logging: always write metrics JSONL (into log_path by default)
    metrics_path = os.path.join(cfg.log_path, "metrics.jsonl")
    metrics_f = open(metrics_path, "a")
    # Per-sample log for reproducing training curves / discovery@k
    samples_path = os.path.join(cfg.log_path, "samples.jsonl")
    samples_f = open(samples_path, "a")

    # Load held-out eval set
    eval_set: list[dict[str, Any]] = []
    if cfg.eval_every > 0:
        eval_ds = _load_deepcoder_split(cfg.eval_split)
        eval_indices = list(range(len(eval_ds)))
        random.Random(cfg.shuffle_seed + 1).shuffle(eval_indices)  # different seed from training
        eval_indices = eval_indices[:cfg.eval_size]
        eval_set = [dict(eval_ds[i]) for i in eval_indices]
        logger.info(f"Loaded {len(eval_set)} eval problems from {cfg.eval_split} split")

    # Wandb (optional)
    wb_run = _init_wandb(cfg)

    sandbox_sem = asyncio.Semaphore(cfg.sandbox_concurrency)
    llm_sem = asyncio.Semaphore(cfg.llm_feedback_concurrency)
    n_batches = (len(indices) + cfg.batch_size - 1) // cfg.batch_size

    if cfg.train_signal != "pure_sdpo":
        raise ValueError(
            f"train_signal={cfg.train_signal!r} is not supported in this benchmark path. "
            "Use train_signal='pure_sdpo'."
        )

    logger.info(
        f"Starting auto-train: {n_batches} batches, num_rollouts={cfg.num_rollouts}, "
        f"llm_feedback={cfg.llm_feedback}"
    )
    logger.info(f"  Metrics: {metrics_path}")
    logger.info(f"  Samples: {samples_path}")
    if cfg.llm_feedback:
        logger.info(f"  LLM feedback model: {cfg.llm_feedback_model}")
    if wb_run:
        logger.info(f"  Wandb: {cfg.wandb_project}")

    total_passed = 0
    total_graded = 0
    total_tokens_generated = 0
    solved_indices: set[int] = set()  # curriculum: track solved problems
    # Accumulate trainable failures (mixed groups only) and pass-context for solution demos.
    accumulated_failures: list[GradeResult] = []
    accumulated_pass_context: list[GradeResult] = []
    # Aggregated training-quality stats for the pending SDPO step.
    accumulated_total_groups = 0
    accumulated_mixed_groups = 0
    accumulated_skipped_all_fail_groups = 0
    accumulated_total_failures = 0
    accumulated_eligible_failures = 0
    # Per-problem difficulty tracking: {problem_idx: {"attempts": int, "first_solve_step": int|None}}
    problem_tracker: dict[int, dict[str, Any]] = {}
    # Group-level stats for GRPO
    total_groups = 0
    total_groups_with_variance = 0
    t_start = time.monotonic()

    for batch_idx in range(start_batch, n_batches):
        t0 = time.monotonic()

        # Filter out solved problems if curriculum is enabled
        batch_indices = indices[batch_idx * cfg.batch_size : (batch_idx + 1) * cfg.batch_size]
        if cfg.skip_solved:
            batch_indices = [i for i in batch_indices if i not in solved_indices]
        if not batch_indices:
            continue

        # 1. Sample N rollouts per problem + grade batch concurrently
        group_tasks = [
            sample_and_grade_group(i, dict(ds[i]), sampling_client, renderer, cfg, sandbox_sem)
            for i in batch_indices
        ]
        group_results: list[list[GradeResult]] = await asyncio.gather(*group_tasks)
        t_sampled = time.monotonic()

        # Flatten groups and compute trainability stats.
        results: list[GradeResult] = []
        batch_group_rewards: list[float] = []
        eligible_failures_batch: list[GradeResult] = []
        pass_context_batch: list[GradeResult] = []
        batch_mixed_groups = 0
        batch_skipped_all_fail_groups = 0
        batch_total_failures = 0
        for group in group_results:
            results.extend(group)
            group_passes = sum(1 for r in group if r.passed)
            group_fails = sum(1 for r in group if not r.passed)
            total_groups += 1
            if group_passes > 0 and group_fails > 0:
                total_groups_with_variance += 1
            batch_group_rewards.append(group_passes / max(1, len(group)))

            group_failures = [r for r in group if (not r.passed) and r.completion is not None]
            group_pass_context = [r for r in group if r.passed and bool(r.response_text)]
            batch_total_failures += len(group_failures)
            pass_context_batch.extend(group_pass_context)
            if group_failures:
                if group_pass_context:
                    batch_mixed_groups += 1
                    eligible_failures_batch.extend(group_failures)
                else:
                    batch_skipped_all_fail_groups += 1

        batch_eligible_failures = len(eligible_failures_batch)
        mixed_group_fraction = batch_mixed_groups / max(1, len(group_results))
        eligible_failure_fraction = batch_eligible_failures / max(1, batch_total_failures)

        # Tally and log per-sample results
        batch_passed = 0
        batch_tokens: list[int] = []
        for r in results:
            is_pass = r.passed
            # Update difficulty tracker (count per-problem, not per-rollout)
            if r.idx not in problem_tracker:
                problem_tracker[r.idx] = {"attempts": 0, "first_solve_step": None}
            problem_tracker[r.idx]["attempts"] += 1
            if is_pass and problem_tracker[r.idx]["first_solve_step"] is None:
                problem_tracker[r.idx]["first_solve_step"] = sdpo_steps

            if is_pass:
                batch_passed += 1
                if cfg.skip_solved:
                    solved_indices.add(r.idx)
            batch_tokens.append(r.response_tokens)
            total_tokens_generated += r.response_tokens

            # Per-sample log
            sample_entry = {
                "sdpo_step": sdpo_steps,
                "batch_idx": batch_idx,
                "problem_idx": r.idx,
                "passed": is_pass,
                "response_tokens": r.response_tokens,
                "has_code": r.code is not None,
                "error_preview": r.error[:200] if r.error else None,
                "wall_time_s": round(time.monotonic() - t_start, 1),
            }
            samples_f.write(json.dumps(sample_entry) + "\n")
        samples_f.flush()

        # Accumulate only failures from mixed groups to ensure sibling solution context.
        accumulated_failures.extend(eligible_failures_batch)
        accumulated_pass_context.extend(pass_context_batch)
        accumulated_total_groups += len(group_results)
        accumulated_mixed_groups += batch_mixed_groups
        accumulated_skipped_all_fail_groups += batch_skipped_all_fail_groups
        accumulated_total_failures += batch_total_failures
        accumulated_eligible_failures += batch_eligible_failures
        total_passed += batch_passed
        total_graded += len(results)

        # Batch-level stats
        resp_len_mean = sum(batch_tokens) / max(1, len(batch_tokens))
        resp_len_max = max(batch_tokens) if batch_tokens else 0
        group_mean_reward = sum(batch_group_rewards) / max(1, len(batch_group_rewards))

        logger.info(
            f"Batch {batch_idx+1}/{n_batches}: "
            f"{batch_passed}/{len(results)} passed ({len(batch_indices)} problems x {cfg.num_rollouts} rollouts), "
            f"{batch_eligible_failures}/{batch_total_failures} eligible failures "
            f"({len(accumulated_failures)} accumulated), "
            f"group_reward={group_mean_reward:.2f}"
        )

        # Log batch metrics to wandb
        batch_metrics = {
            "batch/pass_rate": batch_passed / max(1, len(results)),
            "batch/response_length_mean": resp_len_mean,
            "batch/response_length_max": resp_len_max,
            "batch/cumulative_pass_rate": total_passed / max(1, total_graded),
            "batch/total_sampled": total_graded,
            "batch/solved_problems": len(solved_indices),
            "batch/total_tokens_generated": total_tokens_generated,
            "group/mean_reward": group_mean_reward,
            "group/pass_rate": batch_passed / max(1, len(results)),
            "group/groups_with_variance": total_groups_with_variance,
            "group/total_groups": total_groups,
            "train/mixed_group_fraction": mixed_group_fraction,
            "train/eligible_failure_fraction": eligible_failure_fraction,
            "train/skipped_all_fail_groups": float(batch_skipped_all_fail_groups),
        }
        _log_wandb(batch_metrics, step=total_graded)

        # 2. Check if we have enough trainable examples for an SDPO step
        if len(accumulated_failures) < cfg.min_sdpo_examples:
            continue

        # 3. Generate LLM feedback for accumulated failures only (passes skip feedback)
        failures_for_step = accumulated_failures
        pass_context_for_step = accumulated_pass_context
        step_total_groups = accumulated_total_groups
        step_mixed_groups = accumulated_mixed_groups
        step_skipped_all_fail_groups = accumulated_skipped_all_fail_groups
        step_total_failures = accumulated_total_failures
        step_eligible_failures = accumulated_eligible_failures
        accumulated_failures = []
        accumulated_pass_context = []
        accumulated_total_groups = 0
        accumulated_mixed_groups = 0
        accumulated_skipped_all_fail_groups = 0
        accumulated_total_failures = 0
        accumulated_eligible_failures = 0

        t_feedback = time.monotonic()

        # Build solution lookup: problem_idx -> first successful response text (for demos)
        solution_by_idx: dict[int, str] = {}
        for r in pass_context_for_step:
            if r.idx not in solution_by_idx and r.response_text:
                solution_by_idx[r.idx] = r.response_text

        # Generate feedback for failures
        feedback_by_failure: list[str] = []
        if cfg.llm_feedback and failures_for_step:
            async def _get_feedback(r: GradeResult) -> str:
                async with llm_sem:
                    return await generate_llm_feedback(
                        question=r.question,
                        code=r.code or "(no code extracted)",
                        stderr=r.error,
                        stdout="",
                        model=cfg.llm_feedback_model,
                    )

            feedback_by_failure = list(await asyncio.gather(
                *[_get_feedback(r) for r in failures_for_step]
            ))
        elif failures_for_step:
            feedback_by_failure = [
                cfg.feedback_template.format(error=r.error) for r in failures_for_step
            ]

        # Build 4-tuples: (completion, feedback, solution_demo, env_feedback)
        sdpo_tuples: list[tuple[SampledCompletion, str, str | None, str | None]] = []
        for r, fb in zip(failures_for_step, feedback_by_failure):
            sdpo_tuples.append((
                r.completion,  # type: ignore[arg-type]
                fb,
                solution_by_idx.get(r.idx),
                r.error if sdpo_config.include_environment_feedback else None,
            ))

        # 4. SDPO gradient step
        t_sdpo = time.monotonic()
        metrics = await sdpo_train_step(
            training_client=training_client,
            teacher_sampling_client=teacher_sampling_client,
            renderer=renderer,
            sdpo_config=sdpo_config,
            learning_rate=cfg.learning_rate,
            completions_and_feedback=sdpo_tuples,
        )
        t_done = time.monotonic()

        if metrics:
            # Refresh sampling client with updated weights
            sampling_client = await training_client.save_weights_and_get_sampling_client_async("current")
            if teacher_is_student:
                teacher_sampling_client = sampling_client
            sdpo_steps += 1

            # Response length of all completions we trained on
            all_step_results = failures_for_step  # GradeResult list
            failure_tokens = [r.response_tokens for r in all_step_results]
            failure_resp_mean = sum(failure_tokens) / max(1, len(failure_tokens))

            # Enrich metrics
            metrics["sdpo_step"] = float(sdpo_steps)
            metrics["batch_idx"] = float(batch_idx)
            metrics["n_failures"] = float(len(failures_for_step))
            metrics["n_sdpo_examples"] = float(len(failures_for_step))
            metrics["batch_pass_rate"] = batch_passed / max(1, len(results))
            metrics["cumulative_pass_rate"] = total_passed / max(1, total_graded)
            metrics["response_length_mean"] = resp_len_mean
            metrics["response_length_max"] = float(resp_len_max)
            metrics["failure_response_length_mean"] = failure_resp_mean
            metrics["total_tokens_generated"] = float(total_tokens_generated)
            metrics["solved_problems"] = float(len(solved_indices))
            metrics["total_sampled"] = float(total_graded)
            metrics["group/mean_reward"] = group_mean_reward
            metrics["group/groups_with_variance_pct"] = (
                total_groups_with_variance / max(1, total_groups)
            )
            metrics["train/mixed_group_fraction"] = step_mixed_groups / max(1, step_total_groups)
            metrics["train/eligible_failure_fraction"] = (
                step_eligible_failures / max(1, step_total_failures)
            )
            metrics["train/skipped_all_fail_groups"] = float(step_skipped_all_fail_groups)
            # Time breakdown
            metrics["time_sample_s"] = round(t_sampled - t0, 2)
            metrics["time_feedback_s"] = round(t_sdpo - t_feedback, 2)
            metrics["time_sdpo_s"] = round(t_done - t_sdpo, 2)
            metrics["time_total_s"] = round(t_done - t0, 2)
            metrics["wall_time_s"] = round(t_done - t_start, 1)

            # Difficulty metrics
            solved_problems = [p for p in problem_tracker.values() if p["first_solve_step"] is not None]
            unsolved_problems = [p for p in problem_tracker.values() if p["first_solve_step"] is None]
            metrics["difficulty/unsolved_count"] = float(len(unsolved_problems))
            metrics["difficulty/solved_count"] = float(len(solved_problems))
            if solved_problems:
                metrics["difficulty/avg_attempts_to_solve"] = sum(
                    p["attempts"] for p in solved_problems
                ) / len(solved_problems)
            if unsolved_problems:
                hardest = max(unsolved_problems, key=lambda p: p["attempts"])
                metrics["difficulty/hardest_unsolved_attempts"] = float(hardest["attempts"])

            logger.info(
                f"  SDPO step {sdpo_steps}: loss={metrics.get('loss', 0):.4f} "
                f"kl={metrics.get('sdpo_kl', 0):.4f} "
                f"failures={len(failures_for_step)} "
                f"pass_rate={metrics['cumulative_pass_rate']:.1%} "
                f"[sample={metrics['time_sample_s']:.1f}s feed={metrics['time_feedback_s']:.1f}s sdpo={metrics['time_sdpo_s']:.1f}s]"
            )

            # Write metrics
            metrics_f.write(json.dumps(metrics) + "\n")
            metrics_f.flush()
            _log_wandb(metrics, step=total_graded)

        # 5. Checkpoint (contains weights AFTER this SDPO step)
        if cfg.save_every > 0 and sdpo_steps > 0 and sdpo_steps % cfg.save_every == 0:
            ckpt_name = f"sdpo_{sdpo_steps:06d}"
            logger.info(f"  Saving checkpoint: {ckpt_name}")
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=ckpt_name,
                log_path=cfg.log_path,
                kind="both",
                loop_state={
                    "sdpo_steps": sdpo_steps,
                    "batch_idx": batch_idx + 1,
                    "total_passed": total_passed,
                    "total_graded": total_graded,
                    "solved_count": len(solved_indices),
                },
            )

        # 6. Held-out eval
        if eval_set and cfg.eval_every > 0 and sdpo_steps > 0 and sdpo_steps % cfg.eval_every == 0:
            logger.info(f"  Running eval on {len(eval_set)} held-out problems...")
            t_eval = time.monotonic()
            eval_pass_rate = await run_eval_pass(eval_set, sampling_client, renderer, cfg)
            t_eval_done = time.monotonic()
            eval_metrics = {
                "eval/pass@1": eval_pass_rate,
                "eval/time_s": round(t_eval_done - t_eval, 2),
            }
            logger.info(
                f"  Eval pass@1={eval_pass_rate:.1%} ({t_eval_done - t_eval:.1f}s)"
            )
            _log_wandb(eval_metrics, step=total_graded)
            metrics_f.write(json.dumps({"sdpo_step": sdpo_steps, **eval_metrics}) + "\n")
            metrics_f.flush()

    # Final checkpoint
    if sdpo_steps > 0:
        logger.info("Training complete. Saving final checkpoint...")
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={
                "sdpo_steps": sdpo_steps,
                "batch_idx": n_batches,
                "total_passed": total_passed,
                "total_graded": total_graded,
                "solved_count": len(solved_indices),
            },
        )

    # Write per-problem difficulty log
    difficulty_path = os.path.join(cfg.log_path, "problem_difficulty.jsonl")
    with open(difficulty_path, "w") as df:
        for pidx, pdata in sorted(problem_tracker.items()):
            df.write(json.dumps({"problem_idx": pidx, **pdata}) + "\n")
    logger.info(f"Wrote difficulty data for {len(problem_tracker)} problems to {difficulty_path}")

    elapsed = time.monotonic() - t_start
    logger.info(
        f"Done: {sdpo_steps} SDPO steps, {total_graded} sampled, "
        f"pass_rate={total_passed}/{total_graded} ({total_passed/max(1,total_graded):.1%}), "
        f"{len(solved_indices)} unique problems solved, "
        f"{elapsed:.0f}s elapsed"
    )

    metrics_f.close()
    samples_f.close()
    if _wandb_run is not None:
        _wandb_run.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config: AutoTrainConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(run_auto_train(config))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
