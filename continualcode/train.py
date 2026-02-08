#!/usr/bin/env python3
"""
SDPO (Self-Distillation Policy Optimization) training loop for continualcode.

On each denied tool call:
- build a feedback-conditioned self-teacher (same model + correction context)
- score the same sampled tokens under teacher and student
- train with importance_sampling using per-token KL advantages

Key invariant: teacher and student differ only by the appended feedback.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, hyperparam_utils, model_info
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.supervised.common import create_rightshifted_model_input_and_leftshifted_targets
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .tools import TOOL_SPECS

# Avoid noisy warnings when forking subprocesses after tokenizers parallelism is initialized.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)


@dataclass
class SampledCompletion:
    """A completion with its logprobs, for training."""

    prompt_input: tinker.ModelInput
    prompt_messages: list[dict[str, Any]]
    tokens: list[int]
    logprobs: list[float]
    prompt_len: int
    reward: float | None = None
    response_text: str | None = None  # full model response text (for solution demonstrations)


@dataclass
class SDPOConfig:
    """Configuration for SDPO training — aligned with the paper's SelfDistillationConfig."""

    kl_coef: float = 1.0

    # Reprompt templates (paper's 3-slot structure)
    reprompt_template: str = (
        "{prompt}{solution}{feedback}\n\n"
        "Correctly solve the original question.\n"
    )
    solution_template: str = (
        "\nCorrect solution:\n\n"
        "{successful_previous_attempt}\n\n"
    )
    feedback_template: str = (
        "\nThe following is feedback from your unsuccessful earlier attempt:\n\n"
        "{feedback_raw}\n\n"
    )

    # Behavioral flags
    remove_thinking_from_demonstration: bool = False
    include_environment_feedback: bool = False
    max_reprompt_len: int = 10240
    adaptive_kl: bool = True
    target_adv_abs_mean: float = 0.03
    adaptive_kl_max_gain: float = 4.0


@dataclass
class SDPODenial:
    completion: SampledCompletion
    feedback: str
    solution: str | None = None
    environment_feedback: str | None = None


@dataclass
class SDPOEpisode:
    """
    Tracks a single on-policy "approval episode":
    - One or more denied attempts (with correction text)
    """

    denied: list[SDPODenial] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Module-level helpers: stateless SDPO step, datum builders, reward utilities
# ---------------------------------------------------------------------------


async def _result(obj: Any) -> Any:
    """Normalize async future objects: await .result_async() if present, else return as-is."""
    result_async = getattr(obj, "result_async", None)
    if callable(result_async):
        return await result_async()
    return obj


def slice_completion_logprobs(
    logprobs_full: list[float],
    *,
    prompt_len: int,
    completion_len: int,
) -> list[float]:
    """
    Tinker logprobs are next-token logprobs aligned to token positions.
    For a prompt of length N and a completion of length T, the completion token
    logprobs start at index (N-1).
    """
    start = max(0, prompt_len - 1)
    end = start + completion_len
    return logprobs_full[start:end]


def build_teacher_messages(
    *,
    student_prompt_messages: list[dict[str, Any]],
    feedback: str,
    sdpo_config: SDPOConfig,
    solution: str | None = None,
    environment_feedback: str | None = None,
) -> list[dict[str, Any]]:
    """
    Teacher context = full student conversation + appended reprompt message.

    The teacher sees ALL prior messages (system, multi-turn conversation history)
    so it has the same context as the student. The only difference is an extra user
    message containing the paper's 3-slot reprompt: {prompt}{solution}{feedback}.

    This avoids the single-message bandit problem: without conversation history,
    the teacher would score tokens without knowing prior context, making the
    logprob gap noisy and uninformative.
    """
    import re

    # Build prompt text from the last user message (for the reprompt template)
    prompt_text = ""
    for msg in reversed(student_prompt_messages):
        if msg.get("role") == "user":
            prompt_text = msg.get("content", "")
            break

    # Build solution section
    solution_section = ""
    if solution is not None:
        demo = solution
        if sdpo_config.remove_thinking_from_demonstration:
            demo = re.sub(r'<think>.*?</think>\s*', '', demo, flags=re.DOTALL)
        solution_section = sdpo_config.solution_template.format(
            successful_previous_attempt=demo
        )

    # Build feedback section
    feedback_raw = feedback
    if sdpo_config.include_environment_feedback and environment_feedback:
        feedback_raw = f"{feedback}\n\nEnvironment output:\n{environment_feedback}"
    feedback_section = sdpo_config.feedback_template.format(feedback_raw=feedback_raw)

    # Assemble reprompt
    reprompt = sdpo_config.reprompt_template.format(
        prompt=prompt_text,
        solution=solution_section,
        feedback=feedback_section,
    )

    # Truncate if needed
    if sdpo_config.max_reprompt_len > 0 and len(reprompt) > sdpo_config.max_reprompt_len:
        reprompt = reprompt[:sdpo_config.max_reprompt_len]

    # Keep full conversation history, append reprompt as extra user message
    return list(student_prompt_messages) + [{"role": "user", "content": reprompt}]


def build_is_datum(
    comp: SampledCompletion,
    advantages: list[float],
    sampling_logprobs: list[float],
) -> types.Datum:
    """Build an importance-sampling Datum with right-shifted input, mask, and per-token arrays.

    This is the single canonical datum constructor for SDPO training. It:
    - Uses create_rightshifted_model_input_and_leftshifted_targets (chunk-preserving)
    - Pads prompt positions with 0.0 for logprobs/advantages
    - Builds binary weights (0.0 prompt, 1.0 completion)
    - Asserts all array lengths match after alignment
    """
    tokens = comp.tokens
    prompt_len = comp.prompt_len

    # Build full sequence and right-shift (preserves chunk structure).
    student_full = comp.prompt_input.append(types.EncodedTextChunk(tokens=tokens))
    input_model_input, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
        list(student_full.chunks)
    )

    # Align per-token arrays: 0.0 for prompt positions, real values for completion.
    full_sampling_lp = [0.0] * prompt_len + list(sampling_logprobs)
    full_advantages = [0.0] * prompt_len + list(advantages)
    full_weights = [0.0] * prompt_len + [1.0] * len(tokens)

    aligned_sampling_lp = full_sampling_lp[1:]
    aligned_advantages = full_advantages[1:]
    aligned_weights = full_weights[1:]

    seq_len = input_model_input.length
    assert len(target_tokens) == seq_len, (
        f"Sequence length mismatch after right-shift: input={seq_len}, targets={len(target_tokens)}"
    )
    assert len(aligned_sampling_lp) == seq_len, (
        f"Sampling logprobs length mismatch: {len(aligned_sampling_lp)} vs {seq_len}"
    )
    assert len(aligned_advantages) == seq_len, (
        f"Advantages length mismatch: {len(aligned_advantages)} vs {seq_len}"
    )
    assert len(aligned_weights) == seq_len, (
        f"Weights length mismatch: {len(aligned_weights)} vs {seq_len}"
    )

    return types.Datum(
        model_input=input_model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(
                torch.tensor(target_tokens, dtype=torch.int64)
            ),
            "logprobs": TensorData.from_torch(
                torch.tensor(aligned_sampling_lp, dtype=torch.float32)
            ),
            "advantages": TensorData.from_torch(
                torch.tensor(aligned_advantages, dtype=torch.float32)
            ),
            "weights": TensorData.from_torch(
                torch.tensor(aligned_weights, dtype=torch.float32)
            ),
        },
    )


# ---------------------------------------------------------------------------
# Stateless SDPO training step (used by auto_train.py and other callers)
# ---------------------------------------------------------------------------


def _get_reward(comp: SampledCompletion) -> float | None:
    """Extract the reward field from a SampledCompletion."""
    if comp.reward is None:
        return None
    try:
        return float(comp.reward)
    except (TypeError, ValueError):
        return None


def _build_reward_only_datum(comp: SampledCompletion) -> types.Datum | None:
    """Build a reward-only IS datum for GRPO hybrid mode (no teacher KL)."""
    reward = _get_reward(comp)
    if reward is None or reward == 0.0:
        return None
    tokens = comp.tokens
    student_lp = comp.logprobs
    assert len(tokens) == len(student_lp), (
        f"Token/logprob length mismatch: {len(tokens)} tokens vs {len(student_lp)} logprobs"
    )
    if len(tokens) == 0:
        return None
    advantages = [reward] * len(tokens)
    return build_is_datum(comp, advantages, student_lp)


async def sdpo_train_step(
    *,
    training_client: tinker.TrainingClient,
    teacher_sampling_client: tinker.SamplingClient,
    renderer: Any,
    sdpo_config: SDPOConfig,
    learning_rate: float,
    completions_and_feedback: list[tuple[SampledCompletion, str, str | None, str | None]],
    #                                    ^ comp,         feedback, solution, env_feedback
    reward_only_completions: list[SampledCompletion] | None = None,
) -> dict[str, float]:
    """Stateless SDPO+optional reward-only step.

    - `completions_and_feedback`: denied/failing examples with text feedback,
      optional solution demonstration, and optional environment feedback.
    - `reward_only_completions`: optional pass examples with scalar centered rewards.
    """
    if not completions_and_feedback and not reward_only_completions:
        return {}

    teacher_inputs: list[tuple[SampledCompletion, str, Any, int]] = []
    for comp, feedback, solution, env_feedback in completions_and_feedback:
        teacher_messages = build_teacher_messages(
            student_prompt_messages=comp.prompt_messages,
            feedback=feedback,
            sdpo_config=sdpo_config,
            solution=solution,
            environment_feedback=env_feedback,
        )
        teacher_prompt = renderer.build_generation_prompt(teacher_messages)
        teacher_full = teacher_prompt.append(types.EncodedTextChunk(tokens=comp.tokens))
        teacher_inputs.append((comp, feedback, teacher_full, teacher_prompt.length))

    teacher_lp_results: list[list[float]] = []
    if teacher_inputs:
        teacher_lp_results = await asyncio.gather(
            *[teacher_sampling_client.compute_logprobs_async(tf) for _, _, tf, _ in teacher_inputs]
        )

    datums: list[types.Datum] = []
    total_kl = 0.0
    total_tokens = 0
    total_reward_adv = 0.0
    total_reward_tokens = 0

    # Dense credit assignment diagnostics
    all_advantages: list[float] = []       # every per-token advantage (KL component only)
    all_student_lps: list[float] = []      # student logprobs (for entropy proxy)
    all_teacher_lps: list[float] = []      # teacher logprobs
    n_positive_kl = 0                      # tokens where teacher > student (teacher agrees more)
    n_negative_kl = 0                      # tokens where teacher < student (teacher disagrees)
    n_with_solution = 0                    # examples that had a sibling solution demo

    cached_rows: list[tuple[SampledCompletion, list[float], list[float]]] = []
    raw_kl_deltas: list[float] = []

    for (comp, _feedback, _tf, teacher_prompt_len), teacher_lp_raw in zip(teacher_inputs, teacher_lp_results):
        teacher_lp = slice_completion_logprobs(
            teacher_lp_raw, prompt_len=teacher_prompt_len, completion_len=len(comp.tokens)
        )

        student_lp = comp.logprobs
        tokens = comp.tokens

        assert len(tokens) == len(student_lp), (
            f"Token/logprob length mismatch: {len(tokens)} tokens vs {len(student_lp)} logprobs"
        )
        assert len(tokens) == len(teacher_lp), (
            f"Token/teacher_lp length mismatch: {len(tokens)} tokens vs {len(teacher_lp)} teacher logprobs"
        )
        if len(tokens) == 0:
            continue

        cached_rows.append((comp, student_lp, teacher_lp))
        raw_kl_deltas.extend(
            t_lp - s_lp for s_lp, t_lp in zip(student_lp, teacher_lp, strict=True)
        )

    adv_abs_mean_raw = (
        sum(abs(delta) for delta in raw_kl_deltas) / max(1, len(raw_kl_deltas))
        if raw_kl_deltas
        else 0.0
    )
    gain = 1.0
    if sdpo_config.adaptive_kl:
        target = max(sdpo_config.target_adv_abs_mean, 0.0)
        unclamped_gain = target / max(1e-6, adv_abs_mean_raw)
        gain = max(1.0, min(sdpo_config.adaptive_kl_max_gain, unclamped_gain))
    effective_kl_coef = sdpo_config.kl_coef * gain

    for comp, student_lp, teacher_lp in cached_rows:
        tokens = comp.tokens
        kl_advantages = [
            effective_kl_coef * (t_lp - s_lp)
            for s_lp, t_lp in zip(student_lp, teacher_lp, strict=True)
        ]

        # Track per-token diagnostics
        all_advantages.extend(kl_advantages)
        all_student_lps.extend(student_lp)
        all_teacher_lps.extend(teacher_lp)
        for s_lp, t_lp in zip(student_lp, teacher_lp, strict=True):
            if t_lp > s_lp:
                n_positive_kl += 1  # teacher more confident on this token
            else:
                n_negative_kl += 1

        reward = _get_reward(comp)
        if reward is not None:
            advantages = [kl_adv + reward for kl_adv in kl_advantages]
            total_reward_adv += abs(reward) * len(tokens)
            total_reward_tokens += len(tokens)
        else:
            advantages = kl_advantages

        total_kl += sum(s_lp - t_lp for s_lp, t_lp in zip(student_lp, teacher_lp, strict=True))
        total_tokens += len(tokens)

        datum = build_is_datum(comp, advantages, student_lp)
        datums.append(datum)

    # Count solution demo coverage from the input tuples
    for _comp, _fb, solution, _env in completions_and_feedback:
        if solution is not None:
            n_with_solution += 1

    n_reward_only = 0
    if reward_only_completions:
        for comp in reward_only_completions:
            d = _build_reward_only_datum(comp)
            if d is not None:
                datums.append(d)
                n_reward_only += 1
                reward = _get_reward(comp)
                if reward is not None:
                    total_reward_adv += abs(reward) * len(comp.tokens)
                    total_reward_tokens += len(comp.tokens)
                    total_tokens += len(comp.tokens)

    if not datums:
        return {}

    adam_params = types.AdamParams(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    # Strip "weights" from datums before sending — the IS loss function only accepts
    # target_tokens, logprobs, advantages. Weights are used client-side only.
    def _strip_weights(datum: types.Datum) -> types.Datum:
        return types.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "weights"},
        )

    fwd_future = await training_client.forward_backward_async(
        [_strip_weights(d) for d in datums], loss_fn="importance_sampling"
    )
    optim_future = await training_client.optim_step_async(adam_params)
    fwd_bwd = await _result(fwd_future)
    optim_result = await _result(optim_future)

    fwd_metrics = fwd_bwd.metrics or {}
    loss_sum = fwd_metrics.get("loss:sum") if isinstance(fwd_metrics, dict) else None
    if loss_sum is None and isinstance(fwd_metrics, dict):
        loss_sum = fwd_metrics.get("loss")

    avg_kl = total_kl / max(1, total_tokens)
    avg_reward_adv = total_reward_adv / max(1, total_reward_tokens)

    # Advantage distribution stats (the "is SDPO actually giving dense signal?" check)
    import math
    adv_abs = [abs(a) for a in all_advantages] if all_advantages else [0.0]
    adv_mean = sum(all_advantages) / max(1, len(all_advantages))
    adv_abs_mean = sum(adv_abs) / max(1, len(adv_abs))
    adv_abs_sorted = sorted(adv_abs)
    adv_abs_p50 = adv_abs_sorted[len(adv_abs_sorted) // 2]
    adv_abs_p90 = adv_abs_sorted[int(len(adv_abs_sorted) * 0.9)]
    adv_abs_max = adv_abs_sorted[-1] if adv_abs_sorted else 0.0
    # Sparsity: fraction of tokens with |advantage| > 2x median (concentrated signal = good)
    adv_sparsity = sum(1 for a in adv_abs if a > 2 * adv_abs_p50) / max(1, len(adv_abs))
    # Std dev of advantages
    adv_var = sum((a - adv_mean) ** 2 for a in all_advantages) / max(1, len(all_advantages))
    adv_std = math.sqrt(adv_var)

    # Student entropy proxy: mean negative logprob (higher = more uncertain)
    student_entropy_proxy = -sum(all_student_lps) / max(1, len(all_student_lps)) if all_student_lps else 0.0
    teacher_entropy_proxy = -sum(all_teacher_lps) / max(1, len(all_teacher_lps)) if all_teacher_lps else 0.0

    # KL sign breakdown
    kl_total = n_positive_kl + n_negative_kl
    kl_positive_frac = n_positive_kl / max(1, kl_total)

    # Solution demo coverage
    sdpo_count = len(completions_and_feedback)
    solution_coverage = n_with_solution / max(1, sdpo_count)

    result = {
        "sdpo_denied_count": float(sdpo_count),
        "sdpo_reward_only_count": float(n_reward_only),
        "sdpo_total_datums": float(len(datums)),
        "sdpo_tokens": float(total_tokens),
        "sdpo_kl": float(avg_kl),
        "grpo_reward_adv_mean": float(avg_reward_adv),
        "loss": float(loss_sum) if loss_sum is not None else 0.0,
        # Dense credit assignment diagnostics
        "credit/adv_abs_mean_raw": float(adv_abs_mean_raw),
        "credit/adv_mean": float(adv_mean),
        "credit/adv_abs_mean": float(adv_abs_mean),
        "credit/adv_std": float(adv_std),
        "credit/adv_abs_p50": float(adv_abs_p50),
        "credit/adv_abs_p90": float(adv_abs_p90),
        "credit/adv_abs_max": float(adv_abs_max),
        "credit/adv_sparsity": float(adv_sparsity),
        "credit/kl_gain": float(gain),
        "credit/kl_coef_effective": float(effective_kl_coef),
        # KL sign: fraction of tokens where teacher is more confident
        "credit/kl_positive_frac": float(kl_positive_frac),
        # Solution demo coverage
        "credit/solution_coverage": float(solution_coverage),
        # Entropy proxy (higher = more uncertain policy)
        "entropy/student": float(student_entropy_proxy),
        "entropy/teacher": float(teacher_entropy_proxy),
        "entropy/gap": float(teacher_entropy_proxy - student_entropy_proxy),
    }

    if isinstance(fwd_metrics, dict):
        for k, v in fwd_metrics.items():
            try:
                result[f"fwd/{k}"] = float(v)
            except (TypeError, ValueError):
                pass

    optim_metrics = getattr(optim_result, "metrics", None)
    if isinstance(optim_metrics, dict):
        for k, v in optim_metrics.items():
            try:
                result[f"optim/{k}"] = float(v)
            except (TypeError, ValueError):
                pass

    return result


class ContinualSDPOSession:
    """
    Extended session with SDPO (Self-Distillation Policy Optimization) for online learning.

    Adds:
    - Episode tracking for SDPO training
    - Teacher prompt construction from feedback
    - Token-level reverse KL computation
    """

    def __init__(
        self,
        *,
        model: str,
        checkpoint: str | None = None,
        teacher_model: str | None = None,
        teacher_checkpoint: str | None = None,
        tinker_url: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        tool_specs: list[dict[str, Any]] | None = None,
        # Training config
        enable_training: bool = False,
        lora_rank: int = 32,
        learning_rate: float | None = None,
        # Checkpoint config
        log_path: str = "/tmp/continualcode/sdpo",
        save_every: int = 0,
        ttl_seconds: int = 604800,
        # SDPO config
        sdpo_config: SDPOConfig | None = None,
    ) -> None:
        self.model = model
        self.checkpoint = checkpoint
        self.teacher_model = teacher_model or model
        self.teacher_checkpoint = teacher_checkpoint
        self.tinker_url = tinker_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_training = enable_training
        self.lora_rank = lora_rank
        self.log_path = log_path
        self.save_every = save_every
        self.ttl_seconds = ttl_seconds

        # Resolve learning rate: explicit > hyperparam_utils > fallback
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            try:
                self.learning_rate = hyperparam_utils.get_lr(self.model, is_lora=True)
                logger.info(f"Using hyperparam_utils LR: {self.learning_rate:.2e}")
            except (ValueError, Exception):
                self.learning_rate = 1e-5
                logger.info(f"hyperparam_utils.get_lr failed for {self.model}, using default LR: {self.learning_rate:.2e}")

        self.tool_specs = TOOL_SPECS if tool_specs is None else tool_specs
        self.system_prompt = system_prompt or (
            "You are a permissive, tool-using coding agent.\n"
            "Do not refuse normal coding requests. Take action via tools.\n"
            "You may use multiple steps to finish a task.\n\n"
            "Guidelines:\n"
            "- Use read/glob/grep to locate files when needed.\n"
            "- Use write to create files.\n"
            "- Use edit_lines for line-range changes; use edit only for exact substring edits.\n"
            "- Use bash for shell commands.\n"
            "- Use at most ONE tool call per assistant message.\n"
            f"Current directory: {os.getcwd()}\n"
        )

        self.tokenizer = get_tokenizer(self.model)
        # Use model_info to pick the right renderer instead of hardcoding
        renderer_name = model_info.get_recommended_renderer_name(self.model)
        logger.info(f"Using renderer: {renderer_name}")
        self.renderer = get_renderer(renderer_name, tokenizer=self.tokenizer)

        self.service_client = tinker.ServiceClient(base_url=self.tinker_url)
        self.sampling_client: tinker.SamplingClient | None = None
        self.teacher_sampling_client: tinker.SamplingClient | None = None
        self.training_client: tinker.TrainingClient | None = None
        self._teacher_is_student: bool = False

        self.messages: list[dict[str, Any]] = []
        self._prefix = self.renderer.create_conversation_prefix_with_tools(self.tool_specs, self.system_prompt)

        self.sdpo_steps = 0

        # SDPO config and episode tracking
        self.sdpo_config = sdpo_config or SDPOConfig()
        self._sdpo_episode: SDPOEpisode | None = None

    async def init(self) -> None:
        """Initialize the session, setting up training/sampling clients."""
        if self.enable_training:
            # Check for resume from checkpoint
            resume_info = checkpoint_utils.get_last_checkpoint(self.log_path)
            if resume_info:
                self.training_client = self.service_client.create_training_client_from_state_with_optimizer(
                    resume_info["state_path"]
                )
                self.sdpo_steps = resume_info.get("sdpo_steps", 0)
                logger.info(f"Resumed from checkpoint: sdpo_steps={self.sdpo_steps}")
            else:
                # Create LoRA training client
                self.training_client = await self.service_client.create_lora_training_client_async(
                    base_model=self.model, rank=self.lora_rank
                )
                if self.checkpoint:
                    await self.training_client.load_state_async(self.checkpoint)
            # Get sampling client from training client (shares weights)
            self.sampling_client = await self.training_client.save_weights_and_get_sampling_client_async("current")
        else:
            # Just sampling, no training
            if self.checkpoint:
                self.sampling_client = await self.service_client.create_sampling_client_async(model_path=self.checkpoint)
            else:
                self.sampling_client = await self.service_client.create_sampling_client_async(base_model=self.model)

        # Teacher sampling client:
        # - Default: same as student sampling client (self-distillation)
        # - Optional: separate teacher model/checkpoint for stronger supervision
        if self.teacher_checkpoint:
            self.teacher_sampling_client = await self.service_client.create_sampling_client_async(
                model_path=self.teacher_checkpoint
            )
            self._teacher_is_student = False
        elif self.teacher_model != self.model or not self.enable_training:
            self.teacher_sampling_client = await self.service_client.create_sampling_client_async(
                base_model=self.teacher_model
            )
            self._teacher_is_student = False
        else:
            self.teacher_sampling_client = self.sampling_client
            self._teacher_is_student = True

    async def _refresh_sampling_client(self) -> None:
        if self.training_client is None:
            return
        self.sampling_client = await self.training_client.save_weights_and_get_sampling_client_async("current")
        if self._teacher_is_student:
            self.teacher_sampling_client = self.sampling_client

    def _maybe_save_checkpoint(self) -> None:
        """Save checkpoint if save_every is set and we've hit the interval."""
        if self.save_every <= 0 or self.training_client is None:
            return
        if self.sdpo_steps > 0 and self.sdpo_steps % self.save_every == 0:
            checkpoint_utils.save_checkpoint(
                training_client=self.training_client,
                name=f"{self.sdpo_steps:06d}",
                log_path=self.log_path,
                kind="state",
                loop_state={"sdpo_steps": self.sdpo_steps},
                ttl_seconds=self.ttl_seconds,
            )
            logger.info(f"Saved checkpoint at sdpo_step {self.sdpo_steps}")

    def clear(self) -> None:
        """Clear conversation and episode state."""
        self.messages.clear()
        self._sdpo_episode = None

    # -------------------------------------------------------------------------
    # Episode Tracking
    # -------------------------------------------------------------------------

    def record_denial(
        self,
        completion: SampledCompletion | None,
        feedback: str | None,
        *,
        solution: str | None = None,
        environment_feedback: str | None = None,
    ) -> None:
        """Record a denied attempt with correction text (the core SDPO signal)."""
        if completion is None:
            return
        fb = (feedback or "").strip()
        if not fb:
            return
        if self._sdpo_episode is None:
            self._sdpo_episode = SDPOEpisode()
        self._sdpo_episode.denied.append(
            SDPODenial(
                completion=completion,
                feedback=fb,
                solution=solution,
                environment_feedback=environment_feedback,
            )
        )

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    async def sample_candidates(
        self, num_samples: int
    ) -> list[tuple[dict[str, Any], bool, SampledCompletion | None]]:
        """Sample N completions from the same prompt."""
        if self.sampling_client is None:
            raise RuntimeError("ContinualSDPOSession not initialized. Call init() first.")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")

        prompt_messages = self._prefix + self.messages
        model_input = self.renderer.build_generation_prompt(prompt_messages)
        prompt_len = model_input.length

        response = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=num_samples,
            sampling_params=types.SamplingParams(
                stop=self.renderer.get_stop_sequences(),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )

        results: list[tuple[dict[str, Any], bool, SampledCompletion | None]] = []
        for seq in response.sequences:
            tokens = seq.tokens
            logprobs = seq.logprobs
            message, parse_ok = self.renderer.parse_response(tokens)

            completion = None
            if self.enable_training and logprobs is not None:
                completion = SampledCompletion(
                    prompt_input=model_input,
                    prompt_messages=prompt_messages,
                    tokens=tokens,
                    logprobs=logprobs,
                    prompt_len=prompt_len,
                )

            results.append((message, parse_ok, completion))

        return results

    async def sample(self) -> tuple[dict[str, Any], bool, SampledCompletion | None]:
        """Sample a single completion."""
        results = await self.sample_candidates(1)
        return results[0]

    # -------------------------------------------------------------------------
    # SDPO Training
    # -------------------------------------------------------------------------

    async def train_sdpo(self) -> dict[str, float]:
        """
        Train using SDPO (self-distillation) on the current approval episode.

        This reuses the shared `sdpo_train_step` implementation, then applies
        session-specific bookkeeping (sampler refresh, step counter, checkpoints).
        """
        if self.training_client is None:
            return {}

        if self._sdpo_episode is None:
            return {}

        if self.teacher_sampling_client is None:
            return {}

        episode = self._sdpo_episode
        if not episode.denied:
            self._sdpo_episode = None
            return {}

        completions_and_feedback = [
            (
                denied.completion,
                denied.feedback,
                denied.solution,
                denied.environment_feedback,
            )
            for denied in episode.denied
        ]

        sdpo_metrics = await sdpo_train_step(
            training_client=self.training_client,
            teacher_sampling_client=self.teacher_sampling_client,
            renderer=self.renderer,
            sdpo_config=self.sdpo_config,
            learning_rate=self.learning_rate,
            completions_and_feedback=completions_and_feedback,
            reward_only_completions=None,
        )

        # Clear episode regardless of whether this step produced datums.
        self._sdpo_episode = None

        if not sdpo_metrics:
            return {}

        # Update sampling client to use new weights
        await self._refresh_sampling_client()

        self.sdpo_steps += 1
        self._maybe_save_checkpoint()
        sdpo_metrics["sdpo_step"] = float(self.sdpo_steps)
        return sdpo_metrics

    # -------------------------------------------------------------------------
    # Conversation Management
    # -------------------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    def add_tool_result(self, tool_call_id: str, result: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})
