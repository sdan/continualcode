#!/usr/bin/env python3
"""
SDPO (Self-Distillation Policy Optimization) training loop for a tool-using coding agent.

This module is the “cookbook core”:
- sample a tool-call action from the current policy
- when the user provides a correction, build a feedback-conditioned self-teacher
- score the *same sampled tokens* under teacher and student
- train with `loss_fn="importance_sampling"` using per-token advantages derived from the logprob gap

Key invariant for clean credit assignment: teacher and student differ only by the appended feedback.
"""

from __future__ import annotations

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


@dataclass
class SDPOConfig:
    """Configuration for SDPO training."""

    kl_coef: float = 1.0            # KL penalty coefficient
    is_clip: float = 2.0            # Optional IS ratio clipping (effective if doing off-policy updates)
    feedback_template: str = "User correction: {feedback}"
    reprompt_suffix: str = "Correctly solve the original question."


@dataclass
class SDPODenial:
    completion: SampledCompletion
    feedback: str


@dataclass
class SDPOEpisode:
    """
    Tracks a single on-policy "approval episode":
    - One or more denied attempts (with correction text)
    """

    denied: list[SDPODenial] = field(default_factory=list)


class ContinualSDPOSession:
    """
    Extended session with SDPO (Self-Distillation Policy Optimization) for online learning.

    Adds:
    - Episode tracking for SDPO training
    - Teacher prompt construction from feedback
    - Token-level reverse KL computation
    - IS ratio clipping
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
            "- Use execute (alias of bash) for shell commands.\n"
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

        # Training step counters
        self.train_steps = 0
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
                self.train_steps = resume_info.get("train_steps", 0)
                self.sdpo_steps = resume_info.get("sdpo_steps", 0)
                logger.info(f"Resumed from checkpoint: train_steps={self.train_steps}, sdpo_steps={self.sdpo_steps}")
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
        total_steps = self.train_steps + self.sdpo_steps
        if total_steps > 0 and total_steps % self.save_every == 0:
            checkpoint_utils.save_checkpoint(
                training_client=self.training_client,
                name=f"{total_steps:06d}",
                log_path=self.log_path,
                kind="state",
                loop_state={"train_steps": self.train_steps, "sdpo_steps": self.sdpo_steps},
                ttl_seconds=self.ttl_seconds,
            )
            logger.info(f"Saved checkpoint at step {total_steps}")

    @staticmethod
    async def _result(obj: Any) -> Any:
        """
        Tinker async APIs sometimes return Future-like objects (with .result_async()).
        This helper normalizes either a Future-like object or an already-materialized result.
        """
        result_async = getattr(obj, "result_async", None)
        if callable(result_async):
            return await result_async()
        return obj

    def clear(self) -> None:
        """Clear conversation and episode state."""
        self.messages.clear()
        self._sdpo_episode = None

    @property
    def prefix_messages(self) -> list[dict[str, Any]]:
        return list(self._prefix)

    # -------------------------------------------------------------------------
    # Episode Tracking
    # -------------------------------------------------------------------------

    def reset_sdpo_episode(self) -> None:
        """Reset SDPO episode state (call after finishing an approval episode)."""
        self._sdpo_episode = None

    def record_denial(self, completion: SampledCompletion | None, feedback: str | None) -> None:
        """Record a denied attempt with correction text (the core SDPO signal)."""
        if completion is None:
            return
        fb = (feedback or "").strip()
        if not fb:
            return
        if self._sdpo_episode is None:
            self._sdpo_episode = SDPOEpisode()
        self._sdpo_episode.denied.append(SDPODenial(completion=completion, feedback=fb))

    def record_sdpo_denial(self, completion: SampledCompletion | None, feedback: str | None) -> None:
        """Backward-compatible alias for record_denial."""
        self.record_denial(completion, feedback)

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

    def _build_teacher_messages(
        self,
        *,
        student_prompt_messages: list[dict[str, Any]],
        feedback: str,
    ) -> list[dict[str, Any]]:
        """
        Teacher context = student context + an extra user message containing:
        - the user correction text
        - an instruction to solve correctly

        This keeps credit assignment clean: teacher/student differ ONLY by the added feedback,
        but we score the student's actual sampled tokens under both.
        """
        parts: list[str] = []
        parts.append(self.sdpo_config.feedback_template.format(feedback=feedback))
        parts.append(self.sdpo_config.reprompt_suffix)
        extra = "\n\n".join(parts)
        return list(student_prompt_messages) + [{"role": "user", "content": extra}]

    @staticmethod
    def _slice_completion_logprobs(
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

    async def train_sdpo(self) -> dict[str, float]:
        """
        Train using SDPO (self-distillation) on the current approval episode.

        For each denied attempt with correction text:
        1. Teacher prompt = student prompt + correction
        2. Compute teacher logprobs on the student's sampled tokens
        3. Per-token advantage = -kl_coef * (student_lp - teacher_lp)
        4. Train with tinker loss_fn="importance_sampling" using sampling logprobs as behavior policy.
        """
        if self.training_client is None:
            return {}

        if self._sdpo_episode is None:
            return {}

        if self.sampling_client is None or self.teacher_sampling_client is None:
            return {}

        episode = self._sdpo_episode
        if not episode.denied:
            self._sdpo_episode = None
            return {}

        datums: list[types.Datum] = []
        total_kl = 0.0
        total_tokens = 0
        ratio_sum = 0.0
        ratio_count = 0

        for denied in episode.denied:
            comp = denied.completion
            feedback = denied.feedback

            # Teacher prompt: same context + extra feedback message
            teacher_messages = self._build_teacher_messages(
                student_prompt_messages=comp.prompt_messages,
                feedback=feedback,
            )
            teacher_prompt = self.renderer.build_generation_prompt(teacher_messages)
            teacher_prompt_len = teacher_prompt.length

            # Teacher logprobs on student's sampled tokens
            teacher_full = teacher_prompt.append(types.EncodedTextChunk(tokens=comp.tokens))
            teacher_lp_raw = await self.teacher_sampling_client.compute_logprobs_async(teacher_full)
            teacher_lp = self._slice_completion_logprobs(
                teacher_lp_raw, prompt_len=teacher_prompt_len, completion_len=len(comp.tokens)
            )

            student_lp = comp.logprobs
            tokens = comp.tokens

            min_len = min(len(tokens), len(student_lp), len(teacher_lp))
            if min_len <= 0:
                continue
            tokens = tokens[:min_len]
            student_lp = student_lp[:min_len]
            teacher_lp = teacher_lp[:min_len]

            # Only reverse-KL shaping is supported for now.
            # advantage[t] = -kl_coef * (student_lp[t] - teacher_lp[t])
            advantages = [
                -self.sdpo_config.kl_coef * (s_lp - t_lp) for s_lp, t_lp in zip(student_lp, teacher_lp)
            ]

            total_kl += sum(s_lp - t_lp for s_lp, t_lp in zip(student_lp, teacher_lp))
            total_tokens += len(tokens)

            # Build full token sequence and right-shift for importance_sampling loss.
            student_full = comp.prompt_input.append(types.EncodedTextChunk(tokens=tokens))
            full_tokens = list(student_full.to_ints())
            prompt_len = comp.prompt_len

            input_tokens = full_tokens[:-1]
            target_tokens = full_tokens[1:]

            full_sampling_lp = [0.0] * prompt_len + list(student_lp)
            full_advantages = [0.0] * prompt_len + advantages

            aligned_sampling_lp = full_sampling_lp[1:]
            aligned_advantages = full_advantages[1:]

            min_full = min(len(input_tokens), len(target_tokens), len(aligned_sampling_lp), len(aligned_advantages))
            input_tokens = input_tokens[:min_full]
            target_tokens = target_tokens[:min_full]
            aligned_sampling_lp = aligned_sampling_lp[:min_full]
            aligned_advantages = aligned_advantages[:min_full]

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(input_tokens),
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
                },
            )

            # Optional IS clipping: replace ratio with clipped ratio by scaling advantages.
            if self.sdpo_config.is_clip and self.sdpo_config.is_clip > 0:
                try:
                    fwd_future = await self.training_client.forward_async(
                        [datum], loss_fn="importance_sampling"
                    )
                    fwd = await self._result(fwd_future)
                    out0 = fwd.loss_fn_outputs[0] if fwd.loss_fn_outputs else {}
                    td = out0.get("logprobs")
                    if td is not None:
                        target_lp = td.to_torch().flatten().to(torch.float32)
                        sampling_lp = datum.loss_fn_inputs["logprobs"].to_torch().flatten().to(torch.float32)
                        adv = datum.loss_fn_inputs["advantages"].to_torch().flatten().to(torch.float32)
                        mask = adv != 0
                        if mask.any():
                            ratio = torch.exp(target_lp[mask] - sampling_lp[mask])
                            ratio_clipped = ratio.clamp(max=float(self.sdpo_config.is_clip))
                            scale = ratio_clipped / torch.clamp(ratio, min=1e-12)
                            adv2 = adv.clone()
                            adv2[mask] = adv[mask] * scale
                            datum.loss_fn_inputs["advantages"] = TensorData.from_torch(adv2)
                            ratio_sum += ratio.mean().item()
                            ratio_count += 1
                except Exception:
                    pass

            datums.append(datum)

        if not datums:
            self._sdpo_episode = None
            return {}

        # Forward-backward with importance sampling loss (async submit)
        fwd_future = await self.training_client.forward_backward_async(
            datums, loss_fn="importance_sampling"
        )

        # Optimizer step (submit immediately to overlap clock cycle)
        adam_params = types.AdamParams(
            learning_rate=self.learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )
        optim_future = await self.training_client.optim_step_async(adam_params)

        # Retrieve results
        fwd_bwd = await self._result(fwd_future)
        await self._result(optim_future)

        # Update sampling client to use new weights
        await self._refresh_sampling_client()

        self.sdpo_steps += 1
        self._maybe_save_checkpoint()

        # Compute metrics
        metrics = fwd_bwd.metrics or {}
        loss_sum = metrics.get("loss:sum") if isinstance(metrics, dict) else None
        if loss_sum is None and isinstance(metrics, dict):
            loss_sum = metrics.get("loss")

        avg_kl = total_kl / max(1, total_tokens)
        ratio_mean = (ratio_sum / ratio_count) if ratio_count > 0 else 1.0

        sdpo_metrics = {
            "sdpo_step": float(self.sdpo_steps),
            "sdpo_denied_count": float(len(episode.denied)),
            "sdpo_tokens": float(total_tokens),
            "sdpo_kl": float(avg_kl),
            "sdpo_ratio_mean": float(ratio_mean),
            "loss": float(loss_sum) if loss_sum is not None else 0.0,
        }

        # Clear episode
        self._sdpo_episode = None

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
