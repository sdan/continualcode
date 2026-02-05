#!/usr/bin/env python3
"""
SDPO (Self-Distillation Policy Optimization) backend for online continual learning.

	Core insight: When user denies a diff and gives feedback:
	    User: "fix the test"
	    Model: diff_1 → User: "no, wrong file - bug is in foo.py"
	    Model: diff_2 → User: "yes" ✓

	    SDPO Training:
	    - Sample once (diff_1) from the current policy (student).
	    - Build a self-teacher by conditioning the SAME model on the feedback.
	    - Evaluate the SAME sampled diff_1 tokens under both student and teacher (dense token-level signal).
	    - Distill teacher → student by minimizing per-token KL:
	        KL(student_next_token || stopgrad(teacher_next_token))
	      (often described as "reverse KL" in words).
	    - Intuition: if feedback makes the teacher less confident in wrong-file tokens, the student gets pushed away from them.

The model learns to anticipate corrections without needing them at inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Avoid noisy warnings when forking subprocesses after tokenizers parallelism is initialized.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import TrainOnWhat
from tinker_cookbook.supervised.common import compute_mean_nll, datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import get_tokenizer

try:
    from tools import TOOL_SPECS
except ImportError:  # pragma: no cover
    from .tools import TOOL_SPECS


@dataclass
class SampledCompletion:
    """A completion with its logprobs, for training."""
    prompt_input: tinker.ModelInput
    prompt_messages: list[dict[str, Any]]  # Full message list used to build prompt (includes prefix/tool defs)
    tokens: list[int]
    logprobs: list[float]
    prompt_len: int  # Token length of prompt_input


@dataclass
class SDPOConfig:
    """Configuration for SDPO training."""
    kl_coef: float = 1.0            # KL penalty coefficient
    is_clip: float = 2.0            # Optional IS ratio clipping (effective if doing off-policy updates)

    # Prompt templates for teacher construction
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


class SDPOSession:
    """
    Extended session with SDPO (Self-Distillation Policy Optimization) for online learning.

    Extends the basic TinkerSession with:
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
        learning_rate: float = 1e-5,
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
        self.learning_rate = learning_rate

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
        self.renderer = get_renderer("qwen3_instruct", tokenizer=self.tokenizer)

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
            # If training is enabled and teacher_model==student model, reuse.
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

    def record_sdpo_denial(self, completion: SampledCompletion | None, feedback: str | None) -> None:
        """Record a denied attempt with correction text (the core SDPO signal)."""
        if completion is None:
            return
        fb = (feedback or "").strip()
        if not fb:
            return
        if self._sdpo_episode is None:
            self._sdpo_episode = SDPOEpisode()
        self._sdpo_episode.denied.append(SDPODenial(completion=completion, feedback=fb))

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    async def sample_candidates(
        self, num_samples: int
    ) -> list[tuple[dict[str, Any], bool, SampledCompletion | None]]:
        """Sample N completions from the same prompt."""
        if self.sampling_client is None:
            raise RuntimeError("SDPOSession not initialized. Call init() first.")
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
        but we score the student's *actual sampled tokens* under both.
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
    # Standard RL Training (from parent TinkerSession)
    # -------------------------------------------------------------------------

    async def train_on_episode(self, steps: list[SampledCompletion], rewards: list[float]) -> dict[str, float]:
        """
        Train online on an episode/trajectory using importance-weighted policy gradient.

        Each step is an action sampled under the current policy; we push up/down that action
        proportional to (reward - baseline).
        """
        if self.training_client is None:
            return {}

        if len(steps) != len(rewards):
            raise ValueError(f"steps/rewards length mismatch: {len(steps)} vs {len(rewards)}")
        if not steps:
            return {}

        t = len(steps)
        mean_reward = sum(rewards) / t
        baseline = mean_reward if t > 1 else 0.0
        step_advantages = [r - baseline for r in rewards]

        datums: list[types.Datum] = []
        per_step_mean_logprob: list[float] = []

        for comp, advantage in zip(steps, step_advantages):
            per_step_mean_logprob.append(sum(comp.logprobs) / max(1, len(comp.logprobs)))

            # Build full token sequence and right-shift for importance_sampling loss.
            student_full = comp.prompt_input.append(types.EncodedTextChunk(tokens=comp.tokens))
            full_tokens = list(student_full.to_ints())
            prompt_len = comp.prompt_len

            input_tokens = full_tokens[:-1]
            target_tokens = full_tokens[1:]

            full_sampling_lp = [0.0] * prompt_len + list(comp.logprobs)
            full_advantages = [0.0] * prompt_len + [advantage] * len(comp.tokens)

            aligned_sampling_lp = full_sampling_lp[1:]
            aligned_advantages = full_advantages[1:]

            min_len = min(len(input_tokens), len(target_tokens), len(aligned_sampling_lp), len(aligned_advantages))
            input_tokens = input_tokens[:min_len]
            target_tokens = target_tokens[:min_len]
            aligned_sampling_lp = aligned_sampling_lp[:min_len]
            aligned_advantages = aligned_advantages[:min_len]

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(input_tokens),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.int64)),
                    "logprobs": TensorData.from_torch(torch.tensor(aligned_sampling_lp, dtype=torch.float32)),
                    "advantages": TensorData.from_torch(torch.tensor(aligned_advantages, dtype=torch.float32)),
                },
            )
            datums.append(datum)

        # Forward-backward with importance sampling loss
        fwd_future = await self.training_client.forward_backward_async(datums, loss_fn="importance_sampling")

        # Optimizer step
        adam_params = types.AdamParams(
            learning_rate=self.learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )
        optim_future = await self.training_client.optim_step_async(adam_params)

        fwd_bwd = await self._result(fwd_future)
        await self._result(optim_future)

        # Compute approx KL for diagnostics
        approx_kl = 0.0
        ratio_mean = 1.0
        try:
            fwd_after_future = await self.training_client.forward_async(
                datums, loss_fn="importance_sampling"
            )
            fwd_after = await self._result(fwd_after_future)
            old_logps: list[torch.Tensor] = []
            new_logps: list[torch.Tensor] = []
            for i, out in enumerate(fwd_after.loss_fn_outputs):
                td = out.get("logprobs")
                if td is None:
                    continue
                new_lp = td.to_torch().flatten().to(torch.float32)
                old_lp = datums[i].loss_fn_inputs["logprobs"].to_torch().flatten().to(torch.float32)
                adv = datums[i].loss_fn_inputs["advantages"].to_torch().flatten().to(torch.float32)
                mask = adv != 0
                if mask.any():
                    new_logps.append(new_lp[mask])
                    old_logps.append(old_lp[mask])
            if old_logps and new_logps:
                old_cat = torch.cat(old_logps)
                new_cat = torch.cat(new_logps)
                approx_kl = (old_cat - new_cat).mean().item()
                ratio_mean = torch.exp(new_cat - old_cat).mean().item()
        except Exception:
            pass

        # Update sampling client to use new weights
        await self._refresh_sampling_client()

        self.train_steps += 1

        metrics = fwd_bwd.metrics or {}
        loss_sum = metrics.get("loss:sum") if isinstance(metrics, dict) else None
        if loss_sum is None and isinstance(metrics, dict):
            loss_sum = metrics.get("loss")

        return {
            "step": float(self.train_steps),
            "t": float(t),
            "reward_sum": float(sum(rewards)),
            "reward_mean": float(mean_reward),
            "baseline": float(baseline),
            "adv_mean": float(sum(step_advantages) / t),
            "logprob_mean": float(sum(per_step_mean_logprob) / t),
            "loss:sum": float(loss_sum) if loss_sum is not None else 0.0,
            "approx_kl": float(approx_kl),
            "ratio_mean": float(ratio_mean),
        }

    def train_sft_on_messages(self, messages: list[dict[str, Any]]) -> dict[str, float]:
        """Supervised update on the last assistant message."""
        if self.training_client is None:
            return {}

        model_input, weights = self.renderer.build_supervised_example(
            messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        datum = datum_from_model_input_weights(model_input, weights, max_length=None)

        fwd_bwd = self.training_client.forward_backward([datum], loss_fn="cross_entropy").result()
        adam_params = types.AdamParams(
            learning_rate=self.learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )
        self.training_client.optim_step(adam_params).result()
        self._refresh_sampling_client()

        self.train_steps += 1

        logprobs = [x.get("logprobs") for x in fwd_bwd.loss_fn_outputs]
        logprobs_td = [x for x in logprobs if x is not None]
        weights_td = datum.loss_fn_inputs["weights"]
        train_nll = (
            compute_mean_nll(logprobs_td, [weights_td] * len(logprobs_td)) if logprobs_td else float("nan")
        )
        metrics = fwd_bwd.metrics or {}
        loss_sum = metrics.get("loss:sum", metrics.get("loss", 0.0)) if isinstance(metrics, dict) else 0.0
        return {
            "step": float(self.train_steps),
            "sft": 1.0,
            "loss:sum": float(loss_sum),
            "nll": float(train_nll),
        }

    # -------------------------------------------------------------------------
    # Conversation Management
    # -------------------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    def add_tool_result(self, tool_call_id: str, result: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})
