# Architecture: SDPO in continualcode

## Paper alignment

Our SDPO implementation targets the reference at [github.com/self-distillation/SDPO](https://github.com/self-distillation/SDPO) (verl-based, full-batch RL). We run on Tinker instead of verl, which constrains some features but enables online single-example updates.

### What matches the paper

| Feature | Reference | Ours | Notes |
|---------|-----------|------|-------|
| 3-slot reprompt template | `{prompt}{solution}{feedback}` | Same | `SDPOConfig.reprompt_template` |
| Solution demonstrations | First successful sibling rollout | Same | Built in `auto_train.py` solution map |
| `dont_distill_on_self_success` | Exclude self from demo pool | Same | Config flag |
| `remove_thinking_from_demonstration` | Strip `<think>` tags from demo | Same | Regex in `build_teacher_messages` |
| Environment feedback injection | Append env output to feedback | Same | `include_environment_feedback` flag |
| IS ratio clipping | `exp(π_current - π_old)` clamped | Same | Forward pass to get current logprobs, clamp ratio |
| GRPO group reward centering | Binary pass/fail, subtract group mean | Same | `sample_and_grade_group` |
| Per-token KL advantages | `adv = -(student_lp - teacher_lp)` | Same | Scalar logprob gap |

### What we can't do (Tinker API limitations)

**Full-logit KL / JSD (alpha interpolation)**
- Paper supports `alpha ∈ [0,1]`: forward KL (α=0), reverse KL (α=1), generalized JSD (0<α<1)
- Requires full vocabulary logit distributions from both student and teacher
- Tinker's `compute_logprobs` returns a **scalar per-token logprob**, not the full softmax distribution
- We're limited to reverse KL via the scalar logprob gap: `student_lp - teacher_lp`
- Impact: less expressive distillation signal, but still effective for code tasks where the correct token is usually high-probability

**EMA teacher (exponential moving average)**
- Paper uses `θ_teacher = (1-τ)·θ_teacher + τ·θ_student` with τ=0.05
- Tinker doesn't expose `model.parameters()` — weights live server-side
- Workaround would be checkpoint save → load → blend, but that's expensive (~seconds per step vs milliseconds for in-memory EMA)
- Our default: teacher IS the student (`self._teacher_is_student = True`), updated after each `optim_step` via `save_weights_and_get_sampling_client`
- This is equivalent to EMA with τ=1.0 (instant update), which is more aggressive than the paper's τ=0.05

**Top-k distillation**
- Paper supports `distillation_topk` for approximate full-distribution matching using only top-k logits
- Same API limitation as full-logit KL — Tinker returns scalar logprobs only

### What we chose not to implement (not blocked)

| Feature | Why skipped |
|---------|-------------|
| `environment_feedback_only_without_solution` | Niche flag — easy to add if needed |
| `trust-region` teacher regularization | Adds complexity, EMA is the paper's default |
| Loss aggregation modes (`seq-mean-token-sum`, etc.) | Tinker's `importance_sampling` loss_fn handles internally |
| `reprompt_truncation` direction (left/right) | We truncate the reprompt string; Tinker handles token-level truncation |

## Architecture: two training paths

### 1. Interactive (ContinualSDPOSession)

```
User → Agent proposes tool call → User denies with correction
                                        ↓
                              record_denial(completion, feedback)
                                        ↓
                              train_sdpo() → build teacher prompt
                                           → compute_logprobs (teacher)
                                           → KL advantages
                                           → forward_backward (IS loss)
                                           → optim_step
                                           → refresh sampling client
```

- Multi-turn: teacher sees full conversation history + appended reprompt
- Preserves extension property: student prompt is a prefix of teacher prompt → backend KV-cache reuse
- No solution demonstrations (human feedback IS the signal)
- Immediate single-example updates (no batching)

### 2. Benchmark (auto_train.py)

```
Sample N rollouts per problem → Sandbox grade → LLM feedback for failures
                                                        ↓
                                              Build solution map (passes)
                                                        ↓
                                              sdpo_train_step(
                                                failures + feedback + solution demos,
                                                reward_only passes
                                              )
```

- Single-turn: each problem is a fresh prompt (matches paper's setting)
- Solution demonstrations from successful sibling rollouts
- Batched updates: accumulate `min_sdpo_examples` before stepping
- Two training signals: `pure_sdpo` (KL only) or `hybrid` (KL + GRPO rewards)

## Key invariant

Teacher and student see the same conversation context. The only difference is the appended reprompt message containing feedback (and optionally a solution demonstration). This ensures the logprob gap `student_lp - teacher_lp` reflects the feedback signal, not context mismatch.

## Scaling considerations

**EMA teacher (τ=1.0 vs τ=0.05)**

We use τ=1.0 (instant teacher update) because Tinker doesn't expose weight-level operations. The paper uses τ=0.05. This is fine for our current setup (small batches, single gradient step, immediate refresh) but matters at scale:

- **Catastrophic forgetting**: Without EMA, a hard correction can shift weights enough to regress on already-solved problems. EMA keeps the teacher as a smoothed historical average that resists swings.
- **Feedback amplification**: Student overcorrects → teacher (= student) reflects the overcorrection → next signal is based on an already-drifted target → oscillation. EMA damps this by making the teacher move slowly.
- **Off-policy staleness**: In batched mode (`auto_train.py`), sampling logprobs from example 1 are stale by the time we process example 8 in the same step. EMA doesn't fix staleness directly but keeps the KL target stable.

Current mitigations: single gradient step per batch, immediate `save_weights_and_get_sampling_client` refresh, small `min_sdpo_examples`. If scaling to larger batches (32+) or multi-epoch training, EMA would help. A Tinker-compatible approximation: save a checkpoint every K steps and use it as a frozen teacher, refreshing periodically (coarse-grained EMA).

## Files

| File | Role |
|------|------|
| `train.py` | `SDPOConfig`, `SampledCompletion`, `build_teacher_messages`, `sdpo_train_step`, `ContinualSDPOSession` |
| `benchmarks/auto_train.py` | Automated LCB training loop, solution map construction, LLM feedback generation |
| `tui.py` | Interactive CLI, approval/denial flow, calls `record_denial` + `train_sdpo` |
| `tools.py` | Tool implementations + structured tool feedback |
