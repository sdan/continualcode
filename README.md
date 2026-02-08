# continualcode

A coding agent that learns from your corrections in real-time. Built on [Tinker](https://thinkingmachines.ai/tinker).

When you deny a tool call with feedback, the model uses your correction as context to teach itself via [on-policy self-distillation](https://self-distillation.github.io/SDPO), takes a gradient step on LoRA, and retries with updated weights.

```
You: "fix the test"
Agent: write(test.py, ...)       # overwrites the file
You: n → "use edit_lines; don't overwrite"
  → SDPO update runs immediately
  → agent retries with updated weights
Agent: edit_lines(test.py, 14, 17, ...)
You: y
```

## Install

```bash
pip install continualcode
export TINKER_API_KEY=<your-key>
continualcode
```

## How it works

Four feedback types, one training signal. Your correction becomes privileged context for a self-teacher (same model, richer input). Per-token KL between teacher and student = dense training signal — O(N) bits per correction, not O(1). One gradient step on LoRA, retry with updated weights.

**[Full explanation →](docs/)**

## Why not DPO / GRPO / PPO / SFT

**DPO** needs preference pairs and has no per-token credit. **GRPO** needs 64 samples per prompt — absurd UX for a CLI. **PPO** doubles memory with a critic network. **SFT on corrections** is off-policy and causes catastrophic forgetting. Self-distillation is the unique intersection: dense signal, on-policy, no extra models, mode-seeking stability.

**[Full reasoning →](docs/design.md)**

## Code layout

- `train.py` — SDPO core: teacher prompt, logprob scoring, IS update, sampler refresh
- `tui.py` — interactive CLI: approve/deny/edit, correction prompt, `/metrics`
- `tools.py` — tool implementations + structured feedback
- `benchmarks/auto_train.py` — automated training loop (LCB, multi-rollout GRPO + SDPO)
- `demo/` — deny → train → retry end-to-end

## References

- [SDPO](https://self-distillation.github.io/SDPO) — Hübotter et al. 2026
- [SDFT](https://self-distillation.github.io/SDFT.html) — Shenfeld, Damani, Guestrin 2026
- [GKD](https://arxiv.org/abs/2306.13649) — Agarwal et al. 2023
- [Tinker](https://thinkingmachines.ai/tinker) — training API
