# Continual Code

We provide a small library and CLI for building a **self‑improving coding agent** with **online SDPO (Self‑Distillation Policy Optimization)** on top of **Tinker**.

- `continualcode` is a CLI coding agent with tool use (`read`, `write`, `edit_lines`, `glob`, `grep`, `bash`/`execute`). You approve or deny each tool call.
- When you deny with a correction (or a tool fails and you correct it), the agent performs an immediate SDPO update and retries in the same session.

References:
- SDPO project page: https://self-distillation.github.io/SDPO
- SDPO paper: https://arxiv.org/abs/2601.20802
- Tinker: https://thinkingmachines.ai/tinker

## Installation

1. Create a Tinker API key from the console and export it as `TINKER_API_KEY`.
2. Install the package.

```bash
pip install continualcode
export TINKER_API_KEY=<your-key>
```

## Quick start

```bash
continualcode
```

Example interaction:

```
You: "fix the test"
Agent proposes: write(test.py, ...)  # overwrites the file
You: n → "use edit_lines; don't overwrite"
  → SDPO update runs immediately
  → agent retries with updated weights
```

## Code layout

- `continualcode/train.py`: SDPO training loop (teacher prompt, logprob scoring, `importance_sampling` update, sampler refresh).
- `continualcode/tui.py`: interactive CLI (approve/deny/edit args, correction prompt, metrics).
- `continualcode/tools.py`: tool implementations + structured tool feedback.
- `demo/`: a tiny project to run “deny → train → retry” end-to-end.

## How SDPO works (in this repo)

SDPO converts text feedback (“wrong file”, “read first”, “use edit_lines”) into a dense training signal.

Per denied attempt:
1. **Student**: the policy that sampled the tool‑call tokens.
2. **Self‑teacher**: the same model, conditioned on your correction text.
3. **Dense credit assignment**: evaluate the **same sampled tokens** under both student and teacher (no re‑sampling).
4. **Update**: distill teacher → student by minimizing per‑token KL along the sampled rollout:

   `KL(student_next_token || stopgrad(teacher_next_token))`

Equivalently (the view used in code), we compute token advantages from the logprob gap:
`adv[t] = -kl_coef * (student_lp[t] - teacher_lp[t])`,
then train with `loss_fn="importance_sampling"` using the student sampling logprobs as the behavior policy.

Learning speed is empirical (model/task dependent), so don’t assume a fixed number of corrections.

## Demo

We include a tiny project in `demo/` to show online updates:

```bash
cd demo/
export TINKER_API_KEY=<your-key>
python -m continualcode
```

See `demo/README.md` for the walkthrough.

## Documentation

This repo is intentionally small. For the underlying training/sampling APIs, see the Tinker docs:
https://tinker-docs.thinkingmachines.ai/training-sampling

## Configuration

```bash
continualcode --help
```

`continualcode` uses `chz` config‑style arguments (pass `key=value`).

Examples:

```bash
# inference only
continualcode enable_training=false

# change base model + LoRA rank
continualcode model_name=Qwen/Qwen3-4B-Instruct-2507 lora_rank=64

# enable checkpointing every 10 SDPO/RL steps (writes to /tmp by default)
continualcode save_every=10
```

Common pitfalls:
- **One tool call per message**: SDPO credit assignment assumes one tool call per assistant message.
- **Teacher compatibility**: a separate teacher must be tokenizer‑compatible with the student (it scores the student’s token IDs). If unsure, prefer a teacher checkpoint from the same base model.

## Citation

```bibtex
@misc{continualcode,
  title = {continualcode: online SDPO for coding agents},
  url = {https://github.com/sdan/continualcode},
}
```
