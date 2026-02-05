# continualcode

Self-improving coding agent with online SDPO (Self-Distillation Policy Optimization).

The agent learns from your corrections in real-time: when you deny a tool call and explain why, the model trains on that feedback immediately — no offline dataset curation needed.

## Install

```bash
pip install continualcode
```

## Quick start

```bash
export TINKER_API_KEY=<your-key>
continualcode
```

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen3-4B-Instruct-2507` | Base model |
| `--checkpoint` | — | LoRA checkpoint path |
| `--teacher-model` | same as model | Teacher model for SDPO |
| `--teacher-checkpoint` | — | Teacher checkpoint |
| `--tinker-url` | — | Custom Tinker API URL |
| `--tinker-api-key` | `$TINKER_API_KEY` | API key |
| `--max-tokens` | 4096 | Max generation tokens |
| `--temperature` | 0.7 | Sampling temperature |
| `--no-training` | false | Inference only |
| `--no-sdpo` | false | Disable SDPO |
| `--enable-rl` | false | Enable RL training |
| `--lora-rank` | 32 | LoRA rank |
| `--learning-rate` | 1e-5 | Learning rate |
| `--kl-coef` | 1.0 | KL penalty coefficient |
| `--is-clip` | 2.0 | IS ratio clipping |
| `--auto-approve-readonly` | false | Auto-approve read/glob/grep |

## TUI controls

- **y**: Approve tool call
- **n**: Deny (requires correction text)
- **e**: Edit tool args in `$EDITOR`
- **/c**: Clear conversation
- **/metrics**: Show training metrics
- **/q**: Quit

## Python API

```python
from continualcode import SDPOConfig, SDPOSession

session = SDPOSession(
    model="Qwen/Qwen3-4B-Instruct-2507",
    enable_training=True,
    sdpo_config=SDPOConfig(kl_coef=1.0),
)
await session.init()

# Sample completions
message, ok, completion = await session.sample()

# Record denials and train
session.record_sdpo_denial(completion, "wrong file")
metrics = await session.train_sdpo()
```

## How SDPO works

```
User: "fix the test"
Model: diff_1 -> User: "no, wrong file - bug is in foo.py"

SDPO Training (per denied attempt):
- Student context = the exact prompt snapshot that produced diff_1
- Teacher context = that same snapshot + feedback (and an instruction like "solve correctly")
- Evaluate the SAME sampled diff_1 tokens under both student and teacher (dense, token-level signal)
- Distill teacher → student by minimizing per-token KL:
  KL(student_next_token || stopgrad(teacher_next_token))
  which is equivalent to an importance-sampling style loss with advantages like:
  adv[t] = -kl_coef * (student_lp[t] - teacher_lp[t])
```

The model learns to anticipate corrections without needing them at inference time.
