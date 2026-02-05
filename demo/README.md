# SDPO Demo

A tiny project to demonstrate continualcode learning from corrections. Results are empirical and depend on the model and the task.

## Setup

```bash
cd demo/
export TINKER_API_KEY=<your-key>
python -m continualcode
```

## Demo script (repeat corrections → observe learning)

### 1. Ask it to add divide

```
❯ add a divide function to main.py that handles zero division
```

It will probably try `write` (overwrite the whole file). **Deny it:**

```
[y]es [n]o [e]dit args: n
Reason: use edit_lines to add the function after multiply, don't overwrite the file
```

It trains, retries with `edit_lines`. **Approve.**

### 2. Ask it to add the CLI

```
❯ add a simple CLI to main.py using argparse
```

If it uses `write` again — deny with the same correction. If it uses `edit_lines`, that’s evidence the update helped.

### 3. Check metrics

```
❯ /metrics
```

You should see loss decreasing and kl > 0, meaning the model updated.

## What's happening

Each time you deny a tool call with a correction:
1. The **student** (model) already sampled tokens for the wrong action
2. The **teacher** (same model + your correction) re-scores those tokens
3. Where the teacher disagrees → per-token learning signal
4. One gradient step via `importance_sampling` loss

No reward model. No offline dataset. Just your corrections → dense supervision.
