# SDPO Demo (90-Second Video Runbook)

This demo is designed for a short, edited recording aimed at ML engineers. It
shows:
1. Scoped agent workflow (`demo/` only).
2. One minimal root-cause patch.
3. One correction/denial that becomes online SDPO training signal.

Interview scenario in this folder:
- The CLI currently has a subtle decimal-precision bug (integer coercion of operands).
- A focused test catches it and creates a clean root-cause -> minimal-fix narrative.

## Setup

```bash
cd /Users/sdan/Developer/continualcode
export TINKER_API_KEY=<your-key>
python -m continualcode
```

## Shot Plan (90 seconds)

### 0:00 - 0:10 Hook

Show:
- Terminal
- `demo/main.py` and `demo/test_main.py` briefly in editor

Voice/text:
- "Can an agent apply a minimal root-cause patch, not just generate code?"

### 0:10 - 0:20 Constrain scope

Paste this prompt:

```text
Work only in /Users/sdan/Developer/continualcode/demo.
Find the root cause of the failing demo behavior.
Apply the smallest viable patch only.
Run the most targeted verification command first.
Then summarize root cause, file changed, and why minimal.
```

### 0:20 - 0:35 Intentional denial (learning signal)

If the model proposes a broad `write` or risky edit, deny once:

```text
Read first, then use edit_lines. Do not overwrite full file.
```

Approve after it proposes a minimal `edit_lines` patch.

### 0:35 - 0:55 Patch execution + targeted verification

Ensure it runs a targeted check first, typically:

```bash
python -m pytest demo/test_main.py -q
```

Optional only if needed: broader demo checks after targeted pass.

### 0:55 - 1:15 Proof and metrics

Run:

```text
/metrics
```

Highlight only 1-2 metrics:
- `loss`
- `sdpo_kl`

### 1:15 - 1:30 Close

If needed, ask:

```text
Summarize in 3 bullets: root cause, exact patch, verification results.
```

Final overlay text:
- Scoped prompts
- Minimal patch
- Verify first
- Correction becomes training signal

## Acceptance Checklist

- Agent explicitly constrained to `/Users/sdan/Developer/continualcode/demo`.
- One clear root-cause diagnosis appears on-screen.
- Exactly one minimal patch is applied (prefer single-file change).
- Targeted verification passes on-screen.
- One correction/denial moment is captured.
- `/metrics` appears briefly with `loss` and `sdpo_kl`.
- Final summary ties behavior to learning signal.

## Fallbacks

If model asks for context again:

```text
Focus on failing tests in /demo only. Diagnose and patch minimally.
```

If no failing tests exist:
- Reframe task to implement one TODO minimally and verify with one focused test.

If tool output parse is messy:
- Re-prompt once, keep recording, trim dead time in edit.
