#!/usr/bin/env python3
"""
tui.py — Terminal UI for continualcode.

Accepts an argparse.Namespace from cli.py instead of module-level globals.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
from typing import Any

# Avoid noisy warnings when forking subprocesses after tokenizers parallelism is initialized.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .train import SDPOConfig, SDPOSession
from .tools import READONLY_TOOLS, ToolResult, execute_tool


RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, CYAN, GREEN, YELLOW, RED = "\033[34m", "\033[36m", "\033[32m", "\033[33m", "\033[31m"
USE_SEPARATOR = False
USE_ASSIST_BLOCK = True
NO_BG = False

# Subtle gray background (24-bit RGB) for the user input block.
BG_ASSIST = "\033[48;2;38;38;38m"
FG_MUTED = "\033[38;2;210;210;210m"

CORRECTION_CHIPS: dict[str, str] = {
    "f": "Wrong file — use read/glob/grep to locate the correct file first.",
    "r": "Read the file first before editing.",
    "e": "Use edit_lines instead of edit.",
    "t": "Tool error — fix args and retry.",
}


def _summarize_tool_args(tool_name: str, tool_args: dict[str, Any]) -> str:
    if tool_name in ("bash", "execute"):
        cmd = str(tool_args.get("cmd", "")).strip()
        return f"$ {cmd}".strip()
    if tool_name == "edit_lines":
        path = str(tool_args.get("path", "")).strip()
        start_line = tool_args.get("start_line")
        end_line = tool_args.get("end_line")
        return f"{path} @ {start_line}-{end_line}".strip()
    if "path" in tool_args:
        return str(tool_args.get("path", "")).strip()
    if tool_name in ("glob", "grep"):
        pat = str(tool_args.get("pat", "")).strip()
        base = str(tool_args.get("path", "")).strip()
        return f"{base} {pat}".strip()
    return ""


def _extract_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return str(content)


def _parse_tool_calls_from_text(text: str) -> tuple[list[dict[str, Any]], str]:
    tool_calls: list[dict[str, Any]] = []
    remaining = text
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    for match in pattern.finditer(text):
        try:
            payload = json.loads(match.group(1))
            if "function" in payload:
                tool_calls.append(payload)
            remaining = remaining.replace(match.group(0), "", 1)
        except json.JSONDecodeError:
            continue
    return tool_calls, remaining.strip()


def _separator() -> str:
    width = min(os.get_terminal_size().columns, 80)
    return f"{DIM}{'─' * width}{RESET}"


def _print_system(msg: str) -> None:
    print(f"{DIM}{msg}{RESET}")


def _terminal_width(default: int = 80) -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default


def _supports_ansi() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _bg(code: str) -> str:
    if NO_BG or not _supports_ansi():
        return ""
    return code


def _wrap_lines(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw in (text or "").splitlines() or [""]:
        if not raw.strip():
            lines.append("")
            continue
        lines.extend(
            textwrap.wrap(raw, width=width, break_long_words=False, break_on_hyphens=False) or [""]
        )
    return lines


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_len(s: str) -> int:
    return len(_ANSI_RE.sub("", s))


def _print_bg_block(*, text: str, prefix: str, bg_code: str, enabled: bool, muted: bool) -> None:
    """Print a padded, full-width message block."""
    if not enabled:
        _print_wrapped(prefix, text, indent=0)
        return

    width = max(40, _terminal_width(100))
    pad_left = 2
    usable = max(10, width - pad_left - _visible_len(prefix))

    bg = _bg(bg_code)
    fg = FG_MUTED if muted and _supports_ansi() else ""

    def emit(raw: str) -> None:
        if bg:
            visible = _visible_len(raw)
            trailing = " " * max(0, width - pad_left - visible)
            print(f"{bg}{' ' * pad_left}{fg}{raw}{trailing}{RESET}")
        else:
            if _supports_ansi():
                print(f"{' ' * pad_left}{fg}{raw}{RESET}")
            else:
                print(f"{' ' * pad_left}{raw}")

    for line in _wrap_lines(text, usable):
        emit(f"{prefix}{line}")


def _print_wrapped(prefix: str, text: str, *, indent: int = 2) -> None:
    width = max(40, _terminal_width(100))
    usable = max(10, width - indent - _visible_len(prefix))
    for i, line in enumerate(_wrap_lines(text, usable)):
        if i == 0:
            print(f"{' ' * indent}{prefix}{line}")
        else:
            print(f"{' ' * indent}{' ' * _visible_len(prefix)}{line}")


def _truncate_line(s: str, width: int) -> str:
    if len(s) <= width:
        return s
    if width <= 1:
        return s[:width]
    return s[: max(0, width - 1)] + "…"


def _print_change_block(*, old_content: str | None, new_content: str, start_line: int | None) -> None:
    """Diff-ish preview with line numbers and +/- coloring (no background fill)."""
    width = max(40, _terminal_width(100))
    pad_left = 4
    usable = max(10, width - pad_left - 2)
    line_no_width = 5 if start_line is not None else 0

    def emit(line_no: int | None, sign: str, color: str, text: str) -> None:
        gutter = f"{DIM}{line_no:>{line_no_width}} {RESET}" if line_no is not None else ""
        body = _truncate_line(text, usable - len(gutter) - 2)
        print(f"{' ' * pad_left}{gutter}{color}{sign} {body}{RESET}")

    print()
    if old_content is not None:
        for i, line in enumerate(str(old_content).splitlines(), 0):
            emit((start_line + i) if start_line is not None else None, "-", RED, line)
    for i, line in enumerate(str(new_content).splitlines(), 0):
        emit((start_line + i) if start_line is not None else None, "+", GREEN, line)
    print()


async def _shimmer(label: str, stop: asyncio.Event) -> None:
    """Animated shimmer to indicate training work in progress."""
    if not _supports_ansi():
        return
    width = max(40, _terminal_width() - len(label) - 4)
    pos = 0
    direction = 1
    bar_len = max(3, min(8, width // 8))
    while not stop.is_set():
        head = ">" if direction >= 0 else "<"
        dist_from_edge = min(pos, width - bar_len - pos)
        trail = max(1, min(bar_len, dist_from_edge + 1))
        bar = head * trail
        line = " " * pos + bar
        line = line.ljust(width)
        print(f"\r{DIM}{label} {line}{RESET}", end="", flush=True)
        await asyncio.sleep(0.04)
        pos += direction
        if pos >= width - bar_len:
            direction = -1
            pos = width - bar_len
        elif pos <= 0:
            direction = 1
            pos = 0
    # Clear line
    print("\r" + (" " * (width + len(label) + 2)) + "\r", end="", flush=True)


def _print_assistant(msg: str) -> None:
    _print_bg_block(
        text=msg,
        prefix="",
        bg_code=BG_ASSIST,
        enabled=USE_ASSIST_BLOCK,
        muted=True,
    )


def _print_tool(name: str, summary: str) -> None:
    print()
    if summary:
        print(f"{GREEN}●{RESET} {BOLD}{name}{RESET}({DIM}{summary}{RESET})")
        return
    print(f"{GREEN}●{RESET} {BOLD}{name}{RESET}")


def _print_tool_result(status: str, preview: str) -> None:
    status_color = GREEN if status == "ok" else RED if status == "FAILED" else DIM
    print(f"  {DIM}└─{RESET} {status_color}{status}{RESET}: {preview}")


def _print_tool_output(text: str, *, max_lines: int = 14) -> None:
    lines = (text or "").splitlines()
    if not lines:
        return
    shown = lines[:max_lines]
    for ln in shown:
        print(f"  {DIM}│{RESET} {ln}")
    if len(lines) > max_lines:
        more = len(lines) - max_lines
        print(f"  {DIM}└─{RESET} {DIM}... (+{more} more lines){RESET}")


def _prompt_correction(*, prompt: str, suggested: str | None) -> str:
    if suggested:
        _print_system("Suggested (Enter to use):")
        _print_wrapped("", suggested, indent=2)
    else:
        _print_system(
            "Keep it short & specific. Shortcuts: f wrong file | r read first | e use edit_lines | t tool error"
        )
    while True:
        raw = input(prompt).strip()
        if not raw and suggested:
            return suggested.strip()
        if raw in CORRECTION_CHIPS:
            return CORRECTION_CHIPS[raw]
        if raw:
            return raw


def _edit_args_in_editor(args_to_edit: dict[str, Any]) -> dict[str, Any] | None:
    editor_raw = os.environ.get("EDITOR") or "vim"
    editor_cmd = shlex.split(editor_raw)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(args_to_edit, f, indent=2, ensure_ascii=False)
            f.write("\n")
            tmp_path = f.name

        subprocess.run([*editor_cmd, tmp_path], check=False)
        with open(tmp_path, "r", encoding="utf-8") as f:
            edited = json.load(f)
        return edited if isinstance(edited, dict) else None
    except Exception:
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


async def run(args: Any) -> None:
    """Main TUI loop. Receives parsed argparse.Namespace from cli.py."""
    model = args.model
    enable_training = not args.no_training
    enable_sdpo = not args.no_sdpo
    enable_rl = args.enable_rl

    print(f"{BOLD}continualcode{RESET} {DIM}| {model} | {os.getcwd()}{RESET}")
    last_sdpo_metrics: dict[str, Any] | None = None
    last_rl_metrics: dict[str, Any] | None = None

    mode_parts: list[str] = []
    if enable_training:
        mode_parts.append("training")
    if enable_sdpo:
        mode_parts.append("sdpo")
    mode = "+".join(mode_parts) if mode_parts else "inference"
    _print_system(f"Initializing session ({mode})...")

    sdpo_config = SDPOConfig(kl_coef=args.kl_coef, is_clip=args.is_clip)
    session = SDPOSession(
        model=model,
        checkpoint=args.checkpoint,
        teacher_model=args.teacher_model,
        teacher_checkpoint=args.teacher_checkpoint,
        tinker_url=args.tinker_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_training=enable_training,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        sdpo_config=sdpo_config,
    )

    try:
        await session.init()
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "auth" in err_msg.lower() or "api" in err_msg.lower():
            print(
                f"\n{RED}Authentication failed.{RESET}\n"
                f"Check that your TINKER_API_KEY is valid.\n"
                f"\n  Error: {err_msg}\n",
                file=sys.stderr,
            )
        else:
            print(
                f"\n{RED}Failed to initialize session.{RESET}\n"
                f"\n  Error: {err_msg}\n",
                file=sys.stderr,
            )
        sys.exit(1)

    _print_system(f"Ready. Model: {model}")
    if enable_training:
        _print_system("Training enabled.")
    if enable_sdpo:
        _print_system("SDPO enabled.")
    if enable_training and not enable_rl:
        _print_system("RL disabled (sdpo-only).")

    last_tool_feedback: str | None = None

    while True:
        try:
            if USE_SEPARATOR:
                print(_separator())
            user_input = input(f"{BLUE}>{RESET} ").strip()
            if not user_input:
                continue
            if user_input in ("/q", "exit"):
                break
            if user_input == "/c":
                session.clear()
                _print_system("Cleared conversation.")
                continue
            if user_input == "/metrics":
                if last_sdpo_metrics:
                    print(f"{DIM}sdpo metrics:{RESET}")
                    for k in sorted(last_sdpo_metrics.keys()):
                        print(f"  {k}: {last_sdpo_metrics[k]}")
                if last_rl_metrics:
                    print(f"{DIM}rl metrics:{RESET}")
                    for k in sorted(last_rl_metrics.keys()):
                        print(f"  {k}: {last_rl_metrics[k]}")
                if not last_sdpo_metrics and not last_rl_metrics:
                    _print_system("No training metrics yet.")
                continue

            session.add_user_message(user_input)

            while True:
                _print_system("thinking...")
                message, ok, completion = await session.sample()
                if not ok:
                    _print_system("Parse failed.")
                    break

                text = _extract_text(message)
                tool_calls = message.get("tool_calls", [])
                if not tool_calls and "<tool_call>" in text:
                    parsed_calls, display_text = _parse_tool_calls_from_text(text)
                    if parsed_calls:
                        tool_calls = parsed_calls
                        text = display_text

                if text.strip():
                    _print_assistant(text)

                if not tool_calls:
                    session.add_assistant_message(message)
                    break

                if len(tool_calls) > 1:
                    _print_system("Model emitted multiple tool calls; retrying with ONE tool call.")
                    session.add_user_message("Retry: output exactly ONE tool call.")
                    continue

                session.add_assistant_message(message)
                tc = tool_calls[0]
                tool_call_id = ""
                try:
                    name = tc.function.name
                    model_args = json.loads(tc.function.arguments or "{}")
                    tool_call_id = getattr(tc, "id", "") or ""
                except Exception:
                    name = tc.get("function", {}).get("name", "tool")
                    model_args = tc.get("function", {}).get("arguments", {}) or {}
                    tool_call_id = tc.get("id", "")

                summary = _summarize_tool_args(name, model_args)
                _print_tool(name, summary)

                # Show compact diff preview for edit tools
                if name == "edit_lines":
                    path = model_args.get("path", "")
                    start_line = model_args.get("start_line", 1)
                    end_line = model_args.get("end_line", start_line)
                    new_content = model_args.get("content", "")
                    old_content = model_args.get("old_content")

                    if not old_content and path and os.path.isfile(path):
                        try:
                            with open(path, "r", encoding="utf-8", errors="replace") as f:
                                file_lines = f.readlines()
                            old_lines = file_lines[start_line - 1 : end_line]
                            old_content = "".join(old_lines).rstrip("\n")
                        except Exception:
                            old_content = None

                    _print_change_block(
                        old_content=old_content,
                        new_content=str(new_content),
                        start_line=int(start_line),
                    )
                elif name == "edit":
                    old_content = model_args.get("old", "")
                    new_content = model_args.get("new", "")
                    _print_change_block(
                        old_content=str(old_content),
                        new_content=str(new_content),
                        start_line=None,
                    )
                elif name == "write":
                    new_content = model_args.get("content", "")
                    _print_change_block(
                        old_content=None,
                        new_content=str(new_content),
                        start_line=1,
                    )

                approved = False
                edited_args = None
                correction: str | None = None

                if args.auto_approve_readonly and name in READONLY_TOOLS:
                    approved = True
                else:
                    while True:
                        decision = input(f"{DIM}[y]es [n]o [e]dit args{RESET}: ").strip().lower()
                        if decision == "y":
                            approved = True
                            break
                        if decision == "e":
                            edited = _edit_args_in_editor(model_args)
                            if edited:
                                edited_args = edited
                                model_args = edited
                                _print_system("args updated")
                            continue
                        if decision == "n":
                            correction = _prompt_correction(
                                prompt=f"{YELLOW}Reason for denying (required){RESET}: ",
                                suggested=last_tool_feedback,
                            )
                            last_tool_feedback = None
                            break

                final_args = edited_args if (approved and edited_args) else model_args

                tool_output = ""
                tool_success = True
                tool_feedback = None
                failure_correction: str | None = None

                if approved:
                    tool_result = execute_tool(name, final_args)
                    if isinstance(tool_result, ToolResult):
                        tool_output = tool_result.output
                        tool_success = tool_result.success
                        tool_feedback = tool_result.feedback
                    else:
                        tool_output = str(tool_result)
                        tool_success = not tool_output.lower().startswith("error:")
                        tool_feedback = tool_output if not tool_success else None

                    lines = tool_output.split("\n") if tool_output else []
                    preview = lines[0][:80] if lines else ""
                    if len(lines) > 1:
                        preview += f" ... +{len(lines)-1} lines"
                    status = "ok" if tool_success else "FAILED"
                    _print_tool_result(status, preview)
                    if tool_success and tool_output and name in ("read", "glob", "grep", "bash", "execute"):
                        stripped = tool_output.strip()
                        if stripped and stripped != "ok":
                            _print_tool_output(tool_output)

                    if not tool_success and tool_feedback:
                        last_tool_feedback = tool_feedback
                        failure_correction = _prompt_correction(
                            prompt=f"{YELLOW}Tool failed. Correction (required){RESET}: ",
                            suggested=tool_feedback,
                        )

                else:
                    tool_output = json.dumps(
                        {
                            "status": "denied",
                            "tool_name": name,
                            "tool_args": model_args,
                            "reason": correction,
                            "correction": correction,
                        },
                        ensure_ascii=False,
                    )

                session.add_tool_result(tool_call_id, tool_output)
                if failure_correction:
                    session.add_user_message(failure_correction)

                # SDPO training on correction
                if (
                    enable_sdpo
                    and completion is not None
                    and session.training_client is not None
                    and (correction or failure_correction)
                ):
                    _print_system("training(sdpo)...")
                    if correction:
                        session.record_sdpo_denial(completion, correction)
                    if failure_correction:
                        session.record_sdpo_denial(completion, failure_correction)
                    stop = asyncio.Event()
                    shimmer_task = asyncio.create_task(_shimmer("training(sdpo)", stop))
                    metrics = await session.train_sdpo()
                    stop.set()
                    await shimmer_task
                    print()
                    if metrics:
                        last_sdpo_metrics = metrics
                        print(
                            f"{GREEN}sdpo{RESET} step={int(metrics.get('sdpo_step',0))} "
                            f"denied={int(metrics.get('sdpo_denied_count',0))} "
                            f"tok={int(metrics.get('sdpo_tokens',0))} "
                            f"loss={CYAN}{metrics.get('loss',0.0):.3f}{RESET} "
                            f"kl={CYAN}{metrics.get('sdpo_kl',0.0):.4f}{RESET} "
                            f"r={CYAN}{metrics.get('sdpo_ratio_mean',1.0):.2f}{RESET}"
                        )

                # RL training (optional)
                if (
                    enable_rl
                    and completion is not None
                    and session.training_client is not None
                    and (approved or correction or failure_correction)
                ):
                    rewards = [1.0 if approved else 0.0]
                    _print_system("training(rl)...")
                    stop = asyncio.Event()
                    shimmer_task = asyncio.create_task(_shimmer("training(rl)", stop))
                    metrics = await session.train_on_episode([completion], rewards)
                    stop.set()
                    await shimmer_task
                    print()
                    if metrics:
                        last_rl_metrics = metrics
                        print(
                            f"{GREEN}rl{RESET} step={int(metrics.get('step',0))} "
                            f"R={CYAN}{metrics.get('reward_sum',0.0):.1f}{RESET} "
                            f"loss={CYAN}{metrics.get('loss:sum',0.0):.3f}{RESET} "
                            f"kl={CYAN}{metrics.get('approx_kl',0.0):.4f}{RESET} "
                            f"r={CYAN}{metrics.get('ratio_mean',1.0):.2f}{RESET}"
                        )
                print()

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as err:
            print(f"{RED}Error: {err}{RESET}")
