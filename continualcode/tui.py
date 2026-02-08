#!/usr/bin/env python3
"""
continualcode TUI (terminal UI).

This is an approval-gated coding agent:
- The model proposes at most one tool call per message.
- You approve or deny the tool call.
- Denials require a short correction/reason.
- On each correction, SDPO runs immediately (on-policy) and the agent retries.

This file is intentionally “operator-facing”: it prints compact diffs and prompts for
high-signal feedback.
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

from .train import SDPOConfig, ContinualSDPOSession
from .tools import READONLY_TOOLS, ToolResult, execute_tool


# --- ANSI ---

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, CYAN, GREEN, YELLOW, RED = "\033[34m", "\033[36m", "\033[32m", "\033[33m", "\033[31m"
USE_ASSIST_BLOCK = True
BG_ASSIST = "\033[48;2;38;38;38m"
FG_MUTED = "\033[38;2;210;210;210m"

CORRECTION_CHIPS: dict[str, str] = {
    "f": "Wrong file — use read/glob/grep to locate the correct file first.",
    "r": "Read the file first before editing.",
    "e": "Use edit_lines instead of edit.",
    "t": "Tool error — fix args and retry.",
}


# --- Helpers ---

def _summarize_tool_args(tool_name: str, tool_args: dict[str, Any]) -> str:
    if tool_name == "bash":
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


def _summarize_arg_edit(old_args: dict[str, Any], new_args: dict[str, Any]) -> str:
    changed_keys = sorted(
        key for key in (set(old_args.keys()) | set(new_args.keys()))
        if old_args.get(key) != new_args.get(key)
    )
    preview_new_values = {key: new_args.get(key) for key in changed_keys[:5]}
    return json.dumps(
        {"changed_keys": changed_keys, "new_values": preview_new_values},
        ensure_ascii=False,
        sort_keys=True,
    )


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


def _print_system(msg: str) -> None:
    print(f"{DIM}{msg}{RESET}")


def _terminal_width(default: int = 80) -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default


def _supports_ansi() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _clear_line() -> None:
    """Erase current line and return cursor to start."""
    if _supports_ansi():
        print("\033[2K\r", end="", flush=True)


def _bg(code: str) -> str:
    if not _supports_ansi():
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
        # Erase entire line first, then rewrite — prevents scrolling on wide terminals
        print(f"\033[2K\r{DIM}{label} {line}{RESET}", end="", flush=True)
        await asyncio.sleep(0.04)
        pos += direction
        if pos >= width - bar_len:
            direction = -1
            pos = width - bar_len
        elif pos <= 0:
            direction = 1
            pos = 0
    _clear_line()


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


def _print_tool_result(success: bool, preview: str) -> None:
    icon = f"{GREEN}✓{RESET}" if success else f"{RED}✗{RESET}"
    print(f"  {icon} {DIM}{preview}{RESET}")


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


# --- Main loop ---

class ContinualCodeApp:
    """Wrapper that accepts CLI args and runs the async main loop."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def run(self) -> None:
        asyncio.run(self._main())

    async def _main(self) -> None:
        cfg = self.config
        model = cfg.model_name
        auto_approve_readonly = cfg.auto_approve_readonly

        print(f"\n{BOLD}continualcode{RESET} {DIM}| {model}{RESET}")
        print(f"{DIM}{os.getcwd()}{RESET}\n")
        last_sdpo_metrics: dict[str, Any] | None = None

        print(f"{DIM}initializing...{RESET}", end="", flush=True)

        sdpo_config = SDPOConfig(kl_coef=cfg.kl_coef)
        session = ContinualSDPOSession(
            model=model,
            checkpoint=cfg.load_checkpoint_path,
            teacher_model=cfg.teacher_model,
            teacher_checkpoint=cfg.teacher_checkpoint,
            tinker_url=cfg.base_url,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            enable_training=cfg.enable_training,
            lora_rank=cfg.lora_rank,
            learning_rate=cfg.learning_rate,
            sdpo_config=sdpo_config,
        )
        await session.init()

        _clear_line()
        print(f"{DIM}ready{RESET}\n")

        last_tool_feedback: str | None = None
        session_allowed_tools: set[str] = set()

        while True:
            try:
                user_input = input(f"{BLUE}❯{RESET} ").strip()
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
                        for k in sorted(last_sdpo_metrics.keys()):
                            print(f"  {k}: {last_sdpo_metrics[k]}")
                    else:
                        _print_system("No training metrics yet.")
                    continue

                session.add_user_message(user_input)

                while True:
                    print(f"{DIM}thinking...{RESET}", end="", flush=True)
                    message, ok, completion = await session.sample()
                    _clear_line()
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

                    # Diff preview for edit tools
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

                    # Approval
                    approved = False
                    edited_args = None
                    correction: str | None = None
                    edit_correction: str | None = None
                    edit_solution: str | None = None
                    original_model_args = json.loads(json.dumps(model_args))

                    if auto_approve_readonly and name in READONLY_TOOLS:
                        approved = True
                    elif name in session_allowed_tools:
                        approved = True
                    else:
                        while True:
                            print(f"{DIM}Approve `{name}`?{RESET}")
                            print("  1. Yes")
                            print(f"  2. Yes, allow `{name}` during this session")
                            print("  3. No")
                            print("  4. Edit args")
                            decision = input(f"{DIM}Select 1-4{RESET}: ").strip()
                            if decision == "1":
                                approved = True
                                break
                            if decision == "2":
                                approved = True
                                session_allowed_tools.add(name)
                                _print_system(f"Auto-approving `{name}` for this session.")
                                break
                            if decision == "3":
                                correction = _prompt_correction(
                                    prompt=f"{YELLOW}Reason for denying (required){RESET}: ",
                                    suggested=last_tool_feedback,
                                )
                                last_tool_feedback = None
                                break
                            if decision == "4":
                                edited = _edit_args_in_editor(model_args)
                                if edited:
                                    edited_args = edited
                                    model_args = edited
                                    _print_system("args updated")
                                continue
                            _print_system("Invalid choice. Select 1, 2, 3, or 4.")

                    final_args = edited_args if (approved and edited_args) else model_args
                    if approved and edited_args is not None and original_model_args != final_args:
                        edit_summary = _summarize_arg_edit(original_model_args, final_args)
                        edit_correction = (
                            "Use these tool arguments instead of the previous proposal: "
                            f"{edit_summary}"
                        )
                        edit_solution = (
                            "Approved tool call:\n"
                            f"{name}({json.dumps(final_args, ensure_ascii=False, sort_keys=True)})"
                        )

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
                            preview += f" +{len(lines)-1} lines"
                        _print_tool_result(tool_success, preview)
                        if tool_success and tool_output and name in ("read", "glob", "grep", "bash"):
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
                        completion is not None
                        and session.training_client is not None
                        and (correction or failure_correction or edit_correction)
                    ):
                        if correction:
                            session.record_denial(completion, correction)
                        if failure_correction:
                            session.record_denial(
                                completion,
                                failure_correction,
                                environment_feedback=tool_feedback or tool_output,
                            )
                        if edit_correction:
                            session.record_denial(
                                completion,
                                edit_correction,
                                solution=edit_solution,
                            )
                        stop = asyncio.Event()
                        shimmer_task = asyncio.create_task(_shimmer("training", stop))
                        metrics = await session.train_sdpo()
                        stop.set()
                        await shimmer_task
                        if metrics:
                            last_sdpo_metrics = metrics
                            print(
                                f"{CYAN}trained{RESET} {DIM}#{int(metrics.get('sdpo_step',0))} "
                                f"L={metrics.get('loss',0.0):.3f} "
                                f"kl={metrics.get('sdpo_kl',0.0):.4f} "
                                f"t={int(metrics.get('sdpo_tokens',0))}{RESET}"
                            )
                    print()

            except KeyboardInterrupt:
                print()
                continue
            except EOFError:
                break
            except Exception as err:
                print(f"{RED}Error: {err}{RESET}")
