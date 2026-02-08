#!/usr/bin/env python3
"""
Core tool implementations and schemas shared by tinkercode harnesses.

Tool names intentionally match the "Claude Code"-style set:
read, write, edit, edit_lines, glob, grep, bash
"""

from __future__ import annotations

import glob as globlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    # Only required by the Tinker renderer; keep optional for simple imports.
    from tinker_cookbook.renderers import ToolSpec
except Exception:  # pragma: no cover
    ToolSpec = dict  # type: ignore[misc,assignment]


@dataclass
class ToolResult:
    """Structured tool result with feedback for rich textual feedback."""
    output: str
    success: bool
    feedback: str | None = None

    def __str__(self) -> str:
        return self.output


def _error_result(message: str, feedback: str | None = None) -> ToolResult:
    return ToolResult(output=message, success=False, feedback=feedback or message)


def tool_read(args: dict[str, Any]) -> ToolResult:
    path = args["path"]
    if not os.path.isfile(path):
        return _error_result(
            f"error: file not found: {path}",
            feedback=f"File not found: {path}\nCheck the path or use glob to locate the file.",
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

    offset = max(0, int(args.get("offset", 0) or 0))
    limit = args.get("limit")
    limit_int = len(lines) if limit is None else max(0, int(limit))
    selected = lines[offset : offset + limit_int]
    output = "".join(f"{offset + idx + 1:4}| {line}" for idx, line in enumerate(selected))
    return ToolResult(output=output, success=True)


def tool_write(args: dict[str, Any]) -> ToolResult:
    path = args["path"]
    content = args["content"]
    try:
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResult(output="ok", success=True)
    except PermissionError:
        return _error_result(
            f"error: permission denied: {path}",
            feedback=f"Permission denied writing to {path}\nCheck file permissions or choose a different path.",
        )
    except Exception as e:
        return _error_result(f"error: {e}", feedback=f"Write failed: {e}")


def tool_edit(args: dict[str, Any]) -> ToolResult:
    path = args["path"]
    if not os.path.isfile(path):
        return _error_result(
            f"error: file not found: {path}",
            feedback=f"File not found: {path}\nUse 'write' to create new files, or check the path.",
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    old_raw = args["old"]
    new_raw = args["new"]
    replace_all = bool(args.get("all", False))

    def _strip_numbered_prefixes(s: str) -> tuple[str, bool]:
        """Remove common line-number prefixes that models copy from the UI."""
        changed = False
        out_lines: list[str] = []
        for line in s.splitlines():
            m = re.match(r"^\s*\d+\s*(\||[-+])\s?", line)
            if m:
                changed = True
                out_lines.append(line[m.end() :])
            else:
                out_lines.append(line)
        return ("\n".join(out_lines), changed)

    old, old_was_numbered = _strip_numbered_prefixes(old_raw)
    new, _new_was_numbered = _strip_numbered_prefixes(new_raw)

    old_candidates: list[tuple[str, bool]] = [(old_raw, False)]
    if old_was_numbered:
        old_candidates.append((old, True))

    chosen_old: str | None = None
    chosen_old_was_stripped = False
    for cand, stripped in old_candidates:
        if cand and cand in text:
            chosen_old = cand
            chosen_old_was_stripped = stripped
            break

    if chosen_old is None:
        hint = "error: old_string not found."
        probe_source, _ = _strip_numbered_prefixes(old_raw)
        probe_lines = [ln.strip() for ln in probe_source.splitlines() if ln.strip()]

        feedback_lines = [
            "old_string not found in file.",
            "",
            "The 'old' field must match the file contents exactly (whitespace included).",
            "Use 'read' first to see the exact file contents.",
            "If the edit spans lines, prefer edit_lines.",
        ]

        if probe_lines:
            probe = probe_lines[0]
            file_lines = text.splitlines()
            hits: list[int] = []
            for i, line in enumerate(file_lines, 1):
                if probe in line:
                    hits.append(i)
                    if len(hits) >= 3:
                        break
            if hits:
                hint += f" First matching line fragment appears near: {', '.join(map(str, hits))}."
                feedback_lines.append(f"\nPartial match found near lines: {', '.join(map(str, hits))}")
                feedback_lines.append("Try using 'edit_lines' with these line numbers instead.")

        return _error_result(hint, feedback="\n".join(feedback_lines))

    replacement_new = new if chosen_old_was_stripped else new_raw

    count = text.count(chosen_old)
    if not replace_all and count > 1:
        return _error_result(
            f"error: old_string appears {count} times, must be unique (use all=true)",
            feedback=f"Ambiguous edit: '{old_raw[:50]}...' appears {count} times.\nAdd more context or use edit_lines.",
        )

    replacement = (
        text.replace(chosen_old, replacement_new)
        if replace_all
        else text.replace(chosen_old, replacement_new, 1)
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(replacement)
    return ToolResult(output="ok", success=True)


def tool_edit_lines(args: dict[str, Any]) -> ToolResult:
    """Edit a file by replacing an inclusive line range (1-based)."""
    path = args["path"]
    if not os.path.isfile(path):
        return _error_result(
            f"error: file not found: {path}",
            feedback=f"File not found: {path}\nUse 'write' to create new files, or check the path.",
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

    start_line = int(args["start_line"])
    end_line = int(args["end_line"])
    total_lines = len(lines)

    if start_line < 1 or end_line < 1 or end_line < start_line:
        return _error_result(
            "error: invalid line range",
            feedback=(
                f"Invalid line range: {start_line}-{end_line}\n"
                "start_line and end_line must be >= 1, and end_line >= start_line."
            ),
        )
    if start_line > total_lines + 1:
        return _error_result(
            "error: start_line beyond end of file",
            feedback=(
                f"start_line {start_line} is beyond file end (file has {total_lines} lines).\n"
                "Use 'read' to check file contents first."
            ),
        )
    if end_line > total_lines:
        return _error_result(
            "error: end_line beyond end of file",
            feedback=(
                f"end_line {end_line} is beyond file end (file has {total_lines} lines).\n"
                "Use 'read' to check file contents first."
            ),
        )

    new_content = str(args.get("content", ""))
    new_lines = new_content.splitlines(keepends=True)
    if new_lines and not new_lines[-1].endswith("\n") and (start_line <= total_lines):
        new_lines[-1] = new_lines[-1] + "\n"

    before = lines[: start_line - 1]
    after = lines[end_line:]
    out_lines = before + new_lines + after

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    return ToolResult(output="ok", success=True)


def tool_glob(args: dict[str, Any]) -> ToolResult:
    base = args.get("path", ".") or "."
    pattern = (str(base) + "/" + args["pat"]).replace("//", "/")
    files = globlib.glob(pattern, recursive=True)
    files = sorted(
        files,
        key=lambda p: os.path.getmtime(p) if os.path.isfile(p) else 0,
        reverse=True,
    )
    return ToolResult(output="\n".join(files[:50]) or "none", success=True)


def tool_grep(args: dict[str, Any]) -> ToolResult:
    try:
        pattern = re.compile(args["pat"])
    except re.error:
        return _error_result(
            "error: invalid regex pattern",
            feedback="Invalid regex pattern. Check escaping or simplify the pattern.",
        )

    base = args.get("path", ".") or "."
    hits: list[str] = []

    for filepath in globlib.glob(str(base) + "/**", recursive=True):
        if not os.path.isfile(filepath):
            continue
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if pattern.search(line):
                        hits.append(f"{filepath}:{line_num}:{line.rstrip()}")
                        if len(hits) >= 50:
                            return ToolResult(output="\n".join(hits), success=True)
        except Exception:
            continue

    return ToolResult(output="\n".join(hits) or "none", success=True)


def tool_bash(args: dict[str, Any]) -> ToolResult:
    cmd = args["cmd"]
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        output = (result.stdout or "") + (result.stderr or "")
        output = output.strip() or "(empty output)"
        if result.returncode == 0:
            return ToolResult(output=output, success=True)
        return _error_result(
            output,
            feedback=f"Command failed (exit {result.returncode}). Fix the command and retry.",
        )
    except subprocess.TimeoutExpired:
        return _error_result(
            "error: command timed out after 30s",
            feedback="Command timed out after 30s. Simplify the command or increase timeout.",
        )
    except Exception as e:
        return _error_result(f"error: {e}", feedback=f"Command failed: {e}")

TOOL_FUNCTIONS: dict[str, Any] = {
    "read": tool_read,
    "write": tool_write,
    "edit": tool_edit,
    "edit_lines": tool_edit_lines,
    "glob": tool_glob,
    "grep": tool_grep,
    "bash": tool_bash,
}


def execute_tool(name: str, args: dict[str, Any]) -> ToolResult:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return _error_result(
            f"error: unknown tool '{name}'",
            feedback=f"Unknown tool '{name}'. Use one of: {', '.join(TOOL_FUNCTIONS.keys())}.",
        )
    try:
        return fn(args)
    except Exception as e:
        return _error_result(f"error: {e}", feedback=f"Tool crashed: {e}")


TOOL_SPECS: list[ToolSpec] = [
    {
        "name": "read",
        "description": "Read a file and return its contents with line numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
                "offset": {"type": "integer", "description": "Line number to start from (0-indexed)"},
                "limit": {"type": "integer", "description": "Maximum number of lines to read"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write",
        "description": "Write content to a file, creating it if it doesn't exist",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit",
        "description": "Edit a file by replacing old with new. old must appear once unless all=true.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit"},
                "old": {"type": "string", "description": "The exact string to find and replace"},
                "new": {"type": "string", "description": "The string to replace it with"},
                "all": {"type": "boolean", "description": "Replace all occurrences (default false)"},
            },
            "required": ["path", "old", "new"],
        },
    },
    {
        "name": "edit_lines",
        "description": "Edit a file by replacing an inclusive line range (1-based) with new content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit"},
                "start_line": {"type": "integer", "description": "First line to replace (1-based, inclusive)"},
                "end_line": {"type": "integer", "description": "Last line to replace (1-based, inclusive)"},
                "content": {"type": "string", "description": "Replacement text for that line range"},
            },
            "required": ["path", "start_line", "end_line", "content"],
        },
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern",
        "parameters": {
            "type": "object",
            "properties": {
                "pat": {"type": "string", "description": "Glob pattern (e.g., '**/*.py')"},
                "path": {"type": "string", "description": "Base path to search from"},
            },
            "required": ["pat"],
        },
    },
    {
        "name": "grep",
        "description": "Search for a regex pattern in files",
        "parameters": {
            "type": "object",
            "properties": {
                "pat": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Base path to search from"},
            },
            "required": ["pat"],
        },
    },
    {
        "name": "bash",
        "description": "Run a shell command",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "The shell command to run"},
            },
            "required": ["cmd"],
        },
    },
]


READONLY_TOOLS = {"read", "glob", "grep"}
