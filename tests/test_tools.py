import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS_PATH = Path(__file__).resolve().parents[1] / "continualcode" / "tools.py"
SPEC = importlib.util.spec_from_file_location("continualcode_tools", TOOLS_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Failed to load tools module from {TOOLS_PATH}")
tools = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tools
SPEC.loader.exec_module(tools)

execute_tool = tools.execute_tool
tool_edit = tools.tool_edit
tool_edit_lines = tools.tool_edit_lines
tool_glob = tools.tool_glob
tool_grep = tools.tool_grep
tool_read = tools.tool_read


class ToolsTests(unittest.TestCase):
    def test_edit_strips_line_number_prefixes_when_needed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("line one\nline two\n", encoding="utf-8")

            result = tool_edit(
                {
                    "path": str(path),
                    "old": "2| line two",
                    "new": "2| line changed",
                }
            )

            self.assertTrue(result.success)
            self.assertEqual(path.read_text(encoding="utf-8"), "line one\nline changed\n")

    def test_edit_rejects_ambiguous_replace_without_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("dup\nkeep\ndup\n", encoding="utf-8")

            result = tool_edit({"path": str(path), "old": "dup", "new": "new"})

            self.assertFalse(result.success)
            self.assertIn("must be unique", result.output)
            self.assertIn("Ambiguous edit", result.feedback or "")

    def test_edit_lines_rejects_invalid_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("a\nb\n", encoding="utf-8")

            result = tool_edit_lines(
                {
                    "path": str(path),
                    "start_line": 3,
                    "end_line": 2,
                    "content": "x\n",
                }
            )

            self.assertFalse(result.success)
            self.assertIn("invalid line range", result.output)

    def test_edit_lines_rejects_start_line_beyond_eof(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("a\nb\n", encoding="utf-8")

            result = tool_edit_lines(
                {
                    "path": str(path),
                    "start_line": 5,
                    "end_line": 5,
                    "content": "x\n",
                }
            )

            self.assertFalse(result.success)
            self.assertIn("start_line beyond end of file", result.output)

    def test_edit_lines_rejects_end_line_beyond_eof(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("a\nb\n", encoding="utf-8")

            result = tool_edit_lines(
                {
                    "path": str(path),
                    "start_line": 2,
                    "end_line": 5,
                    "content": "x\n",
                }
            )

            self.assertFalse(result.success)
            self.assertIn("end_line beyond end of file", result.output)

    def test_grep_rejects_invalid_regex(self):
        result = tool_grep({"pat": "("})
        self.assertFalse(result.success)
        self.assertIn("invalid regex pattern", result.output)

    def test_glob_rejects_absolute_path(self):
        result = tool_glob({"pat": "**/*.py", "path": "/tmp"})
        self.assertFalse(result.success)
        self.assertIn("absolute path not allowed", result.output)

    def test_read_returns_missing_file_error(self):
        result = tool_read({"path": "does-not-exist.txt"})
        self.assertFalse(result.success)
        self.assertIn("file not found", result.output)

    def test_execute_tool_rejects_unknown_tool(self):
        result = execute_tool("not_a_tool", {})
        self.assertFalse(result.success)
        self.assertIn("unknown tool", result.output)


if __name__ == "__main__":
    unittest.main()
