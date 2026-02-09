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

tool_edit_lines = tools.tool_edit_lines


class EditLinesAppendTests(unittest.TestCase):
    def test_edit_lines_can_append_at_eof(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("a\nb\n", encoding="utf-8")

            result = tool_edit_lines(
                {
                    "path": str(path),
                    "start_line": 3,
                    "end_line": 3,
                    "content": "c\n",
                }
            )

            self.assertTrue(result.success)
            self.assertEqual(path.read_text(encoding="utf-8"), "a\nb\nc\n")

    def test_edit_lines_rejects_partial_beyond_eof_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("a\nb\n", encoding="utf-8")

            result = tool_edit_lines(
                {
                    "path": str(path),
                    "start_line": 3,
                    "end_line": 4,
                    "content": "c\n",
                }
            )

            self.assertFalse(result.success)
            self.assertIn("end_line beyond end of file", result.output)


if __name__ == "__main__":
    unittest.main()
