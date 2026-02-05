"""
CLI entry point for continualcode.

Validates TINKER_API_KEY before anything heavy imports,
then hands off to the TUI.
"""

from __future__ import annotations

import argparse
import os
import sys


def _check_tinker_api_key() -> str | None:
    """Return the API key or None if missing."""
    return os.environ.get("TINKER_API_KEY")


def main() -> None:
    p = argparse.ArgumentParser(
        prog="continualcode",
        description="Self-improving coding agent with online SDPO",
    )
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--teacher-model", default=None)
    p.add_argument("--teacher-checkpoint", default=None)
    p.add_argument("--tinker-url", default=os.environ.get("TINKER_URL"))
    p.add_argument("--tinker-api-key", default=None, help="Tinker API key (or set TINKER_API_KEY env var)")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--no-training", action="store_true", help="Disable training (inference only)")
    p.add_argument("--no-sdpo", action="store_true", help="Disable SDPO")
    p.add_argument("--enable-rl", action="store_true", help="Enable RL training")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--kl-coef", type=float, default=1.0)
    p.add_argument("--is-clip", type=float, default=2.0)
    p.add_argument("--auto-approve-readonly", action="store_true")
    args = p.parse_args()

    # Resolve API key: CLI flag > env var
    if args.tinker_api_key:
        os.environ["TINKER_API_KEY"] = args.tinker_api_key

    api_key = _check_tinker_api_key()
    if not api_key:
        print(
            "\033[31mError: TINKER_API_KEY not set.\033[0m\n"
            "\n"
            "continualcode requires a Tinker API key to run.\n"
            "\n"
            "  export TINKER_API_KEY=<your-key>\n"
            "  continualcode\n"
            "\n"
            "Or pass it directly:\n"
            "\n"
            "  continualcode --tinker-api-key <your-key>\n"
            "\n"
            "Get a key at https://tinker.dev",
            file=sys.stderr,
        )
        sys.exit(1)

    # Only import heavy modules after validation passes
    import asyncio
    from .tui import run

    asyncio.run(run(args))
