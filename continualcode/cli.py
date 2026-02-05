"""
CLI entry point for continualcode.

`continualcode` uses `chz` for config parsing (pass `key=value`).
See `continualcode --help` for available parameters.
"""

from __future__ import annotations

import os
import sys

import chz


@chz.chz
class Config:
    """Top-level configuration for continualcode."""

    # Model
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    load_checkpoint_path: str | None = None
    teacher_model: str | None = None
    teacher_checkpoint: str | None = None
    base_url: str | None = None

    # Generation
    max_tokens: int = 4096
    temperature: float = 0.7

    # Training
    learning_rate: float | None = None  # None = auto from hyperparam_utils
    lora_rank: int = 32
    kl_coef: float = 1.0
    is_clip: float = 2.0

    # Feature toggles
    enable_training: bool = True
    enable_sdpo: bool = True
    auto_approve_readonly: bool = False


def main(config: Config) -> None:
    # Require TINKER_API_KEY
    if not os.environ.get("TINKER_API_KEY"):
        print(
            "\033[31mError: TINKER_API_KEY not set.\033[0m\n"
            "\n"
            "continualcode requires a Tinker API key to run.\n"
            "\n"
            "  export TINKER_API_KEY=<your-key>\n"
            "  continualcode\n"
            "\n"
            "Or pass it directly as an env var:\n"
            "\n"
            "  TINKER_API_KEY=<your-key> continualcode\n"
            "\n"
            "Get a key at https://tinker-console.thinkingmachines.ai",
            file=sys.stderr,
        )
        sys.exit(1)

    from .tui import SDPOCodeApp

    app = SDPOCodeApp(config)
    app.run()


def _entrypoint() -> None:
    """Wrapper for pyproject.toml [project.scripts] — invokes chz CLI parsing."""
    chz.nested_entrypoint(main)
