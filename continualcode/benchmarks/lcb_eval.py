#!/usr/bin/env python3
"""
LiveCodeBench-style evaluation using the DeepCoder+LCB mix from tinker_cookbook.

Runs pass@1 (default) and optionally pass@k by sampling multiple completions
per problem and checking correctness with the same sandboxed runner as the
code_rl recipe.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import chz
import tinker

from tinker_cookbook import model_info
from tinker_cookbook.renderers import Renderer, get_renderer, get_text_content
from tinker_cookbook.recipes.code_rl.code_grading import (
    extract_code_from_model,
    sandbox_check_correctness,
    taco_to_lcb_format,
)
from tinker_cookbook.recipes.code_rl.lcb_utils import fetch_live_code_bench_system_prompt
from tinker_cookbook.sandbox import SandboxBackend
from tinker_cookbook.tokenizer_utils import get_tokenizer

try:
    from datasets import Dataset, concatenate_datasets, load_dataset  # type: ignore
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "Missing dependency: datasets. Install with `pip install datasets`."
    ) from exc


def _load_deepcoder_split(split: Literal["train", "test"]) -> Dataset:
    if split == "train":
        datasets = [
            load_dataset("agentica-org/DeepCoder-Preview-Dataset", name=name, split="train")
            for name in ("primeintellect", "taco", "lcbv5")
        ]
    else:
        datasets = [
            load_dataset("agentica-org/DeepCoder-Preview-Dataset", name=name, split="test")
            for name in ("codeforces", "lcbv5")
        ]
    return concatenate_datasets(datasets)


def _ensure_dict(metadata: Any) -> dict[str, Any]:
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
    if isinstance(metadata, dict):
        return metadata
    return {}


def _normalize_tests(raw_tests: Any, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    tests = raw_tests
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except json.JSONDecodeError:
            return []
    if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
        tests = taco_to_lcb_format(tests)
    if isinstance(tests, dict):
        tests = [tests]

    normalized: list[dict[str, Any]] = []
    for test in tests or []:
        if not isinstance(test, dict):
            continue
        testtype = test.get("testtype") or "stdin_stdout"
        test_metadata = _ensure_dict(test.get("metadata", {}))
        if testtype == "functional":
            func_name = test_metadata.get("func_name") or metadata.get("func_name")
            if func_name is not None:
                test_metadata["func_name"] = str(func_name)
        normalized.append(
            {
                "input": str(test.get("input", "")),
                "output": str(test.get("output", "")),
                "testtype": testtype,
                "metadata": test_metadata or {"func_name": None},
            }
        )
    return normalized


def _build_question(example: dict[str, Any]) -> str | None:
    question = example.get("question") or example.get("prompt") or example.get("problem")
    if not isinstance(question, str) or not question.strip():
        return None
    starter_code = example.get("starter_code")
    if isinstance(starter_code, str) and starter_code.strip():
        return fetch_live_code_bench_system_prompt(question, starter_code)
    return fetch_live_code_bench_system_prompt(question)


@dataclass
class EvalResult:
    idx: int
    pass1: bool
    passk: bool
    tokens: int
    latency_s: float
    skipped: bool = False
    error: str | None = None


@chz.chz
class CLIConfig:
    model_path: str | None = None
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    split: Literal["train", "test"] = "test"
    max_samples: int = 100
    num_samples: int = 1
    max_tokens: int = 1024
    temperature: float = 0.7
    seed: int = 42
    sandbox_backend: SandboxBackend = SandboxBackend.SANDBOXFUSION
    concurrency: int = 16
    log_path: str | None = None


async def _evaluate_one(
    *,
    idx: int,
    example: dict[str, Any],
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    num_samples: int,
    max_tokens: int,
    temperature: float,
    sandbox_backend: SandboxBackend,
) -> EvalResult:
    metadata = _ensure_dict(example.get("metadata", {}))
    tests = _normalize_tests(example.get("tests") or example.get("ground_truth"), metadata)
    question = _build_question(example)
    if not tests or question is None:
        return EvalResult(idx=idx, pass1=False, passk=False, tokens=0, latency_s=0.0, skipped=True)

    prompt_messages = [{"role": "user", "content": question}]
    prompt = renderer.build_generation_prompt(prompt_messages)

    start = time.perf_counter()
    response = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=num_samples,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),
        ),
    )
    latency_s = time.perf_counter() - start

    sequences = response.sequences
    total_tokens = sum(len(seq.tokens) for seq in sequences)

    pass1 = False
    passk = False

    for i, seq in enumerate(sequences):
        message, parse_ok = renderer.parse_response(seq.tokens)
        content = get_text_content(message)
        if not parse_ok:
            continue
        code = extract_code_from_model(content)
        if code is None:
            continue
        ok, _details = await sandbox_check_correctness(
            tests, code, backend=sandbox_backend
        )
        if i == 0:
            pass1 = ok
        if ok:
            passk = True
            if num_samples == 1:
                break

    return EvalResult(
        idx=idx, pass1=pass1, passk=passk, tokens=total_tokens, latency_s=latency_s
    )


async def main(config: CLIConfig) -> None:
    if config.sandbox_backend == SandboxBackend.SANDBOXFUSION:
        if not os.environ.get("SANDBOX_URL"):
            raise RuntimeError(
                "SANDBOX_URL is not set. Start SandboxFusion and export SANDBOX_URL, "
                "e.g. http://localhost:8080/run_code"
            )
    elif config.sandbox_backend == SandboxBackend.MODAL:
        try:
            import modal  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "Modal backend selected but `modal` is not installed. "
                "Install with `pip install modal` and run `modal token new`."
            ) from exc

    dataset = _load_deepcoder_split(config.split)
    dataset = dataset.shuffle(seed=config.seed)
    if config.max_samples > 0:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))

    renderer_name = model_info.get_recommended_renderer_name(config.base_model)
    tokenizer = get_tokenizer(config.base_model)
    renderer = get_renderer(renderer_name, tokenizer=tokenizer)

    service_client = tinker.ServiceClient()
    if config.model_path:
        sampling_client = service_client.create_sampling_client(
            model_path=config.model_path, base_model=config.base_model
        )
    else:
        sampling_client = service_client.create_sampling_client(base_model=config.base_model)

    semaphore = asyncio.Semaphore(config.concurrency)
    results: list[EvalResult] = []

    async def _run_one(i: int, ex: dict[str, Any]) -> None:
        async with semaphore:
            try:
                res = await _evaluate_one(
                    idx=i,
                    example=ex,
                    sampling_client=sampling_client,
                    renderer=renderer,
                    num_samples=config.num_samples,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    sandbox_backend=config.sandbox_backend,
                )
            except Exception as exc:  # pragma: no cover - robust eval
                res = EvalResult(
                    idx=i,
                    pass1=False,
                    passk=False,
                    tokens=0,
                    latency_s=0.0,
                    skipped=True,
                    error=str(exc),
                )
            results.append(res)

    tasks = [asyncio.create_task(_run_one(i, ex)) for i, ex in enumerate(dataset)]
    await asyncio.gather(*tasks)

    results = sorted(results, key=lambda r: r.idx)
    total = len(results)
    skipped = sum(1 for r in results if r.skipped)
    valid = total - skipped
    pass1 = sum(1 for r in results if r.pass1)
    passk = sum(1 for r in results if r.passk)
    avg_tokens = sum(r.tokens for r in results) / max(1, valid)
    avg_latency = sum(r.latency_s for r in results) / max(1, valid)

    if config.log_path:
        with open(config.log_path, "w") as f:
            for r in results:
                f.write(
                    json.dumps(
                        {
                            "idx": r.idx,
                            "pass1": r.pass1,
                            "passk": r.passk,
                            "tokens": r.tokens,
                            "latency_s": r.latency_s,
                            "skipped": r.skipped,
                            "error": r.error,
                        }
                    )
                    + "\n"
                )

    print("\n" + "=" * 72)
    print("LCB-STYLE EVAL (DeepCoder+LCB mix)")
    print("=" * 72)
    print(f"model_path: {config.model_path or '(base model)'}")
    print(f"base_model: {config.base_model}")
    print(f"sandbox_backend: {config.sandbox_backend}")
    print(f"split: {config.split}")
    print(f"samples: {total} (skipped={skipped})")
    print(f"pass@1: {pass1 / max(1, valid):.3f} ({pass1}/{max(1, valid)})")
    if config.num_samples > 1:
        print(f"pass@{config.num_samples}: {passk / max(1, valid):.3f} ({passk}/{max(1, valid)})")
    print(f"avg_tokens: {avg_tokens:.1f}")
    print(f"avg_latency_s: {avg_latency:.2f}")
    if config.log_path:
        print(f"log_path: {config.log_path}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    cfg = chz.entrypoint(CLIConfig)
    asyncio.run(main(cfg))
