"""Interview-style calculator demo with one subtle bug for debugging demos."""

from __future__ import annotations

import argparse


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a - b


def multiply(a: float, b: float) -> float:
    return a * b


def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("division by zero")
    return a / b


def calculate(op: str, a: float, b: float) -> float:
    if op == "add":
        return add(a, b)
    if op == "subtract":
        return subtract(a, b)
    if op == "multiply":
        return multiply(a, b)
    if op == "divide":
        return divide(a, b)
    raise ValueError(f"unknown operation: {op}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple calculator")
    parser.add_argument("--op", choices=["add", "subtract", "multiply", "divide"], required=True)
    parser.add_argument("--a", required=True, help="first operand")
    parser.add_argument("--b", required=True, help="second operand")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Intentional bug for demo: integer conversion loses decimal precision.
    a = float(args.a)
    b = float(args.b)
    try:
        result = calculate(args.op, a, b)
    except ValueError as err:
        print(f"error: {err}")
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
