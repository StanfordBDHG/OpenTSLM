#!/usr/bin/env python3
"""
evaluate_accuracy.py

Compute exact‑match accuracy for the JSONL file written by *train_updated.py*.
Each line of the file must be a JSON object with keys:
    {"prompt": ..., "generated": ..., "gold": ...}

Usage examples
--------------
# plain exact‑match (case‑sensitive, keeps EOS)
python evaluate_accuracy.py test_predictions.jsonl

# ignore case and strip a trailing EOS token before comparison
python evaluate_accuracy.py --file test_predictions.jsonl --ignore-case --strip-eos --eos-token "</s>"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _normalise(text: str, *, ignore_case: bool = False, strip_eos: bool = False, eos: str = "</s>") -> str:
    """Trim whitespace, optionally remove EOS, optionally lowercase."""
    text = text.strip()
    if strip_eos and text.endswith(eos):
        text = text[: -len(eos)].rstrip()
    return text.lower() if ignore_case else text


def compute_accuracy(
    file_path: Path | str,
    *,
    ignore_case: bool = False,
    strip_eos: bool = False,
    eos_token: str = "</s>",
) -> tuple[int, int, float]:
    """Return (#correct, #total, accuracy)."""
    correct = total = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            gen  = _normalise(obj["generated"], ignore_case=ignore_case, strip_eos=strip_eos, eos=eos_token)
            gold = _normalise(obj["gold"],      ignore_case=ignore_case, strip_eos=strip_eos, eos=eos_token)
            
            gen = gen.strip()
            gold = gold.strip()
    
            total += 1
            if gold.startswith(gen):
                correct += 1
            else:
                print(gen, gold)
           
    accuracy = correct / total if total else 0.0
    return correct, total, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact‑match accuracy for TSQA predictions.")
    parser.add_argument("--file", type=Path, default="test_predictions.jsonl", help="Path to JSONL predictions.")
    parser.add_argument("--ignore-case", action="store_true", help="Ignore case when comparing strings.")
    parser.add_argument("--strip-eos", action="store_true", help="Strip trailing EOS token before comparison.")
    parser.add_argument("--eos-token", type=str, default="</s>", help="EOS token to strip when --strip-eos is set.")
    args = parser.parse_args()

    if not args.file.exists():
        sys.exit(f"❌ File '{args.file}' does not exist.")

    correct, total, acc = compute_accuracy(
        args.file,
        ignore_case=args.ignore_case,
        strip_eos=args.strip_eos,
        eos_token=args.eos_token,
    )
    print(f"Exact‑match accuracy: {correct} / {total} = {acc * 100:.2f}%")


if __name__ == "__main__":
    main()