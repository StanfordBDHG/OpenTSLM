#!/usr/bin/env python3
"""
evaluate_accuracy.py

Compute exact‑match (or *starts‑with*) accuracy for prediction files written in **either**

* JSON‑Lines format (one JSON object per line)  
* a single JSON **array** of objects

Each object must contain keys
    {"prompt": ..., "generated": ..., "gold": ...}

Changes vs. the original script
-------------------------------
* Gracefully handles files that are a JSON **list** instead of JSONL (fixes the
  "extra data" decode error).
* Skips blank lines and prints a short warning instead of crashing on a bad
  line.
* `starts‑with` comparison is now an **opt‑in** flag (`--startswith`).  Default
  remains strict equality.

Usage examples
--------------
# exact match, case‑sensitive
python evaluate_accuracy.py test_predictions.jsonl

# generated string only needs to be a *prefix* of gold, ignore case, strip EOS
python evaluate_accuracy.py \
       --file results.jsonl --startswith --ignore-case --strip-eos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Dict


# -----------------------------------------------------------------------------
# Normalisation helpers
# -----------------------------------------------------------------------------

def _normalise(
    text: str,
    *,
    ignore_case: bool = False,
    strip_eos: bool = False,
    eos: str = "</s>",
) -> str:
    """Trim whitespace, optionally remove EOS, optionally lowercase."""
    text = text.strip()
    if strip_eos and text.endswith(eos):
        text = text[: -len(eos)].rstrip()
    return text.lower() if ignore_case else text


# -----------------------------------------------------------------------------
# Robust input reader
# -----------------------------------------------------------------------------

def _iter_records(file_path: Path) -> Iterable[Dict[str, str]]:
    """Yield JSON objects from *either* JSONL or a single JSON array."""
    with open(file_path, "r", encoding="utf-8") as f:
        # peek first non‑whitespace char to decide format
        while True:
            ch = f.read(1)
            if not ch:  # empty file
                return
            if not ch.isspace():
                first_char = ch
                break
        f.seek(0)

        if first_char == "[":
            # ---------- JSON array ----------
            try:
                data: List[Dict[str, str]] = json.load(f)
                for obj in data:
                    yield obj
            except json.JSONDecodeError as e:
                sys.exit(f"❌ Failed to parse JSON array: {e}")
        else:
            # ---------- JSON Lines ----------
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # skip blank

                # Fast path: try normal json.loads
                try:
                    yield json.loads(line)
                    continue
                except json.JSONDecodeError as e:
                    # Some writers concatenate objects on a single line:
                    # {...}{...}{...}
                    # or they forget to put line breaks between records.
                    # We'll attempt to decode sequentially.
                    decoder = json.JSONDecoder()
                    idx = 0
                    raw = line
                    recovered = False
                    while idx < len(raw):
                        try:
                            obj, next_idx = decoder.raw_decode(raw, idx)
                            recovered = True
                            yield obj
                            idx = next_idx
                            # skip any whitespace between objects
                            while idx < len(raw) and raw[idx].isspace():
                                idx += 1
                        except json.JSONDecodeError:
                            break  # give up on remainder
                    if not recovered:
                        print(
                            f"⚠️  Skipping line {lineno}: {e}",
                            file=sys.stderr,
                        )
# -----------------------------------------------------------------------------
# Accuracy computation
# -----------------------------------------------------------------------------

def compute_accuracy(
    file_path: Path,
    *,
    startswith: bool = False,
    ignore_case: bool = False,
    strip_eos: bool = False,
    eos_token: str = "</s>",
) -> tuple[int, int, float]:
    """Return (#correct, #total, acc)."""
    correct = total = 0
    for obj in _iter_records(file_path):
        gen  = _normalise(obj["generated"], ignore_case=ignore_case, strip_eos=strip_eos, eos=eos_token)
        gold = _normalise(obj["gold"],      ignore_case=ignore_case, strip_eos=strip_eos, eos=eos_token)

        total += 1
        if (gold.startswith(gen) if startswith else gen == gold):
            correct += 1
        else:
            print(gold, gen)
    acc = correct / total if total else 0.0
    return correct, total, acc


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Exact‑match or prefix accuracy for TSQA predictions.")
    p.add_argument("--file", type=Path, default="test_predictions.jsonl", help="Path to prediction file.")
    p.add_argument("--startswith", action="store_true", help="Count as correct when gold starts with generated.")
    p.add_argument("--ignore-case", action="store_true", help="Case‑insensitive comparison.")
    p.add_argument("--strip-eos", action="store_true", help="Strip trailing EOS token before comparison.")
    p.add_argument("--eos-token", type=str, default="</s>", help="EOS token to strip when --strip-eos is set.")
    args = p.parse_args()

    if not args.file.exists():
        sys.exit(f"❌ File '{args.file}' does not exist.")

    correct, total, acc = compute_accuracy(
        args.file,
        startswith=args.startswith,
        ignore_case=args.ignore_case,
        strip_eos=args.strip_eos,
        eos_token=args.eos_token,
    )
    print(f"Accuracy: {correct} / {total} = {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
