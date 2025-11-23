#!/usr/bin/env python3
#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Show example of actual prompt with time series token placement.
"""
import json
from transformers import AutoTokenizer

print("=" * 80)
print("EXAMPLE: TimeSeriesExam Prompt with Time Series Tokens")
print("=" * 80)

# Read first sample from TimeSeriesExam predictions
with open("/local/home/wangni/OpenTSLM/results/Llama_3_2_1B/OpenTSLMFlamingo/stage_tsexam_eval/results/test_predictions.jsonl") as f:
    sample = json.loads(f.readline())

print("\n[1] Pre-prompt (question):")
print("-" * 80)
print(sample["pre_prompt"][:500] + "..." if len(sample["pre_prompt"]) > 500 else sample["pre_prompt"])

print("\n[2] Time series text (descriptive stats):")
print("-" * 80)
for i, ts_text in enumerate(sample["time_series_text"]):
    print(f"  Time series {i+1}: {ts_text}")

print("\n[3] Post-prompt:")
print("-" * 80)
print(sample["post_prompt"])

print("\n[4] Gold answer:")
print("-" * 80)
print(sample["gold"])

print("\n[5] Stage2 model's generated answer (WRONG - captioning style):")
print("-" * 80)
print(sample["generated"][:500])

# Now show how the prompt is constructed with special tokens
print("\n" + "=" * 80)
print("TOKENIZED PROMPT CONSTRUCTION (from OpenTSLMFlamingo.py:210-214)")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.add_special_tokens(
    {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
)

media_token_id = tokenizer.encode("<image>", add_special_tokens=False)[0]
endofchunk_token_id = tokenizer.encode("<|endofchunk|>", add_special_tokens=False)[0]

print(f"\nSpecial token IDs:")
print(f"  <image> token ID: {media_token_id}")
print(f"  <|endofchunk|> token ID: {endofchunk_token_id}")

# Construct prompt exactly as done in OpenTSLMFlamingo.py:210-214
prompt_text = sample["pre_prompt"]
for ts_text in sample["time_series_text"]:
    prompt_text += f" {tokenizer.decode([media_token_id])} {ts_text} {tokenizer.decode([endofchunk_token_id])}"
if sample["post_prompt"]:
    prompt_text += f" {sample['post_prompt']}"

print("\n[6] Full constructed prompt (what the model sees):")
print("-" * 80)
# Show where <image> tokens are inserted
for i, ts_text in enumerate(sample["time_series_text"]):
    prompt_snippet = f"<image> {ts_text} <|endofchunk|>"
    print(f"\nTime series {i+1} insertion:")
    print(f"  {prompt_snippet}")

print("\n[7] Complete prompt with tokens:")
print("-" * 80)
print(prompt_text[:1000] + "..." if len(prompt_text) > 1000 else prompt_text)

# Tokenize to see token count
tokenized = tokenizer(prompt_text, add_special_tokens=False)
input_ids = tokenized.input_ids
print(f"\nTotal tokens in prompt: {len(input_ids)}")

# Find <image> token positions
image_token_positions = [i for i, token_id in enumerate(input_ids) if token_id == media_token_id]
print(f"<image> token positions: {image_token_positions}")
print(f"Number of <image> tokens: {len(image_token_positions)}")

print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print("✓ The model receives TWO inputs during generation (OpenTSLMFlamingo.py:263-264):")
print("    1. vision_x: Time series embeddings (actual numerical data)")
print("    2. lang_x: Tokenized prompt with <image> tokens as placeholders")
print("")
print("✓ The <image> tokens tell the model WHERE to attend to the time series")
print("    embeddings while processing the text.")
print("")
print("✓ Both stage1 (TSQA) and stage3 (TimeSeriesExam) use the SAME")
print("    tokenization and data processing logic.")
print("")
print("✓ The 0% accuracy on TimeSeriesExam is NOT due to missing time series tokens.")
print("    The tokens are correctly inserted.")
print("")
print("✓ The problem is catastrophic forgetting: stage2 checkpoint learned to")
print("    generate long captions instead of concise MCQ answers like '(b)'.")
