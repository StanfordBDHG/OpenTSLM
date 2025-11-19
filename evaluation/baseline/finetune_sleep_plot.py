#!/usr/bin/env python3
#
# Minimal: load SleepEDF train split and plot first sample
#

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io
 
# Ensure project src/ is on sys.path so we can import time_series_datasets
PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

"""Also add project root so we can import sibling modules when running script directly."""
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset

# Prefer local import when running from evaluation/baseline, fall back to package path
try:
    from common_finetune_sft import run_sft  # when cwd is this folder or script path is used
except ModuleNotFoundError:
    from evaluation.baseline.common_finetune_sft import run_sft


def plot_time_series(time_series, out_path=None):
    """
    Minimal version of evaluate_sleep_plot.generate_time_series_plot:
    Accepts a list/array of 1D series or a 2D array (rows = channels).
    """
    if time_series is None:
        print("No time_series provided")
        return

    # Normalize to list of 1D arrays
    if isinstance(time_series, np.ndarray):
        if time_series.ndim == 1:
            ts_list = [time_series]
        elif time_series.ndim == 2:
            ts_list = [time_series[i] for i in range(time_series.shape[0])]
        else:
            raise ValueError(f"Unsupported ndarray shape: {time_series.shape}")
    else:
        ts_list = list(time_series)
        if len(ts_list) > 0 and not hasattr(ts_list[0], "__len__"):
            ts_list = [ts_list]

    axis_names = {0: "EEG", 1: "EOG", 2: "EMG"}
    n = len(ts_list)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, s in enumerate(ts_list):
        s = np.asarray(s)
        axes[i].plot(s, marker="o", linestyle="-", markersize=0)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(axis_names.get(i, f"Axis {i+1}"))

    plt.tight_layout()
    plt.show()


def _time_series_to_pil(time_series) -> Image.Image:
    """Render a 1D or 2D time series array to a PIL RGB image (no disk I/O)."""
    # Normalize to list of 1D arrays
    if isinstance(time_series, np.ndarray):
        if time_series.ndim == 1:
            ts_list = [time_series]
        elif time_series.ndim == 2:
            ts_list = [time_series[i] for i in range(time_series.shape[0])]
        else:
            raise ValueError(f"Unsupported ndarray shape: {time_series.shape}")
    else:
        ts_list = list(time_series)
        if len(ts_list) > 0 and not hasattr(ts_list[0], "__len__"):
            ts_list = [ts_list]

    n = len(ts_list)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), dpi=100, sharex=True)
    if n == 1:
        axes = [axes]
    for i, s in enumerate(ts_list):
        s = np.asarray(s)
        axes[i].plot(s, marker="o", linestyle="-", markersize=0)
        axes[i].axis("off")
    plt.tight_layout(pad=0.1)

    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img

def _build_messages_from_sample(sample: dict) -> dict:
    """Build chat-style messages using pre/post prompts and an image of the time series.

    We intentionally do NOT include the raw time series as text. The assistant content
    is the provided sample["answer"].
    """
    pre = (sample.get("pre_prompt") or "").strip()
    post = (sample.get("post_prompt") or "").strip()
    ans = (sample.get("answer") or "").strip()

    # Prefer original_data field injected by SleepEDFCoTQADataset
    ts = sample.get("original_data", sample.get("time_series", None))
    img = _time_series_to_pil(ts)

    user_text = "\n\n".join([p for p in [pre, post] if p])
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful medical AI that analyzes sleep EEG."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": img},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": ans}],
        },
    ]
    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(
        description="Plot the first SleepEDF train sample (default) or run Gemma SFT with --sft."
    )
    # SFT options
    parser.add_argument("--output-dir", type=str, default="runs/gemma3-4b-pt-sleep-lora")
    parser.add_argument("--llm-id", type=str, default="google/gemma-3-4b-pt")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=4096)

    args = parser.parse_args()

    # Build training chat examples with images from SleepEDF train split
    ds = SleepEDFCoTQADataset(split="train", EOS_TOKEN="")
    n = len(ds) if args.max_samples is None else min(args.max_samples, len(ds))
    train_examples = [_build_messages_from_sample(ds[i]) for i in range(n)]

    # print(_build_messages_from_sample(ds[0]))
    run_sft(
        train_examples,
        output_dir=args.output_dir,
        llm_id=args.llm_id,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_len=args.max_seq_len,
    )

    # SleepEDFCoTQADataset._format_sample adds 'original_data'
    # time_series = sample.get("original_data", None)
    # if time_series is None:
    #     # Fallback: try the raw dataset field if present
    #     time_series = sample.get("time_series", None)

    # plot_time_series(time_series)


if __name__ == "__main__":
    main()