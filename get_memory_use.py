import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import torch

# Ensure src is on path
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.append(REPO_DIR)
SRC_DIR = os.path.join(REPO_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Models
from model.llm.EmbedHealthFlamingo import EmbedHealthFlamingo
from model.llm.EmbedHealthSP import EmbedHealthSP

# Datasets
from time_series_datasets.TSQADataset import TSQADataset
from time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
from time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
from time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset


def get_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def measure_peak_cuda_bytes() -> int:
    if not torch.cuda.is_available():
        return -1
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())



def get_first_batch(dataset, batch_size: int = 1) -> List[Dict[str, any]]:
    # QADataset returns dict samples compatible with model.compute_loss
    batch: List[Dict[str, any]] = []
    for i in range(min(batch_size, len(dataset))):
        batch.append(dataset[i])
    return batch


def one_iter_train(model, batch: List[Dict[str, any]]) -> Tuple[float, int]:
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    # Fallback to a tiny learning rate; we only do a single step
    optimizer = torch.optim.AdamW(params, lr=1e-4) if len(params) > 0 else None

    if optimizer:
        optimizer.zero_grad(set_to_none=True)

    loss = model.compute_loss(batch)
    # Backward only if there are trainable params
    if optimizer and loss.requires_grad:
        loss.backward()
        optimizer.step()

    peak_bytes = measure_peak_cuda_bytes()
    return float(loss.detach().item()), peak_bytes


def ensure_csv(path: str, header: List[str]):
    exists = os.path.exists(path)
    if not exists:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def append_row(path: str, row: List[any]):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def run_for_dataset(model_name: str, model, dataset_name: str, dataset_obj) -> Dict[str, any]:
    result: Dict[str, any] = {
        "model": model_name,
        "dataset": dataset_name,
        "loss": None,
        "peak_cuda_bytes": None,
        "status": "ok",
        "error": "",
    }
    try:
        batch = get_first_batch(dataset_obj, batch_size=1)
        loss, peak = one_iter_train(model, batch)
        result["loss"] = loss
        result["peak_cuda_bytes"] = peak
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    return result


def main():
    parser = argparse.ArgumentParser(description="Measure memory use for a single training iteration for a chosen model and dataset.")
    parser.add_argument("-llm_id", required=True, help="HuggingFace model id for the language model")
    parser.add_argument("--model", required=True, choices=["EmbedHealthFlamingo", "EmbedHealthSP"], help="Model to instantiate")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["TSQADataset", "HARCoTQADataset", "SleepEDFCoTQADataset", "ECGQACoTQADataset"],
        help="Dataset to use",
    )
    parser.add_argument("--device", default=None, help="Device to run on (e.g., cuda, cuda:0, cpu)")
    parser.add_argument("--results_csv", default=os.path.join(REPO_DIR, "memory_use.csv"), help="Path to CSV file to append results")
    args = parser.parse_args()

    device = get_device(args.device)

    # CSV header and file
    header = [
        "timestamp",
        "llm_id",
        "device",
        "model",
        "dataset",
        "loss",
        "peak_cuda_bytes",
        "status",
        "error",
    ]
    ensure_csv(args.results_csv, header)

    # Instantiate selected model
    if args.model == "EmbedHealthFlamingo":
        model = EmbedHealthFlamingo(device=device, llm_id=args.llm_id)
        eos = model.get_eos_token()
    elif args.model == "EmbedHealthSP":
        model = EmbedHealthSP(llm_id=args.llm_id, device=device, cross_attn_every_n_layers=1,
                gradient_checkpointing=True)
        eos = model.get_eos_token()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Instantiate selected dataset
    if args.dataset == "TSQADataset":
        dataset = TSQADataset(split="train", EOS_TOKEN=eos)
        dataset_name = "TSQA"
    elif args.dataset == "HARCoTQADataset":
        dataset = HARCoTQADataset(split="train", EOS_TOKEN=eos)
        dataset_name = "HAR-CoT"
    elif args.dataset == "SleepEDFCoTQADataset":
        dataset = SleepEDFCoTQADataset(split="train", EOS_TOKEN=eos)
        dataset_name = "SleepEDF-CoT"
    elif args.dataset == "ECGQACoTQADataset":
        dataset = ECGQACoTQADataset(split="train", EOS_TOKEN=eos, max_samples=1, preload_processed_data=False)
        dataset_name = "ECG-QA-CoT"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Run one iteration and append results
    res = run_for_dataset(args.model, model, dataset_name, dataset)
    append_row(
        args.results_csv,
        [
            datetime.utcnow().isoformat(),
            args.llm_id,
            device,
            res["model"],
            res["dataset"],
            res["loss"],
            res["peak_cuda_bytes"],
            res["status"],
            res["error"],
        ],
    )

    print(f"Done. Results appended to: {args.results_csv}")


if __name__ == "__main__":
    main()


