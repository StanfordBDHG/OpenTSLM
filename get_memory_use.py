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
from time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate


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
    # Ensure time series tensors are padded and converted
    batch = extend_time_series_to_match_patch_size_and_aggregate(batch)
    return batch


def one_iter_train(model, model_type: str, batch: List[Dict[str, any]]) -> Tuple[float, int]:
    model.train()
    # Parameter selection and grouping similar to curriculum_learning.py
    base_lr = 2e-4
    optimizer = None

    if model_type == "EmbedHealthSP":
        # SP: optimize encoder and projector only (LLM is frozen by class init)
        enc_params = [p for p in getattr(model, "encoder").parameters() if p.requires_grad]
        proj_params = [p for p in getattr(model, "projector").parameters() if p.requires_grad]
        param_groups = []
        if len(enc_params) > 0:
            param_groups.append({"params": enc_params, "weight_decay": 0.1})
        if len(proj_params) > 0:
            param_groups.append({"params": proj_params, "weight_decay": 0.1})
        if len(param_groups) > 0:
            optimizer = torch.optim.AdamW(param_groups, lr=base_lr)
    else:
        # Flamingo: filter trainable params and split by gated_cross_attn for WD
        named_params = list(model.named_parameters())
        trainable = list(
            filter(
                lambda np: np[1].requires_grad and not getattr(np[1], "exclude_from_optimizer", False),
                named_params,
            )
        )

        params_with_wd, params_without_wd = [], []
        for name, p in trainable:
            if "gated_cross_attn" in name:
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)

        if len(params_with_wd) + len(params_without_wd) > 0:
            optimizer = torch.optim.AdamW(
                [
                    {"params": params_with_wd, "weight_decay": 0.1},
                    {"params": params_without_wd, "weight_decay": 0.0},
                ],
                lr=base_lr,
            )

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
        loss, peak = one_iter_train(model, model_name, batch)
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
    parser.add_argument("--device", default="cuda", help="Device to run on (e.g., cuda, cuda:0, cpu)")
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
        model = EmbedHealthFlamingo(device=device, llm_id=args.llm_id, cross_attn_every_n_layers=1,
                gradient_checkpointing=True)
        eos = model.get_eos_token()
    elif args.model == "EmbedHealthSP":
        model = EmbedHealthSP(llm_id=args.llm_id, device=device)
        eos = model.get_eos_token()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Make absolutely sure parameters are on the requested device
    model.to(device)

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


