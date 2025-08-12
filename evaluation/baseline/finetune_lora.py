from dataclasses import dataclass
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Type

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from torch.utils.data import Dataset as Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset as HFDataset

from common_evaluator import CommonEvaluator

# Ensure project src is on path (this script is under evaluation/baseline/)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from time_series_datasets.TSQADataset import TSQADataset
from time_series_datasets.pamap2.PAMAP2AccQADataset import PAMAP2AccQADataset
from time_series_datasets.pamap2.PAMAP2CoTQADataset import PAMAP2CoTQADataset
#from time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset

# --------------------
# Configuration
# --------------------

MODEL_IDS: List[str] = [
    # "google/gemma-3n-e2b",
    # "google/gemma-3n-e2b-it",
    "meta-llama/Llama-3.2-1B",
]

DATASETS: List[Type[Dataset]] = [
    #TSQADataset,
    #PAMAP2AccQADataset,
    PAMAP2CoTQADataset,
    #SleepEDFCoTQADataset
]

# Training hyperparameters (defaults)
EPOCHS = 1
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRAD_ACC_STEPS = 8
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
TRAIN_SAMPLES_LIMIT = 0  # 0 or negative = no limit
EVAL_SAMPLES_LIMIT = 0  # number of test samples to evaluate; 0 => all

# LoRA config (defaults)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0  # 0.05

# Generation/eval settings
GEN_TEMPERATURE = 0.1
GEN_MAX_NEW_TOKENS = 50
EVAL_BATCH_SIZE = 2  # Batch size for evaluation to improve GPU efficiency


FORCE_RETRAIN = False  # Set to True to force retraining even if output exists

# --------------------
# Utilities
# --------------------


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_pad_token(tokenizer, model_id: str):
    if tokenizer.pad_token is None:
        if "llama" in model_id.lower():
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        elif "gemma" in model_id.lower():
            tokenizer.pad_token = "<pad>"
        else:
            tokenizer.pad_token = tokenizer.unk_token or "<pad>"
    return tokenizer


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "-", name.lower())


def prepare_train_dataset(
    common_evaluator: CommonEvaluator, tokenizer, dataset_class
) -> HFDataset:
    """
    Prepare your dataset for prompt-answer fine-tuning.
    Dataset format: [{"prompt": "question", "answer": "response"}, ...]
    """
    dataset_file = os.path.join(
        os.path.dirname(__file__),
        "datasets",
        f"hf_train_dataset_{dataset_class.__name__}",
    )
    if os.path.exists(dataset_file):
        print(f"Loading cached dataset from disk: {dataset_file}")
        tokenized_dataset = HFDataset.load_from_disk(dataset_file)
    else:
        # Load your data
        # Use same formatting approach as baseline

        train_ds = common_evaluator.load_dataset(
            dataset_class=dataset_class,
            split="train",
            format_sample_str=True,
        )

        items = list(train_ds)

        # Convert to Hugging Face dataset
        dataset = HFDataset.from_list(items)

        # compute max_tokens in training dataset
        def compute_length(batch):
            prompts = batch["prompt"]
            answers = batch["answer"]
            eos = tokenizer.eos_token
            full_texts = [p + "\n" + a + eos for p, a in zip(prompts, answers)]
            tokens = tokenizer(full_texts, add_special_tokens=False)["input_ids"]
            return {"token_length": list(map(len, tokens))}

        # Map across dataset
        token_lens = dataset.map(
            compute_length,
            batched=True,
            num_proc=2,
            batch_size=1000,
            remove_columns=dataset.column_names,
        )

        # Find max token length in
        max_tokens = max(token_lens["token_length"])
        print(f"Maximum token length in dataset: {max_tokens}")

        def tokenize_function(batch):
            prompts = batch["prompt"]
            answers = batch["answer"]

            # Build full texts (we append eos to answers to mark their end)
            eos = tokenizer.eos_token
            full_texts = [p + "\n" + a + eos for p, a in zip(prompts, answers)]

            # Tokenize full texts in one batched call -> uses fast tokenizer path
            full_enc = tokenizer(
                full_texts,
                padding="max_length",  # pad to max_length (consistent tensor sizes)
                truncation=True,
                max_length=max_tokens,
                add_special_tokens=False,
            )

            # Tokenize prompts alone (no padding) to compute prompt lengths
            prompt_enc = tokenizer(
                [p + "\n" for p in prompts],
                padding=False,
                truncation=True,
                max_length=max_tokens,
                add_special_tokens=False,
            )
            prompt_lengths = [len(x) for x in prompt_enc["input_ids"]]

            # Return the tokenized/padded arrays and prompt length
            return {
                "input_ids": full_enc["input_ids"],
                "attention_mask": full_enc["attention_mask"],
                "prompt_length": prompt_lengths,
            }

        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            num_proc=2,
            batched=True,
            remove_columns=dataset.column_names,
        )
        tokenized_dataset.save_to_disk(dataset_file)

    if TRAIN_SAMPLES_LIMIT > 0:
        tokenized_dataset = tokenized_dataset.take(TRAIN_SAMPLES_LIMIT)

    return tokenized_dataset


def build_test_dataset(
    common_evaluator: CommonEvaluator, dataset_class: Type[Dataset]
) -> HFDataset:
    test_ds = common_evaluator.load_dataset(
        dataset_class=dataset_class,
        split="test",
        format_sample_str=True,
    )

    items = list(test_ds)
    if EVAL_SAMPLES_LIMIT > 0:
        items = items[:EVAL_SAMPLES_LIMIT]

    pattern = (
        r"This is the time series, it has mean (-?\d+\.\d{4}) and std (-?\d+\.\d{4})\."
    )
    replacement = "This is the time series:"

    cleaned_items = []
    for idx, item in enumerate(items):
        cleaned_prompt = re.sub(pattern, replacement, item["prompt"])
        cleaned_items.append(
            {
                "sample_idx": idx,
                "input_text": cleaned_prompt,
                "target_answer": item["answer"],
            }
        )

    return HFDataset.from_list(cleaned_items)


def prepare_eval_dataset(test_items: List[Dict[str, str]]) -> HFDataset:
    """
    Prepare test items for batch evaluation by cleaning prompts and converting to HF Dataset.
    """
    # Clean up prompts for TSQADataset to match baseline
    pattern = (
        r"This is the time series, it has mean (-?\d+\.\d{4}) and std (-?\d+\.\d{4})\."
    )
    replacement = "This is the time series:"

    cleaned_items = []
    for idx, item in enumerate(test_items):
        cleaned_prompt = re.sub(pattern, replacement, item["prompt"])
        cleaned_items.append(
            {
                "sample_idx": idx,
                "input_text": cleaned_prompt,
                "target_answer": item["answer"],
            }
        )

    return HFDataset.from_list(cleaned_items)


def apply_lora(model: AutoModelForCausalLM) -> PeftModel:
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # https://huggingface.co/blog/mlabonne/sft-llama3,
        modules_to_save=None,
    )
    return get_peft_model(model, lora_cfg)


@dataclass
class SimpleCollator:
    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        features is a list of dicts with keys: 'input_ids' (list), 'attention_mask' (list), 'prompt_length' (int)
        They are already padded to the same length by prepare_dataset.
        """
        # Stack into tensors
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor(
            [f["attention_mask"] for f in features], dtype=torch.long
        )
        prompt_lengths = [f["prompt_length"] for f in features]

        # Build labels: copy input_ids and mask prompt tokens and pad tokens with -100
        labels = input_ids.clone()
        for i, prompt_length in enumerate(prompt_lengths):
            if prompt_length > 0:
                labels[i, :prompt_length] = self.label_pad_token_id

        # mask any pad tokens
        labels[labels == self.tokenizer.pad_token_id] = self.label_pad_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_lora_for_model_and_dataset(
    model_id: str, dataset_class: Type[Dataset], force_retrain: bool = False
) -> str:
    print(
        f"Starting LoRA fine-tuning for model {model_id} on dataset {dataset_class.__name__}"
    )

    # Prepare training output directory
    normalized_model_id = normalize_name(model_id)
    normalized_dataset_name = normalize_name(dataset_class.__name__)
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "models",
        f"lora_{normalized_model_id}_{normalized_dataset_name}",
    )
    if (
        not force_retrain
        and os.path.exists(output_dir)
        and glob.glob("*.safetensors", root_dir=output_dir)
    ):
        print(f"Output directory {output_dir} already exists. Skipping fine-tuning.")
        return output_dir

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    device = detect_device()
    print(f"Using device: {device}")

    # Load tokenizer and model
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer = ensure_pad_token(tokenizer, model_id)
    tokenizer.padding_side = "right"

    # Load model with 4-bit quantization for memory efficiency (optional)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA
    print("Applying LoRA adapters...")
    model = apply_lora(model)
    model.print_trainable_parameters()
    model.train()

    # Prepare datasets and collator
    print("Preparing training dataset...")
    common_evaluator = CommonEvaluator(device=device)
    common_evaluator.current_model_name = model_id
    train_dataset = prepare_train_dataset(common_evaluator, tokenizer, dataset_class)
    data_collator = SimpleCollator(tokenizer=tokenizer)

    # Training
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            optim="adamw_8bit"
            if device == "cuda"
            else None,  # Use 8-bit AdamW optimizer for memory efficiency
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            # weight_decay=WEIGHT_DECAY,
            # warmup_ratio=WARMUP_RATIO,
            logging_steps=10,
            fp16=(device == "cuda"),
            remove_unused_columns=False,
            # save_strategy="epoch",
            # dataloader_pin_memory=False,
        ),
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    print("Saving LoRA adapters...")
    trainer.model.save_pretrained(output_dir)  # type: ignore
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")

    return output_dir


def evaluate_finetuned_model(
    model_id: str, dataset_class: Type[Dataset], adapter_dir: str
):
    print(
        f"\nEvaluating fine-tuned model (adapters from {adapter_dir}) on dataset {dataset_class.__name__}"
    )
    print("=" * 80)

    device = detect_device()
    print(f"Using device: {device}")

    # Load base and attach adapters for inference
    print("Loading base model and attaching LoRA adapters...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    tokenizer = ensure_pad_token(tokenizer, model_id)
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    if (
        getattr(base_model.config, "pad_token_id", None) is None
        and tokenizer.pad_token_id is not None
    ):
        base_model.config.pad_token_id = tokenizer.pad_token_id
    ft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    # Build pipeline for generation (match baseline style)
    print("Building generation pipeline...")
    pipe = pipeline(
        task="text-generation",
        model=ft_model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
    )

    # Quick sanity check
    print("Quick generation sanity check...")
    try:
        out = pipe("The capital of France is", max_new_tokens=20)
        print(out)
    except Exception as e:
        print(f"Sanity check generation failed: {e}")
        raise e

    # Load test dataset
    print("Loading test dataset...")
    common_evaluator = CommonEvaluator(device=detect_device())
    common_evaluator.current_model_name = model_id
    test_dataset = build_test_dataset(common_evaluator, dataset_class)
    print(f"Loaded {len(test_dataset)} test samples")

    # Metrics
    total_samples = 0
    successful_inferences = 0
    results: List[Dict[str, str]] = []

    # Prepare dataset for batch processing

    # Process in batches
    print("\nRunning inference on test samples...")
    print("=" * 80)

    # Process all samples in batches
    all_outputs = pipe(
        KeyDataset(test_dataset, "input_text"),
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        return_full_text=False,
        batch_size=EVAL_BATCH_SIZE,
    )

    # Process results
    for idx, output in tqdm(
        enumerate(all_outputs), total=len(test_dataset), desc="Eval"
    ):
        try:
            sample = test_dataset[idx]
            if output and len(output) > 0:
                generated_text = output[0]["generated_text"].strip()
                successful_inferences += 1

                result = {
                    "sample_idx": sample["sample_idx"],
                    "input_text": sample["input_text"],
                    "target_answer": sample["target_answer"],
                    "generated_answer": generated_text,
                }
                results.append(result)

                if idx < 5:
                    print(f"\nSAMPLE {idx + 1}:")
                    print(f"PROMPT: {sample['input_text'][:1000]}...")
                    print(f"ANSWER: {sample['target_answer']}")
                    print(f"OUTPUT: {generated_text}")
                    print("=" * 80)

            total_samples += 1

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            raise e
            # continue

    if successful_inferences > 0:
        success_rate = successful_inferences / total_samples

        print("\n" + "=" * 80)
        print("FINE-TUNED MODEL EVALUATION RESULTS")
        print("=" * 80)
        print(f"Model: {model_id}")
        print(f"Total samples processed: {total_samples}")
        print(f"Successful inferences: {successful_inferences}")
        print(f"Success rate: {success_rate:.2%}")

        # Simple accuracy metrics (exact and partial match) to mirror baseline
        exact_matches = 0
        partial_matches = 0

        for result in results:
            target = result["target_answer"].lower().strip()
            generated = result["generated_answer"].lower().strip()

            if target == generated:
                exact_matches += 1
            elif target in generated or generated in target:
                partial_matches += 1

        exact_accuracy = exact_matches / successful_inferences
        partial_accuracy = (exact_matches + partial_matches) / successful_inferences

        print("\nAccuracy Metrics:")
        print(f"  Exact match accuracy: {exact_accuracy:.2%}")
        print(f"  Partial match accuracy: {partial_accuracy:.2%}")

        # Save detailed results
        normalized_model_id = normalize_name(model_id)
        normalized_dataset_name = normalize_name(dataset_class.__name__)
        normalized_adapter = normalize_name(os.path.basename(adapter_dir.rstrip("/")))
        results_file = f"baseline_test_results_{normalized_model_id}-{normalized_adapter}_{normalized_dataset_name}.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "model_name": model_id,
                    "adapter_dir": adapter_dir,
                    "total_samples": total_samples,
                    "successful_inferences": successful_inferences,
                    "success_rate": success_rate,
                    "exact_accuracy": exact_accuracy,
                    "partial_accuracy": partial_accuracy,
                    "results": results,
                },
                f,
                indent=2,
            )

        print(f"\nDetailed evaluation results saved to: {results_file}")

    else:
        print("No successful inferences completed!")
    print("\nEvaluation completed.")


def run():
    for model_id in MODEL_IDS:
        for dataset_class in DATASETS:
            adapter_dir = train_lora_for_model_and_dataset(
                model_id, dataset_class, force_retrain=FORCE_RETRAIN
            )
            evaluate_finetuned_model(model_id, dataset_class, adapter_dir)


if __name__ == "__main__":
    run()
