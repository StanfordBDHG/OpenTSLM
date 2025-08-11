import glob
import json
import os
import re
import sys
from typing import Dict, List, Type

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
    DataCollatorForLanguageModeling,
)
from datasets import Dataset as HFDataset

# Ensure project src is on path (this script is under evaluation/baseline/)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from time_series_datasets.TSQADataset import TSQADataset
# from time_series_datasets.pamap2.PAMAP2AccQADataset import PAMAP2AccQADataset

# --------------------
# Configuration
# --------------------

MODEL_IDS: List[str] = [
    # "google/gemma-3n-e2b",
    # "google/gemma-3n-e2b-it",
    "meta-llama/Llama-3.2-1B",
]

DATASETS: List[Type[Dataset]] = [
    TSQADataset,
    # PAMAP2AccQADataset,
]

# Training hyperparameters (defaults)
EPOCHS = 1
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRAD_ACC_STEPS = 8
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
MAX_SEQ_LEN = 1024 * 16
TRAIN_SAMPLES_LIMIT = 10  # 512  # 0 or negative = no limit
EVAL_SAMPLES_LIMIT = 50  # number of test samples to evaluate; 0 => all

# LoRA config (defaults)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0  # 0.05

# Generation/eval settings
GEN_TEMPERATURE = 0.1
GEN_MAX_NEW_TOKENS = 100


FORCE_RETRAIN = True  # Set to True to force retraining even if output exists

# --------------------
# Utilities
# --------------------


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        # Fall back to eos token for padding
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "-", name.lower())


def ts_format_function(arr: np.ndarray) -> str:
    """
    Match the baseline's time series formatting, to keep consistency between training and evaluation.
    """
    return (
        np.array2string(
            arr,
            separator=" ",
            formatter={"all": lambda x: f'"{x:.2f}"'.replace(".", "")},
            threshold=sys.maxsize,
            max_line_width=sys.maxsize,
        )
        .removeprefix("[")
        .removesuffix("]")
    )


def prepare_train_dataset(tokenizer, dataset_class, max_length=MAX_SEQ_LEN):
    """
    Prepare your dataset for prompt-answer fine-tuning.
    Dataset format: [{"prompt": "question", "answer": "response"}, ...]
    """
    # Load your data
    # Use same formatting approach as baseline
    train_ds = dataset_class(
        "train",
        "",
        format_sample_str=True,
        time_series_format_function=ts_format_function,
    )
    items = list(train_ds)
    if TRAIN_SAMPLES_LIMIT > 0:
        items = items[:TRAIN_SAMPLES_LIMIT]

    # Convert to Hugging Face dataset
    dataset = HFDataset.from_list(items)

    def tokenize_function(examples):
        batch_size = len(examples["prompt"])
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for i in range(batch_size):
            prompt = examples["prompt"][i]
            answer = examples["answer"][i]

            # Create the full conversation
            # You can customize this format based on your needs
            full_text = f"{prompt}\n{answer}"

            # Tokenize prompt and full text separately to get lengths
            prompt_tokens = tokenizer(
                prompt + "\n",  # Include newline in prompt
                truncation=False,
                padding=False,
                return_tensors=None,
            )

            full_tokens = tokenizer(
                full_text,
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None,
            )

            input_ids = full_tokens["input_ids"]
            attention_mask = full_tokens["attention_mask"]

            # Create labels: -100 for prompt tokens (ignore in loss), actual tokens for answer
            labels = input_ids.copy()
            prompt_length = (
                len(prompt_tokens["input_ids"]) - 1
            )  # -1 because we don't want to ignore the newline

            # Set prompt tokens to -100 (ignored in loss computation)
            for j in range(min(prompt_length, len(labels))):
                labels[j] = -100

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    return tokenized_dataset


def build_test_dataset(dataset_class: Type[Dataset]) -> List[Dict[str, str]]:
    test_ds = dataset_class(
        "test",
        "",
        format_sample_str=True,
        time_series_format_function=ts_format_function,
    )
    items = list(test_ds)
    if EVAL_SAMPLES_LIMIT > 0:
        items = items[:EVAL_SAMPLES_LIMIT]
    return items


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
    tokenizer = ensure_pad_token(tokenizer)
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
    # train_dataset = build_train_dataset(dataset_class, tokenizer.eos_token)
    train_dataset = prepare_train_dataset(
        tokenizer, dataset_class, max_length=MAX_SEQ_LEN
    )
    # data_collator = DataCollatorForCausalLMWithPromptMask(
    #    tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN
    # )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # False for causal language modeling
        # pad_to_multiple_of=8,  # Optional: for better performance on some hardware
    )

    # Training
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            # optim="paged_adamw_8bit",  # Use 8-bit AdamW optimizer for memory efficiency
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            logging_steps=10,
            fp16=(device == "cuda"),
            # remove_unused_columns=False,
            # dataloader_pin_memory=False,
            # label_names=["answer"],
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
    tokenizer = ensure_pad_token(tokenizer)
    tokenizer.padding_side = "right"

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
        temperature=GEN_TEMPERATURE,
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
    test_items = build_test_dataset(dataset_class)
    print(f"Loaded {len(test_items)} test samples")

    # Metrics
    total_samples = 0
    successful_inferences = 0
    results: List[Dict[str, str]] = []

    print("\nRunning inference on test samples...")
    print("=" * 80)

    # Process each sample
    for idx in tqdm(range(len(test_items)), desc="Evaluating samples"):
        try:
            sample = test_items[idx].copy()

            # Clean up prompt for TSQADataset to match baseline
            pattern = r"This is the time series, it has mean (-?\d+\.\d{4}) and std (-?\d+\.\d{4})\."
            replacement = "This is the time series:"
            sample["prompt"] = re.sub(pattern, replacement, sample["prompt"])

            input_text = sample["prompt"]
            target_answer = sample["answer"]

            outputs = pipe(
                input_text,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                return_full_text=False,
            )

            if outputs and len(outputs) > 0:
                generated_text = outputs[0]["generated_text"].strip()
                successful_inferences += 1

                result = {
                    "sample_idx": idx,
                    "input_text": input_text,
                    "target_answer": target_answer,
                    "generated_answer": generated_text,
                }
                results.append(result)

                if idx < 5:
                    print(f"\nSAMPLE {idx + 1}:")
                    print(f"PROMPT: {sample['prompt'][:1000]}...")
                    print(f"ANSWER: {target_answer}")
                    print(f"OUTPUT: {generated_text}")
                    print("=" * 80)

            total_samples += 1

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

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
