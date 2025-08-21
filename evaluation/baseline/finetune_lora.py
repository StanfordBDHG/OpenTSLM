from dataclasses import dataclass
import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Type

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from torch.utils.data import Dataset as Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset as HFDataset

os.environ["NCCL_P2P_LEVEL"] = "NVL"

from common_evaluator import CommonEvaluator

# Ensure project src is on path (this script is under evaluation/baseline/)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from time_series_datasets.TSQADataset import TSQADataset
from time_series_datasets.pamap2.PAMAP2AccQADataset import PAMAP2AccQADataset
from time_series_datasets.pamap2.PAMAP2CoTQADataset import PAMAP2CoTQADataset
from time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
from time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset

# --------------------
# Configuration
# --------------------
FALLBACK_MODEL_ID = "meta-llama/Llama-3.2-1B"  # Default model if none specified
FALLBACK_DATASET = PAMAP2AccQADataset

# MODEL_IDS: List[str] = [
#     # "google/gemma-3n-e2b",
#     # "google/gemma-3n-e2b-it",
#     "meta-llama/Llama-3.2-1B",
#     #"meta-llama/Llama-3.2-3B",
# ]

# DATASETS: List[Type[Dataset]] = [
#     #TSQADataset,
#     #PAMAP2AccQADataset,
#     PAMAP2CoTQADataset,
#     #SleepEDFCoTQADataset
#]

# Training hyperparameters (defaults)
EPOCHS = 1
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRAD_ACC_STEPS = 8
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
TRAIN_SAMPLES_LIMIT = 0  # 0 or negative = no limit
EVAL_SAMPLES_LIMIT = 0  # number of test samples to evaluate; 0 => all

DO_VALIDATION = False

# LoRA config (defaults)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0  # 0.05

# Generation/eval settings
GEN_TEMPERATURE = 0.1
GEN_MAX_NEW_TOKENS = 300
EVAL_BATCH_SIZE = 2  # Batch size for evaluation to improve GPU efficiency


FORCE_RETRAIN = True  # Set to True to force retraining even if output exists

# --------------------
# Distributed Utilities
# --------------------
def is_distributed() -> bool:
    return "LOCAL_RANK" in os.environ

def setup_distributed():
    """Initialize distributed training if running in distributed mode."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True
    return False

def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0

def get_rank():
    """Get the current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size():
    """Get the total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1

def print_main(message: str):
    """Print message only from the main process."""
    if is_main_process():
        print(message)

# --------------------
# Utilities
# --------------------


def detect_device() -> str:
    """Detect the appropriate device, considering distributed setup."""
    if torch.cuda.is_available():
        if "LOCAL_RANK" in os.environ:
            # In distributed mode, use the local rank as device
            return f"cuda:{int(os.environ['LOCAL_RANK'])}"
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


def prepare_dataset(
    common_evaluator: CommonEvaluator,
    tokenizer,
    dataset_class,
    split: str = "train",
) -> HFDataset:
    """
    Prepare your dataset for prompt-answer fine-tuning.
    Dataset format: [{"prompt": "question", "answer": "response"}, ...]
    """
    dataset_file = os.path.join(
        os.path.dirname(__file__),
        "datasets",
        f"hf_{split}_dataset_{dataset_class.__name__}",
    )
    
    # Only main process should create/save datasets to avoid race conditions
    if os.path.exists(dataset_file):
        print_main(f"Loading cached dataset from disk: {dataset_file}")
        tokenized_dataset = HFDataset.load_from_disk(dataset_file)
    else:
        # Only main process creates the dataset
        if is_main_process():
            print(f"Creating dataset for {dataset_class.__name__} {split} split...")
            
            # Load your data
            # Use same formatting approach as baseline
            split_ds = common_evaluator.load_dataset(
                dataset_class=dataset_class,
                split=split,
                format_sample_str=True,
            )

            items = list(split_ds)

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
        
        # Wait for main process to finish creating dataset
        if dist.is_initialized():
            dist.barrier()
        
        # Now all processes can load the dataset
        tokenized_dataset = HFDataset.load_from_disk(dataset_file)

    if TRAIN_SAMPLES_LIMIT > 0:
        tokenized_dataset = tokenized_dataset.take(TRAIN_SAMPLES_LIMIT)

    return tokenized_dataset


def build_test_dataset(
    common_evaluator: CommonEvaluator, dataset_class: Type[Dataset]
) -> HFDataset:

    dataset_file = os.path.join(
        os.path.dirname(__file__),
        "datasets",
        f"hf_test_dataset_{dataset_class.__name__}",
    )
    
    # Only main process should create/save datasets to avoid race conditions
    if os.path.exists(dataset_file):
        print_main(f"Loading cached dataset from disk: {dataset_file}")
        test_dataset = HFDataset.load_from_disk(dataset_file)
    else:
        # Only main process creates the dataset
        if is_main_process():
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

            test_dataset = HFDataset.from_list(cleaned_items)
            test_dataset.save_to_disk(dataset_file)
        
        # Wait for main process to finish creating dataset
        if dist.is_initialized():
            dist.barrier()
        
        # Now all processes can load the dataset
        test_dataset = HFDataset.load_from_disk(dataset_file)

    return test_dataset


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

    print_main(f"Starting LoRA fine-tuning for model {model_id} on dataset {dataset_class.__name__}")

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
        print_main(f"Output directory {output_dir} already exists. Skipping fine-tuning.")
        return output_dir

    os.makedirs(output_dir, exist_ok=True)

    device = detect_device()
    print(f"Using device: {device}")

    # Load tokenizer and model
    print_main("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, device=device)
    tokenizer = ensure_pad_token(tokenizer, model_id)
    tokenizer.padding_side = "right"

    # Load model with 4-bit quantization for memory efficiency (optional)
    model = AutoModelForCausalLM.from_pretrained(model_id, max_memory="38000MB",)#quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    model.to(device)
    model.resize_token_embeddings(len(tokenizer))


    # Apply LoRA
    model = apply_lora(model)
    if is_main_process():
        model.print_trainable_parameters()
    model.train()

    if is_distributed():
        print_main("Wrapping model in DistributedDataParallel...")
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    # Prepare datasets and collator
    print_main("Preparing training dataset...")
    common_evaluator = CommonEvaluator(device=device)
    common_evaluator.current_model_name = model_id
    train_dataset = prepare_dataset(
        common_evaluator, tokenizer, dataset_class, split="train"
    )
    data_collator = SimpleCollator(tokenizer=tokenizer)

    trainer_kwargs = dict()
    training_arguments_kwargs = dict()
    if DO_VALIDATION:
        validation_dataset = prepare_dataset(
            common_evaluator, tokenizer, dataset_class, split="validation"
        )
        trainer_kwargs = dict(
            eval_dataset=validation_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        training_arguments_kwargs = dict(
            # Validation & Early Stopping Configuration
            eval_strategy="epoch",  # or "epoch"
            eval_steps=1,  # Evaluate every 50 steps
            per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            save_strategy="epoch",
            save_steps=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",  # or custom metric
            save_total_limit=3,  # Keep only 3 best checkpoints
        )

    # DDP-specific training arguments
    ddp_kwargs = {}
    if is_distributed():
        ddp_kwargs.update({
            "ddp_backend": "nccl",
            "dataloader_num_workers": 2,
            "ddp_find_unused_parameters": False,  # LoRA doesn't use all parameters
            "dataloader_pin_memory": True,
        })
    
    # Training
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            optim="adamw_8bit" if "cuda" in device else "adamw_torch",
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            logging_steps=10,
            fp16=not torch.cuda.is_bf16_supported() if "cuda" in device else False,
            bf16=torch.cuda.is_bf16_supported() if "cuda" in device else False,
            remove_unused_columns=False,
            **ddp_kwargs,
            **training_arguments_kwargs,
        ),
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        **trainer_kwargs,
    )


    print_main("Starting training...")
    trainer.train()
    print_main("Training complete.")

    if is_main_process():
        print_main("Saving LoRA adapters...")
        if isinstance(model, DDP) and hasattr(model, "module"):
            trainer.model.module.save_pretrained(output_dir)  # type: ignore
        else:
            trainer.model.save_pretrained(output_dir)  # type: ignore
        tokenizer.save_pretrained(output_dir)
        print_main(f"Saved to {output_dir}")

    return output_dir


def evaluate_finetuned_model(
    model_id: str, dataset_class: Type[Dataset], adapter_dir: str
):
    print_main(f"\nEvaluating fine-tuned model (adapters from {adapter_dir}) on dataset {dataset_class.__name__}")
    print_main("=" * 80)

    device = detect_device()
    print(f"Using device: {device}")

    # Load base and attach adapters for inference
    print_main("Loading base model and attaching LoRA adapters...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    tokenizer = ensure_pad_token(tokenizer, model_id)
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    base_model.to(device)
    if (
        getattr(base_model.config, "pad_token_id", None) is None
        and tokenizer.pad_token_id is not None
    ):
        base_model.config.pad_token_id = tokenizer.pad_token_id
    ft_model = PeftModel.from_pretrained(base_model, adapter_dir, torch_device=device, tp_plan="auto" if is_distributed() else None)

    # Build pipeline for generation (match baseline style)
    pipe = pipeline(
        task="text-generation",
        model=ft_model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
    )

    # Quick sanity check
    print_main("Quick generation sanity check...")
    try:
        out = pipe("The capital of France is", max_new_tokens=20)
        print(out)
    except Exception as e:
        print(f"Sanity check generation failed: {e}")
        raise e

    # Load test dataset
    print_main("Loading test dataset...")
    common_evaluator = CommonEvaluator(device=detect_device())
    common_evaluator.current_model_name = model_id
    test_dataset = build_test_dataset(common_evaluator, dataset_class)
    print_main(f"Loaded {len(test_dataset)} test samples")

    # Distribute dataset across GPUs for multi-GPU inference
    if is_distributed():
        world_size = get_world_size()
        rank = get_rank()
        
        # Calculate samples per GPU
        total_samples = len(test_dataset)
        samples_per_gpu = total_samples // world_size
        remainder = total_samples % world_size
        
        # Handle uneven distribution - give extra samples to first few GPUs
        if rank < remainder:
            start_idx = rank * (samples_per_gpu + 1)
            end_idx = start_idx + samples_per_gpu + 1
        else:
            start_idx = remainder * (samples_per_gpu + 1) + (rank - remainder) * samples_per_gpu
            end_idx = start_idx + samples_per_gpu
        
        # Create local dataset for this GPU
        local_test_dataset = test_dataset.select(range(start_idx, end_idx))
        
        print_main(f"Distributed inference across {world_size} GPUs")
        print(f"GPU {rank}: processing samples {start_idx} to {end_idx-1} ({len(local_test_dataset)} samples)")
    else:
        local_test_dataset = test_dataset
        if is_main_process():
            print("Single GPU inference")

    # Process in batches
    print(f"\nRunning inference on {len(local_test_dataset)} local samples...")
    print("=" * 80)

    # Process local samples in batches
    local_outputs = pipe(
        KeyDataset(local_test_dataset, "input_text"),
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        return_full_text=False,
        batch_size=EVAL_BATCH_SIZE,
    )

    # Process local results
    local_results = []
    local_successful_inferences = 0
    
    for idx, output in tqdm(
        enumerate(local_outputs), 
        total=len(local_test_dataset), 
        desc=f"Eval GPU {get_rank()}" if is_distributed() else "Eval"
    ):
        try:
            sample = local_test_dataset[idx]
            if output and len(output) > 0:
                generated_text = output[0]["generated_text"].strip()
                local_successful_inferences += 1

                result = {
                    "sample_idx": sample["sample_idx"],
                    "input_text": sample["input_text"],
                    "target_answer": sample["target_answer"],
                    "generated_answer": generated_text,
                }
                local_results.append(result)

                # Show first few samples only on main process
                if idx < 5 and is_main_process():
                    print(f"\nSAMPLE {sample['sample_idx'] + 1}:")
                    print(f"PROMPT: {sample['input_text'][:1000]}...")
                    print(f"ANSWER: {sample['target_answer']}")
                    print(f"OUTPUT: {generated_text}")
                    print("=" * 80)

        except Exception as e:
            print(f"Error processing sample {idx} on GPU {get_rank()}: {e}")
            continue

    # Gather results from all GPUs
    if is_distributed():
        # Synchronize all processes before gathering results
        dist.barrier()
        
        # Gather all results to main process
        all_results = [None] * get_world_size()
        all_successful_counts = [None] * get_world_size()
        
        dist.all_gather_object(all_results, local_results)
        dist.all_gather_object(all_successful_counts, local_successful_inferences)
        
        if is_main_process():
            # Combine results from all GPUs
            results = []
            total_successful_inferences = sum(count for count in all_successful_counts if count is not None)
            
            for gpu_results in all_results:
                if gpu_results is not None:
                    results.extend(gpu_results)
            
            # Sort results by original sample index to maintain order
            results.sort(key=lambda x: x["sample_idx"])
            total_samples = len(test_dataset)
            
            print(f"\nCombined results from {get_world_size()} GPUs:")
            print(f"Total samples: {total_samples}")
            print(f"Total successful inferences: {total_successful_inferences}")
        else:
            # Non-main processes don't need to process final results
            print(f"GPU {get_rank()} completed processing {len(local_results)} samples")
            return
    else:
        # Single GPU case
        results = local_results
        total_samples = len(local_test_dataset)
        total_successful_inferences = local_successful_inferences

    # Final evaluation metrics (only on main process)
    if is_main_process() and total_successful_inferences > 0:
        success_rate = total_successful_inferences / total_samples

        print("\n" + "=" * 80)
        print("FINE-TUNED MODEL EVALUATION RESULTS")
        print("=" * 80)
        print(f"Model: {model_id}")
        print(f"Total samples processed: {total_samples}")
        print(f"Successful inferences: {total_successful_inferences}")
        print(f"Success rate: {success_rate:.2%}")

        # Simple accuracy metrics (exact and partial match) to mirror baseline
        exact_matches = 0
        partial_matches = 0

        for result in results:
            target = (
                result["target_answer"].lower().split("answer:", maxsplit=1)[-1].strip()
            )
            generated = (
                result["generated_answer"]
                .lower()
                .split("answer:", maxsplit=1)[-1]
                .strip()
            )

            if target == generated:
                exact_matches += 1
            elif target in generated or generated in target:
                partial_matches += 1

        exact_accuracy = exact_matches / total_successful_inferences
        partial_accuracy = (exact_matches + partial_matches) / total_successful_inferences

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
                    "successful_inferences": total_successful_inferences,
                    "success_rate": success_rate,
                    "exact_accuracy": exact_accuracy,
                    "partial_accuracy": partial_accuracy,
                    "results": results,
                },
                f,
                indent=2,
            )

        print(f"\nDetailed evaluation results saved to: {results_file}")

    elif is_main_process():
        print("No successful inferences completed!")
    
    print("\nEvaluation completed.")


def run_single_experiment(model_id: str, dataset_class: Type[Dataset]):
    """Run a single fine-tuning experiment for the given model and dataset."""
    adapter_dir = train_lora_for_model_and_dataset(
        model_id, dataset_class, force_retrain=FORCE_RETRAIN
    )
    # Wait for all processes
    if dist.is_initialized():
        dist.barrier()
    evaluate_finetuned_model(model_id, dataset_class, adapter_dir)

def main():
    """Main function with argument parsing and distributed setup."""
    parser = argparse.ArgumentParser(description="Fine-tune models with LoRA on time series datasets")
    parser.add_argument(
        '--model', 
        type=str, 
        default='meta-llama/Llama-3.2-1B',
        help='Model ID to fine-tune (default: meta-llama/Llama-3.2-1B)'
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='PAMAP2CoTQADataset',
        choices=['PAMAP2CoTQADataset', 'HARCoTQADataset', 'SleepEDFCoTQADataset', 'PAMAP2AccQADataset'],
        help='Dataset class name to use (default: PAMAP2CoTQADataset)'
    )
    
    args = parser.parse_args()
    
    # Initialize distributed training if available
    is_distributed = setup_distributed()
    
    try:
        # Dataset class mapping
        dataset_classes = {
            'PAMAP2AccQADataset': PAMAP2AccQADataset,
            'PAMAP2CoTQADataset': PAMAP2CoTQADataset,
            'HARCoTQADataset': HARCoTQADataset,
            'SleepEDFCoTQADataset': SleepEDFCoTQADataset
        }
        
        dataset_class = dataset_classes[args.dataset]
        
        print_main(f"Starting fine-tuning experiment:")
        print_main(f"  Model: {args.model}")
        print_main(f"  Dataset: {args.dataset}")
        if is_distributed:
            print_main(f"  Distributed training: {get_world_size()} GPUs")
            print_main(f"  Current rank: {get_rank()}")
        print_main("=" * 80)
        
        run_single_experiment(args.model, dataset_class)
        
    finally:
        # Clean up distributed training
        if is_distributed:
            cleanup_distributed()


if __name__ == "__main__":
    main()

    # fallback for testing
    # run_single_experiment(FALLBACK_MODEL_ID, FALLBACK_DATASET)
