"""
Wrapper script to run fine-tuning experiments across multiple models and datasets.
This script loops over all specified models and datasets, running finetune_lora.py
for each combination and logging the output to separate files.
"""

import subprocess
import os
import sys
from datetime import datetime

import torch

RUN_DISTRIBUTED = True  # Set to True to run in distributed mode


def normalize_model_name(model_name: str) -> str:
    """Normalize model name for use in filenames."""
    return model_name.replace("/", "_").replace("-", "_")


def run_all_experiments():
    """Run fine-tuning experiments for all model/dataset combinations."""

    # Models to test
    models = [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.1-8B",
        "google/gemma-3-270m",
        "google/gemma-3-1b-pt",
        "google/gemma-3-4b-pt",
    ]

    # Datasets to test
    datasets = ["HARCoTQADataset", "SleepEDFCoTQADataset"]

    # Create logs directory
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Track experiment results
    total_experiments = len(models) * len(datasets)
    completed_experiments = 0
    failed_experiments = []

    print("=" * 80)
    print("FINE-TUNING EXPERIMENT BATCH")
    print("=" * 80)
    print(f"Total experiments to run: {total_experiments}")
    print(f"Models: {len(models)}")
    print(f"Datasets: {len(datasets)}")
    print(f"Logs directory: {logs_dir}")
    print("=" * 80)

    # Run experiments
    for model_idx, model in enumerate(models, 1):
        for dataset_idx, dataset in enumerate(datasets, 1):
            experiment_num = (model_idx - 1) * len(datasets) + dataset_idx

            print(f"\n[{experiment_num}/{total_experiments}] Starting experiment:")
            print(f"  Model: {model}")
            print(f"  Dataset: {dataset}")

            # Generate log filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            normalized_model = normalize_model_name(model)
            log_filename = f"{normalized_model}_{dataset}_{timestamp}.log"
            log_filepath = os.path.join(logs_dir, log_filename)

            print(f"  Log file: {log_filepath}")
            print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            try:
                # Run the experiment
                with open(log_filepath, "w") as log_file:
                    # Write experiment header
                    log_file.write(f"Fine-tuning Experiment Log\n")
                    log_file.write(f"=" * 80 + "\n")
                    log_file.write(f"Model: {model}\n")
                    log_file.write(f"Dataset: {dataset}\n")
                    log_file.write(f"Start time: {datetime.now()}\n")
                    log_file.write(
                        f"Experiment: {experiment_num}/{total_experiments}\n"
                    )
                    log_file.write("=" * 80 + "\n\n")
                    log_file.flush()

                    gpu_count = torch.cuda.device_count()

                    # Run finetune_lora.py with arguments
                    if RUN_DISTRIBUTED and gpu_count > 1:
                        print(f"Running in distributed mode with {gpu_count} CUDA devices...\n")
                        command = [
                            "torchrun",
                            f"--nproc-per-node={gpu_count}",
                        ]
                    else:
                        print("  Running in single-GPU mode...\n")
                        command = [sys.executable]
                    command += [
                        "finetune_lora.py",
                        "--model",
                        model,
                        "--dataset",
                        dataset,
                    ]
                    print("  Running command:", command)
                    result = subprocess.run(
                        command,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                        cwd=os.path.dirname(
                            __file__
                        ),  # Ensure we run in the correct directory
                    )

                    # Write experiment footer
                    log_file.write(f"\n\n" + "=" * 80 + "\n")
                    log_file.write(f"End time: {datetime.now()}\n")
                    log_file.write(f"Exit code: {result.returncode}\n")
                    log_file.write(
                        f"Status: {'SUCCESS' if result.returncode == 0 else 'FAILED'}\n"
                    )
                    log_file.write(f"=" * 80 + "\n")

                if result.returncode == 0:
                    print(
                        f"  ✅ SUCCESS - Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    completed_experiments += 1
                else:
                    print(f"  ❌ FAILED - Exit code: {result.returncode}")
                    failed_experiments.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "exit_code": result.returncode,
                            "log_file": log_filename,
                        }
                    )

            except Exception as e:
                print(f"  ❌ ERROR - Exception: {str(e)}")
                failed_experiments.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "exit_code": -1,
                        "error": str(e),
                        "log_file": log_filename,
                    }
                )

                # Write error to log file
                try:
                    with open(log_filepath, "a") as log_file:
                        log_file.write(f"\n\nEXCEPTION OCCURRED:\n")
                        log_file.write(f"Error: {str(e)}\n")
                        log_file.write(f"End time: {datetime.now()}\n")
                        log_file.write(f"Status: EXCEPTION\n")
                except:
                    pass

            print(
                f"  Progress: {completed_experiments + len(failed_experiments)}/{total_experiments}"
            )

    # Print final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT BATCH SUMMARY")
    print("=" * 80)
    print(f"Total experiments: {total_experiments}")
    print(f"Completed successfully: {completed_experiments}")
    print(f"Failed: {len(failed_experiments)}")
    print(f"Success rate: {completed_experiments / total_experiments * 100:.1f}%")

    if failed_experiments:
        print(f"\nFailed experiments:")
        for i, failure in enumerate(failed_experiments, 1):
            print(f"  {i}. {failure['model']} + {failure['dataset']}")
            print(f"     Exit code: {failure.get('exit_code', 'N/A')}")
            print(f"     Log file: {failure['log_file']}")
            if "error" in failure:
                print(f"     Error: {failure['error']}")

    print(f"\nAll logs saved to: {logs_dir}")
    print(f"Batch completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return completed_experiments, failed_experiments


def main():
    """Main function."""
    print("Fine-tuning Experiment Batch Runner")
    print(
        "This script will run fine-tuning experiments for multiple model/dataset combinations."
    )

    # Check if finetune_lora.py exists
    finetune_script = os.path.join(os.path.dirname(__file__), "finetune_lora.py")
    if not os.path.exists(finetune_script):
        print(f"ERROR: finetune_lora.py not found at {finetune_script}")
        sys.exit(1)

    try:
        completed, failed = run_all_experiments()

        # Exit with appropriate code
        if len(failed) == 0:
            print("All experiments completed successfully!")
            sys.exit(0)
        elif completed > 0:
            print("Some experiments completed successfully, but some failed.")
            sys.exit(1)
        else:
            print("All experiments failed!")
            sys.exit(2)

    except KeyboardInterrupt:
        print("\n\nExperiment batch interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error in batch runner: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
