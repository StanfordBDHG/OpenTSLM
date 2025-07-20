import json
import os
import re
import sys
from typing import Type, Callable, Dict, List, Any, Optional
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.pipelines import pipeline

# Add src to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)


class CommonEvaluator:
    """
    A common evaluation framework for testing LLMs on time series datasets.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            device: Device to use for inference ('cuda', 'mps', 'cpu', or None for auto)
        """
        self.device = device or self._get_best_device()
        
    def _get_best_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self, model_name: str, **pipeline_kwargs) -> pipeline:
        """
        Load a model using transformers pipeline.
        
        Args:
            model_name: Name of the model to load
            **pipeline_kwargs: Additional arguments for the pipeline
            
        Returns:
            Loaded pipeline
        """
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Default pipeline arguments
        default_kwargs = {
            "task": "text-generation",
            "device": self.device,
            "temperature": 0.1,
            "max_new_tokens": 10000,
        }
        
        # Update with provided kwargs
        default_kwargs.update(pipeline_kwargs)
        
        pipe = pipeline(model=model_name, **default_kwargs)
        print(f"Model loaded successfully: {model_name}")
        
        return pipe
    

    
    def load_dataset(self, dataset_class: Type[Dataset], split: str = "test", 
                    format_sample_str: bool = True, **dataset_kwargs) -> Dataset:
        """
        Load a dataset with proper formatting.
        
        Args:
            dataset_class: Dataset class to instantiate
            split: Dataset split to load
            format_sample_str: Whether to format samples as strings
            **dataset_kwargs: Additional arguments for dataset initialization
            
        Returns:
            Loaded dataset
        """
        print(f"Loading dataset: {dataset_class.__name__}")
        
        # Import the gruver formatter
        from gruver_llmtime_tokenizer import gruver_et_al_formatter
        
        # Default dataset arguments
        default_kwargs = {
            "split": split,
            "EOS_TOKEN": "",
            "format_sample_str": format_sample_str,
            "time_series_format_function": gruver_et_al_formatter,
        }
        
        # Update with provided kwargs
        default_kwargs.update(dataset_kwargs)
        
        dataset = dataset_class(**default_kwargs)
        print(f"Loaded {len(dataset)} {split} samples")
        
        return dataset
    
    def evaluate_model_on_dataset(
        self,
        model_name: str,
        dataset_class: Type[Dataset],
        evaluation_function: Callable[[str, str], Dict[str, Any]],
        max_samples: Optional[int] = None,
        **pipeline_kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a model on a dataset using a custom evaluation function.
        
        Args:
            model_name: Name of the model to evaluate
            dataset_class: Dataset class to use
            evaluation_function: Function that takes (ground_truth, prediction) and returns metrics
            max_samples: Maximum number of samples to evaluate (None for all)
            **pipeline_kwargs: Additional arguments for model pipeline
            
        Returns:
            Dictionary containing evaluation results
        """
        print(f"Starting evaluation with model {model_name} on dataset {dataset_class.__name__}")
        print("=" * 60)
        
        # Load model
        pipe = self.load_model(model_name, **pipeline_kwargs)
        
        # Load dataset
        dataset = self.load_dataset(dataset_class)
        
        # Limit samples if specified
        if max_samples is not None:
            dataset_size = min(len(dataset), max_samples)
            print(f"Processing first {dataset_size} samples...")
        else:
            dataset_size = len(dataset)
            print(f"Processing all {dataset_size} samples...")
        
        # Initialize tracking
        total_samples = 0
        successful_inferences = 0
        all_metrics = []
        results = []
        first_error_printed = False  # Track if we've printed the first error
        
        print("\nRunning inference...")
        print("=" * 80)
        
        # Process each sample
        for idx in tqdm(range(dataset_size), desc="Processing samples"):
            try:
                sample = dataset[idx]
                
                # Clean up prompt for TSQADataset (if needed)
                if hasattr(sample, 'get') and sample.get('prompt'):
                    pattern = r"This is the time series, it has mean (-?\d+\.\d{4}) and std (-?\d+\.\d{4})\."
                    replacement = "This is the time series:"
                    sample["prompt"] = re.sub(pattern, replacement, sample["prompt"])
                
                # Create input text
                input_text = sample["prompt"]
                target_answer = sample["answer"]
                
                # Generate prediction
                outputs = pipe(
                    input_text,
                    max_new_tokens=10000,
                    return_full_text=False,
                )
                
                # Extract generated text
                if outputs and len(outputs) > 0:
                    generated_text = outputs[0]["generated_text"].strip()
                    successful_inferences += 1
                    
                    # Evaluate using custom function
                    metrics = evaluation_function(target_answer, generated_text)
                    all_metrics.append(metrics)
                    
                    # Store detailed results
                    result = {
                        "sample_idx": idx,
                        "input_text": input_text,
                        "target_answer": target_answer,
                        "generated_answer": generated_text,
                        "metrics": metrics,
                    }
                    results.append(result)
                    
                    # Print progress for first few samples
                    if idx < 3:
                        print(f"\nSAMPLE {idx + 1}:")
                        print(f"PROMPT: {input_text}...")
                        print(f"TARGET: {target_answer}")
                        print(f"PREDICTION: {generated_text}")
                        print(f"METRICS: {metrics}")
                        print("=" * 80)
                    
                    # Print first error for debugging
                    if not first_error_printed and metrics.get('accuracy', 1) == 0:
                        print(f"\n❌ FIRST ERROR (Sample {idx + 1}):")
                        print(f"TARGET: {target_answer}")
                        print(f"PREDICTION: {generated_text}")
                        print("=" * 80)
                        first_error_printed = True
                
                total_samples += 1
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        # Calculate aggregate metrics
        if successful_inferences > 0:
            # Aggregate metrics across all samples
            aggregate_metrics = self._aggregate_metrics(all_metrics)
            
            # Calculate success rate
            success_rate = successful_inferences / total_samples
            
            # Prepare final results
            final_results = {
                "model_name": model_name,
                "dataset_name": dataset_class.__name__,
                "total_samples": total_samples,
                "successful_inferences": successful_inferences,
                "success_rate": success_rate,
                "metrics": aggregate_metrics,
                "detailed_results": results,
            }
            
            # Print summary
            self._print_summary(final_results)
            
            # Save results
            self._save_results(final_results)
            
            return final_results
        else:
            print("No successful inferences completed!")
            return {
                "model_name": model_name,
                "dataset_name": dataset_class.__name__,
                "total_samples": total_samples,
                "successful_inferences": 0,
                "success_rate": 0.0,
                "metrics": {},
                "detailed_results": [],
            }
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate metrics across all samples.
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Aggregated metrics
        """
        if not metrics_list:
            return {}
        
        # Get all unique metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        aggregated = {}
        for key in all_keys:
            values = [metrics.get(key, 0) for metrics in metrics_list]
            if all(isinstance(v, (int, float)) for v in values):
                # Calculate overall accuracy/percentage
                accuracy = np.mean(values) * 100
                aggregated[key] = accuracy
            else:
                # For non-numeric metrics, just count occurrences
                aggregated[key] = {
                    "values": values,
                    "count": len(values),
                }
        
        return aggregated
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"Model: {results['model_name']}")
        print(f"Dataset: {results['dataset_name']}")
        print(f"Total samples processed: {results['total_samples']}")
        print(f"Successful inferences: {results['successful_inferences']}")
        print(f"Success rate: {results['success_rate']:.2%}")
        
        if results['metrics']:
            print("\nAggregated Metrics:")
            for metric_name, metric_values in results['metrics'].items():
                if isinstance(metric_values, (int, float)):
                    print(f"  {metric_name}: {metric_values:.1f}%")
                else:
                    print(f"  {metric_name}: {metric_values}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save detailed results to file."""
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        detailed_dir = os.path.join(current_dir, "..", "results", "baseline", "detailed")
        os.makedirs(detailed_dir, exist_ok=True)
        normalized_model_id = re.sub(r"[^a-z0-9]", "-", results['model_name'].lower())
        normalized_dataset_name = re.sub(r"[^a-z0-9]", "-", results['dataset_name'].lower())
        results_file = os.path.join(detailed_dir, f"evaluation_results_{normalized_model_id}_{normalized_dataset_name}.json")
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
    
    def evaluate_multiple_models(
        self,
        model_names: List[str],
        dataset_classes: List[Type[Dataset]],
        evaluation_functions: Dict[str, Callable[[str, str], Dict[str, Any]]],
        max_samples: Optional[int] = None,
        **pipeline_kwargs
    ) -> pd.DataFrame:
        """
        Evaluate multiple models on multiple datasets.
        
        Args:
            model_names: List of model names to evaluate
            dataset_classes: List of dataset classes to evaluate on
            evaluation_functions: Dictionary mapping dataset class names to evaluation functions
            max_samples: Maximum number of samples per evaluation
            **pipeline_kwargs: Additional arguments for model pipeline
            
        Returns:
            DataFrame with results for all model-dataset combinations
        """
        all_results = []
        
        # Generate filename once at the beginning
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(current_dir, "..", "results", "baseline")
        os.makedirs(results_dir, exist_ok=True)
        df_filename = os.path.join(results_dir, f"evaluation_results_{timestamp}.csv")
        print(f"Results will be saved to: {df_filename}")
        
        for model_name in model_names:
            for dataset_class in dataset_classes:
                dataset_name = dataset_class.__name__
                
                if dataset_name not in evaluation_functions:
                    print(f"Warning: No evaluation function found for {dataset_name}")
                    continue
                
                evaluation_function = evaluation_functions[dataset_name]
                
                print(f"\n{'='*80}")
                print(f"Evaluating {model_name} on {dataset_name}")
                print(f"{'='*80}")
                
                try:
                    results = self.evaluate_model_on_dataset(
                        model_name=model_name,
                        dataset_class=dataset_class,
                        evaluation_function=evaluation_function,
                        max_samples=max_samples,
                        **pipeline_kwargs
                    )
                    
                    # Extract key metrics for DataFrame
                    row = {
                        "model": model_name,
                        "dataset": dataset_name,
                        "total_samples": results["total_samples"],
                        "successful_inferences": results["successful_inferences"],
                        "success_rate": results["success_rate"],
                    }
                    
                    # Add specific metrics
                    if results["metrics"]:
                        for metric_name, metric_values in results["metrics"].items():
                            if isinstance(metric_values, (int, float)):
                                row[metric_name] = metric_values
                            else:
                                row[metric_name] = str(metric_values)
                    
                    all_results.append(row)
                    
                    # Save DataFrame after each model-dataset evaluation
                    df = pd.DataFrame(all_results)
                    df.to_csv(df_filename, index=False)
                    print(f"✅ Results updated: {df_filename}")
                    
                except Exception as e:
                    print(f"Error evaluating {model_name} on {dataset_name}: {e}")
                    all_results.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "error": str(e),
                    })
                    
                    # Save DataFrame even after errors
                    df = pd.DataFrame(all_results)
                    df.to_csv(df_filename, index=False)
                    print(f"⚠️  Results updated (with error): {df_filename}")
        
        print(f"\nFinal results saved to: {df_filename}")
        return df 