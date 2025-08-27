import json
import os
import io
import re
import sys
import base64
from typing import Type, Callable, Dict, List, Any, Optional
from abc import ABC, abstractmethod

from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.pipelines import pipeline
import matplotlib.pyplot as plt
from time import sleep
from PIL import Image

# Add src to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

# Import OpenAIPipeline
from openai_pipeline import OpenAIPipeline
from common_evaluator import CommonEvaluator

class CommonEvaluatorPlot(CommonEvaluator):
    """
    A common evaluation framework for testing LLMs on time series datasets with plot generation.
    """
    
    def load_model(self, model_name: str, **pipeline_kwargs) -> pipeline:
        """
        Load a model using transformers pipeline or OpenAI API.
        """
        self.current_model_name = model_name  # Track the current model name for formatter selection
        if model_name.startswith("openai-"):
            # Use OpenAI API
            openai_model = model_name.replace("openai-", "")
            return OpenAIPipeline(model_name=openai_model, **pipeline_kwargs)
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        # Default pipeline arguments
        default_kwargs = {
            "task": "image-text-to-text",
            "device": self.device,
            "temperature": 0.1,
        }
        default_kwargs.update(pipeline_kwargs)
        pipe = pipeline(model=model_name, **default_kwargs)
        print(f"Model loaded successfully: {model_name}")
        return pipe
    

    
    def load_dataset(self, dataset_class: Type[Dataset], split: str = "test", 
                    format_sample_str: bool = True, **dataset_kwargs) -> Dataset:
        """
        Load a dataset with proper formatting.
        """
        print(f"Loading dataset: {dataset_class.__name__}")
        
        formatter = None

        # Default dataset arguments
        default_kwargs = {
            "split": split,
            "EOS_TOKEN": "",
            "format_sample_str": format_sample_str,
            "time_series_format_function": formatter,
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
        dataset = self.load_dataset(dataset_class, format_sample_str=False)
        
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
        
        # Get max_new_tokens for generation (default 1000)
        max_new_tokens = pipeline_kwargs.pop('max_new_tokens', 1000)
        
        for idx in tqdm(range(dataset_size), desc="Processing samples"):
            try:
                sample = dataset[idx]
                plot_data = None

                if isinstance(sample, dict) and 'time_series' in sample:
                    plot_data = self.get_plot_from_timeseries(sample["time_series"])

                input_text = """
You are given accelerometer data in all three dimensions. Your task is to classify the activity based on analysis of the data.
Instructions:
- Begin by analyzing the time series without assuming a specific label.
- Think step-by-step about what the observed patterns suggest regarding movement intensity and behavior.
- Write your rationale as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.
- Do **not** mention any class label until the final sentence.
The following activities (class labels) are possible: lying, sitting, standing, walking, ascending stairs, descending stairs, running, cycling, nordic walking, ironing, vacuum cleaning, rope jumping,

- You MUST end your response with "Answer: <class label>"
"""

                target_answer = sample["answer"]
                
                # Generate prediction
                if isinstance(pipe, OpenAIPipeline):
                    outputs = pipe(
                        input_text,
                        max_new_tokens=max_new_tokens,
                        return_full_text=False,
                        plot_data=plot_data,
                    )
                else: # For Hugging Face pipelines, convert plot_data (base64) to PIL and pass via images
                    try:
                        img_bytes = base64.b64decode(plot_data)
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                        messages = [
                            {"role": "user", "content": [
                                {"type": "image", "image": img}, 
                                {"type": "text", "text": input_text}
                            ]}
                        ]
                        outputs = pipe(
                            text=messages,
                            max_new_tokens=max_new_tokens,
                            return_full_text=False,
                        )
                    except Exception as e:
                        raise RuntimeError(f"Failed to decode plot image: {e}")

                
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
                    if idx < 10:
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
            print("❌ No successful inferences completed!")
            return {
                "model_name": model_name,
                "dataset_name": dataset_class.__name__,
                "total_samples": total_samples,
                "successful_inferences": 0,
                "success_rate": 0.0,
                "metrics": {},
                "detailed_results": [],
            }
    
    
    def get_plot_from_timeseries(self, time_series):
        """
        Create a base64 PNG plot from a list/tuple of 1D numpy arrays (e.g., [x, y, z]).
        """
        # Normalize input to a list of 1D arrays
        if time_series is None:
            return None
        ts_list = list(time_series)

        # Create the plot
        num_series = len(ts_list)
        fig, axes = plt.subplots(num_series, 1, figsize=(10, 4 * num_series), sharex=True)
        if num_series == 1:
            axes = [axes]

        # Plot each time series in its own subplot
        axis_names = {0: 'X-axis', 1: 'Y-axis', 2: 'Z-axis'}
        for i, series in enumerate(ts_list):
            axes[i].plot(series, marker='o', linestyle='-', markersize=0)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f"Accelerometer - {axis_names.get(i, f'Axis {i+1}')}" )

        plt.tight_layout()

        # Convert plot to base64 image
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        img_buffer.seek(0)
        image_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return image_data