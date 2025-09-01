from datasets import Dataset
from typing import List, Tuple, Literal
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset
from time_series_datasets.ecg_qa.ecgqa_cot_loader import load_ecg_qa_cot_splits
import numpy as np

class ECGQACoTQADataset(QADataset):
    """
    ECG-QA Chain-of-Thought Dataset for question answering with electrocardiogram data.
    
    This dataset combines ECG time series data from PTB-XL with 
    question-answer pairs and chain-of-thought reasoning from the ECG-QA CoT dataset.
    
    Requires: pip install wfdb
    """
    
    def __init__(self, split: Literal["train", "test", "validation"], EOS_TOKEN: str, 
                 format_sample_str: bool = False, time_series_format_function=None,
                 max_samples: int = None, exclude_comparison: bool = False):
        """
        Initialize ECG-QA CoT Dataset.
        
        Args:
            split: Dataset split to load
            EOS_TOKEN: End-of-sequence token
            format_sample_str: Whether to format samples as strings
            time_series_format_function: Function to format time series data
            max_samples: Maximum number of samples per split (for testing)
            exclude_comparison: If True, exclude comparison questions (question_type starting with "comparison_")
        """
        self.max_samples = max_samples
        self.exclude_comparison = exclude_comparison
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load the ECG-QA CoT dataset splits."""
        print("Loading ECG-QA CoT dataset splits...")
        train, val, test = load_ecg_qa_cot_splits()
        
        # Filter out comparison questions if requested
        if self.exclude_comparison:
            print("Filtering out comparison questions...")
            
            def filter_comparison(dataset):
                filtered_data = []
                for sample in dataset:
                    question_type = sample.get("question_type")
                    if question_type is None:
                        raise ValueError(f"Sample missing required 'question_type' field: {sample}")
                    if not question_type.startswith("comparison"):
                        filtered_data.append(sample)
                return Dataset.from_list(filtered_data)
            
            original_train_len = len(train)
            original_val_len = len(val)
            original_test_len = len(test)
            
            train = filter_comparison(train)
            val = filter_comparison(val)
            test = filter_comparison(test)
            
            print(f"Filtered out comparison questions:")
            print(f"  Train: {original_train_len} -> {len(train)} ({original_train_len - len(train)} removed)")
            print(f"  Val: {original_val_len} -> {len(val)} ({original_val_len - len(val)} removed)")
            print(f"  Test: {original_test_len} -> {len(test)} ({original_test_len - len(test)} removed)")
        
        # Limit samples for faster testing if requested
        if self.max_samples:
            print(f"Limiting to {self.max_samples} samples per split for testing...")
            if len(train) > self.max_samples:
                train = train.select(range(self.max_samples))
            if len(val) > self.max_samples:
                val = val.select(range(self.max_samples))
            if len(test) > self.max_samples:
                test = test.select(range(self.max_samples))
        
        return train, val, test

    def _get_answer(self, row) -> str:
        """Get the answer from the row, which is the chain-of-thought reasoning."""
        return row.get("rationale", "No chain-of-thought reasoning available.")

    def _get_pre_prompt(self, row) -> str:
        """Generate the pre-prompt explaining the task with clinical context."""
        question_type = row.get("question_type")
        if question_type is None:
            raise ValueError(f"Sample missing required 'question_type' field: {row}")
        
        question = row.get("question")
        if question is None:
            raise ValueError(f"Sample missing required 'question' field: {row}")
        
        # Get clinical context if available
        clinical_contexts = row.get("clinical_contexts", [])
        if not clinical_contexts:
            raise ValueError(f"Sample missing required 'clinical_contexts' field: {row}")
        clinical_context = clinical_contexts[0]
        
        base_prompt = f"""You are an expert cardiologist analyzing an ECG (electrocardiogram). 

Clinical Context: {clinical_context}

Your task is to examine the ECG signal and answer the following medical question:

Question: {question}

Instructions:
- Begin by analyzing the time series without assuming a specific answer.
- Think step-by-step about what the observed patterns suggest regarding the cardiac condition.
- Write your rationale as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.
- Do **not** mention any final answer until the very end.
- Consider the ECG morphology, intervals, and any abnormalities that relate to the question."""
        
        if question_type == "single-verify":
            task_specific = """

Please analyze the ECG carefully and provide a clear, definitive answer with your reasoning."""
        
        elif question_type == "single-choice":
            task_specific = """

Analyze the patterns, waves, intervals, and any abnormalities to determine the correct answer."""
        
        elif question_type.startswith("comparison"):
            task_specific = """

This question requires comparison between different ECG recordings.
Look for differences, similarities, and changes between the ECGs to answer the question."""
        
        else:
            raise ValueError(f"Unknown question type: {question_type}. Expected: single-verify, single-choice, or comparison_*")
        
        return base_prompt + task_specific

    def _get_post_prompt(self, row) -> str:
        """Generate the post-prompt with possible answers and instructions."""
        # Try to get template-specific answers first
        template_id = row.get("template_id")
        if template_id is None:
            raise ValueError(f"Sample missing required 'template_id' field: {row}")
        
        possible_answers = ECGQACoTQADataset.get_possible_answers_for_template(template_id)
        
        if possible_answers:
            answers_text = ", ".join(possible_answers)
            prompt = f"""
Based on your analysis of the ECG data, select your answer from the following options:
{answers_text}

- Make sure that your last word is the answer. You MUST end your response with "Answer: "
"""
        else:
            prompt = """
Based on your analysis of the ECG data, provide your answer.
Make sure that your last word is the answer. You MUST end your response with "Answer: "
"""
        
        return prompt.strip()

    @staticmethod
    def get_possible_answers_for_template(template_id: int) -> List[str]:
        """Get possible answers for a specific template ID."""
        try:
            import pandas as pd
            import ast
            from time_series_datasets.ecg_qa.ecgqa_loader import ECG_QA_DIR
            
            # Load template answers directly
            template_answers_path = os.path.join(ECG_QA_DIR, "ecgqa", "ptbxl", "answers_for_each_template.csv")
            template_df = pd.read_csv(template_answers_path)
            
            # Find the row for this template_id
            template_row = template_df[template_df.template_id == template_id]
            if len(template_row) > 0:
                # Parse the string list back to actual list
                answers_str = template_row.iloc[0]['classes']
                return ast.literal_eval(answers_str)
            else:
                print(f"Warning: Template ID {template_id} not found in answers mapping")
                return []
                
        except Exception as e:
            print(f"Error loading template answers: {e}")
            return []
    
    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """Load ECG data and convert to TextTimeSeriesPrompt format."""
        
        ecg_prompts = []
        ecg_paths = row.get("ecg_paths")
        if ecg_paths is None:
            raise ValueError(f"Sample missing required 'ecg_paths' field: {row}")
        
        if not ecg_paths:
            # Fallback: single ECG path
            ecg_id = row.get("ecg_id")
            if ecg_id is None:
                raise ValueError(f"Sample missing required 'ecg_id' field: {row}")
            
            if not isinstance(ecg_id, list) or len(ecg_id) == 0:
                raise ValueError(f"Sample 'ecg_id' must be a non-empty list: {ecg_id}")
            
            from time_series_datasets.ecg_qa.ecgqa_loader import get_ptbxl_ecg_path
            ecg_path = get_ptbxl_ecg_path(ecg_id[0]) + ".dat"
            ecg_paths = [ecg_path]
        
        for i, ecg_path in enumerate(ecg_paths):
            # Load ECG data using wfdb
            base_path = ecg_path.replace('.dat', '').replace('.hea', '')
            
            if not os.path.exists(base_path + '.dat'):
                raise FileNotFoundError(f"ECG data file not found: {base_path}.dat")
            
            if not os.path.exists(base_path + '.hea'):
                raise FileNotFoundError(f"ECG header file not found: {base_path}.hea")
            
            try:
                # Read the ECG record
                import wfdb
                record = wfdb.rdrecord(base_path)
            except Exception as e:
                raise RuntimeError(f"Failed to read ECG record from {base_path}: {str(e)}")
            
            # Get the signal data - shape is (samples, leads)
            ecg_signal = record.p_signal  # Physical signal
            
            if ecg_signal is None:
                raise ValueError(f"ECG signal is None for file {base_path}")
            
            if ecg_signal.shape[0] == 0:
                raise ValueError(f"ECG signal is empty (0 samples) for file {base_path}")
            
            # PTB-XL typically has 12 leads, sample at 500Hz for 10 seconds = 5000 samples
            # We want to use 100Hz data for consistency and efficiency
            
            # Take first few leads (I, II, III, aVR, aVL, aVF) which are most common
            if len(ecg_signal.shape) == 1:
                # Single lead case
                n_leads = 1
            elif len(ecg_signal.shape) == 2:
                n_leads = min(6, ecg_signal.shape[1])
                if ecg_signal.shape[1] < 6:
                    print(f"Warning: ECG file {base_path} has only {ecg_signal.shape[1]} leads, expected at least 6")
            else:
                raise ValueError(f"Unexpected ECG signal shape {ecg_signal.shape} for file {base_path}")
            
            for lead_idx in range(n_leads):
                if len(ecg_signal.shape) > 1:
                    lead_signal = ecg_signal[:, lead_idx]
                else:
                    lead_signal = ecg_signal
                
                if len(lead_signal) == 0:
                    raise ValueError(f"Lead {lead_idx} is empty for file {base_path}")
                
                # Use 100Hz data - PTB-XL has both 100Hz and 500Hz versions
                # If we have 500Hz data, downsample to 100Hz (take every 5th sample)
                # If we already have 100Hz data, use as is
                if len(lead_signal) > 1000:  # Likely 500Hz data (5000 samples for 10 seconds)
                    downsampled_signal = lead_signal[::5]  # Downsample to 100Hz
                    original_freq = "500Hz"
                    target_freq = "100Hz"
                else:  # Likely already 100Hz data (1000 samples for 10 seconds)
                    downsampled_signal = lead_signal
                    original_freq = "100Hz"
                    target_freq = "100Hz"
                
                if len(downsampled_signal) == 0:
                    raise ValueError(f"Downsampled signal is empty for lead {lead_idx} in file {base_path}")
                
                # Verify we have exactly 1000 samples (10 seconds at 100Hz)
                if len(downsampled_signal) != 1000:
                    print(f"Warning: Lead {lead_idx} in file {base_path} has {len(downsampled_signal)} samples, expected 1000 for 100Hz")
                
                # Normalize the signal
                mean_val = float(np.mean(downsampled_signal))
                std_val = float(np.std(downsampled_signal))
                
                if np.isnan(mean_val) or np.isnan(std_val):
                    raise ValueError(f"NaN values detected in ECG signal statistics for lead {lead_idx} in file {base_path}")
                
                if std_val > 1e-6:  # Avoid division by zero
                    normalized_signal = (downsampled_signal - mean_val) / std_val
                else:
                    print(f"Warning: Lead {lead_idx} in file {base_path} has very low std deviation ({std_val}), signal may be flat")
                    normalized_signal = downsampled_signal - mean_val
                
                # Verify normalized signal is valid
                if np.any(np.isnan(normalized_signal)) or np.any(np.isinf(normalized_signal)):
                    raise ValueError(f"Invalid values (NaN/Inf) in normalized signal for lead {lead_idx} in file {base_path}")
                
                # Create lead name
                lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
                lead_name = lead_names[lead_idx] if lead_idx < len(lead_names) else f"Lead_{lead_idx}"
                
                ecg_label = f"ECG Lead {lead_name}"
                if len(ecg_paths) > 1:
                    ecg_label += f" (Recording {i+1})"
                    
                ecg_label += f" - sampled at 100Hz, normalized (mean={mean_val:.3f}, std={std_val:.3f})"
                
                try:
                    ecg_prompts.append(
                        TextTimeSeriesPrompt(ecg_label, normalized_signal.tolist())
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to create TextTimeSeriesPrompt for lead {lead_name} in file {base_path}: {str(e)}")
        
        if not ecg_prompts:
            raise RuntimeError(f"No ECG prompts were created for sample. ECG paths attempted: {ecg_paths}")
        
        return ecg_prompts

    @staticmethod
    def get_labels() -> List[str]:
        """Get all possible answer labels for ECG-QA CoT dataset."""
        # These are common answers in ECG-QA - could be loaded from answers.csv
        return [
            "yes", "no", "not sure",
            "normal", "abnormal", "borderline",
            "conduction disturbance", "hypertrophy", "ischemia", "infarction",
            "arrhythmia", "axis deviation", "non-specific changes"
        ]
    
    def _format_sample(self, row):
        # Call parent method to get the standard formatted sample
        formatted_sample = super()._format_sample(row)
        
        # Add CoT-specific fields if they exist in the original row
        if 'rationale' in row:
            formatted_sample['rationale'] = row['rationale']
        if 'cot_question_id' in row:
            formatted_sample['cot_question_id'] = row['cot_question_id']
        if 'cot_template_id' in row:
            formatted_sample['cot_template_id'] = row['cot_template_id']
        if 'cot_question_type' in row:
            formatted_sample['cot_question_type'] = row['cot_question_type']
        
        # Add original ECG-QA fields
        if 'template_id' in row:
            formatted_sample['template_id'] = row['template_id']
        if 'question_type' in row:
            formatted_sample['question_type'] = row['question_type']
        if 'question' in row:
            formatted_sample['question'] = row['question']
        
        # Add ECG data fields
        if 'ecg_id' in row:
            formatted_sample['ecg_id'] = row['ecg_id']
        if 'ecg_paths' in row:
            formatted_sample['ecg_paths'] = row['ecg_paths']
        if 'clinical_contexts' in row:
            formatted_sample['clinical_contexts'] = row['clinical_contexts']
        
        return formatted_sample


if __name__ == "__main__":
    # Test the dataset with limited samples
    print("Testing ECGQACoTQADataset...")
    
    try:
        # Test with just 5 samples per split for faster testing
        dataset = ECGQACoTQADataset(split="train", EOS_TOKEN="", max_samples=5)
        dataset_val = ECGQACoTQADataset(split="validation", EOS_TOKEN="", max_samples=5)
        dataset_test = ECGQACoTQADataset(split="test", EOS_TOKEN="", max_samples=5)
        
        print(f"Dataset sizes: Train: {len(dataset)}, Validation: {len(dataset_val)}, Test: {len(dataset_test)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print("\nSample keys:", sample.keys())
            print("Sample question:", sample.get("question", "N/A"))
            print("Sample answer (rationale):", sample["answer"][:200] + "..." if len(sample["answer"]) > 200 else sample["answer"])
            print("Sample question type:", sample.get("question_type", "N/A"))
            print("Sample ECG IDs:", sample.get("ecg_id", "N/A"))
            if "time_series_text" in sample:
                print("Time series prompts:", len(sample["time_series_text"]))
                if len(sample["time_series_text"]) > 0:
                    first_ts = sample["time_series_text"][0]
                    if hasattr(first_ts, 'text'):
                        print("First time series label:", first_ts.text)
                        print("First time series length:", len(first_ts.time_series))
                    else:
                        print("Time series format:", type(first_ts))
            print("Pre prompt:", sample["pre_prompt"][:100] + "..." if len(sample["pre_prompt"]) > 100 else sample["pre_prompt"])
            print("Post prompt:", sample["post_prompt"])
            
            # Show CoT-specific fields
            if 'rationale' in sample:
                print("CoT Rationale:", sample['rationale'][:100] + "..." if len(sample['rationale']) > 100 else sample['rationale'])
            if 'cot_question_id' in sample:
                print("CoT Question ID:", sample['cot_question_id'])
                
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
