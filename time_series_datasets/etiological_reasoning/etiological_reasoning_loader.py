from constants import RAW_DATA_PATH
import os
import subprocess
import requests
import json
from datasets import Dataset


def load_etiological_reasoning_dataset():
    """
    Download and save the etiological reasoning dataset directly from the JSON files.
    Returns train, validation, and test datasets.
    """
    # Load the etiological reasoning dataset
    etiological_reasoning_dir = os.path.join(RAW_DATA_PATH, "etiological_reasoning")
    
    # Create directory if it doesn't exist
    os.makedirs(etiological_reasoning_dir, exist_ok=True)
    
    # URLs for the three dataset splits
    urls = {
        "train": "https://huggingface.co/datasets/mikeam/time-series-reasoning/resolve/main/TS_Dataset_MCQ/train.json",
        "val": "https://huggingface.co/datasets/mikeam/time-series-reasoning/resolve/main/TS_Dataset_MCQ/val.json", 
        "test": "https://huggingface.co/datasets/mikeam/time-series-reasoning/resolve/main/TS_Dataset_MCQ/test.json"
    }
    
    datasets = {}
    
    for split_name, url in urls.items():
        split_dir = os.path.join(etiological_reasoning_dir, split_name)
        
        # Check if dataset already exists
        if os.path.exists(split_dir):
            print(f"Loading existing {split_name} dataset from: {split_dir}")
            try:
                dataset = Dataset.load_from_disk(split_dir)
                datasets[split_name] = dataset
                print(f"Loaded {len(dataset)} examples for {split_name}")
                continue
            except Exception as e:
                print(f"Error loading existing {split_name} dataset: {e}")
                # Continue to download if loading fails
        
        print(f"Downloading {split_name} dataset from: {url}")
        
        try:
            # Download the JSON file
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Parse the JSON file
            data = []
            for line in response.text.strip().split('\n'):
                if line.strip():  # Skip empty lines
                    example = json.loads(line)
                    data.append(example)
            
            print(f"Downloaded {len(data)} examples for {split_name}")
            
            # Create dataset from the downloaded data
            dataset = Dataset.from_list(data)
            
            # Save the dataset to a subdirectory
            dataset.save_to_disk(split_dir)
            print(f"{split_name.capitalize()} dataset saved successfully to: {split_dir}")
            
            datasets[split_name] = dataset
            
        except Exception as e:
            print(f"Error downloading or processing {split_name} dataset: {e}")
            datasets[split_name] = None
    
    return datasets


if __name__ == "__main__":
    dataset = load_etiological_reasoning_dataset()
