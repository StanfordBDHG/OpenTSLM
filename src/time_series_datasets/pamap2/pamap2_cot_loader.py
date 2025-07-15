import os
import pandas as pd
from datasets import Dataset
from typing import Tuple, Dict
import ast


DATA_DIR = "./data/"
COT_CSV = os.path.join(DATA_DIR, "pamap2_cot.csv")


TEST_FRAC = 0.1
VAL_FRAC = 0.1


def load_pamap2_cot_splits(seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load the PAMAP2 CoT dataset and split it into train, validation, and test sets.
    
    Args:
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train, validation, test) datasets
    """
    if not os.path.exists(COT_CSV):
        raise FileNotFoundError(f"CoT CSV not found: {COT_CSV}")
        
    df = pd.read_csv(COT_CSV)
    
    def parse_series(s):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []
    
    if 'x_axis' in df.columns:
        df['x_axis'] = df['x_axis'].apply(parse_series)
    if 'y_axis' in df.columns:
        df['y_axis'] = df['y_axis'].apply(parse_series)
    if 'z_axis' in df.columns:
        df['z_axis'] = df['z_axis'].apply(parse_series)
    
    full_dataset = Dataset.from_pandas(df)
    
    train_val, test = full_dataset.train_test_split(test_size=TEST_FRAC, seed=seed).values()
    val_frac_adj = VAL_FRAC / (1.0 - TEST_FRAC)
    train, val = train_val.train_test_split(test_size=val_frac_adj, seed=seed+1).values()
    
    return train, val, test


def get_label_distribution(dataset: Dataset) -> Dict[str, int]:
    """
    Get the distribution of labels in a dataset.
    
    Args:
        dataset: The dataset to analyze.
    Returns:
        Dictionary mapping label to count.
    """
    labels = dataset['label']
    return dict(pd.Series(labels).value_counts())

def print_dataset_info(dataset: Dataset, name: str):
    """
    Print information about a dataset split.
    
    Args:
        dataset: The dataset split.
        name: Name of the split (e.g., 'Train').
    """
    label_dist = get_label_distribution(dataset)
    print(f"\n{name} dataset:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Label distribution:")
    for label, count in label_dist.items():
        print(f"    {label}: {count} ({count/len(dataset)*100:.1f}%)")


if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_pamap2_cot_splits()
    
    print_dataset_info(train_ds, "Train")
    print_dataset_info(val_ds, "Validation")
    print_dataset_info(test_ds, "Test")
    
    if len(train_ds) > 0:
        sample = train_ds[0]
        print("\nSample data:")
        for key, value in sample.items():
            if key in ['x_axis', 'y_axis', 'z_axis']:
                print(f"{key}: {value[:5]}... (truncated)")
            else:
                print(f"{key}: {value}")
