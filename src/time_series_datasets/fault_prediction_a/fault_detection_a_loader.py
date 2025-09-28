import os
import pandas as pd
from datasets import Dataset
from typing import Tuple, Dict
import urllib.request
import ssl
import zipfile
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from time_series_datasets.constants import RAW_DATA
from tqdm.auto import tqdm

# Directory and file paths
FAULT_DETECTION_A_DATA_DIR = os.path.join(RAW_DATA, "fault_detection_a")
FAULT_DETECTION_A_ZIP = os.path.join(FAULT_DETECTION_A_DATA_DIR, "FaultDetectionA.zip")
FAULT_DETECTION_A_TRAIN_TS = os.path.join(
    FAULT_DETECTION_A_DATA_DIR, "FaultDetectionA_TRAIN.ts"
)
FAULT_DETECTION_A_TEST_TS = os.path.join(
    FAULT_DETECTION_A_DATA_DIR, "FaultDetectionA_TEST.ts"
)
FAULT_DETECTION_A_VAL_TS = os.path.join(FAULT_DETECTION_A_DATA_DIR, "val.ts")
FAULT_DETECTION_A_RELEASE_URL = (
    "https://www.timeseriesclassification.com/aeon-toolkit/FaultDetectionA.zip"
)


def download_and_extract_fault_detection_a():
    """
    Download the FaultDetectionA dataset zip file and extract the TS files if not already present.
    """
    # Check if all TS files already exist
    if (
        os.path.exists(FAULT_DETECTION_A_TRAIN_TS)
        and os.path.exists(FAULT_DETECTION_A_TEST_TS)
        and os.path.exists(FAULT_DETECTION_A_VAL_TS)
    ):
        print(f"FaultDetectionA dataset already exists at {FAULT_DETECTION_A_DATA_DIR}")
        return

    os.makedirs(FAULT_DETECTION_A_DATA_DIR, exist_ok=True)

    # Download the zip file if it doesn't exist
    if not os.path.exists(FAULT_DETECTION_A_ZIP):
        print(
            f"Downloading FaultDetectionA dataset from {FAULT_DETECTION_A_RELEASE_URL}..."
        )
        try:
            # Create SSL context that doesn't verify certificates (for development)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            request = urllib.request.Request(FAULT_DETECTION_A_RELEASE_URL)
            with urllib.request.urlopen(request, context=ssl_context) as response:
                total = int(response.headers.get("content-length", 0))
                with (
                    open(FAULT_DETECTION_A_ZIP, "wb") as f,
                    tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc="Downloading FaultDetectionA.zip",
                    ) as pbar,
                ):
                    for chunk in iter(lambda: response.read(8192), b""):
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
            print("Download completed successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to download FaultDetectionA dataset: {e}")

    # Extract the zip file
    print("Extracting FaultDetectionA dataset...")
    try:
        with zipfile.ZipFile(FAULT_DETECTION_A_ZIP, "r") as zip_ref:
            zip_ref.extractall(FAULT_DETECTION_A_DATA_DIR)
        print("Extraction completed successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract FaultDetectionA dataset: {e}")

    # Verify all TS files exist
    if not all(
        [
            os.path.exists(FAULT_DETECTION_A_TRAIN_TS),
            os.path.exists(FAULT_DETECTION_A_TEST_TS),
            os.path.exists(FAULT_DETECTION_A_VAL_TS),
        ]
    ):
        raise FileNotFoundError(
            f"TS files not found after extraction in {FAULT_DETECTION_A_DATA_DIR}"
        )


def ensure_fault_detection_a_dataset():
    """
    Ensure the FaultDetectionA dataset is available in data/fault_detection_a/.
    Download and extract if necessary.
    """
    if not (
        os.path.exists(FAULT_DETECTION_A_TRAIN_TS)
        and os.path.exists(FAULT_DETECTION_A_TEST_TS)
        and os.path.exists(FAULT_DETECTION_A_VAL_TS)
    ):
        download_and_extract_fault_detection_a()


def parse_ts_line(line):
    """
    Parse a line from the .ts file format.
    Format: value1,value2,...,valueN:label

    Args:
        line: String line from the .ts file

    Returns:
        Tuple of (time_series_values, label)
    """
    line = line.strip()
    if ":" not in line:
        return None, None

    # Split on the last colon to separate values from label
    values_str, label_str = line.rsplit(":", 1)

    try:
        # Parse time series values
        time_series = [float(x) for x in values_str.split(",")]
        # Parse label
        label = float(label_str)
        return time_series, label
    except ValueError as e:
        print(f"Error parsing line: {e}")
        print(f"Line: {line[:100]}...")
        return None, None


def load_fault_detection_a_ts(ts_path: str) -> pd.DataFrame:
    """
    Load and preprocess a FaultDetectionA TS file.

    Args:
        ts_path: Path to the TS file

    Returns:
        Processed DataFrame with time series data and labels
    """
    print(f"Loading {ts_path}...")

    # Read the file and skip metadata lines
    processed_data = []
    with open(ts_path, "r") as f:
        for line_num, line in enumerate(f):
            # Skip metadata lines that start with @
            if line.startswith("@"):
                continue

            time_series, label = parse_ts_line(line)
            if time_series is not None and label is not None:
                processed_data.append({"time_series": time_series, "label": label})

    processed_df = pd.DataFrame(processed_data)

    if len(processed_df) > 0:
        print(
            f"Loaded {len(processed_df)} samples with {processed_df['label'].nunique()} unique labels"
        )
        print(f"Time series length: {len(processed_df['time_series'].iloc[0])}")
    else:
        print("Warning: No data was loaded from the file")

    return processed_df


def load_fault_detection_a_splits() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load the FaultDetectionA dataset splits using the original split.

    According to the official dataset description:
    - Original train: 8,184 samples
    - Original validation: 2,728 samples
    - Original test: 2,728 samples

    The train file contains train+validation concatenated (10,912 total).
    We split the first 8,184 as train and the last 2,728 as validation.

    Returns:
        Tuple of (train, validation, test) Dataset objects
    """
    # Ensure dataset is available
    ensure_fault_detection_a_dataset()

    # Load the combined train file and test data
    combined_train_df = load_fault_detection_a_ts(FAULT_DETECTION_A_TRAIN_TS)
    test_df = load_fault_detection_a_ts(FAULT_DETECTION_A_TEST_TS)

    # Split according to original split: first 8,184 = train, last 2,728 = validation
    train_df = combined_train_df.iloc[:8184].copy()
    val_df = combined_train_df.iloc[8184:].copy()

    print(f"Using original dataset split:")
    print(f"  Train samples: {len(train_df)} (original train set)")
    print(f"  Validation samples: {len(val_df)} (original validation set)")
    print(f"  Test samples: {len(test_df)} (original test set)")

    # Convert to Dataset objects
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    return train_dataset, val_dataset, test_dataset


def get_label_distribution(dataset: Dataset) -> Dict[str, int]:
    """
    Get label distribution for a dataset.

    Args:
        dataset: Dataset object

    Returns:
        Dictionary mapping labels to counts
    """
    labels = dataset["label"]
    return dict(pd.Series(labels).value_counts())


def get_label_names() -> Dict[float, str]:
    """
    Get the mapping from numeric labels to descriptive names.
    Based on the dataset description:
    - undamaged (9.09%)
    - inner damaged (45.55%)
    - outer damaged (45.55%)

    Returns:
        Dictionary mapping numeric labels to descriptive names
    """
    return {0.0: "undamaged", 1.0: "inner_damaged", 2.0: "outer_damaged"}


def print_dataset_info(dataset: Dataset, name: str):
    """
    Print information about a dataset split.

    Args:
        dataset: The dataset split.
        name: Name of the split (e.g., 'Train').
    """
    label_dist = get_label_distribution(dataset)
    label_names = get_label_names()

    print(f"\n{name} dataset:")
    print(f"  Total samples: {len(dataset)}")
    print("  Label distribution:")
    for label, count in sorted(label_dist.items()):
        label_name = label_names.get(label, f"unknown_{label}")
        print(
            f"    {label} ({label_name}): {count} ({count / len(dataset) * 100:.1f}%)"
        )


if __name__ == "__main__":
    print("=== FaultDetectionA Dataset Loading Demo ===\n")

    # Load the dataset splits
    train_ds, val_ds, test_ds = load_fault_detection_a_splits()

    # Print dataset information
    print_dataset_info(train_ds, "Train")
    print_dataset_info(val_ds, "Validation")
    print_dataset_info(test_ds, "Test")

    # Show sample data
    if len(train_ds) > 0:
        print("\n" + "=" * 50 + "\n")
        print("Sample data from training set:")
        sample = train_ds[0]
        for key, value in sample.items():
            if key == "time_series":
                if isinstance(value, list) and len(value) > 0:
                    print(f"{key}: {value[:5]}... (length: {len(value)})")
                else:
                    print(f"{key}: {value}")
            else:
                print(f"{key}: {value}")
