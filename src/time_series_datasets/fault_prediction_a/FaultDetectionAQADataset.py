from datasets import Dataset
from typing import List, Tuple, Literal
import torch

from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset
from time_series_datasets.fault_prediction_a.fault_detection_a_loader import (
    load_fault_detection_a_splits,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)

# Time series label for single-channel sensor data
TIME_SERIES_LABEL = "Motor current signal data from an electromechanical drive system"


class FaultDetectionAQADataset(QADataset):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        super().__init__(
            split, EOS_TOKEN, format_sample_str, time_series_format_function
        )

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load the FaultDetectionA dataset splits using the fault_detection_a_loader.

        Returns:
            Tuple of (train, validation, test) datasets
        """
        return load_fault_detection_a_splits()

    def _get_answer(self, row) -> str:
        """Convert numeric label to descriptive string"""
        label = row["label"]
        label_mapping = {0.0: "undamaged", 1.0: "inner_damaged", 2.0: "outer_damaged"}
        return label_mapping.get(label, f"unknown_{label}")

    def _get_pre_prompt(self, _row) -> str:
        return "You are given motor current signal data from an electromechanical drive system monitoring rolling bearings. This single-channel sensor data was recorded at 64 kHz sampling frequency over 4-second intervals from an electromechanical test setup. The system operates under varying conditions including different rotational speeds, load torques, and radial forces. The motor current signatures reflect the mechanical health of rolling bearings and can reveal characteristic patterns indicating undamaged bearings, inner race damage, or outer race damage. Your task is to detect bearing damage conditions."

    def _get_post_prompt(self, _row) -> str:
        damage_types = ", ".join(self.get_labels())
        text = f"""
Instructions:
- Begin by analyzing the time series signal without assuming a specific condition.
- Think step-by-step about what the observed patterns in the motor current suggest regarding bearing health.
- Write your rationale as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.
- Consider signal characteristics like amplitude variations, frequency patterns, and anomalies that might indicate bearing damage.
- Do **not** mention any damage condition until the final sentence.
The following bearing conditions are possible: {damage_types}
- You MUST end your response with "Answer: <condition>"
"""
        return text

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        Convert the time series data into a list of TextTimeSeriesPrompt objects.
        FaultDetectionA has single-channel data.
        """
        time_series = torch.tensor(row["time_series"], dtype=torch.float32)

        return [TextTimeSeriesPrompt(TIME_SERIES_LABEL, time_series.tolist())]

    @staticmethod
    def get_labels() -> List[str]:
        """Return the possible bearing condition labels"""
        return ["undamaged", "inner_damaged", "outer_damaged"]

    def _format_sample(self, row):
        # Call the parent method to get the formatted sample
        sample = super()._format_sample(row)
        # Add any additional fields we need
        sample["label"] = row["label"]
        return sample


if __name__ == "__main__":
    dataset = FaultDetectionAQADataset(split="train", EOS_TOKEN="")
    dataset_val = FaultDetectionAQADataset(split="validation", EOS_TOKEN="")
    dataset_test = FaultDetectionAQADataset(split="test", EOS_TOKEN="")

    print(
        f"Dataset sizes: Train: {len(dataset)}, Validation: {len(dataset_val)}, Test: {len(dataset_test)}"
    )

    # Print label distribution
    print("\n=== Label Distribution ===")
    for split_name, split_dataset in [
        ("Train", dataset),
        ("Validation", dataset_val),
        ("Test", dataset_test),
    ]:
        labels = [sample["label"] for sample in split_dataset]
        label_counts = {
            0.0: labels.count(0.0),
            1.0: labels.count(1.0),
            2.0: labels.count(2.0),
        }
        total = len(labels)
        print(f"{split_name}:")
        print(
            f"  undamaged (0.0): {label_counts[0.0]} ({label_counts[0.0] / total * 100:.1f}%)"
        )
        print(
            f"  inner_damaged (1.0): {label_counts[1.0]} ({label_counts[1.0] / total * 100:.1f}%)"
        )
        print(
            f"  outer_damaged (2.0): {label_counts[2.0]} ({label_counts[2.0] / total * 100:.1f}%)"
        )

    # Show 5 samples per label from the test set
    print("\n=== Sample Examples from Test Set ===")
    samples_per_label = {0.0: [], 1.0: [], 2.0: []}

    # Collect samples by label
    for i, sample in enumerate(dataset_test):
        label = sample["label"]
        if len(samples_per_label[label]) < 5:
            samples_per_label[label].append((i, sample))

        # Stop when we have 5 samples for each label
        if all(len(samples) >= 5 for samples in samples_per_label.values()):
            break

    # Print samples for each label
    for label_val, label_name in [
        (0.0, "undamaged"),
        (1.0, "inner_damaged"),
        (2.0, "outer_damaged"),
    ]:
        print(f"\n{label_name.upper()} samples:")
        for idx, (sample_idx, sample) in enumerate(samples_per_label[label_val]):
            print(f"  Sample {idx + 1} (dataset index {sample_idx}):")
            print(f"    Answer: {sample['answer']}")
            print(f"    Label: {sample['label']}")
            if "time_series" in sample and len(sample["time_series"]) > 0:
                ts_length = (
                    len(sample["time_series"][0])
                    if isinstance(sample["time_series"], list)
                    else len(sample["time_series"])
                )
                print(f"    Time series length: {ts_length}")

    # Test dataloader with single batch
    print("\n=== Dataloader Test ===")
    dataloader = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=4
        ),
    )

    for batch in tqdm(dataloader, total=1):
        print("Batch keys:", batch[0].keys())
        print("Batch time_series shape:", batch[0]["time_series"].shape)
        print("Batch answer:", batch[0]["answer"])
        break
