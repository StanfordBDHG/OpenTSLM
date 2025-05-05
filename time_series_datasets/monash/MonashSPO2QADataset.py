from typing import Literal, Optional
from datasets import Dataset, Value, Sequence, Features
from time_series_datasets.monash.MonashDataset import MonashDataset
from time_series_datasets.QADataset import QADataset
from time_series_datasets.util import (
    SingletonDataset,
    collate_fn,
    load_qa_dataset,
    torch_to_hf_generator,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


features = Features(
    {
        "Series": Sequence(Value("float32")),
        "Question": Value("string"),
        "Answer": Value("string"),
    }
)

file_mapping = {"test": "TEST", "train": "TRAIN", "val": "TRAIN", "validation": "TRAIN"}


@SingletonDataset
class MonashSPO2QADataset:
    last_file_suffix_loaded = None
    hugging_face_dataset = None

    def load(self, split: Literal["train", "validation", "test"], EOS_TOKEN):
        if (
            self.hugging_face_dataset is None
            or self.last_file_suffix_loaded is None
            or file_mapping[split] != self.last_file_suffix_loaded
        ):
            file_suffix = file_mapping[split]
            self.last_file_suffix_loaded = file_suffix
            self.hugging_face_dataset = Dataset.from_generator(
                torch_to_hf_generator,
                gen_kwargs={
                    "torch_ds": QADataset(
                        MonashDataset(
                            _data_dir="monash_datasets",
                            data_name=f"BIDMC32SpO2/BIDMC32SpO2_{file_suffix}",
                        ),
                        "Given the following PPG data, what is the oxygen saturation of the blood?",
                    )
                },
                features=features,
            )

        # Since we have different files for train and test we are setting
        # val_frac only when we want the training dataset and train_frac to always 0
        val_frac = 0.1 if split in ["train", "val", "validation"] else 0
        test_frac = 0

        self.qa_dataset = load_qa_dataset(
            self.hugging_face_dataset,
            split=split,
            EOS_TOKEN=EOS_TOKEN,
            max_samples=None,
            val_frac=val_frac,
            test_frac=test_frac,
        )
        return self.qa_dataset

    def __len__(self) -> int:
        return len(self.qa_dataset)

    def __getitem__(self, idx):
        return self.qa_dataset[idx]


if __name__ == "__main__":
    dataset = MonashSPO2QADataset()
    dataset.load(split="train", EOS_TOKEN="")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, patch_size=4),
    )

    m = float("-inf")
    series = []
    for ts_batch, prompts, answers in tqdm(dataloader):
        print(ts_batch, prompts, answers)
        series = max(series, ts_batch[0], key=len)
        m = max(m, len(ts_batch[0]))

    print(m)
    print(series.tolist())
    print(len(series))
