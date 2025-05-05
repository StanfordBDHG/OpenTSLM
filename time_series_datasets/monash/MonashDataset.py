import logging
import os

import numpy as np

from tqdm.auto import tqdm


from time_series_datasets.monash.monash_utils import (
    download_and_extract_monash_ucr,
    load_from_tsfile_to_dataframe,
)


class MonashDataset:
    def __init__(self, _data_dir=None, data_name=None):
        self.logger = logging.getLogger(__name__)
        self._data_dir = _data_dir
        self.data_name = data_name

        if not os.path.exists(_data_dir):
            download_and_extract_monash_ucr(destination="monash_datasets")
        dataset_file = os.path.join(_data_dir, f"{data_name}.ts")

        # Load the dataset
        print(f"Loading dataset: {data_name}")
        X, y = load_from_tsfile_to_dataframe(dataset_file, return_separate_X_and_y=True)

        # Convert sktime format to numpy arrays
        X_np = np.stack([x.to_numpy() for x in X.iloc[:, 0]])
        y_np = np.array(y)

        self.feature = X_np
        self.target = y_np

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        item = self.feature[idx]
        label = self.target[idx]
        item = np.expand_dims(item, axis=0)

        return item, label


if __name__ == "__main__":
    loader = MonashDataset(
        _data_dir="monash_datasets", data_name="IEEEPPG/IEEEPPG_TRAIN"
    )

    prog = tqdm(loader)

    for item, label in prog:
        print("item", item, "; label", label)
