import sys
import os

# Ensure that the parent directory (containing datasets/) is visible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from time_series_datasets.m4 import get_m4_loader


def test_loader():
    loader = get_m4_loader(frequency="Monthly", split="train", batch_size=2)
    for series_batch, ids in loader:
        print(series_batch.shape)
        print(ids)
        break


if __name__ == "__main__":
    test_loader()
