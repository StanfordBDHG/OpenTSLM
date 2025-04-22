# data.py
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import ast

def load_tsqa(split='train', max_samples=None, val_frac=0.1, seed=42):
    """
    Load ChengsenWang/TSQA and split off a validation subset.
    Args:
        split (str): 'train' or 'validation'
        max_samples (int, optional): limit on total samples (applied after splitting)
        val_frac (float): fraction of data to use for validation
        seed (int): random seed for splitting
    Returns:
        Dataset: with columns ['ts','question','answer']
    """
    # 1) load the single 'train' split
    ds = load_dataset("ChengsenWang/TSQA", split="train")

    # 2) train/test split
    splits = ds.train_test_split(test_size=val_frac, seed=seed)
    if split == 'train':
        ds = splits['train']
    elif split in ('validation', 'val'):
        ds = splits['test']
    else:
        raise ValueError(f"Unknown split '{split}'; choose 'train' or 'validation'")

    # 3) optional sample limit
    if max_samples is not None and max_samples < len(ds):
        ds = ds.select(list(range(max_samples)))

    # 4) preprocessing
    def preprocess(ex):
        series = torch.tensor(ast.literal_eval(ex['Series']), dtype=torch.float32)
        m, s = series.mean(), series.std()
        series = (series - m) / (s + 1e-8)
        return {
            'ts': series,
            'question': ex['Question'],
            'answer': ex['Answer']
        }

    ds = ds.map(preprocess)
    ds.set_format(type='torch', columns=['ts','question','answer'])
    return ds


def collate_fn(batch, patch_size=4):
    # ceil-pad to nearest multiple of patch_size
    max_len = max(ex['ts'].size(0) for ex in batch)
    max_len = ((max_len + patch_size - 1) // patch_size) * patch_size

    ts_list, qs, ans = [], [], []
    for ex in batch:
        ts = ex['ts']
        if ts.size(0) < max_len:
            pad = max_len - ts.size(0)
            ts = torch.nn.functional.pad(ts, (0, pad), 'constant', 0)
        else:
            ts = ts[:max_len]
        ts_list.append(ts)
        qs.append(ex['question'] + "\nAnswer:")
        ans.append(ex['answer'])

    ts_batch = torch.stack(ts_list)
    return ts_batch, qs, ans


def get_loader(split='train', batch_size=8, patch_size=4, max_samples=None):
    """
    Returns a DataLoader over TSQA.
    split: 'train' or 'validation'
    """
    ds = load_tsqa(split=split, max_samples=max_samples)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split=='train'),
        collate_fn=lambda b: collate_fn(b, patch_size)
    )
