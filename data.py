from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import ast

def load_tsqa(split='train'):
    # Load the ChengsenWang/TSQA dataset
    dataset = load_dataset("ChengsenWang/TSQA", split=split)

    def preprocess(example):
        # The TSQA CSV uses columns: 'Series', 'Question', 'Answer'
        # Parse the 'Series' string into a list of floats
        series_str = example['Series']
        series_list = ast.literal_eval(series_str)
        ts = torch.tensor(series_list, dtype=torch.float32)
        question = example['Question']
        answer = example['Answer']
        return {'ts': ts, 'question': question, 'answer': answer}

    dataset = dataset.map(preprocess)
    dataset.set_format(type='torch', columns=['ts', 'question', 'answer'])
    return dataset

def collate_fn(batch, patch_size=4):
    # Ensure each series is padded/truncated to length divisible by patch_size
    max_len = max(item['ts'].size(0) for item in batch)
    max_len = (max_len // patch_size) * patch_size

    ts_batch = []
    prompts = []
    answers = []
    for item in batch:
        ts = item['ts']
        if ts.size(0) < max_len:
            pad_len = max_len - ts.size(0)
            ts = torch.nn.functional.pad(ts, (0, pad_len), 'constant', 0)
        else:
            ts = ts[:max_len]
        ts_batch.append(ts)
        prompts.append(item['question'] + "\nAnswer:")
        answers.append(item['answer'])

    ts_batch = torch.stack(ts_batch)
    return ts_batch, prompts, answers

def get_loader(split='train', batch_size=8, patch_size=4):
    dataset = load_tsqa(split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split=='train'),
        collate_fn=lambda batch: collate_fn(batch, patch_size)
    )
    return loader
