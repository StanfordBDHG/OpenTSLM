from torch.utils.data import Dataset


class QADataset(Dataset):
    def __init__(self, dataset: Dataset, question: str):
        self.dataset = dataset
        self.question = question

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        timeseries, output = self.dataset[idx]

        return timeseries, self.question, output
