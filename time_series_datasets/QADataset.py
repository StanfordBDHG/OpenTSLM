
from prompt.full_prompt import FullPrompt
from prompt.prompt_with_answer import PromptWithAnswer
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from torch.utils.data import Dataset


class QADataset(Dataset):
    def __init__(self, dataset: Dataset, question: str):
        self.dataset = dataset
        self.question = question

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        timeseries, output = self.dataset[idx]

        # print(timeseries)
        # print(str(output))
        # print("jo")
        # print(
        #     PromptWithAnswer(
        #         "",
        #         [TextTimeSeriesPrompt(self.question, timeseries[0])],
        #         "",
        #         str(output),
        #     ).to_dict()
        # )
        # print("huh")
        return PromptWithAnswer(
            "",
            [TextTimeSeriesPrompt(self.question, timeseries[0])],
            "",
            str(output),
        ).to_dict()
