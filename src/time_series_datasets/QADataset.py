from abc import ABC, abstractmethod
from typing import List, Literal, Tuple
from prompt.prompt_with_answer import PromptWithAnswer
from prompt.text_prompt import TextPrompt
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from torch.utils.data import Dataset


class QADataset(Dataset, ABC):
    def __init__(self, split: Literal["train", "test", "validation"], EOS_TOKEN: str):
        self.EOS_TOKEN = EOS_TOKEN
        if not hasattr(self.__class__, "loaded"):
            train, val, test = self._load_splits()

            self.__class__._train_dataset = list(map(self._format_sample, train))
            self.__class__._validation_dataset = list(map(self._format_sample, val))
            self.__class__._test_dataset = list(map(self._format_sample, test))

            self.__class__.loaded = True

        match split:
            case "train":
                self.dataset = self.__class__._train_dataset
            case "validation":
                self.dataset = self.__class__._validation_dataset
            case "test":
                self.dataset = self.__class__._test_dataset
            case _:
                raise RuntimeError(
                    "Split must be a literal of 'train', 'training', or 'validation'"
                )

    @abstractmethod
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        pass

    @abstractmethod
    def _get_answer(self, row) -> str:
        pass

    @abstractmethod
    def _get_pre_prompt(self, row) -> str:
        pass

    @abstractmethod
    def _get_post_prompt(self, row) -> str:
        pass

    @abstractmethod
    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        pass

    def _format_sample(self, row):
        answer = self._get_answer(row)
        if not answer.endswith(self.EOS_TOKEN):
            answer += self.EOS_TOKEN

        return PromptWithAnswer(
            TextPrompt(self._get_pre_prompt(row).strip()),
            self._get_text_time_series_prompt_list(row),
            TextPrompt(self._get_post_prompt(row).strip()),
            answer.strip(),
        ).to_dict()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
