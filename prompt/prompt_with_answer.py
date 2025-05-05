from typing import List
from prompt.text_prompt import TextPrompt
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from .full_prompt import FullPrompt


class PromptWithAnswer:
    """
    A wrapper for a FullPrompt + a single answer string,
    intended for training (loss computation).
    """

    def __init__(
        self,
        pre_prompt: TextPrompt,
        text_time_series_prompt_list: List[TextTimeSeriesPrompt],
        post_prompt: TextPrompt,
        answer: str,
    ):
        # assert isinstance(full_prompt, FullPrompt), "Prompt must be a FullPrompt."
        assert isinstance(answer, str), "Answer must be a string."

        self.pre_prompt = pre_prompt
        self.text_time_series_prompt_texts = list(
            map(lambda x: x.get_text(), text_time_series_prompt_list)
        )
        self.text_time_series_prompt_time_series = list(
            map(lambda x: x.get_time_series(), text_time_series_prompt_list)
        )
        self.post_prompt = post_prompt
        self.answer = answer

    def to_dict(self):
        return {
            "pre_prompt": self.pre_prompt,
            "text_time_series_prompt_texts": self.text_time_series_prompt_texts,
            "text_time_series_prompt_time_series": self.text_time_series_prompt_time_series,
            "post_prompt": self.post_prompt,
            "answer": self.answer,
        }
