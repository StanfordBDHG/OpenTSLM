from .prompt import Prompt
from typing import List


class FullPrompt(object):
    """
    A wrapper for a sequence of Prompt objects (TextPrompt or TextTimeSeriesPrompt),
    intended for inference.
    """

    def __init__(self, prompts: List[Prompt]):
        if not all(isinstance(p, Prompt) for p in prompts):
            raise TypeError("All elements must be subclasses of Prompt.")
        self.prompts = prompts

    def get_texts(self) -> List[str]:
        return [p.get_text() for p in self.prompts]
