from .prompt import Prompt


class TextPrompt(Prompt):
    def __init__(self, text: str):
        assert isinstance(text, str), "Text must be a string!"
        self.__text = text

    def get_text(self) -> str:
        return self.__text
