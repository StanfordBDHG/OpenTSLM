from abc import ABC, abstractmethod


class Prompt(ABC):
    @abstractmethod
    def get_text(self):
        pass
