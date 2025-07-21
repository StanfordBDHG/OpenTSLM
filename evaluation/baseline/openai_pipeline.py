import os
from openai import OpenAI

class OpenAIPipeline:
    def __init__(self, model_name: str, api_key: str = None, temperature: float = 0.1, max_new_tokens: int = 1000):
        """
        A small wrapper around the new OpenAI v1 client for chat completions.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided via argument or OPENAI_API_KEY env var")
        # instantiate the v1 client
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature

    def __call__(self, prompt: str, max_new_tokens: int = 1000, return_full_text: bool = False):
        """
        Send a single-user-message chat completion request and return the generated text in HuggingFace pipeline format.
        """
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_new_tokens,
        )
        # Return in HuggingFace pipeline format
        return [{"generated_text": resp.choices[0].message.content.strip()}]
