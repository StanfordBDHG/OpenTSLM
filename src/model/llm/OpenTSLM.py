import torch
from typing import List, Dict, Optional
from huggingface_hub import hf_hub_download
from transformers import AutoConfig

from .TimeSeriesLLM import TimeSeriesLLM
from .EmbedHealthSP import EmbedHealthSP
from .EmbedHealthFlamingo import EmbedHealthFlamingo
from prompt.full_prompt import FullPrompt


class OpenTSLM:
    """
    Unified interface for loading and using EmbedHealth models from Hugging Face Hub.

    Automatically detects model type based on repository ID suffix:
    - Repository IDs ending with "-sp" load EmbedHealthSP models
    - Repository IDs ending with "-flamingo" load EmbedHealthFlamingo models

    Args:
        repo_id: Hugging Face repository ID (e.g., "OpenTSLM/gemma-3-270m-pt-sleep-flamingo")
        device: Device to load the model on (default: auto-detect)
        cache_dir: Directory to cache downloaded models (optional)
        **kwargs: Additional arguments passed to the underlying model

    Example:
        >>> # Load an model
        >>> model = OpenTSLM("<hugging_face_repo_id>") # available models can be found under https://huggingface.co/OpenTSLM
        >>> # Generate predictions
        >>> from prompt.full_prompt import FullPrompt
        >>> prompt = FullPrompt(...)
        >>> response = model.eval_prompt(prompt)
    """

    def __init__(
        self,
        repo_id: str,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        self.repo_id = repo_id
        self.device = self._get_device(device)
        self.cache_dir = cache_dir

        # Detect model type from repository ID
        self.model_type = self._detect_model_type(repo_id)

        # Download and load the model
        self.model = self._load_model(**kwargs)

    def _get_device(self, device: Optional[str]) -> str:
        """Auto-detect device if not specified."""
        if device is not None:
            return device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _detect_model_type(self, repo_id: str) -> str:
        """Detect model type from repository ID suffix."""
        if repo_id.endswith("-sp"):
            return "sp"
        elif repo_id.endswith("-flamingo"):
            return "flamingo"
        else:
            raise ValueError(
                f"Repository ID '{repo_id}' must end with either '-sp' or '-flamingo' "
                f"to indicate the model type."
            )

    def _download_model_files(self) -> str:
        """Download model checkpoint from Hugging Face Hub."""
        try:
            # Download the main model checkpoint file
            checkpoint_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="best_model.pt",
                cache_dir=self.cache_dir,
                local_files_only=False,
            )
            print(f"✅ Downloaded model checkpoint from {self.repo_id}")
            return checkpoint_path

        except Exception as e:
            # Try alternative filename
            try:
                checkpoint_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename="pytorch_model.bin",
                    cache_dir=self.cache_dir,
                    local_files_only=False,
                )
                print(f"✅ Downloaded model checkpoint from {self.repo_id}")
                return checkpoint_path
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to download model from {self.repo_id}. "
                    f"Tried 'best_model.pt' and 'pytorch_model.bin'. "
                    f"Original error: {e}, Secondary error: {e2}"
                )

    def _get_base_llm_id(self) -> str:
        """Get the base LLM ID from static mapping based on repository ID pattern."""
        repo_name = self.repo_id.split("/")[-1] if "/" in self.repo_id else self.repo_id

        # Extract base model from repository name pattern
        if repo_name.startswith("llama-3.2-3b"):
            return "meta-llama/Llama-3.2-3B"
        elif repo_name.startswith("llama-3.2-1b"):
            return "meta-llama/Llama-3.2-1B"
        elif repo_name.startswith("gemma-3-1b"):
            return "google/gemma-3-1b"
        elif repo_name.startswith("gemma-3-270m"):
            return "google/gemma-3-270m"
        else:
            # Raise exception if pattern doesn't match
            raise ValueError(
                f"Unable to determine base LLM ID from repository name '{repo_name}'. "
                f"Repository name must start with one of: 'llama-3.2-3b', 'llama-3.2-1b', "
                f"'gemma-3-1b', or 'gemma-3-270m'."
            )

    def _load_model(self, **kwargs) -> TimeSeriesLLM:
        """Load the appropriate model class and checkpoint."""
        checkpoint_path = self._download_model_files()
        base_llm_id = self._get_base_llm_id()

        print(f"🚀 Loading {self.model_type.upper()} model...")
        print(f"   Repository: {self.repo_id}")
        print(f"   Base LLM: {base_llm_id}")
        print(f"   Device: {self.device}")

        # Instantiate the appropriate model class
        if self.model_type == "sp":
            model = EmbedHealthSP(llm_id=base_llm_id, device=self.device, **kwargs)
        elif self.model_type == "flamingo":
            model = EmbedHealthFlamingo(
                device=self.device, llm_id=base_llm_id, **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Load the checkpoint
        model.load_from_file(checkpoint_path)
        model.eval()

        print(f"✅ {self.model_type.upper()} model loaded successfully!")
        return model

    # Proxy methods to delegate to the underlying model
    def generate(
        self, batch: List[Dict[str, any]], max_new_tokens: int = 50, **generate_kwargs
    ) -> List[str]:
        """Generate text for a batch of inputs."""
        return self.model.generate(batch, max_new_tokens, **generate_kwargs)

    def compute_loss(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """Compute loss for a batch of inputs."""
        return self.model.compute_loss(batch)

    def eval_prompt(self, prompt: FullPrompt, max_new_tokens: int = 30000) -> str:
        """Evaluate a single prompt and return the generated text."""
        return self.model.eval_prompt(prompt, max_new_tokens)

    def get_eos_token(self) -> str:
        """Get the end-of-sequence token."""
        return self.model.get_eos_token()

    def to(self, device: str):
        """Move model to device."""
        self.device = device
        self.model.to(device)
        return self

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        """Set model to training mode."""
        self.model.train(mode)
        return self

    def parameters(self):
        """Get model parameters."""
        return self.model.parameters()

    def named_parameters(self):
        """Get named model parameters."""
        return self.model.named_parameters()

    def state_dict(self):
        """Get model state dictionary."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        """Load model state dictionary."""
        return self.model.load_state_dict(state_dict, strict)

    def __getattr__(self, name):
        """Delegate any other attribute access to the underlying model."""
        return getattr(self.model, name)

    def __repr__(self):
        return f"OpenTSLM(repo_id='{self.repo_id}', model_type='{self.model_type}', device='{self.device}')"
