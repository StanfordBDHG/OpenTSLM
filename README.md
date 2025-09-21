# EmbedHealth

<div align="center">
  <img src="assets/stanford_biodesign_logo.png" alt="Stanford Biodesign" height="120">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/eth_cdhi_logo.png" alt="ETH Centre for Digital Health Interventions" height="120">
</div>

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/StanfordBDHG/EmbedHealth.git --recurse-submodules
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## LLM Setup

EmbedHealth uses the Llama 3.2 1B model, which is stored in a Hugging Face repository which is restricted. Follow these steps to gain access and download:

1. **Request Access**  
   Submit a request to the repository administrator to gain read access.

2. **Authenticate with Hugging Face**  
   Log in to your Hugging Face account and configure the CLI:

   ```bash
   huggingface-cli login
   ```

3. **Create an API Token**
   - Go to your Hugging Face settings: https://huggingface.co/settings/tokens
   - Generate a new token with `read` scope.
   - Copy the token for CLI login.

## Multi-stage training (Curriculum)

EmbedHealth uses curriculum learning with progressive training stages:

### Training Stages

1. **Stage 1 (MCQ)**: Multiple choice questions on time series data
2. **Stage 2 (Captioning)**: Generate detailed captions for time series

> **⚠️ MPS/CUDA Compatibility Warning:**
>
> If you are using Apple's MPS (Metal Performance Shaders) backend (e.g., on Mac with Apple Silicon), you may encounter issues with training or inference. **Checkpoints trained with CUDA (NVIDIA GPUs) may not yield good results or may not be fully compatible when loaded and run on MPS.** For best results, use the same device type (CUDA or MPS) for both training and inference. CUDA is preferred in general.

### Quick Start

```bash
# Run full curriculum with EmbedHealthFlamingo
python curriculum_learning.py --model EmbedHealthSP

# Run full curriculum with EmbedHealthSP
python curriculum_learning.py --model EmbedHealthFlamingo

# Run only MCQ stage
python curriculum_learning.py --model EmbedHealthFlamingo --stages stage1_mcq

# Run only captioning stage
python curriculum_learning.py --model EmbedHealthFlamingo --stages stage2_captioning

# Specify device
python curriculum_learning.py --model EmbedHealthFlamingo --device cuda

# Run only evaluation
python curriculum_learning.py --model EmbedHealthFlamingo --eval_only
```

### Command Line Arguments

- `--model`: Model type (`EmbedHealthSP` or `EmbedHealthFlamingo`)
- `--stages`: Stages to run (`stage1_mcq`, `stage2_captioning`, or both)
- `--device`: Device to use (`cuda`, `mps`, `cpu`)
- `--eval_only`: Run evaluation only (requires an existing checkpoint for the stage)

## 🚀 Using Pre-trained Models

EmbedHealth provides a factory class called `OpenTSLM` for easily loading pre-trained models from Hugging Face Hub. The `load_pretrained` method automatically detects the model type and returns the appropriate model instance.

### Quick Usage

```python
from model.llm.OpenTSLM import OpenTSLM
from prompt.full_prompt import FullPrompt
import torch

# Load a pre-trained model using the factory method
# Available models can be found at: https://huggingface.co/OpenTSLM
# The method returns either an EmbedHealthSP or EmbedHealthFlamingo instance
# depending on the repository ID
model = OpenTSLM.load_pretrained("<hugging_face_repo_id>")

# Create a prompt with time series data
prompt = FullPrompt(
    pre_prompt="Analyze this time series:",
    time_series_text=["Heart rate measurement"],
    time_series=[torch.randn(100)],  # Your time series data
    post_prompt="What does this indicate?"
)

# Generate response using the loaded model
response = model.eval_prompt(prompt)
print(response)
```

### Repository Naming Convention

- Repository IDs ending with `-sp` will load and return `EmbedHealthSP` models
- Repository IDs ending with `-flamingo` will load and return `EmbedHealthFlamingo` models

### Features

- **Automatic Model Detection**: Detects model type from repository name suffix
- **Device Auto-detection**: Automatically selects best available device (CUDA > MPS > CPU)
- **Hugging Face Integration**: Downloads models directly from Hugging Face Hub

## 📁 Results Structure

During training, the scripts creates a structured results directory:

```
results/
├── EmbedHealthSP/
│   ├── stage1_mcq/
│   │   ├── checkpoints/
│   │   │   └── best_model.pt
│   │   └── results/
│   │       ├── test_predictions.jsonl
│   │       └── metrics.json
│   ├── stage2_captioning/
│   │   ├── checkpoints/
│   │   │   └── best_model.pt
│   │   └── results/
│   │       ├── test_predictions.jsonl
│   │       └── metrics.json
│   └── curriculum_results.json
└── EmbedHealthFlamingo/
    ├── stage1_mcq/
    │   ├── checkpoints/
    │   │   └── best_model.pt
    │   └── results/
    │       ├── test_predictions.jsonl
    │       └── metrics.json
    ├── stage2_captioning/
    │   ├── checkpoints/
    │   │   └── best_model.pt
    │   └── results/
    │       ├── test_predictions.jsonl
    │       └── metrics.json
    └── curriculum_results.json
```

Each stage automatically loads the best model from the previous stage, ensuring proper curriculum progression. Results are saved in `results/{model_name}/{stage_name}/`.
