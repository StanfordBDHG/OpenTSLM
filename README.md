# OpenTSLM

<div align="center">
  <img src="assets/stanford_biodesign_logo.png" alt="Stanford Biodesign" height="120">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/eth_cdhi_logo.png" alt="ETH Centre for Digital Health Interventions" height="120">
</div>

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/StanfordBDHG/OpenTSLM.git --recurse-submodules
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## LLM Setup

OpenTSLM is designed to work with Llama and Gemma models, with Llama 3.2 1B as the default. These models are stored in Hugging Face repositories which may require access permissions. Follow these steps to gain access and download:

1. **Request Access (for Llama models)**  
   Visit the Llama model repository (e.g., https://huggingface.co/meta-llama/Llama-3.2-1B) or Gemma models repository (https://huggingface.co/google/gemma-3-270m) and request access from Meta.

2. **Authenticate with Hugging Face**  
   Log in to your Hugging Face account and configure the CLI:

   ```bash
   huggingface-cli login
   ```

3. **Create an API Token**
   - Go to your Hugging Face settings: https://huggingface.co/settings/tokens
   - Generate a new token with `read` scope.
   - Copy the token for CLI login.

### Supported Models

OpenTSLM has been tested and works with the following models:

**Llama Models:**

- **meta-llama/Llama-3.2-1B** (default)
- **meta-llama/Llama-3.2-3B**

**Gemma Models:**

- **google/gemma-3-270m**
- **google/gemma-3-1b-pt**

Other variants may work but have not been extensively tested.

## Multi-stage training (Curriculum)

OpenTSLM uses curriculum learning with progressive training stages:

### Training Stages

1. **Stage 1 (MCQ)**: Multiple choice questions on time series data (TSQA dataset)
2. **Stage 2 (Captioning)**: Generate detailed captions for time series (M4 dataset)
3. **Stage 3 (CoT)**: Chain-of-thought reasoning on human activity recognition (HAR dataset)
4. **Stage 4 (Sleep CoT)**: Chain-of-thought reasoning on sleep stage classification (SleepEDF dataset)
5. **Stage 5 (ECG CoT)**: Chain-of-thought reasoning on ECG question answering (ECG QA dataset)

> **⚠️ MPS/CUDA Compatibility Warning:**
>
> If you are using Apple's MPS (Metal Performance Shaders) backend (e.g., on Mac with Apple Silicon), you may encounter issues with training or inference. **Checkpoints trained with CUDA (NVIDIA GPUs) may not yield good results or may not be fully compatible when loaded and run on MPS.** For best results, use the same device type (CUDA or MPS) for both training and inference. CUDA is preferred in general.

### Quick Start

```bash
# Run full curriculum with OpenTSLMFlamingo
python curriculum_learning.py --model OpenTSLMSP

# Run full curriculum with OpenTSLMSP
python curriculum_learning.py --model OpenTSLMFlamingo

# Run specific stages
python curriculum_learning.py --model OpenTSLMFlamingo --stages stage1_mcq
python curriculum_learning.py --model OpenTSLMFlamingo --stages stage2_captioning
python curriculum_learning.py --model OpenTSLMFlamingo --stages stage3_cot
python curriculum_learning.py --model OpenTSLMFlamingo --stages stage4_sleep_cot
python curriculum_learning.py --model OpenTSLMFlamingo --stages stage5_ecg_cot

# Run multiple stages
python curriculum_learning.py --model OpenTSLMFlamingo --stages stage1_mcq stage2_captioning stage3_cot

# Specify device
python curriculum_learning.py --model OpenTSLMFlamingo --device cuda

# Use different models
python curriculum_learning.py --model OpenTSLMFlamingo --llm_id meta-llama/Llama-3.2-1B
python curriculum_learning.py --model OpenTSLMFlamingo --llm_id google/gemma-3-270m

# Run only evaluation
python curriculum_learning.py --model OpenTSLMFlamingo --eval_only
```

### Command Line Arguments

- `--model`: Model type (`OpenTSLMSP` or `OpenTSLMFlamingo`)
- `--stages`: Stages to run (any combination of: `stage1_mcq`, `stage2_captioning`, `stage3_cot`, `stage4_sleep_cot`, `stage5_ecg_cot`)
- `--device`: Device to use (`cuda`, `mps`, `cpu`)
- `--eval_only`: Run evaluation only (requires an existing checkpoint for the stage)
- `--llm_id`: Model ID (default: `meta-llama/Llama-3.2-1B`, supports Llama and Gemma models)
- `--batch_size`: Batch size for training
- `--gradient_checkpointing`: Enable gradient checkpointing for memory efficiency
- `--verbose`: Enable verbose logging

## 🚀 Using Pre-trained Models

EmbedHealth provides a factory class called `OpenTSLM` for easily loading pre-trained models from Hugging Face Hub. The `load_pretrained` method automatically detects the model type and returns the appropriate model instance.

### Quick Usage

```python
# Load model
model = OpenTSLM.load_pretrained("OpenTSLM/repo-id")

# Create prompt with raw time series data (normalization handled automatically)
prompt = FullPrompt(
    pre_prompt=TextPrompt("You are an expert in HAR analysis."),
    text_time_series_prompt_list=[
        TextTimeSeriesPrompt("X-axis accelerometer", [2.34, 2.34, 7.657, 3.21, -1.2])
    ],
    post_prompt=TextPrompt("What activity is this?")
)

# Generate response
output = model.eval_prompt(prompt, normalize=True)
print(output)
```

### Repository Naming Convention

- Repository IDs ending with `-sp` will load and return `EmbedHealthSP` models
- Repository IDs ending with `-flamingo` will load and return `EmbedHealthFlamingo` models

### Features

- **Automatic Model Detection**: Detects model type from repository name suffix
- **Device Auto-detection**: Automatically selects best available device (CUDA > MPS > CPU)
- **Automatic Normalization**: Time series data is automatically normalized (zero mean, unit variance) during inference
- **Hugging Face Integration**: Downloads models directly from Hugging Face Hub

## 📁 Results Structure

During training, the scripts creates a structured results directory:

```
results/
├── {llm_id}/
│   ├── OpenTSLMSP/
│   │   ├── stage1_mcq/
│   │   │   ├── checkpoints/
│   │   │   │   ├── best_model.pt
│   │   │   │   └── loss_history.txt
│   │   │   └── results/
│   │   │       ├── test_predictions.jsonl
│   │   │       └── metrics.json
│   │   ├── stage2_captioning/
│   │   │   ├── checkpoints/
│   │   │   │   ├── best_model.pt
│   │   │   │   └── loss_history.txt
│   │   │   └── results/
│   │   │       ├── test_predictions.jsonl
│   │   │       └── metrics.json
│   │   ├── stage3_cot/
│   │   │   ├── checkpoints/
│   │   │   │   ├── best_model.pt
│   │   │   │   └── loss_history.txt
│   │   │   └── results/
│   │   │       ├── test_predictions.jsonl
│   │   │       └── metrics.json
│   │   ├── stage4_sleep_cot/
│   │   │   ├── checkpoints/
│   │   │   │   ├── best_model.pt
│   │   │   │   └── loss_history.txt
│   │   │   └── results/
│   │   │       ├── test_predictions.jsonl
│   │   │       └── metrics.json
│   │   ├── stage5_ecg_cot/
│   │   │   ├── checkpoints/
│   │   │   │   ├── best_model.pt
│   │   │   │   └── loss_history.txt
│   │   │   └── results/
│   │   │       ├── test_predictions.jsonl
│   │   │       └── metrics.json
│   │   └── curriculum_results.json
│   └── OpenTSLMFlamingo/
│       ├── stage1_mcq/
│       ├── stage2_captioning/
│       ├── stage3_cot/
│       ├── stage4_sleep_cot/
│       ├── stage5_ecg_cot/
│       └── curriculum_results.json
```

Each stage automatically loads the best model from the previous stage, ensuring proper curriculum progression. Results are organized by model ID (sanitized), then by model type and stage. The `{llm_id}` directory name is derived from the `--llm_id` parameter (e.g., `meta-llama/Llama-3.2-1B` becomes `Llama3_2_1B`, `google/gemma-3-1b-pt` becomes `gemma_3_1b_pt`).
