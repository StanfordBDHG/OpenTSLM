# EmbedHealth

<div align="center">
  <img src="assets/stanford_biodesign_logo.png" alt="Stanford Biodesign" height="120">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/eth_cdhi_logo.png" alt="ETH Centre for Digital Health Interventions" height="120">
</div>

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/StanfordBDHG/EmbedHealth.git
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

4. **Download the Model**  
   ```bash
   python scripts/download_model.py --model llama-3.2-1b
   ```

## Multi-stage training (Curriculum)

EmbedHealth uses curriculum learning with progressive training stages:

### Training Stages
1. **Stage 1 (MCQ)**: Multiple choice questions on time series data
2. **Stage 2 (Captioning)**: Generate detailed captions for time series

### Quick Start
```bash
# Run full curriculum with EmbedHealthFlamingo
python curriculum_learning.py --model EmbedHealthFlamingo

# Run full curriculum with EmbedHealthSP
python curriculum_learning.py --model EmbedHealthSP

# Run only MCQ stage
python curriculum_learning.py --model EmbedHealthFlamingo --stages stage1_mcq

# Run only captioning stage
python curriculum_learning.py --model EmbedHealthFlamingo --stages stage2_captioning

# Specify device
python curriculum_learning.py --model EmbedHealthFlamingo --device cuda
```

### Command Line Arguments

- `--model`: Model type (`EmbedHealthSP` or `EmbedHealthFlamingo`)
- `--stages`: Stages to run (`stage1_mcq`, `stage2_captioning`, or both)
- `--device`: Device to use (`cuda`, `mps`, `cpu`)

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
