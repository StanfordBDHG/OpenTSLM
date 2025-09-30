# OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data
[![DOI](https://img.shields.io/badge/DOI-10.13140/RG.2.2.14827.60963-blue.svg)](https://doi.org/10.13140/RG.2.2.14827.60963)

Large Language Models (LLMs) have emerged as powerful tools for interpreting multimodal data (e.g., images, audio, text), often surpassing specialized models. In medicine, they hold particular promise for synthesizing large volumes of clinical information into actionable insights and patient-facing digital health applications.  Yet, a major limitation remains their inability to handle time series data. To overcome this gap, we present OpenTSLM, a family of Time Series Language Models (TSLMs) created by integrating time series as a native modality to pretrained Large Language Models, enabling natural-language prompting and reasoning over multiple time series of any length... **[🔗 Read the full paper](https://www.researchgate.net/publication/395975374_OpenTSLM_Time-Series_Language_Models_for_Reasoning_over_Multivariate_Medical_Text-_and_Time-Series_Data)**  

<p align="center">
  <img src="assets/schematic_overview_3.png" alt="Schematic Overview" width="100%">
</p>


## Examples
OpenTSLM models can reason over multiple time series of any length at once, generating findings, captions, and rationales in natural language. We tested these models across a wide range of tasks spanning Human Activity Recognition (HAR) from 3-axis acceleration data, sleep stating from EEG readings, 12-lead ECG question answering, and time series captioning. Some examples are shown below, more are available in the paper.
<p align="center">
  <img src="assets/ecg_rationale.png" alt="ECG Rationale" width="32%">
  <img src="assets/har_rationale.png" alt="HAR Rationale" width="32%">
    <img src="assets/m4_caption.png" alt="M4 Caption" width="34%">

</p>


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


## Contributors
This work was made possible only by joint effort of many awesome collaborators:

- Patrick Langer (Stanford, ETH Zurich)
- Thomas Kaar (Stanford, TUM)
- Max Rosenblattl (Stanford, TUM)
- Maxwell A. Xu (Google Research, University of Illinois Urbana-Champaign)
- Winnie Chow (Stanford)
- Martin Maritsch (Amazon)
- Daniel McDuff (Google Research, University Washington)
- Elgar Fleisch (ETH Zurich)
- Filipe Barata (ETH Zurich)
- Paul Schmiedmayer (ETH Zurich)

<div align="center">
  <img src="assets/stanford_biodesign_logo.png" alt="Stanford Biodesign" height="160">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/eth_cdhi_logo.png" alt="ETH Centre for Digital Health Interventions" height="160">
</div>

## Contributing

Contributions to this project are welcome. Please make sure to read the [contribution guidelines](https://github.com/StanfordSpezi/.github/blob/main/CONTRIBUTING.md) and the [contributor covenant code of conduct](https://github.com/StanfordSpezi/.github/blob/main/CODE_OF_CONDUCT.md) first.


## License

This project is licensed under the MIT License. 
