# Curriculum Learning for EmbedHealth Models

This implementation provides a curriculum learning framework for training EmbedHealth models (EmbedHealthSP and EmbedHealthFlamingo) on time-series tasks. The curriculum follows a progressive learning approach, starting with simpler tasks and gradually moving to more complex ones.

## 🎯 Overview

The curriculum learning system trains models in stages:

1. **Stage 1 (MCQ)**: Multiple Choice Question Answering
   - Dataset: TSQADataset (multiple choice questions on time series)
   - Task: Answer questions about time series data
   - Metric: Accuracy (exact match)

2. **Stage 2 (Captioning)**: Time Series Caption Generation
   - Dataset: M4QADataset (time series with captions)
   - Task: Generate detailed captions for time series
   - Metric: Test loss

## 🚀 Quick Start

### Basic Usage

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

The system creates a structured results directory:

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

## 🔧 Implementation Details

### CurriculumTrainer Class

The main class that handles curriculum learning with a simple interface:

```python
from curriculum_learning import CurriculumTrainer

# Initialize trainer
trainer = CurriculumTrainer("EmbedHealthFlamingo")

# Run specific stages
results = trainer.run_curriculum(["stage1_mcq", "stage2_captioning"])

# Or run individual stages directly
mcq_results = trainer.stage1_mcq()
captioning_results = trainer.stage2_captioning()
```

### Key Features

1. **Simple Stage Interface**: Just call `stage1_mcq()` or `stage2_captioning()`
2. **Shared Training Logic**: All training logic is abstracted into `_train_stage()`
3. **Model Abstraction**: Works with both EmbedHealthSP and EmbedHealthFlamingo
4. **Checkpoint Management**: Saves and loads best models for each stage
5. **Previous Stage Loading**: Automatically loads best model from previous stage
6. **Stage-Specific Metrics**: 
   - MCQ: Accuracy (exact match)
   - Captioning: Test loss
7. **Early Stopping**: Prevents overfitting with configurable patience
8. **Resume Training**: Can resume from checkpoints if interrupted

### Training Process

Each stage follows this process:

1. **Previous Stage Loading**: Load best model from previous stage (if exists)
2. **Data Loading**: Load train/validation/test splits
3. **Model Initialization**: Set up optimizer and scheduler
4. **Training Loop**: 
   - Train for specified epochs
   - Validate after each epoch
   - Save best model based on validation loss
   - Early stopping if no improvement
5. **Evaluation**: Test on test set and save results
6. **Checkpoint**: Save model state for next stage

### Previous Stage Loading

The system automatically loads the best model from the previous stage when starting a new stage. This ensures proper curriculum progression:

```
🚀 Starting Captioning (M4) Training with EmbedHealthFlamingo
============================================================
📂 Loading best model from stage1_mcq:
   Achieved at epoch: 15
   Validation loss: 0.1234
   accuracy: 0.8500
   test_loss: 0.2345

🆕 Starting with fresh model (no previous stage found)
```

**Safety Checks:**
- Verifies that previous stage metrics file exists
- Verifies that previous stage checkpoint exists
- Provides clear warnings if files are missing
- Gracefully handles missing previous stages

## 📊 Metrics and Evaluation

### MCQ Stage (Stage 1)
- **Accuracy**: Percentage of exactly correct answers
- **Test Loss**: Cross-entropy loss on test set
- **Output**: JSONL file with predictions and gold answers

### Captioning Stage (Stage 2)
- **Test Loss**: Cross-entropy loss on test set
- **Output**: JSONL file with generated captions and gold captions

## 🔄 Adding New Stages

To add a new stage to the curriculum:

1. **Create Dataset**: Implement a new dataset class inheriting from `QADataset`
2. **Add Stage Method**: Add `stage3_newtask()` method to `CurriculumTrainer`
3. **Update Results Directory**: Add stage directory creation in `_create_results_dir()`
4. **Update Stage Order**: Add stage to the stage order list in `_load_previous_stage_model()`
5. **Update Main Method**: Add stage to the curriculum pipeline

Example:

```python
def stage3_newtask(self) -> Dict[str, Any]:
    """Stage 3: New Task."""
    return self._train_stage(
        stage="stage3_newtask",
        stage_name="New Task",
        dataset_class=NewDataset,
        metric_func=lambda preds, golds: {"custom_metric": self._calculate_custom_metric(preds, golds)}
    )

# Update _create_results_dir method
def _create_results_dir(self):
    # ... existing code ...
    for stage in ["stage1_mcq", "stage2_captioning", "stage3_newtask"]:
        # ... create directories ...

# Update _load_previous_stage_model method
def _load_previous_stage_model(self, current_stage: str):
    stage_order = ["stage1_mcq", "stage2_captioning", "stage3_newtask"]
    # ... rest of method ...

# Update run_curriculum method
def run_curriculum(self, stages: List[str] = None):
    if stages is None:
        stages = ["stage1_mcq", "stage2_captioning", "stage3_newtask"]
    
    # Add stage handling
    elif stage == "stage3_newtask":
        results[stage] = self.stage3_newtask()
```

## 🛠️ Configuration

Training parameters are configured in `model_config.py`:

- `BATCH_SIZE`: Training batch size
- `NUM_EPOCHS`: Maximum training epochs
- `EARLY_STOP_PAT`: Early stopping patience
- `LR_ENCODER`: Learning rate for encoder
- `LR_PROJECTOR`: Learning rate for projector
- `GRAD_CLIP_NORM`: Gradient clipping norm
- `WARMUP_FRAC`: Warmup fraction for scheduler

## 📝 Example Usage

### Python Script Example

```python
from curriculum_learning import CurriculumTrainer

# Initialize trainer
trainer = CurriculumTrainer("EmbedHealthFlamingo", device="cuda")

# Run full curriculum
results = trainer.run_curriculum()

# Or run stages individually
mcq_metrics = trainer.stage1_mcq()
captioning_metrics = trainer.stage2_captioning()

# Print results
for stage, metrics in results.items():
    print(f"{stage}: {metrics}")
```

### Command Line Examples

```bash
# Quick test with MCQ only
python curriculum_learning.py --model EmbedHealthSP --stages stage1_mcq

# Full training on GPU
python curriculum_learning.py --model EmbedHealthFlamingo --device cuda

# Resume training (checkpoints are automatically loaded)
python curriculum_learning.py --model EmbedHealthFlamingo --stages stage2_captioning
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size in `model_config.py`
2. **Slow Training**: Use GPU if available (`--device cuda`)
3. **Checkpoint Loading**: Ensure checkpoint files exist in the correct directory
4. **Dataset Loading**: Verify dataset paths and dependencies
5. **Previous Stage Missing**: The system will warn if previous stage files are missing

### Debug Mode

Add debug prints to the training loop:

```python
# In curriculum_learning.py, add debug prints
print(f"Batch size: {len(batch)}")
print(f"Sample keys: {batch[0].keys()}")
```

## 📈 Monitoring Training

The system provides detailed progress information:

- Previous stage model loading status and metrics
- Epoch progress with loss and learning rate
- Validation loss after each epoch
- Early stopping notifications
- Evaluation results and metrics
- File save locations

## 🔗 Dependencies

- PyTorch
- Transformers
- Datasets
- tqdm
- numpy
- pandas

## 📄 License

This implementation follows the same license as the main EmbedHealth project. 