# TimeSeriesExam Evaluation Documentation

## stage_tsexam_eval

**Location:** `curriculum_learning.py:1315-1337`

**What it does:**
- Loads checkpoint from `stage2_captioning` (trained on M4 captioning dataset, followed by TSQA)
- Evaluates on TimeSeriesExam1QADataset test set
- Uses MCQ accuracy metric

**Results location:**
```
results/Llama_3_2_1B/OpenTSLMFlamingo/stage_tsexam_eval/results/
├── metrics.json
└── test_predictions.jsonl
```

**Results:**
- **Accuracy: 0.0%** (0/150 samples)

---

## tsqa_on_ts_exam

**Location:** `eval_stage1_on_tsexam.py`

**What it does:**
- Standalone evaluation script
- Loads checkpoint from `stage1_mcq` (trained on TSQA MCQ dataset)
- Evaluates on TimeSeriesExam1QADataset test set
- Tests for catastrophic forgetting hypothesis (whether stage2 training destroys MCQ capabilities)

**Results location:**
```
results/Llama_3_2_1B/OpenTSLMFlamingo/tsqa_on_ts_exam/results/
├── metrics.json
└── test_predictions.jsonl
```

**Results:**
- **Accuracy: 39.33%** (59/150 samples)
- Checkpoint: stage1_mcq best_model.pt
- Model: OpenTSLMFlamingo

---

## Key Findings

1. **stage1_mcq → TimeSeriesExam**: 39.33% accuracy
   - Stage1 checkpoint (TSQA MCQ) performs reasonably on TimeSeriesExam

2. **stage2_captioning → TimeSeriesExam**: 0% accuracy
   - Stage2 checkpoint (M4 captioning) completely fails on TimeSeriesExam
   - Suggests catastrophic forgetting of MCQ capabilities during captioning training
