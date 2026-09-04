<!--
SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
SPDX-FileCopyrightText: 2026 This source file is part of the OpenTSLM open-source project.

SPDX-License-Identifier: MIT
-->

# AGENTS.md

## What this repo is

OpenTSLM is a research codebase from Stanford and ETH Zurich for Time Series Language
Models: pretrained LLMs (Llama 3.2, Gemma 3) extended with time series as a native input
modality, so a user can prompt in natural language over one or more time series of any
length and get findings, captions, or rationales back. Two architectures are implemented:
OpenTSLMSP feeds encoded series as soft prompt tokens, OpenTSLMFlamingo injects them via
cross attention. Models are trained with a five stage curriculum on medical and general
time series tasks: MCQ warm up, captioning, human activity recognition from accelerometer
data, sleep staging from EEG, and ECG question answering with chain of thought. Pretrained
checkpoints are published on the Hugging Face `OpenTSLM` org and the package ships on PyPI
as `opentslm`.

Python 3.12, `uv` for everything. Package code lives in `src/opentslm/`; the trainer is the
root script `curriculum_learning.py`.

## Commands

Run from repo root.

| Task | Command | Notes |
|---|---|---|
| Install | `uv sync --all-groups` | slow on cold checkout, run in background. Creates `.venv` |
| Lint | `uv run ruff check <changed files>` | only on files you changed |
| Format check | `uv run ruff format --check <changed files>` | only on files you changed |
| License headers | `uv run reuse lint` | CI gate, must pass |
| Import smoke test | `uv run python test/smoke_test.py` | no network |
| Build wheel | `uv build` | writes `dist/`, ignored by git |
| Smoke test the wheel | `uv run --isolated --no-project --with dist/*.whl test/smoke_test.py` | what CI runs before publish |
| Trainer unit tests | `uv run python -m test.test_curriculum_trainer` | log in to HF first, see Gotchas |
| One loader test | `uv run python -m unittest test.m4_loader_test -v` | slow on first run (download), run in background |
| Trainer CLI help | `uv run python curriculum_learning.py --help` | |

There is no pytest. Tests under `test/` are either `unittest` classes or plain
scripts with a `main()`. Run them as modules (`python -m test.<name>`), not as file
paths, because `curriculum_learning` is imported from the repo root.

Training and demo runs need a GPU and a Hugging Face login for gated models:

```bash
huggingface-cli login
uv run python curriculum_learning.py --model OpenTSLMFlamingo --stages stage1_mcq --llm_id google/gemma-3-270m
uv run python demo/huggingface/01_test_hf_tsqa.py
```

CI (`.github/workflows/static-analysis.yml`) runs only the REUSE check and a
markdown link check. `.github/workflows/publish.yml` is manual and runs `uv build`, both smoke tests,
then `uv publish`. Never run `uv publish` yourself.

## Layout

```
curriculum_learning.py       CurriculumTrainer, argparse entry point, DDP/FSDP setup
src/opentslm/
  model_config.py            hyperparameters (BATCH_SIZE, PATCH_SIZE, LRs, MAX_SAMPLES)
  model/llm/                 OpenTSLMSP (soft prompt), OpenTSLMFlamingo (cross attention)
  model/encoder/             TransformerCNNEncoder
  model/projector/           MLP projectors
  prompt/                    TextPrompt, TextTimeSeriesPrompt, FullPrompt
  time_series_datasets/      one QADataset subclass per task, loaders download on demand
    constants.py             RAW_DATA path (see Gotchas)
test/                        smoke test, trainer tests, per dataset loader tests
demo/huggingface/            load published checkpoints from the OpenTSLM HF org
evaluation/                  baseline (OpenAI/LLMTime), opentslm, clinician, memory studies
scripts/                     memory profiling and plotting helpers
LICENSES/, REUSE.toml        REUSE licensing metadata
```

## Frozen paths

- `uv.lock`: regenerate with `uv lock` (or `uv add <pkg>`), never hand edit.
  `uv lock --check` must stay green.
- `LICENSES/`, `REUSE.toml`, and the SPDX header block on every file: managed by
  `reuse`. Add headers to new files with the `reuse annotate` command in README.md.
- `dist/`, `.venv/`, `src/data/`, `results/`: build and runtime outputs, all ignored.

## Conventions

- Every new source file needs the SPDX header. Copy it from any neighbouring file or
  run the `reuse annotate` snippet from README.md, then `uv run reuse lint`.
- Add dependencies with `uv add` (or `uv add --group dev`). Keep `requirements.txt`
  in sync by hand; it is the pip fallback documented in README.md.
- Model and stage names are CLI enums: `OpenTSLMSP`, `OpenTSLMFlamingo`,
  `stage1_mcq` through `stage5_ecg_cot`. Adding one means updating the argparse
  choices in `curriculum_learning.py` and the matching `CurriculumTrainer` method.
- `ruff` config is in `pyproject.toml` (line length 120). Do not restate it.

## Gotchas

<!-- Append one line per new gotcha. Never restructure. Never delete without a reason.
     Format: - `trigger` -> what to do. why -->

- `ruff check` / `ruff format` -> run only on the files you changed. Never repo wide.
- `python -m test.test_curriculum_trainer` -> log in with `huggingface-cli login` using an
  account that has access to `meta-llama/Llama-3.2-1B` before running. Without it 7 of 8
  tests fail with 401 gated repo errors and exit code 1. Treat those as a login problem, not a code defect.
- Any test under `test/` -> run as a module from the repo root
  (`uv run python -m test.<name>`), never as a file path. File paths fail with
  `ModuleNotFoundError: curriculum_learning`.
- Dataset paths -> expect downloads under `src/data/`, not a root `data/` folder. `RAW_DATA`
  in `src/opentslm/time_series_datasets/constants.py` resolves relative to the package. Both are gitignored
  via `**/data/*`. M4 alone is ~400MB, so do not delete `src/data/` between runs.
- Loader tests (`*_loader_test.py`, `*_cot_test.py`, `ecg_qa*`) -> run one at a time and
  allow network on first run. They pull zips from polybox.ethz.ch, HF datasets, or PhysioNet
  (PTB-XL). SleepEDF and ECG-QA are large.
- Command timeouts -> use your tool's own timeout parameter. The `timeout` binary does
  not exist in the macOS shell.
- First `uv sync --all-groups` -> run it in the background and let it finish. It downloads
  torch, torchvision and jupyter; interrupting leaves a half built `.venv`.
- Adding a dependency -> use `uv add` (or `uv add --group dev`). Never `pip install` into
  `.venv`, it makes `uv.lock` drift.
- Publishing -> never run `uv publish`. Only the manual `.github/workflows/publish.yml` workflow publishes.

## Before you finish

1. `uv run reuse lint` passes (new files have headers).
2. `uv run ruff check <files you changed>` is clean for those files.
3. `uv run python test/smoke_test.py` still imports.
4. If you touched a dataset loader, run its `test/<name>_test.py` as a module.
5. If you touched `pyproject.toml`, `uv lock --check` passes.
