# LoRA vs LoRA+ — Reduced-Compute Experimental Evaluation

Comparison of **standard LoRA** (equal learning rates for adapter matrices A and B) and **LoRA+** (higher learning rate for B) on GLUE SST-2 and QQP, using RoBERTa-base and GPT-2. This is a **reduced-compute study**, not an exact reproduction of Hayou et al. (ICML 2024).

## Repository structure

```text
config.py                 Shared settings and output directory definitions
model.py                  Sequence-classification models, PEFT adapters, AdamW groups
utils.py                  GLUE loading, tokenization, accuracy/F1, runtime/CU estimates
train.py                  Hugging Face Trainer runner and per-run JSON/CSV logging
evaluate_results.py       Screening selection, final three-seed summaries, curves
notebooks/
  roberta_sst2.ipynb      RoBERTa + SST-2
  roberta_qqp.ipynb       RoBERTa + QQP
  gpt2_sst2.ipynb        GPT-2 + SST-2
  gpt2_qqp.ipynb          GPT-2 + QQP (separate notebook implementation)
results/
  roberta_sst2_results/  Exported RoBERTa SST-2 results
  roberta_qqp_results/   Exported RoBERTa QQP results
  gpt2_sst2_results/     Exported GPT-2 SST-2 results
  gpt2_qqp_results/      Exported GPT-2 QQP final results
requirements.txt
```

## Implementation

- Base models: `roberta-base` and `gpt2` (binary sequence classification).
- RoBERTa adapter target modules: `query`, `value`; classification head `classifier` is trainable.
- GPT-2 shared-module target modules: `c_attn`, `c_proj`, `c_fc`; head `score` is trainable. `gpt2_qqp.ipynb` implements its own training runner and uses a distinct physical-batch/gradient-accumulation setup.
- LoRA rank `r=8`, alpha `8`, dropout `0.1`; maximum token length `128`.
- Shared-module training: three epochs, batch size `32`, AdamW with betas `(0.9,0.99)`, epsilon `1e-8`, weight decay `0.01`, linear LR decay and no warmup; task-head LR `1e-4`.
- Validation every `200` optimizer steps, training-loss logging every `50`, checkpoint every `1,000`, maximum two checkpoints retained. FP16 enabled when CUDA is present.
- `torchao` is removed in Colab notebook setup to avoid a previously encountered PEFT incompatibility.

### Datasets and metrics

| Dataset | Train examples used | Validation examples | Primary screening metric |
|---|---:|---:|---|
| SST-2 | 67,349 | 872 | Final validation accuracy |
| QQP | 50,000 fixed training subset (shuffle seed 42) | 40,430 | Final validation **binary F1** (duplicate class = positive) |

These are **validation metrics, not independent GLUE test scores**. The same validation split is used for hyperparameter selection and final result reporting.

### Screening grid

| Method | A LR | B LR | B/A ratio |
|---|---:|---:|---:|
| LoRA | 1e-4 | 1e-4 | 1 |
| LoRA | 2e-4 | 2e-4 | 1 |
| LoRA | 4e-4 | 4e-4 | 1 |
| LoRA+ | 1e-4 | 4e-4 | 4 |
| LoRA+ | 1e-4 | 8e-4 | 8 |
| LoRA+ | 1e-4 | 1.6e-3 | 16 |
| LoRA+ | 2e-4 | 8e-4 | 4 |
| LoRA+ | 2e-4 | 1.6e-3 | 8 |
| LoRA+ | 2e-4 | 3.2e-3 | 16 |

Every configuration is screened using **seed 42** for **3 epochs**. Winners are chosen separately for LoRA and LoRA+ using the dataset's primary final-validation metric; `evaluate_results.py` breaks ties using **final validation loss** (not accuracy for QQP). The selected settings are then rerun at seeds **42, 43, 44**, reporting mean ± sample standard deviation.

## Verified RoBERTa results in this ZIP

All 30 archived RoBERTa records are present: **9 screening and 6 final runs per dataset**. For each dataset, each summary CSV matches its 15 individual JSON result files on the checked numeric fields.

### SST-2 — final three-seed validation results

| Metric | LoRA (A=B=4e-4) | LoRA+ (A=2e-4, B=1.6e-3, ratio 8) |
|---|---:|---:|
| Accuracy | **93.922 ± 0.199%** | 93.616 ± 0.265% |
| Macro-F1 | **93.920 ± 0.199%** | 93.614 ± 0.265% |
| Validation loss | **0.2062** | 0.2137 |
| Mean trainer runtime | 464.47 s | 465.95 s |
| Peak allocated GPU memory | 1.393 GB | 1.393 GB |
| Trainable parameters | 887,042 | 887,042 |

**Observation:** Standard LoRA is higher by **0.306 percentage points** in mean validation accuracy among the selected configurations. This is not evidence of general superiority beyond this experiment.

### QQP — final three-seed validation results

| Metric | LoRA (A=B=4e-4) | LoRA+ (A=2e-4, B=3.2e-3, ratio 16) |
|---|---:|---:|
| Binary F1 | 81.537 ± 0.097% | **82.222 ± 0.046%** |
| Accuracy | 85.732 ± 0.131% | **86.336 ± 0.070%** |
| Macro-F1 | 84.955 ± 0.107% | **85.563 ± 0.060%** |
| Validation loss | 0.3205 | **0.3131** |
| Mean trainer runtime | 987.21 s | 990.57 s |
| Peak allocated GPU memory | 3.116 GB | 3.116 GB |
| Trainable parameters | 887,042 | 887,042 |

**Observation:** LoRA+ leads by **0.685 percentage points** in mean binary F1 under the selected QQP settings. The final runtime is slightly longer (approximately 3.36 s per run on average). Memory and parameter count are effectively the same.

### GPT-2 status in this archive

- **GPT-2 + SST-2:** The repository contains screening and final run artifacts under `results/gpt2_sst2_results/`. The notebook output reports a completed final three-seed comparison: LoRA mean accuracy **91.705%**, LoRA+ **91.820%**.
- **GPT-2 + QQP:** The repository contains six final-run records under `results/gpt2_qqp_results/`. The notebook uses a separate training/output implementation and has `RUN_FINAL = True` in the final-evaluation section. Screening records are not included in the archived result directory, so the screening and final workflow is not yet represented consistently by the shared `evaluate_results.py` module.

## Running in Colab

Reproduce the experiments by opening and running each notebook one at a time in Google Colab:

1. Open the notebook directly from the GitHub repository in Colab.
2. Start a GPU runtime and run the notebook cells sequentially, from top to bottom, without skipping setup or execution cells.
3. The first setup cells clone the repository, initialize project imports and result directories, install `requirements.txt`, and uninstall `torchao`.
4. Continue through dataset preparation and the 50-step sanity test before starting the full experiment.
5. Run the screening cells, select the LoRA and LoRA+ winners, then run the final three seeds (`42`, `43`, and `44`).
6. Run the summary and convergence-plots cells, then use the notebook's download or backup cell to save the completed results.

Use this order for the four model/dataset notebooks:

- `notebooks/roberta_sst2.ipynb`
- `notebooks/roberta_qqp.ipynb`
- `notebooks/gpt2_sst2.ipynb`
- `notebooks/gpt2_qqp.ipynb`

Run each notebook in a separate Colab session so its model, dataset, and result state remain isolated. **Only completed runs write JSON/CSV result records.**

### Results and checkpoints

`train.py` writes `runs/*.json` with `log_history`, a `results_<dataset>.csv` append log, and Hugging Face checkpoints under the output root. The intermediate checkpoints are excluded from Git via `.gitignore` (`results/**/checkpoints/`). Colab local storage is temporary: back up completed results before disconnecting or switching GPUs.

The `utils.py` GPU compute-unit tracker reports **estimated** CU using elapsed time and an assumed GPU-specific hourly rate. It is **not actual Colab billing information** and should not be presented as such.

## Reference

S. Hayou, N. Ghosh, and B. Yu. *LoRA+: Efficient Low Rank Adaptation of Large Models*. ICML 2024. https://proceedings.mlr.press/v235/hayou24a.html