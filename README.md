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
  GPT2_QQP.ipynb         GPT-2 + QQP (separate notebook implementation)
results/
  roberta_sst2_results/  Exported RoBERTa SST-2 results
  roberta_qqp_results/   Exported RoBERTa QQP results
requirements.txt
```

**Important path caveat:** `config.py` currently generates paths under `results/<model>/<dataset>/`, while the two archived RoBERTa results directories in this ZIP use `results/roberta_sst2_results/` and `results/roberta_qqp_results/`. Thus, `evaluate_results.py` cannot discover these archived results without moving/copying the data or changing the path configuration. Do **not** overwrite the original results while reconciling paths.

## Implementation

- Base models: `roberta-base` and `gpt2` (binary sequence classification).
- RoBERTa adapter target modules: `query`, `value`; classification head `classifier` is trainable.
- GPT-2 shared-module target modules: `c_attn`, `c_proj`, `c_fc`; head `score` is trainable. `GPT2_QQP.ipynb` implements its own training runner and uses a distinct physical-batch/gradient-accumulation setup.
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

- **GPT-2 + SST-2:** The notebook output reports a completed final three-seed comparison: LoRA mean accuracy **91.705%**, LoRA+ **91.820%**; unlike RoBERTa results, standalone CSV/JSON artifacts are not included in `results/` for this experiment.
- **GPT-2 + QQP:** The notebook has separate training/output logic. It contains partial screening execution output; final evaluation is guarded by `RUN_FINAL=False` and no completed final summary is demonstrated in the archived notebook.

## Running in Colab

1. Clone the GitHub repository into a known directory (the RoBERTa notebooks currently use different Colab checkout paths; use the actual checkout path consistently).
2. Change into the repository root and install `requirements.txt`; run the environment fix for `torchao` if needed.
3. Open the notebook for the selected model/dataset, initialize the modules, prepare/tokenize the dataset, then run the 50-step smoke test if included.
4. Execute the 9 screening configurations, select the two winners, and execute 3 final seeds for each winner. QQP's LoRA+ screening is split into batches in its notebook; this is a scheduling choice, not a methodological difference.
5. Generate the final CSV summary and convergence plot. **Only completed runs write their final JSON/CSV records.**
6. Download a ZIP of the result files, excluding `checkpoints/`, or authenticate GitHub and push completed result files. Simply saving the notebook or committing locally does not persist results to GitHub.

### Results and checkpoints

`train.py` writes `runs/*.json` with `log_history`, a `results_<dataset>.csv` append log, and Hugging Face checkpoints under the output root. The intermediate checkpoints are excluded from Git via `.gitignore` (`results/**/checkpoints/`). Colab local storage is temporary: back up completed results before disconnecting or switching GPUs.

The `utils.py` GPU compute-unit tracker reports **estimated** CU using elapsed time and an assumed GPU-specific hourly rate. It is **not actual Colab billing information** and should not be presented as such.

## Known issues to reconcile before reproducing archived results

1. The current `evaluate_results.py` **seed-check indentation is incorrect**: the seed equality check is after the `for method` loop, so it validates only the last method. Put the check inside the loop.
2. Archived RoBERTa output folder names do not match `get_output_dir()`.
3. The SST-2 notebook clones to `/content/LoRA_Plus_Project` but its Git commands use `/content/LoRA-Plus`; it also retains outdated Drive/runtime-unassignment text. Standardize paths before using its upload cell.
4. `utils.py` duplicates the `GPU_CU_RATES` dictionary and imports. The second dictionary overrides the first, and GPU detection can misidentify unfamiliar GPUs by memory size. CU estimates need measured rates.
5. Some reported settings/version information is not pinned by `requirements.txt`; recorded JSON software-version fields should be preferred for the archived RoBERTa results.
6. `train.py` skips saved runs based on a limited set of fields (not rank, dropout, target modules, optimizer details, dataset identities, or library versions). Changing those settings while reusing a run name could silently reuse stale results.
7. GPT-2 + QQP uses a separate methodology/runner and is not directly interchangeable with the shared modules; document differences when comparing results.

## Reference

S. Hayou, N. Ghosh, and B. Yu. *LoRA+: Efficient Low Rank Adaptation of Large Models*. ICML 2024. https://proceedings.mlr.press/v235/hayou24a.html