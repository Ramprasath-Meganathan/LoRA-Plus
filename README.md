# LoRA vs LoRA+: Experimental Configuration

## 1. Project Overview

This project evaluates **Low-Rank Adaptation (LoRA)** and **LoRA+** across different pretrained language models and GLUE benchmark datasets.

The objective is to investigate whether assigning different learning rates to LoRA adapter matrices A and B improves:

- Classification accuracy and F1 score
- Convergence speed
- Training efficiency
- GPU memory utilization
- Parameter efficiency

The experiments are inspired by the ICML 2024 paper *LoRA+: Efficient Low Rank Adaptation of Large Models* by Hayou et al.

Our implementation uses a reduced-compute experimental design suitable for Google Colab. It is not an exact reproduction of the paper's full hyperparameter search.

## 2. Model and Dataset Combinations

| Model | Dataset | Task | Primary Metric |
|---|---|---|---|
| RoBERTa-base | SST-2 | Sentiment classification | Accuracy |
| RoBERTa-base | QQP | Paraphrase detection | Binary F1 |
| GPT-2 | SST-2 | Sentiment classification | Accuracy |
| GPT-2 | QQP | Paraphrase detection | Binary F1 |

## 3. Common Experimental Settings

The following settings are standardized across experiments wherever supported.

| Parameter | Configuration |
|---|---|
| LoRA rank (`r`) | 8 |
| LoRA alpha (`lora_alpha`) | 8 |
| LoRA dropout | 0.1 |
| Optimizer | AdamW |
| AdamW beta1 | 0.9 |
| AdamW beta2 | 0.99 |
| AdamW epsilon | 1e-8 |
| Weight decay | 0.01 |
| Learning-rate scheduler | Linear decay |
| Warmup steps | 0 |
| Training batch size | 32 |
| Maximum sequence length | 128 |
| Precision | FP16 when supported |
| Screening seed | 42 |
| Final evaluation seeds | 42, 43, 44 |
| Validation interval | Every 200 optimizer steps |
| LoRA initialization | Random A, zero-initialized B |
| Checkpoint saving | Every 1,000 optimizer steps |
| Checkpoint retention | 2 |

**Note:** Dropout, weight decay, evaluation frequency, and checkpoint settings are choices in our implementation. They should not be interpreted as universally specified settings in the original paper.

## 4. Model-Specific Settings

| Parameter | RoBERTa-base | GPT-2 |
|---|---|---|
| Pretrained checkpoint | `roberta-base` | `gpt2` |
| Architecture | Encoder-only Transformer | Decoder-only Transformer |
| Hidden dimension | 768 | 768 |
| Transformer layers | 12 | 12 |
| LoRA target modules | `query`, `value` | `c_attn`, `c_proj`, `c_fc` |
| Classification head | RoBERTa sequence classifier | GPT-2 sequence classifier |
| Padding | RoBERTa tokenizer padding | Explicit padding-token configuration |
| Number of labels | 2 | 2 |

Both models use the same LoRA/LoRA+ optimization principle, but their adapter target modules differ because of architectural differences.

## 5. Dataset-Specific Settings

| Parameter | SST-2 | QQP |
|---|---|---|
| Task | Binary sentiment classification | Binary paraphrase classification |
| Input | Single sentence | Question pair |
| Labels | Negative / Positive | Not duplicate / Duplicate |
| Primary metric | Accuracy | Binary F1 |
| Secondary metric | Macro-F1 | Accuracy, macro-F1 |
| Full training examples | 67,349 | Approximately 364,000 |
| Training examples used | 67,349 | 50,000 (planned subset) |
| Epochs in original paper | 10 | 3 |
| Epochs in project | 3 | 3 |

For QQP, a fixed 50,000-example training subset is used to reduce computational requirements. The subset must remain identical across all configurations and seeds.

## 6. LoRA vs LoRA+ Optimization

The primary difference between standard LoRA and LoRA+ is how learning rates are assigned to the adapter matrices.

### Standard LoRA

Both matrices use the same learning rate:

\[
\eta_A=\eta_B
\]

### LoRA+

Matrix B uses a larger learning rate:

\[
\eta_B=\lambda\eta_A,\qquad \lambda>1
\]

| Parameter | Standard LoRA | LoRA+ |
|---|---|---|
| Adapter rank | 8 | 8 |
| Adapter alpha | 8 | 8 |
| Adapter initialization | Same | Same |
| Optimizer | AdamW | AdamW |
| A learning rate | Base LR | Base LR |
| B learning rate | Base LR | Base LR × ratio |
| Learning-rate ratio | 1× | 4×, 8×, 16× |
| Trainable parameters | Same for matched model | Same for matched model |

The pretrained backbone remains frozen while the LoRA adapters and task-specific classification head are trained.

## 7. Hyperparameter Screening

### Objective

Identify the best learning-rate configuration separately for standard LoRA and LoRA+.

### Screening Configurations

| Method | Matrix A LR | Matrix B LR | Ratio |
|---|---|---|---|
| LoRA | 1e-4 | 1e-4 | 1× |
| LoRA | 2e-4 | 2e-4 | 1× |
| LoRA | 4e-4 | 4e-4 | 1× |
| LoRA+ | 1e-4 | 4e-4 | 4× |
| LoRA+ | 1e-4 | 8e-4 | 8× |
| LoRA+ | 1e-4 | 1.6e-3 | 16× |
| LoRA+ | 2e-4 | 8e-4 | 4× |
| LoRA+ | 2e-4 | 1.6e-3 | 8× |
| LoRA+ | 2e-4 | 3.2e-3 | 16× |

### Screening Procedure

- Run all nine configurations independently.
- Initialize a fresh pretrained model for each run.
- Use random seed 42.
- Train each configuration for three epochs.
- Record validation metrics after training.
- Select the best standard LoRA and LoRA+ configurations separately.

**Selection criteria:**

- SST-2: Highest final validation accuracy, with validation loss as a tie-breaker.
- QQP: Highest final binary F1, with accuracy as a tie-breaker.

**Total screening runs:** 9 per model–dataset combination.

## 8. Final Multi-Seed Experiments

After screening, the selected standard LoRA and LoRA+ configurations are evaluated across three random seeds.

| Parameter | Final configuration |
|---|---|
| Configurations | Best LoRA and best LoRA+ |
| Random seeds | 42, 43, 44 |
| Epochs | 3 |
| Training dataset | Same as screening |
| Batch size | 32 |
| Optimizer and scheduler | Same as screening |
| Primary output | Mean ± standard deviation |
| Additional outputs | F1, loss, runtime, memory, convergence |

**Total final runs:** 2 configurations × 3 seeds = 6 runs per model–dataset combination.

### Metrics Recorded

| Metric | Purpose |
|---|---|
| Validation accuracy | Classification performance |
| Binary F1 | Positive-class paraphrase detection performance on QQP |
| Macro-F1 | Class-balanced classification performance |
| Training loss | Optimization progress |
| Validation loss | Generalization performance |
| Training runtime | Computational efficiency |
| Peak allocated GPU memory | GPU memory usage |
| Trainable parameter count | Parameter efficiency |
| Accuracy/F1 vs. steps | Convergence behavior |

## 9. Completed Results: RoBERTa-base + SST-2

The three-epoch screening procedure selected the following configurations:

| Parameter | Standard LoRA | LoRA+ |
|---|---|---|
| Matrix A LR | 4e-4 | 2e-4 |
| Matrix B LR | 4e-4 | 1.6e-3 |
| Ratio | 1× | 8× |

### Final Three-Seed Results

| Metric | Standard LoRA | LoRA+ |
|---|---|---|
| Validation accuracy | **93.92 ± 0.20%** | 93.62 ± 0.26% |
| Macro-F1 | **93.92 ± 0.20%** | 93.61 ± 0.27% |
| Validation loss | 0.2062 | 0.2137 |
| Mean training runtime | 461.56 s | 468.34 s |
| Peak allocated GPU memory | 1.393 GB | 1.393 GB |
| Trainable parameters | 887,042 | 887,042 |
| Steps to 93% mean validation accuracy | 1,400 | 1,400 |

Standard LoRA achieved approximately 0.31 percentage points higher mean validation accuracy than LoRA+ under the tested configurations.

Both methods reached 93% mean validation accuracy at the same evaluation step and exhibited nearly identical measured GPU memory allocation.

These results do not establish that standard LoRA is universally superior. The learning-rate search was limited, and hyperparameter selection used one seed.

## 9.1 Completed Results: GPT-2 + SST-2

The GPT-2/SST-2 experiment used the same three-epoch, nine-configuration screening workflow and final three-seed evaluation. The GPT-2-specific LoRA target modules were `c_attn`, `c_proj`, and `c_fc`.

### Selected Screening Configurations

| Parameter | Standard LoRA | LoRA+ |
|---|---:|---:|
| Matrix A learning rate | 0.0002 | 0.0001 |
| Matrix B learning rate | 0.0002 | 0.0008 |
| Ratio (`lambda`) | 1 | 8 |
| Screening validation accuracy | 91.97% | 92.32% |

### Final Three-Seed Results

| Metric | Standard LoRA | LoRA+ |
|---|---:|---:|
| Validation accuracy | 91.70% ± 0.24% | **91.82% ± 0.46%** |
| Binary F1 | 91.94% ± 0.21% | **92.04% ± 0.42%** |
| Macro-F1 | 91.70% ± 0.24% | **91.81% ± 0.47%** |
| Validation loss | **0.2524 ± 0.0098** | 0.2546 ± 0.0021 |
| Mean training runtime | 854.95 ± 1.78 s | 854.88 ± 1.03 s |
| Peak allocated GPU memory | 2.620 ± 0.005 GB | 2.622 ± 0.005 GB |
| Trainable parameters | 1,181,184 | 1,181,184 |

Under this reduced-compute protocol, LoRA+ achieved a small mean validation-accuracy improvement of approximately 0.115 percentage points over standard LoRA, with nearly identical runtime, memory usage, and trainable parameter count. The difference is suggestive rather than statistically conclusive because the final evaluation used three seeds and the same validation split for selection and reporting.

## 10. Comparison with the Original Paper

| Setting | Original LoRA+ paper | Our project |
|---|---|---|
| Learning-rate search | Broad two-dimensional A/B grid | Nine configurations |
| Seeds per reported grid configuration | 3 | 1 during screening |
| Separate screening stage | Not described | Yes |
| Final evaluation | Three-seed grid averages | Three seeds for selected winners |
| SST-2 epochs | 10 | 3 |
| QQP epochs | 3 | 3 |
| Sequence length | 128 | 128 |
| Batch size | 32 | 32 |
| LoRA rank / alpha | 8 / 8 | 8 / 8 |

Our project prioritizes computational feasibility while preserving the core comparison between equal and unequal LoRA adapter learning rates.

### Limitations

1. The hyperparameter search is narrower than the original paper's grid.
2. Screening uses one random seed, which may introduce selection variability.
3. SST-2 training uses three epochs rather than ten.
4. The planned QQP experiment uses a reduced training subset.
5. Hyperparameter selection and final reporting use the same GLUE validation split.
6. Results may vary with GPU hardware and software versions.

## 11. Experimental Reproducibility

To support reproducibility:

- Use identical training and validation splits for LoRA and LoRA+.
- Initialize a fresh model for each configuration.
- Set random seeds before model initialization.
- Use the same training budget and evaluation procedure.
- Keep model and dataset preprocessing consistent within each comparison.
- Record the model, dataset, learning rates, ratio, rank, seed, and epoch count for every run.
- Save training metrics, checkpoints, and results in separate directories.
- Record software versions and GPU hardware.

## 12. Reference

Hayou, S., Ghosh, N., & Yu, B. (2024). **LoRA+: Efficient Low Rank Adaptation of Large Models.** Proceedings of the 41st International Conference on Machine Learning (ICML), PMLR 235.

Paper: https://proceedings.mlr.press/v235/hayou24a.html

Source code: https://github.com/nikhil-ghosh-berkeley/loraplus
