# GPT-2 + SST-2 LoRA/LoRA+ Experiment

This project is Manju Tadi's GPT-2/SST-2 experiment for the LoRA+ proposal. It follows the same staged workflow as the Ram reference project:

1. Install dependencies and mount Google Drive.
2. Run a 50-step sanity test.
3. Screen 3 standard LoRA learning rates and 6 LoRA+ configurations using seed 42 for 3 epochs.
4. Select the best standard LoRA and LoRA+ settings using final validation accuracy, with validation loss as the tie-breaker.
5. Re-run both selected settings with seeds 42, 43, and 44 for 3 epochs.
6. Save CSV/JSON metrics, aggregate mean ± standard deviation, and plot convergence.

## Metrics

The project records validation accuracy, binary F1, macro-F1, validation loss, training loss, runtime, samples/second, optimizer steps, peak GPU memory, trainable parameter count, and library versions. Accuracy is the primary SST-2 metric.

## GPT-2-specific settings

- Checkpoint: `gpt2`
- Dataset: `nyu-mll/glue`, `sst2`
- LoRA target modules: `c_attn`, `c_proj`, `c_fc`
- Classification head saved by PEFT: `score`
- GPT-2 padding token: EOS token reused as the padding token
- Rank / alpha / dropout: `8 / 8 / 0.1`
- Batch size / maximum length: `32 / 128`

## Colab

Place this folder at `/content/LoRA_Plus_Project/gpt2_sst2`, open `notebooks/gpt2_sst2.ipynb`, and run the cells in order. Update the notebook's `PROJECT_DIR` to `/content/LoRA_Plus_Project/gpt2_sst2` when using this repository layout. Results are written to:

`/content/drive/MyDrive/LoRA_Plus_Project/modular_v2/gpt2/sst2/`

If Google Drive is not mounted, the code falls back to the local `results/` directory.
