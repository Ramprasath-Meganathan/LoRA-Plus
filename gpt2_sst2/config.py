"""Experiment configuration for GPT-2 + SST-2 LoRA/LoRA+ runs."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Ram's workflow keeps results/checkpoints on Google Drive.  The local
# fallback makes the project runnable outside Colab as well.
DRIVE_ROOT = Path("/content/drive/MyDrive/LoRA_Plus_Project/modular_v2")
OUTPUT_ROOT = (
    DRIVE_ROOT if DRIVE_ROOT.parent.parent.exists()
    else PROJECT_ROOT / "results"
)

MODELS = {"gpt2": "gpt2"}
TARGET_MODULES = {"gpt2": ["c_attn", "c_proj", "c_fc"]}
CLASSIFIER_MODULES = {"gpt2": ["score"]}

COMMON_CONFIG = {
    "rank": 8,
    "alpha": 8,
    "dropout": 0.1,
    "batch_size": 32,
    "eval_batch_size": 64,
    "max_length": 128,
    "epochs": 3,
    "screening_seed": 42,
    "seeds": [42, 43, 44],
    "weight_decay": 0.01,
    "classifier_lr": 1e-4,
    "adam_beta1": 0.9,
    "adam_beta2": 0.99,
    "adam_epsilon": 1e-8,
    "eval_steps": 200,
    "logging_steps": 50,
    "save_steps": 1000,
    "save_total_limit": 2,
}

DATASET_CONFIGS = {
    "sst2": {"train_subset": None, "primary_metric": "accuracy"}
}

# Ram-aligned screening: 3 standard LoRA + 6 LoRA+ configurations = 9 runs.
LORA_LEARNING_RATES = [1e-4, 2e-4, 4e-4]
LORAPLUS_CONFIGS = [
    (1e-4, 4), (1e-4, 8), (1e-4, 16),
    (2e-4, 4), (2e-4, 8), (2e-4, 16),
]


def get_output_dir(model_name: str, dataset_name: str) -> Path:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return OUTPUT_ROOT / model_name / dataset_name
