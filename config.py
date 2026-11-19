"""Central configuration for models, datasets, LoRA search grids, and output paths."""
from pathlib import Path

# Resolve paths from the repository so results remain portable across runtimes.
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "results"

MODELS = {
    "roberta": "roberta-base",
    "gpt2": "gpt2",
}

TARGET_MODULES = {
    "roberta": ["query", "value"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
}

CLASSIFIER_MODULES = {
    "roberta": ["classifier"],
    "gpt2": ["score"],
}

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
    "sst2": {
        "train_subset": None,
        "primary_metric": "accuracy",
    },
    "qqp": {
        "train_subset": 50000,
        "primary_metric": "f1",
    },
}

# Three standard LoRA configurations.
LORA_LEARNING_RATES = [
    1e-4,
    2e-4,
    4e-4,
]

# Six LoRA+ configurations.
# Each tuple is (A learning rate, B/A ratio).
LORAPLUS_CONFIGS = [
    (1e-4, 4),
    (1e-4, 8),
    (1e-4, 16),
    (2e-4, 4),
    (2e-4, 8),
    (2e-4, 16),
]


def get_output_dir(model_name, dataset_name):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return OUTPUT_ROOT / model_name / dataset_name
