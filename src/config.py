import os
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Local runs default to the repository's results folder.
# Colab can override this to save persistently to Google Drive.
RESULTS_BASE_DIR = Path(
    os.environ.get("LORA_RESULTS_BASE", str(PROJECT_ROOT / "results"))
)

RESULTS_DIR = RESULTS_BASE_DIR / "manju_gpt2_sst2"
RUN_METRICS_DIR = RESULTS_DIR / "run_metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
SCREENING_CSV = RESULTS_DIR / "screening.csv"
FINAL_RESULTS_CSV = RESULTS_DIR / "final_results.csv"


# Dataset and model
MODEL_NAME = "gpt2"
DATASET_NAME = "nyu-mll/glue"
DATASET_CONFIG = "sst2"
MAX_SEQUENCE_LENGTH = 128
TARGET_MODULES = ("c_attn", "c_proj", "c_fc")


# LoRA configuration
LORA_RANK = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0.1


# Optimizer and scheduler
OPTIMIZER_NAME = "adamw"
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.99
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "linear"
WARMUP_STEPS = 0


# Batch configuration: 4 examples × 8 accumulation steps = 32 effective batch
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
EFFECTIVE_TRAIN_BATCH_SIZE = (
    PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
)
EVAL_BATCH_SIZE = 16


# Screening and final run plan
SCREENING_SEEDS = (42,)
SCREENING_EPOCHS = 3
FINAL_SEEDS = (42, 43, 44)
FINAL_EPOCHS = 3


# Logging, evaluation, and checkpoints
LOGGING_STEPS = 50
EVAL_STEPS = 200
SAVE_STEPS = 1_000
SAVE_TOTAL_LIMIT = 2


# The runner should enable FP16 only when CUDA is available.
USE_FP16 = True

# Predefine the target used to compare convergence speed.
CONVERGENCE_TARGET_ACCURACY = 0.90


@dataclass(frozen=True)
class RunSetting:
    method: str
    lr_a: float
    lr_b: float
    lambda_ratio: int


# Ten screening configurations
SCREENING_CONFIGS = (
    RunSetting("standard_lora", 0.0001, 0.0001, 1),
    RunSetting("standard_lora", 0.0002, 0.0002, 1),
    RunSetting("standard_lora", 0.0004, 0.0004, 1),
    RunSetting("loraplus", 0.0001, 0.0002, 2),
    RunSetting("loraplus", 0.0001, 0.0004, 4),
    RunSetting("loraplus", 0.0002, 0.0008, 4),
    RunSetting("loraplus", 0.0001, 0.0008, 8),
    RunSetting("loraplus", 0.0002, 0.0016, 8),
    RunSetting("loraplus", 0.0001, 0.0016, 16),
    RunSetting("loraplus", 0.0002, 0.0032, 16),
)