
# utils.py
import numpy as np
import evaluate

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
)

from config import (
    COMMON_CONFIG,
    MODELS,
    DATASET_CONFIGS,
)


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def prepare_dataset(
    model_name,
    dataset_name,
    pilot=False
):
    """
    Prepare the dataset once per notebook session.

    Returns:
        tokenizer
        train_dataset
        eval_dataset
        data_collator
    """

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODELS[model_name]
    )

    # GPT-2 has no padding token by default.
    # Reuse EOS for padding to avoid changing vocabulary size.
    if model_name == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "nyu-mll/glue",
        dataset_name
    )

    max_length = COMMON_CONFIG["max_length"]

    if dataset_name == "sst2":

        def tokenize_function(batch):
            return tokenizer(
                batch["sentence"],
                truncation=True,
                max_length=max_length
            )

        remove_columns = ["sentence", "idx"]

    elif dataset_name == "qqp":

        def tokenize_function(batch):
            return tokenizer(
                batch["question1"],
                batch["question2"],
                truncation=True,
                max_length=max_length
            )

        remove_columns = [
            "question1",
            "question2",
            "idx"
        ]

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=remove_columns
    )

    tokenized_dataset = tokenized_dataset.rename_column(
        "label",
        "labels"
    )

    tokenized_dataset.set_format("torch")

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"]

    subset_size = DATASET_CONFIGS[
        dataset_name
    ]["train_subset"]

    if subset_size is not None:
        train_dataset = (
            train_dataset
            .shuffle(seed=42)
            .select(
                range(min(subset_size, len(train_dataset)))
            )
        )

    if pilot:
        train_dataset = train_dataset.select(
            range(min(1000, len(train_dataset)))
        )

        eval_dataset = eval_dataset.select(
            range(min(200, len(eval_dataset)))
        )

    # Decoder-only GPT-2 is safer with right padding
    # for this sequence-classification workflow.
    tokenizer.padding_side = "right"

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    print("Model:", model_name)
    print("Dataset:", dataset_name)
    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(eval_dataset))

    return (
        tokenizer,
        train_dataset,
        eval_dataset,
        data_collator
    )


def compute_metrics(eval_prediction):
    """
    Compute accuracy, binary F1, and macro-F1.
    """

    logits, labels = eval_prediction

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(
        predictions=predictions,
        references=labels
    )["accuracy"]

    binary_f1 = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="binary"
    )["f1"]

    macro_f1 = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="macro"
    )["f1"]

    return {
        "accuracy": accuracy,
        "f1": binary_f1,
        "f1_macro": macro_f1,
    }

# --------------------------------------------------
# Colab GPU Runtime and Compute Unit Estimation
# --------------------------------------------------

import time
import torch
from contextlib import contextmanager

# Estimated CU/hour rates (not official billing rates)
GPU_CU_RATES = {
    "T4": 1.15,
    "L4": 1.54,
    "A100 40GB": 5.35,
    "A100 80GB": 7.15,
    "G4": 8.80,
    "H100": 18.00,
}

# --------------------------------------------------
# Colab GPU Runtime and Compute Unit Estimation
# --------------------------------------------------

import time
import torch
from contextlib import contextmanager

# Approximate Google Colab Compute Unit (CU) hourly rates 
# (Subject to minor adjustments by Google; 1 CU = ~$0.10 USD)
GPU_CU_RATES = {
    "T4": 1.19,
    "L4": 1.71,
    "A100 40GB": 5.40,
    "A100 80GB": 7.52,
    "H100": 11.00,  # Varies by tier/availability
}

def detect_gpu():
    """Accurately detect the GPU assigned to the Colab runtime."""
    if not torch.cuda.is_available():
        return "CPU"

    gpu_name = torch.cuda.get_device_name(0).upper()
    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    if "H100" in gpu_name:
        return "H100"
    if "A100" in gpu_name:
        # 80GB variant typically reports ~79+ GB, 40GB variant reports ~39+ GB
        return "A100 80GB" if memory_gb > 50 else "A100 40GB"
    if "L4" in gpu_name:
        return "L4"
    if "T4" in gpu_name:
        return "T4"
    
    # Fallback catch-all based on VRAM size if name string is unfamiliar
    if memory_gb > 70:
        return "A100 80GB"
    elif memory_gb > 35:
        return "A100 40GB"
    elif memory_gb > 20:
        return "L4"
        
    return "T4"  # Default fallback assumption


@contextmanager
def track_compute(label, cu_per_hour=None):
    """
    Measure execution time and estimate compute-unit consumption.
    Automatically detects the assigned GPU and tracks VRAM delta.
    """
    gpu = detect_gpu()
    if cu_per_hour is None:
        cu_per_hour = GPU_CU_RATES.get(gpu)

    # Clear cache and record baseline VRAM usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        start_mem = torch.cuda.memory_allocated() / 1024**2
    else:
        start_mem = 0

    print(f"\n⚡ Starting: {label}")
    print(f"   Detected GPU: {gpu}")
    if cu_per_hour is not None:
        print(f"   Assumed Rate: {cu_per_hour:.2f} CU/hour")
    else:
        print("   CU rate unavailable for this hardware.")

    start_time = time.perf_counter()

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            end_mem = torch.cuda.memory_allocated() / 1024**2
        else:
            peak_mem, end_mem = 0, 0

        print(f"\n{'=' * 45}")
        print(f"🏁 Experiment Finished: {label}")
        print(f"   Hardware: {gpu}")
        print(f"   Execution time: {elapsed / 60:.2f} minutes ({elapsed:.2f}s)")
        print(f"   Peak VRAM Allocated: {peak_mem:.2f} MB")
        
        if cu_per_hour is not None:
            estimated_cu = (elapsed / 3600.0) * cu_per_hour
            print(f"   Estimated Compute Cost: {estimated_cu:.4f} CU (~${estimated_cu * 0.10:.3f})")
        else:
            print("   Estimated CU: Unavailable")
            
        print(f"{'=' * 45}\n")