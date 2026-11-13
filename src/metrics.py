import csv
import json
from pathlib import Path

import numpy as np

from .config import (
    CONVERGENCE_TARGET_ACCURACY,
    FINAL_RESULTS_CSV,
    PLOTS_DIR,
    RUN_METRICS_DIR,
    SCREENING_CSV,
)


def compute_classification_metrics(eval_prediction):
    """Compute accuracy and macro-F1 for a Trainer evaluation."""
    logits = eval_prediction.predictions
    labels = np.asarray(eval_prediction.label_ids)

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)
    accuracy = float(np.mean(predictions == labels))

    f1_scores = []
    for class_id in (0, 1):
        true_positive = np.sum((predictions == class_id) & (labels == class_id))
        false_positive = np.sum((predictions == class_id) & (labels != class_id))
        false_negative = np.sum((predictions != class_id) & (labels == class_id))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(
            float(2 * true_positive / denominator) if denominator else 0.0
        )

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)),
    }


def build_run_result(
    *,
    run_id,
    stage,
    setting,
    seed,
    epochs,
    train_examples,
    validation_examples,
    train_eval,
    final_eval,
    train_output,
    trainer_state,
    elapsed_seconds,
    peak_gpu_memory_mb,
    trainable_parameters,
    total_parameters,
):
    """Build the complete summary and history for one run."""

    logs = trainer_state.log_history
    total_steps = int(trainer_state.global_step)

    training_loss_history = [
        {"step": log.get("step"), "loss": log["loss"]}
        for log in logs
        if "loss" in log and "eval_loss" not in log
    ]

    evaluation_history = [
        {
            "step": log.get("step"),
            "epoch": log.get("epoch"),
            "validation_loss": log.get("eval_loss"),
            "validation_accuracy": log.get("eval_accuracy"),
            "validation_macro_f1": log.get("eval_macro_f1"),
        }
        for log in logs
        if "eval_accuracy" in log or "eval_loss" in log
    ]

    reached_target = next(
        (
            item for item in evaluation_history
            if item["validation_accuracy"] is not None
            and item["validation_accuracy"] >= CONVERGENCE_TARGET_ACCURACY
        ),
        None,
    )

    steps_to_target = reached_target["step"] if reached_target else None
    seconds_to_target = (
        elapsed_seconds * steps_to_target / total_steps
        if reached_target and total_steps
        else None
    )

    trainable_percentage = (
        100.0 * trainable_parameters / total_parameters
        if total_parameters
        else None
    )

    setting_dict = {
        "method": setting.method,
        "lr_a": setting.lr_a,
        "lr_b": setting.lr_b,
        "lambda_ratio": setting.lambda_ratio,
    }

    return {
        "run_id": run_id,
        "stage": stage,
        **setting_dict,
        "seed": seed,
        "epochs": epochs,
        "train_examples": train_examples,
        "validation_examples": validation_examples,
        "training_accuracy": train_eval.get("train_accuracy"),
        "training_macro_f1": train_eval.get("train_macro_f1"),
        "training_eval_loss": train_eval.get("train_loss"),
        "training_loss": train_output.metrics.get("train_loss"),
        "validation_accuracy": final_eval.get("eval_accuracy"),
        "validation_macro_f1": final_eval.get("eval_macro_f1"),
        "validation_loss": final_eval.get("eval_loss"),
        "training_time_seconds": elapsed_seconds,
        "examples_per_second": (
            train_examples * epochs / elapsed_seconds
            if elapsed_seconds
            else None
        ),
        "optimizer_steps_per_second": (
            total_steps / elapsed_seconds if elapsed_seconds else None
        ),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_parameter_percent": trainable_percentage,
        "optimizer_steps": total_steps,
        "convergence_target_accuracy": CONVERGENCE_TARGET_ACCURACY,
        "steps_to_target_accuracy": steps_to_target,
        "estimated_seconds_to_target_accuracy": seconds_to_target,
        "training_loss_history": training_loss_history,
        "evaluation_history": evaluation_history,
    }


CSV_FIELDS = (
    "run_id", "stage", "method", "lr_a", "lr_b", "lambda_ratio",
    "seed", "epochs", "train_examples", "validation_examples",
    "training_accuracy", "training_macro_f1", "training_eval_loss",
    "training_loss", "validation_accuracy", "validation_macro_f1",
    "validation_loss", "training_time_seconds", "examples_per_second",
    "optimizer_steps_per_second", "peak_gpu_memory_mb",
    "trainable_parameters", "total_parameters",
    "trainable_parameter_percent", "optimizer_steps",
    "convergence_target_accuracy", "steps_to_target_accuracy",
    "estimated_seconds_to_target_accuracy",
)


def save_run_outputs(result):
    """Save one run's detailed JSON and update its stage CSV."""
    RUN_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RUN_METRICS_DIR / f"{result['run_id']}.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    csv_path = SCREENING_CSV if result["stage"] == "screening" else FINAL_RESULTS_CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

    rows = [row for row in rows if row.get("run_id") != result["run_id"]]
    rows.append({field: result.get(field) for field in CSV_FIELDS})

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def save_run_plot(result):
    """Save training-loss and validation curves for one run."""
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    train_history = result["training_loss_history"]
    eval_history = result["evaluation_history"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    train_points = [item for item in train_history if item["step"] is not None]
    axes[0].plot(
        [item["step"] for item in train_points],
        [item["loss"] for item in train_points],
    )
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Optimizer step")

    eval_points = [item for item in eval_history if item["step"] is not None]
    axes[1].plot(
        [item["step"] for item in eval_points],
        [item["validation_accuracy"] for item in eval_points],
    )
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Optimizer step")

    axes[2].plot(
        [item["step"] for item in eval_points],
        [item["validation_loss"] for item in eval_points],
    )
    axes[2].set_title("Validation loss")
    axes[2].set_xlabel("Optimizer step")

    fig.suptitle(result["run_id"])
    fig.tight_layout()
    plot_path = PLOTS_DIR / f"{result['run_id']}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path