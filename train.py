
# train.py
import gc
import json
import os
import uuid

import pandas as pd
import torch
import transformers
import peft

from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import COMMON_CONFIG, get_output_dir
from model import create_model, create_optimizer
from utils import compute_metrics


def run_experiment(
    model_name,
    dataset_name,
    tokenizer,
    train_dataset,
    eval_dataset,
    data_collator,
    lam=1,
    seed=42,
    epochs=3,
    base_lr=1e-4,
    stage="screening",
    max_steps=-1,
):
    """
    Train and evaluate LoRA or LoRA+.

    lam=1: standard LoRA
    lam>1: LoRA+

    The tokenizer and datasets are prepared outside this
    function and reused across all experiments.
    """

    root = get_output_dir(model_name, dataset_name)

    runs_dir = root / "runs"
    checkpoints_dir = root / "checkpoints" / model_name / dataset_name
    results_csv = root / f"results_{dataset_name}.csv"

    runs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    bs = COMMON_CONFIG["batch_size"]

    # Include the run settings to avoid mixing experiments.
    run_name = (
        f"{model_name}_{dataset_name}_"
        f"lam{lam}_lr{base_lr:.8g}_"
        f"seed{seed}_ep{epochs}_bs{bs}_"
        f"steps{max_steps}_{stage}"
    )

    json_path = runs_dir / f"{run_name}.json"

    if stage != "test" and json_path.exists():
        with open(json_path, "r") as f:
            previous = json.load(f)

        expected = {
            "model_key": model_name,
            "dataset": dataset_name,
            "lam": lam,
            "eta_A": base_lr,
            "seed": seed,
            "epochs": epochs,
            "batch_size": bs,
            "max_steps": max_steps,
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
        }

        matches = all(
            previous.get(k) == v
            for k, v in expected.items()
        )

        if not matches:
            raise ValueError(
                "Saved result has incompatible settings: "
                + str(json_path)
            )

        print("Already completed:", run_name)
        return previous

    # Seed must be set before model initialization.
    set_seed(seed)

    model = create_model(
        model_name=model_name,
        tokenizer=tokenizer
    )

    optimizer = create_optimizer(
        model=model,
        base_lr=base_lr,
        ratio=lam
    )

    args = TrainingArguments(
        output_dir=str(checkpoints_dir / run_name),
        seed=seed,
        data_seed=seed,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=COMMON_CONFIG[
            "eval_batch_size"
        ],
        eval_strategy="steps",
        eval_steps=COMMON_CONFIG["eval_steps"],
        save_strategy="steps",
        save_steps=COMMON_CONFIG["save_steps"],
        save_total_limit=COMMON_CONFIG[
            "save_total_limit"
        ],
        load_best_model_at_end=False,
        logging_strategy="steps",
        logging_steps=COMMON_CONFIG["logging_steps"],
        lr_scheduler_type="linear",
        warmup_steps=0,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_result = trainer.train()
    final = trainer.evaluate()

    logs = trainer.state.log_history

    evals = [
        x for x in logs
        if "eval_accuracy" in x
    ]

    train_logs = [
        x for x in logs
        if "loss" in x
    ]

    row = {
        "stage": stage,
        "model": model.config._name_or_path,
        "model_key": model_name,
        "dataset": dataset_name,
        "method": "LoRA" if lam == 1 else "LoRA+",
        "lam": lam,
        "eta_A": base_lr,
        "eta_B": base_lr * lam,
        "seed": seed,
        "epochs": epochs,
        "batch_size": bs,
        "max_steps": max_steps,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "final_acc": final["eval_accuracy"],
        "final_f1": final["eval_f1"],
        "final_f1_macro": final["eval_f1_macro"],
        "best_acc": max(
            [x["eval_accuracy"] for x in evals]
            + [final["eval_accuracy"]]
        ),
        "best_f1": max(
            [x["eval_f1"] for x in evals]
            + [final["eval_f1"]]
        ),
        "best_f1_macro": max(
            [x["eval_f1_macro"] for x in evals]
            + [final["eval_f1_macro"]]
        ),
        "final_eval_loss": final["eval_loss"],
        "last_train_loss": (
            train_logs[-1]["loss"]
            if train_logs else None
        ),
        "train_runtime_s": train_result.metrics[
            "train_runtime"
        ],
        "samples_per_s": train_result.metrics[
            "train_samples_per_second"
        ],
        "total_steps": train_result.global_step,
        "peak_mem_gb": (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available()
            else None
        ),
        "trainable_params": sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }

    if stage != "test":
        payload = {
            **row,
            "log_history": logs,
        }

        # Write JSON atomically to avoid partially written
        # records being mistaken for completed experiments.
        temp_path = json_path.with_name(
            json_path.name + "." + uuid.uuid4().hex + ".tmp"
        )

        with open(temp_path, "w") as f:
            json.dump(payload, f, default=str)

        os.replace(temp_path, json_path)

        pd.DataFrame([row]).to_csv(
            results_csv,
            mode="a",
            header=not results_csv.exists(),
            index=False,
        )

        print("Saved result:", json_path)

    del trainer, model, optimizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row
