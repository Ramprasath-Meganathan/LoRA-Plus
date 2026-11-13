import time
from pathlib import Path

import torch
from peft import get_peft_model
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from .config import (
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPSILON,
    EVAL_BATCH_SIZE,
    EVAL_STEPS,
    GRADIENT_ACCUMULATION_STEPS,
    LOGGING_STEPS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    RESULTS_DIR,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    USE_FP16,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    LR_SCHEDULER_TYPE,
)
from .metrics import (
    build_run_result,
    compute_classification_metrics,
    save_run_outputs,
    save_run_plot,
)
from .model import build_model


class EvaluateAtEpochEndCallback(TrainerCallback):
    """Request an evaluation at each epoch boundary in addition to step-based evaluations."""

    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_evaluate = True
        return control


def _build_optimizer(model, setting):
    """Use separate learning rates for LoRA A and B; keep other trainable modules at LR A."""
    lora_a, lora_b, other = [], [], []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_A" in name:
            lora_a.append(parameter)
        elif "lora_B" in name:
            lora_b.append(parameter)
        else:
            other.append(parameter)

    if not lora_a or not lora_b:
        raise RuntimeError("Could not find trainable LoRA A and B parameters.")

    parameter_groups = [
        {"params": lora_a, "lr": setting.lr_a},
        {"params": lora_b, "lr": setting.lr_b},
    ]
    if other:
        parameter_groups.append({"params": other, "lr": setting.lr_a})

    return torch.optim.AdamW(
        parameter_groups,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
    )


def train_one_run(
    *,
    setting,
    seed,
    stage,
    epochs,
    tokenizer,
    train_dataset,
    validation_dataset,
    data_collator,
):
    """Train, evaluate, record, and plot one screening or final run."""

    set_seed(seed)
    model = build_model(tokenizer, seed)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )

    run_id = (
        f"{stage}_{setting.method}"
        f"_a{setting.lr_a:g}_b{setting.lr_b:g}"
        f"_lambda{setting.lambda_ratio}_seed{seed}"
    )
    checkpoint_dir = RESULTS_DIR / "checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = _build_optimizer(model, setting)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        learning_rate=setting.lr_a,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_steps=WARMUP_STEPS,
        fp16=USE_FP16 and torch.cuda.is_available(),
        report_to="none",
        seed=seed,
        data_seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_classification_metrics,
        optimizers=(optimizer, None),
        callbacks=[EvaluateAtEpochEndCallback()],
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    train_output = trainer.train()
    elapsed_seconds = time.perf_counter() - start_time

    # Record metrics on the full training and validation splits.
    train_eval = trainer.evaluate(
        eval_dataset=train_dataset,
        metric_key_prefix="train",
    )
    final_eval = trainer.evaluate(
        eval_dataset=validation_dataset,
        metric_key_prefix="eval",
    )

    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available()
        else None
    )

    result = build_run_result(
        run_id=run_id,
        stage=stage,
        setting=setting,
        seed=seed,
        epochs=epochs,
        train_examples=len(train_dataset),
        validation_examples=len(validation_dataset),
        train_eval=train_eval,
        final_eval=final_eval,
        train_output=train_output,
        trainer_state=trainer.state,
        elapsed_seconds=elapsed_seconds,
        peak_gpu_memory_mb=peak_gpu_memory_mb,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
    )

    json_path, csv_path = save_run_outputs(result)
    plot_path = save_run_plot(result)

    print(f"Saved metrics: {json_path}")
    print(f"Updated summary: {csv_path}")
    print(f"Saved plot: {plot_path}")

    return result