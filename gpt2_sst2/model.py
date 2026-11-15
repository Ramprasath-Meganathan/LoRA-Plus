
# model.py
import torch

from transformers import (
    AutoModelForSequenceClassification,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

from config import (
    COMMON_CONFIG,
    MODELS,
    TARGET_MODULES,
    CLASSIFIER_MODULES,
)


def create_model(model_name, tokenizer):
    """
    Initialize a fresh pretrained classification model
    and attach LoRA adapters.
    """

    if model_name not in MODELS:
        raise ValueError(f"Unsupported model: {model_name}")

    checkpoint = MODELS[model_name]

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    if model_name == "gpt2":
        if tokenizer.pad_token_id is None:
            raise ValueError(
                "GPT-2 tokenizer must have a padding token"
            )

        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=COMMON_CONFIG["rank"],
        lora_alpha=COMMON_CONFIG["alpha"],
        lora_dropout=COMMON_CONFIG["dropout"],
        target_modules=TARGET_MODULES[model_name],
        modules_to_save=CLASSIFIER_MODULES[model_name],
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model


def create_optimizer(model, base_lr, ratio):
    """
    Standard LoRA: ratio = 1
    LoRA+: ratio > 1

    eta_B = ratio * eta_A
    """

    if base_lr <= 0:
        raise ValueError("base_lr must be positive")

    if ratio < 1:
        raise ValueError("ratio must be >= 1")

    lora_a_params = []
    lora_b_params = []
    other_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if "lora_A" in name:
            lora_a_params.append(parameter)

        elif "lora_B" in name:
            lora_b_params.append(parameter)

        else:
            other_params.append(parameter)

    if not lora_a_params:
        raise ValueError("No trainable LoRA A parameters found")

    if not lora_b_params:
        raise ValueError("No trainable LoRA B parameters found")

    if not other_params:
        raise ValueError(
            "No trainable classification-head parameters found"
        )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_a_params,
                "lr": base_lr,
            },
            {
                "params": lora_b_params,
                "lr": base_lr * ratio,
            },
            {
                "params": other_params,
                "lr": COMMON_CONFIG["classifier_lr"],
            },
        ],
        betas=(
            COMMON_CONFIG["adam_beta1"],
            COMMON_CONFIG["adam_beta2"],
        ),
        eps=COMMON_CONFIG["adam_epsilon"],
        weight_decay=COMMON_CONFIG["weight_decay"],
    )

    names = ["A", "B", "other"]

    for name, group in zip(names, optimizer.param_groups):
        count = sum(p.numel() for p in group["params"])

        print(
            name,
            count,
            group["lr"]
        )

    total_optimizer_params = sum(
        p.numel()
        for group in optimizer.param_groups
        for p in group["params"]
    )

    total_trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    assert total_optimizer_params == total_trainable_params, (
        "Optimizer does not contain all trainable parameters"
    )

    return optimizer
