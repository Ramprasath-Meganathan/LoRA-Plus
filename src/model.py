from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, set_seed

from .config import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    MODEL_NAME,
    TARGET_MODULES,
)


def build_model(tokenizer, seed: int):
    """Build a fresh GPT-2 sequence classifier with LoRA adapters."""

    # Make model, head, and adapter initialization repeatable for this seed.
    set_seed(seed)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.problem_type = "single_label_classification"
    base_model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(TARGET_MODULES),
        modules_to_save=["score"],
        bias="none",
        fan_in_fan_out=True,
    )

    model = get_peft_model(base_model, lora_config)

    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    if not trainable_names:
        raise RuntimeError("The model has no trainable parameters.")

    unexpected = [
        name for name in trainable_names
        if not (
            "lora_A" in name
            or "lora_B" in name
            or "modules_to_save" in name
        )
    ]
    if unexpected:
        raise RuntimeError(
            "Unexpected trainable parameters found: "
            + ", ".join(unexpected[:10])
        )

    if not any("lora_A" in name for name in trainable_names):
        raise RuntimeError("No trainable LoRA A parameters were found.")
    if not any("lora_B" in name for name in trainable_names):
        raise RuntimeError("No trainable LoRA B parameters were found.")

    return model