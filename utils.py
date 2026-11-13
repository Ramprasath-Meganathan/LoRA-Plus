
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
