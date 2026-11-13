from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from .config import (
    DATASET_CONFIG,
    DATASET_NAME,
    MAX_SEQUENCE_LENGTH,
    MODEL_NAME,
)


def load_sst2_data():
    """Load and tokenize the full SST-2 train and validation splits."""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2 has no padding token by default, so reuse its EOS token.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_raw = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
    )
    validation_raw = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="validation",
    )

    def tokenize_batch(batch):
        encoded = tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )
        encoded["labels"] = batch["label"]
        return encoded

    train_dataset = train_raw.map(
        tokenize_batch,
        batched=True,
        remove_columns=train_raw.column_names,
    )
    validation_dataset = validation_raw.map(
        tokenize_batch,
        batched=True,
        remove_columns=validation_raw.column_names,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    return tokenizer, train_dataset, validation_dataset, data_collator