"""Lab 3A: fine-tune the Bayan topic classifier."""

import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from bayan.models.data import build_topic_dataset


CHECKPOINT = "xlm-roberta-base"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/topic_classifier",
        help="Where to save the trained classifier artefact.",
    )
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
        ),
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load grouped dataset
    splits = build_topic_dataset()

    train_df = splits["train"].copy()
    val_df = splits["validation"].copy()
    test_df = splits["test"].copy()

    # 2. Build label mappings
    labels = sorted(train_df["topic"].unique())

    label2id = {
        label: idx
        for idx, label in enumerate(labels)
    }

    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    for df in [train_df, val_df, test_df]:
        df["labels"] = df["topic"].map(label2id)

    # 3. Convert pandas -> Hugging Face Dataset
    train_ds = Dataset.from_pandas(
        train_df[["text", "labels"]],
        preserve_index=False,
    )

    val_ds = Dataset.from_pandas(
        val_df[["text", "labels"]],
        preserve_index=False,
    )

    test_ds = Dataset.from_pandas(
        test_df[["text", "labels"]],
        preserve_index=False,
    )

    # 4. Tokenizer from Lab 1 decision
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # 5. Classification model
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    # 6. Training configuration
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=25,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7. Train
    print("=== Lab 3A: Fine-tuning Topic Classifier ===")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Train rows: {len(train_ds)}")
    print(f"Validation rows: {len(val_ds)}")
    print(f"Test rows: {len(test_ds)}")
    print(f"Labels: {labels}")

    trainer.train()

    # 8. Validation evaluation
    val_metrics = trainer.evaluate(val_ds)

    print("\n=== Validation Results ===")
    print(val_metrics)

    # 9. Test evaluation
    test_metrics = trainer.evaluate(
        test_ds,
        metric_key_prefix="test",
    )

    print("\n=== Test Results ===")
    print(test_metrics)

    # 10. Save re-runnable artefact
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\nModel saved to: {output_dir}")


if __name__ == "__main__":
    main()