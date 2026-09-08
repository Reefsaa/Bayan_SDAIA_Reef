"""Lab 4: compare Arabic-centric checkpoints on all/Gulf/MSA slices."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


DATA_PATH = Path("data/raw/bayan_feedback.csv")

MODELS = {
    "CAMeLBERT-mix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "CAMeLBERT-DA": "CAMeL-Lab/bert-base-arabic-camelbert-da",
}

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def load_arabic_data():
    df = pd.read_csv(DATA_PATH)

    required = {
        "text",
        "topic",
        "citizen_group_id",
        "language",
        "dialect",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(
        subset=[
            "text",
            "topic",
            "citizen_group_id",
            "language",
            "dialect",
        ]
    ).copy()

    df = df[
        df["language"].astype(str).str.lower().eq("ar")
    ].copy()

    return df


def grouped_split(df):
    splitter1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=SEED,
    )

    train_idx, temp_idx = next(
        splitter1.split(
            df,
            groups=df["citizen_group_id"],
        )
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=SEED,
    )

    val_idx, test_idx = next(
        splitter2.split(
            temp_df,
            groups=temp_df["citizen_group_id"],
        )
    )

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def build_label_maps(df):
    labels = sorted(df["topic"].astype(str).unique())

    label2id = {
        label: idx
        for idx, label in enumerate(labels)
    }

    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    return label2id, id2label


def make_dataset(df, label2id):
    return Dataset.from_dict({
        "text": df["text"].astype(str).tolist(),
        "labels": [
            label2id[str(label)]
            for label in df["topic"]
        ],
    })


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    return {
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
        ),
    }


def evaluate_slice(
    trainer,
    tokenizer,
    df,
    label2id,
    prefix,
):
    if len(df) == 0:
        return {
            "count": 0,
            "macro_f1": None,
            "accuracy": None,
        }

    ds = make_dataset(
        df,
        label2id,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
        )

    ds = ds.map(
        tokenize,
        batched=True,
    )

    metrics = trainer.evaluate(
        ds,
        metric_key_prefix=prefix,
    )

    return {
        "count": len(df),
        "macro_f1": metrics[f"{prefix}_macro_f1"],
        "accuracy": metrics[f"{prefix}_accuracy"],
    }


def train_and_evaluate(
    model_name,
    checkpoint,
    train_df,
    val_df,
    test_df,
    label2id,
    id2label,
):
    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print(f"CHECKPOINT: {checkpoint}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint
    )

    train_ds = make_dataset(
        train_df,
        label2id,
    )

    val_ds = make_dataset(
        val_df,
        label2id,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
        )

    train_ds = train_ds.map(
        tokenize,
        batched=True,
    )

    val_ds = val_ds.map(
        tokenize,
        batched=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    output_dir = (
        Path("artifacts")
        / "arabic_bakeoff"
        / model_name.lower().replace("-", "_")
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer
        ),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    all_results = evaluate_slice(
        trainer,
        tokenizer,
        test_df,
        label2id,
        "all",
    )

    gulf_df = test_df[
        test_df["dialect"]
        .astype(str)
        .str.lower()
        .eq("gulf")
    ].reset_index(drop=True)

    msa_df = test_df[
        test_df["dialect"]
        .astype(str)
        .str.lower()
        .eq("msa")
    ].reset_index(drop=True)

    gulf_results = evaluate_slice(
        trainer,
        tokenizer,
        gulf_df,
        label2id,
        "gulf",
    )

    msa_results = evaluate_slice(
        trainer,
        tokenizer,
        msa_df,
        label2id,
        "msa",
    )

    trainer.save_model(
        str(output_dir)
    )

    tokenizer.save_pretrained(
        str(output_dir)
    )

    return {
        "model": model_name,
        "all": all_results,
        "gulf": gulf_results,
        "msa": msa_results,
    }


def print_results(results):
    print("\n")
    print("=" * 78)
    print("LAB 4 — ARABIC MODEL BAKE-OFF RESULTS")
    print("=" * 78)

    for result in results:
        print(f"\n{result['model']}")

        for slice_name in [
            "all",
            "gulf",
            "msa",
        ]:
            values = result[slice_name]

            print(
                f"  {slice_name.upper():5s} | "
                f"N={values['count']:4d} | "
                f"Macro-F1={values['macro_f1']:.4f} | "
                f"Accuracy={values['accuracy']:.4f}"
            )


def main():
    set_seed()

    print("Loading Arabic Bayan data...")

    df = load_arabic_data()

    print(f"Arabic records: {len(df)}")

    print("\nDialect distribution:")
    print(df["dialect"].value_counts())

    train_df, val_df, test_df = grouped_split(
        df
    )

    print("\nGrouped split:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    label2id, id2label = build_label_maps(
        train_df
    )

    print(f"\nLabels: {list(label2id.keys())}")

    results = []

    for model_name, checkpoint in MODELS.items():
        result = train_and_evaluate(
            model_name,
            checkpoint,
            train_df,
            val_df,
            test_df,
            label2id,
            id2label,
        )

        results.append(result)

    print_results(results)

    winner = max(
        results,
        key=lambda x: (
            -1
            if x["gulf"]["macro_f1"] is None
            else x["gulf"]["macro_f1"]
        ),
    )

    print("\n" + "=" * 78)
    print(
        "WINNER BY GULF-SLICE MACRO-F1:"
        f" {winner['model']}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()