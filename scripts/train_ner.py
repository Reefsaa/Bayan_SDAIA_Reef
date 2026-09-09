"""Lab 3B / Lab 4: Fine-tune NER token classification."""

import argparse
import random
from pathlib import Path

import numpy as np
from datasets import Dataset
from seqeval.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from bayan.models.ner import align_labels


CHECKPOINT = "xlm-roberta-base"
DATA_PATH = Path("data/models/bayan_ner.conll")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/ner",
        help="Where to save the trained NER artefact.",
    )

    parser.add_argument(
        "--use-segmentation",
        action="store_true",
        help="Apply Arabic clitic segmentation before NER training.",
    )

    return parser.parse_args()


def read_conll(path):
    sentences = []
    labels = []

    current_tokens = []
    current_labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
                continue

            parts = line.split()

            token = parts[0]
            label = parts[-1]

            current_tokens.append(token)
            current_labels.append(label)

    if current_tokens:
        sentences.append(current_tokens)
        labels.append(current_labels)

    return sentences, labels


def apply_segmentation(sentences, labels):
    """
    Apply Arabic clitic segmentation only when explicitly requested.

    Importing segment here avoids requiring camel_tools when
    segmentation is not being used.
    """

    from bayan.preprocessing.arabic import segment

    segmented_sentences = []
    segmented_labels = []

    for tokens, token_labels in zip(sentences, labels):
        new_tokens = []
        new_labels = []

        for token, label in zip(tokens, token_labels):
            pieces = segment(token)

            if not pieces:
                pieces = [token]

            for piece_index, piece in enumerate(pieces):
                new_tokens.append(piece)

                if piece_index == 0:
                    new_labels.append(label)
                else:
                    new_labels.append(label)

        segmented_sentences.append(new_tokens)
        segmented_labels.append(new_labels)

    return segmented_sentences, segmented_labels


def split_data(tokens, labels, seed=42):
    indices = list(range(len(tokens)))

    random.Random(seed).shuffle(indices)

    n = len(indices)

    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    def select(idxs):
        return (
            [tokens[i] for i in idxs],
            [labels[i] for i in idxs],
        )

    return (
        select(train_idx),
        select(val_idx),
        select(test_idx),
    )


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read CoNLL data
    sentences, ner_labels = read_conll(DATA_PATH)

    # 2. Optional Lab 4 Arabic clitic segmentation
    if args.use_segmentation:
        print("Applying Arabic clitic segmentation...")

        sentences, ner_labels = apply_segmentation(
            sentences,
            ner_labels,
        )

    # 3. Build label mappings
    unique_labels = sorted(
        {
            label
            for sequence in ner_labels
            for label in sequence
        }
    )

    label2id = {
        label: idx
        for idx, label in enumerate(unique_labels)
    }

    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    numeric_labels = [
        [
            label2id[label]
            for label in sequence
        ]
        for sequence in ner_labels
    ]

    # 4. Train / validation / test split
    (
        (train_tokens, train_labels),
        (val_tokens, val_labels),
        (test_tokens, test_labels),
    ) = split_data(
        sentences,
        numeric_labels,
    )

    train_ds = Dataset.from_dict({
        "tokens": train_tokens,
        "ner_tags": train_labels,
    })

    val_ds = Dataset.from_dict({
        "tokens": val_tokens,
        "ner_tags": val_labels,
    })

    test_ds = Dataset.from_dict({
        "tokens": test_tokens,
        "ner_tags": test_labels,
    })

    # 5. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT,
        use_fast=True,
    )

    # 6. Tokenize + align labels to subwords
    def tokenize_and_align(batch):
        tokenized = tokenizer(
            batch["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=128,
        )

        aligned_batch = []

        for i, word_labels in enumerate(
            batch["ner_tags"]
        ):
            word_ids = tokenized.word_ids(
                batch_index=i
            )

            aligned = align_labels(
                word_ids,
                word_labels,
            )

            aligned_batch.append(aligned)

        tokenized["labels"] = aligned_batch

        return tokenized

    train_ds = train_ds.map(
        tokenize_and_align,
        batched=True,
    )

    val_ds = val_ds.map(
        tokenize_and_align,
        batched=True,
    )

    test_ds = test_ds.map(
        tokenize_and_align,
        batched=True,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    # 7. Load NER model
    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
    )

    # 8. Entity-level seqeval metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        predictions = np.argmax(
            logits,
            axis=-1,
        )

        true_predictions = []
        true_labels = []

        for prediction, label in zip(
            predictions,
            labels,
        ):
            pred_sequence = []
            label_sequence = []

            for pred_id, label_id in zip(
                prediction,
                label,
            ):
                if label_id == -100:
                    continue

                pred_sequence.append(
                    id2label[int(pred_id)]
                )

                label_sequence.append(
                    id2label[int(label_id)]
                )

            true_predictions.append(
                pred_sequence
            )

            true_labels.append(
                label_sequence
            )

        return {
            "precision": precision_score(
                true_labels,
                true_predictions,
            ),
            "recall": recall_score(
                true_labels,
                true_predictions,
            ),
            "f1": f1_score(
                true_labels,
                true_predictions,
            ),
            "accuracy": accuracy_score(
                true_labels,
                true_predictions,
            ),
        }

    # 9. Training configuration
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,

        # New Transformers API
        eval_strategy="epoch",

        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
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
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("=== NER Fine-tuning ===")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Segmentation: {args.use_segmentation}")
    print(f"Sentences: {len(sentences)}")
    print(f"Train: {len(train_ds)}")
    print(f"Validation: {len(val_ds)}")
    print(f"Test: {len(test_ds)}")
    print(f"Labels: {unique_labels}")

    # 10. Train
    trainer.train()

    # 11. Validation evaluation
    val_metrics = trainer.evaluate(
        val_ds
    )

    print()
    print("=== Validation Results ===")
    print(val_metrics)

    # 12. Test evaluation
    test_metrics = trainer.evaluate(
        test_ds,
        metric_key_prefix="test",
    )

    print()
    print("=== Test Results ===")
    print(test_metrics)

    # 13. Save artefact
    trainer.save_model(
        str(output_dir)
    )

    tokenizer.save_pretrained(
        str(output_dir)
    )

    print()
    print(
        f"NER model saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()