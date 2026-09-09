"""Lab 7: latency and quality benchmark for classifier and NER."""

import os
import time
import random
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

from sklearn.metrics import accuracy_score, f1_score as sklearn_f1
from seqeval.metrics import (
    accuracy_score as ner_accuracy_score,
    f1_score as ner_f1_score,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from bayan.models.data import build_topic_dataset
from bayan.models.ner import align_labels


# ============================================================
# Paths
# ============================================================

CLASSIFIER_MODEL_DIR = Path(
    "artifacts/topic_classifier"
)

CLASSIFIER_FP32_PATH = Path(
    "artifacts/onnx/topic_classifier/model_fp32.onnx"
)

CLASSIFIER_INT8_PATH = Path(
    "artifacts/onnx/topic_classifier/model_int8.onnx"
)

NER_MODEL_DIR = Path(
    "artifacts/ner"
)

NER_FP32_PATH = Path(
    "artifacts/onnx/ner/model_fp32.onnx"
)

NER_INT8_PATH = Path(
    "artifacts/onnx/ner/model_int8.onnx"
)

NER_DATA_PATH = Path(
    "data/models/bayan_ner.conll"
)


# ============================================================
# Benchmark settings
# ============================================================

THREADS = 4
WARMUP_RUNS = 10
BENCH_RUNS = 50

TEXTS = [
    "الخدمة ممتازة وسريعة",
    "الطريق يحتاج إلى صيانة وتحسين الإنارة",
    "الحديقة جميلة ولكن تحتاج إلى المزيد من النظافة",
    "There is a problem with the road near my house.",
    "The public service was excellent and very fast.",
    "نحتاج إلى تحسين الخدمات العامة في المنطقة بشكل عاجل.",
    "The park needs better lighting and regular maintenance.",
    "يوجد ازدحام شديد في الطريق الرئيسي ويحتاج إلى تنظيم أفضل.",
]


# ============================================================
# Helpers
# ============================================================

def make_ort_session(model_path):
    """Create CPU ONNX Runtime session."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX artefact not found: {model_path}"
        )

    options = ort.SessionOptions()

    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1

    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def benchmark_torch(
    model,
    tokenizer,
    *,
    max_length=512,
    dynamic=False,
):
    """Benchmark PyTorch classifier inference."""

    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    os.environ["MKL_NUM_THREADS"] = str(THREADS)

    torch.set_num_threads(THREADS)

    model.to("cpu")
    model.eval()

    def encode(text):

        if dynamic:
            return tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

        return tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    with torch.inference_mode():

        for i in range(WARMUP_RUNS):

            inputs = encode(
                TEXTS[i % len(TEXTS)]
            )

            model(**inputs)

    latencies = []

    with torch.inference_mode():

        for i in range(BENCH_RUNS):

            inputs = encode(
                TEXTS[i % len(TEXTS)]
            )

            start = time.perf_counter()

            model(**inputs)

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            latencies.append(elapsed)

    return {
        "p50": float(
            np.percentile(latencies, 50)
        ),
        "p99": float(
            np.percentile(latencies, 99)
        ),
    }


def benchmark_onnx(
    tokenizer,
    model_path,
    *,
    max_length=128,
):
    """Benchmark ONNX Runtime inference."""

    session = make_ort_session(
        model_path
    )

    def encode(text):

        encoded = tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        return {
            "input_ids":
                encoded["input_ids"].astype(
                    np.int64
                ),

            "attention_mask":
                encoded["attention_mask"].astype(
                    np.int64
                ),
        }

    for i in range(WARMUP_RUNS):

        inputs = encode(
            TEXTS[i % len(TEXTS)]
        )

        session.run(
            None,
            inputs,
        )

    latencies = []

    for i in range(BENCH_RUNS):

        inputs = encode(
            TEXTS[i % len(TEXTS)]
        )

        start = time.perf_counter()

        session.run(
            None,
            inputs,
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed)

    return {
        "p50": float(
            np.percentile(latencies, 50)
        ),
        "p99": float(
            np.percentile(latencies, 99)
        ),
    }


# ============================================================
# Classifier quality
# ============================================================

def evaluate_classifier_quality(
    tokenizer,
    model_path,
    test_df,
    label2id,
):
    """Evaluate classifier Accuracy and Macro-F1."""

    session = make_ort_session(
        model_path
    )

    y_true = []
    y_pred = []

    for _, row in test_df.iterrows():

        encoded = tokenizer(
            row["text"],
            return_tensors="np",
            truncation=True,
            max_length=128,
        )

        inputs = {
            "input_ids":
                encoded["input_ids"].astype(
                    np.int64
                ),

            "attention_mask":
                encoded["attention_mask"].astype(
                    np.int64
                ),
        }

        logits = session.run(
            None,
            inputs,
        )[0]

        prediction = int(
            np.argmax(
                logits,
                axis=-1,
            )[0]
        )

        y_true.append(
            label2id[row["topic"]]
        )

        y_pred.append(
            prediction
        )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "macro_f1": float(
            sklearn_f1(
                y_true,
                y_pred,
                average="macro",
            )
        ),
    }


# ============================================================
# NER data
# ============================================================

def read_ner_conll(path):
    """Read the same CoNLL fixture used during NER training."""

    sentences = []
    labels = []

    current_tokens = []
    current_labels = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:

                if current_tokens:

                    sentences.append(
                        current_tokens
                    )

                    labels.append(
                        current_labels
                    )

                    current_tokens = []
                    current_labels = []

                continue

            parts = line.split()

            current_tokens.append(
                parts[0]
            )

            current_labels.append(
                parts[-1]
            )

    if current_tokens:

        sentences.append(
            current_tokens
        )

        labels.append(
            current_labels
        )

    return sentences, labels


def get_ner_test_split(
    sentences,
    labels,
    seed=42,
):
    """Reproduce the Lab 3B 80/10/10 split."""

    indices = list(
        range(len(sentences))
    )

    random.Random(seed).shuffle(
        indices
    )

    n = len(indices)

    train_end = int(
        n * 0.8
    )

    val_end = int(
        n * 0.9
    )

    test_indices = indices[
        val_end:
    ]

    test_sentences = [
        sentences[i]
        for i in test_indices
    ]

    test_labels = [
        labels[i]
        for i in test_indices
    ]

    return (
        test_sentences,
        test_labels,
    )


# ============================================================
# NER quality
# ============================================================

def evaluate_ner_quality(
    tokenizer,
    model_path,
    test_sentences,
    test_labels,
    label2id,
    id2label,
):
    """
    Evaluate NER using the same label alignment
    and seqeval F1 used in Lab 3B.
    """

    session = make_ort_session(
        model_path
    )

    true_predictions = []
    true_labels = []

    for tokens, labels in zip(
        test_sentences,
        test_labels,
    ):

        encoded = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="np",
            truncation=True,
            max_length=128,
        )

        word_ids = encoded.word_ids(
            batch_index=0
        )

        numeric_word_labels = [
            label2id[label]
            for label in labels
        ]

        aligned_labels = align_labels(
            word_ids,
            numeric_word_labels,
        )

        inputs = {
            "input_ids":
                encoded["input_ids"].astype(
                    np.int64
                ),

            "attention_mask":
                encoded["attention_mask"].astype(
                    np.int64
                ),
        }

        logits = session.run(
            None,
            inputs,
        )[0]

        predictions = np.argmax(
            logits,
            axis=-1,
        )[0]

        pred_sequence = []
        label_sequence = []

        for pred_id, label_id in zip(
            predictions,
            aligned_labels,
        ):

            if label_id == -100:
                continue

            pred_sequence.append(
                id2label[
                    int(pred_id)
                ]
            )

            label_sequence.append(
                id2label[
                    int(label_id)
                ]
            )

        true_predictions.append(
            pred_sequence
        )

        true_labels.append(
            label_sequence
        )

    return {
        "f1": float(
            ner_f1_score(
                true_labels,
                true_predictions,
            )
        ),

        "accuracy": float(
            ner_accuracy_score(
                true_labels,
                true_predictions,
            )
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    os.environ["OMP_NUM_THREADS"] = str(
        THREADS
    )

    os.environ["MKL_NUM_THREADS"] = str(
        THREADS
    )

    torch.set_num_threads(
        THREADS
    )

    # ========================================================
    # CLASSIFIER
    # ========================================================

    print()
    print(
        "========== CLASSIFIER =========="
    )

    classifier_tokenizer = (
        AutoTokenizer.from_pretrained(
            CLASSIFIER_MODEL_DIR
        )
    )

    classifier_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            CLASSIFIER_MODEL_DIR
        )
    )

    print(
        f"CPU threads pinned to: {THREADS}"
    )

    print(
        f"Benchmark runs: {BENCH_RUNS}"
    )

    print()

    print(
        "=== FP32 Torch @512 padded ==="
    )

    baseline = benchmark_torch(
        classifier_model,
        classifier_tokenizer,
        max_length=512,
        dynamic=False,
    )

    print(
        f"p50: {baseline['p50']:.2f} ms"
    )

    print(
        f"p99: {baseline['p99']:.2f} ms"
    )

    print()

    print(
        "=== FP32 Torch @128 dynamic ==="
    )

    dynamic = benchmark_torch(
        classifier_model,
        classifier_tokenizer,
        max_length=128,
        dynamic=True,
    )

    print(
        f"p50: {dynamic['p50']:.2f} ms"
    )

    print(
        f"p99: {dynamic['p99']:.2f} ms"
    )

    del classifier_model

    print()

    print(
        "=== ONNX FP32 @128 dynamic ==="
    )

    classifier_fp32 = benchmark_onnx(
        classifier_tokenizer,
        CLASSIFIER_FP32_PATH,
    )

    print(
        f"p50: {classifier_fp32['p50']:.2f} ms"
    )

    print(
        f"p99: {classifier_fp32['p99']:.2f} ms"
    )

    print()

    print(
        "=== ONNX INT8 @128 dynamic ==="
    )

    classifier_int8 = benchmark_onnx(
        classifier_tokenizer,
        CLASSIFIER_INT8_PATH,
    )

    print(
        f"p50: {classifier_int8['p50']:.2f} ms"
    )

    print(
        f"p99: {classifier_int8['p99']:.2f} ms"
    )

    classifier_speedup = (
        baseline["p99"]
        / classifier_int8["p99"]
    )

    print(
        "p99 speed-up vs Torch @512: "
        f"{classifier_speedup:.2f}x"
    )

    # Classifier quality
    print()
    print(
        "=== Classifier Quality Check ==="
    )

    splits = build_topic_dataset()

    train_df = splits[
        "train"
    ].copy()

    test_df = splits[
        "test"
    ].copy()

    classifier_labels = sorted(
        train_df["topic"].unique()
    )

    classifier_label2id = {
        label: idx
        for idx, label
        in enumerate(classifier_labels)
    }

    classifier_fp32_quality = (
        evaluate_classifier_quality(
            classifier_tokenizer,
            CLASSIFIER_FP32_PATH,
            test_df,
            classifier_label2id,
        )
    )

    classifier_int8_quality = (
        evaluate_classifier_quality(
            classifier_tokenizer,
            CLASSIFIER_INT8_PATH,
            test_df,
            classifier_label2id,
        )
    )

    classifier_tax = (
        classifier_fp32_quality[
            "macro_f1"
        ]
        -
        classifier_int8_quality[
            "macro_f1"
        ]
    ) * 100

    print(
        "FP32 Macro-F1: "
        f"{classifier_fp32_quality['macro_f1']:.4f}"
    )

    print(
        "INT8 Macro-F1: "
        f"{classifier_int8_quality['macro_f1']:.4f}"
    )

    print(
        "Quality tax: "
        f"{classifier_tax:.2f} points"
    )

    classifier_pass = (
        classifier_int8["p99"] <= 25
        and classifier_speedup >= 6
        and classifier_tax <= 1
    )

    print(
        "Classifier decision: "
        f"{'PASS - use INT8' if classifier_pass else 'FAIL'}"
    )

    # ========================================================
    # NER
    # ========================================================

    print()
    print(
        "============== NER =============="
    )

    if not NER_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"NER artefact not found: {NER_MODEL_DIR}"
        )

    ner_tokenizer = (
        AutoTokenizer.from_pretrained(
            NER_MODEL_DIR,
            use_fast=True,
        )
    )

    print()
    print(
        "=== NER ONNX FP32 @128 ==="
    )

    ner_fp32 = benchmark_onnx(
        ner_tokenizer,
        NER_FP32_PATH,
        max_length=128,
    )

    print(
        f"p50: {ner_fp32['p50']:.2f} ms"
    )

    print(
        f"p99: {ner_fp32['p99']:.2f} ms"
    )

    print()

    print(
        "=== NER ONNX INT8 @128 ==="
    )

    ner_int8 = benchmark_onnx(
        ner_tokenizer,
        NER_INT8_PATH,
        max_length=128,
    )

    print(
        f"p50: {ner_int8['p50']:.2f} ms"
    )

    print(
        f"p99: {ner_int8['p99']:.2f} ms"
    )

    ner_speedup = (
        ner_fp32["p99"]
        / ner_int8["p99"]
    )

    print(
        "p99 speed-up vs NER ONNX FP32: "
        f"{ner_speedup:.2f}x"
    )

    # ========================================================
    # NER quality
    # ========================================================

    print()
    print(
        "=== NER Quality Check ==="
    )

    sentences, ner_labels = (
        read_ner_conll(
            NER_DATA_PATH
        )
    )

    test_sentences, test_labels = (
        get_ner_test_split(
            sentences,
            ner_labels,
            seed=42,
        )
    )

    unique_ner_labels = sorted(
        {
            label
            for sequence in ner_labels
            for label in sequence
        }
    )

    ner_label2id = {
        label: idx
        for idx, label
        in enumerate(unique_ner_labels)
    }

    ner_id2label = {
        idx: label
        for label, idx
        in ner_label2id.items()
    }

    print(
        f"NER test sentences: {len(test_sentences)}"
    )

    print(
        f"NER labels: {unique_ner_labels}"
    )

    print()
    print(
        "Evaluating NER FP32..."
    )

    ner_fp32_quality = (
        evaluate_ner_quality(
            ner_tokenizer,
            NER_FP32_PATH,
            test_sentences,
            test_labels,
            ner_label2id,
            ner_id2label,
        )
    )

    print(
        "NER FP32 Accuracy: "
        f"{ner_fp32_quality['accuracy']:.4f}"
    )

    print(
        "NER FP32 F1: "
        f"{ner_fp32_quality['f1']:.4f}"
    )

    print()
    print(
        "Evaluating NER INT8..."
    )

    ner_int8_quality = (
        evaluate_ner_quality(
            ner_tokenizer,
            NER_INT8_PATH,
            test_sentences,
            test_labels,
            ner_label2id,
            ner_id2label,
        )
    )

    print(
        "NER INT8 Accuracy: "
        f"{ner_int8_quality['accuracy']:.4f}"
    )

    print(
        "NER INT8 F1: "
        f"{ner_int8_quality['f1']:.4f}"
    )

    ner_quality_tax = (
        ner_fp32_quality["f1"]
        -
        ner_int8_quality["f1"]
    ) * 100

    print()

    print(
        "NER F1 quality tax: "
        f"{ner_quality_tax:.2f} points"
    )

    ner_quality_pass = (
        ner_quality_tax <= 1.0
    )

    print(
        "NER quality tax <= 1 F1 point: "
        f"{'PASS' if ner_quality_pass else 'FAIL'}"
    )

    # ========================================================
    # NER decision
    # ========================================================

    print()
    print(
        "=== NER Quantisation Decision ==="
    )

    if ner_quality_pass:

        print(
            "PASS - NER INT8 preserves quality."
        )

        print(
            "Decision: use ONNX INT8 for NER."
        )

    else:

        print(
            "FAIL - NER INT8 quality tax "
            "is greater than 1 F1 point."
        )

        print(
            "Decision: keep NER FP32."
        )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print(
        "========== LAB 7 SUMMARY =========="
    )

    print(
        "Classifier INT8: "
        f"p99={classifier_int8['p99']:.2f} ms, "
        f"F1 tax={classifier_tax:.2f} points"
    )

    print(
        "NER FP32: "
        f"p99={ner_fp32['p99']:.2f} ms, "
        f"F1={ner_fp32_quality['f1']:.4f}"
    )

    print(
        "NER INT8: "
        f"p99={ner_int8['p99']:.2f} ms, "
        f"F1={ner_int8_quality['f1']:.4f}, "
        f"tax={ner_quality_tax:.2f} points"
    )


if __name__ == "__main__":
    main()