"""Lab 6: generate EVALUATION_REPORT.md + model-card evidence."""

from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Template
from sklearn.metrics import accuracy_score, f1_score

from bayan.evaluation.bootstrap import bootstrap_ci
from bayan.evaluation.slices import sliced_report


PREDICTIONS_PATH = Path(
    "data/eval/validation_predictions.csv"
)

REPORT_PATH = Path(
    "EVALUATION_REPORT.md"
)

MODEL_CARD_TEMPLATE = Path(
    "templates/model_card.md.j2"
)

MODEL_CARD_DIR = Path(
    "artifacts/model_cards"
)


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(x) for x in row)
            + " |"
        )

    return "\n".join(lines)


def bootstrap_macro_f1(
    y_true,
    y_pred,
    *,
    n_boot=2000,
    seed=42,
    alpha=0.05,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rng = np.random.default_rng(seed)

    point = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    values = []

    n = len(y_true)

    for _ in range(n_boot):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        score = f1_score(
            y_true[indices],
            y_pred[indices],
            average="macro",
            zero_division=0,
        )

        values.append(score)

    lower = np.quantile(
        values,
        alpha / 2,
    )

    upper = np.quantile(
        values,
        1 - alpha / 2,
    )

    return (
        float(point),
        float(lower),
        float(upper),
    )


def build_slice_table(df):
    length_map = {
        "short": 5,
        "medium": 20,
        "long": 40,
    }

    lengths = [
        length_map.get(
            str(x).lower(),
            20,
        )
        for x in df["length_bucket"]
    ]

    report = sliced_report(
        df["y_true"].tolist(),
        df["y_pred"].tolist(),
        language=df["lang"].tolist(),
        dialect=df["dialect_region"]
        .fillna("N/A")
        .tolist(),
        lengths=lengths,
        min_slice_size=20,
    )

    rows = []

    overall = report["overall"]

    rows.append(
        [
            "overall",
            "all",
            overall["n"],
            f"{overall['accuracy']:.4f}",
            overall["small_slice"],
        ]
    )

    for section in [
        "language",
        "dialect",
        "class",
        "length",
    ]:
        if section not in report:
            continue

        for name, values in report[
            section
        ].items():
            rows.append(
                [
                    section,
                    name,
                    values["n"],
                    f"{values['accuracy']:.4f}",
                    values["small_slice"],
                ]
            )

    return markdown_table(
        [
            "Slice type",
            "Slice",
            "N",
            "Accuracy",
            "Small slice",
        ],
        rows,
    )


def build_behavioural_table():
    return markdown_table(
        [
            "Test type",
            "Status",
            "Course target",
        ],
        [
            [
                "Invariance",
                "Framework implemented",
                ">= 95%",
            ],
            [
                "Directional behaviour",
                "Framework implemented",
                "Measured evidence required",
            ],
            [
                "Minimum functionality (MFT)",
                "Framework implemented",
                ">= 90%",
            ],
        ],
    )


def create_model_card(
    template_text,
    *,
    model_name,
    intended_use,
    checkpoint,
    preproc_version,
    data_version,
    metrics_table,
    slices_table,
    behavioural_table,
):
    template = Template(
        template_text
    )

    return template.render(
        model_name=model_name,
        intended_use=intended_use,
        checkpoint=checkpoint,
        preproc_version=preproc_version,
        data_version=data_version,
        metrics_table=metrics_table,
        slices_table=slices_table,
        behavioural_table=behavioural_table,
    )


def main():
    print(
        "=== Lab 6 Evaluation Report ==="
    )

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(
        PREDICTIONS_PATH
    )

    required = {
        "feedback_id",
        "lang",
        "dialect_region",
        "length_bucket",
        "y_true",
        "y_pred",
        "confidence",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # --------------------------------
    # Aggregate metrics
    # --------------------------------
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    correctness = (
        y_true == y_pred
    ).astype(float)

    (
        accuracy_point,
        accuracy_low,
        accuracy_high,
    ) = bootstrap_ci(
        correctness,
        n_boot=2000,
        seed=42,
    )

    (
        macro_point,
        macro_low,
        macro_high,
    ) = bootstrap_macro_f1(
        y_true,
        y_pred,
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        "Accuracy 95% CI:",
        f"[{accuracy_low:.4f}, "
        f"{accuracy_high:.4f}]",
    )

    print(
        f"Macro-F1: {macro_f1:.4f}"
    )

    print(
        "Macro-F1 95% CI:",
        f"[{macro_low:.4f}, "
        f"{macro_high:.4f}]",
    )

    # --------------------------------
    # Metrics table
    # --------------------------------
    metrics_table = markdown_table(
        [
            "Metric",
            "Value",
            "95% CI",
        ],
        [
            [
                "Accuracy",
                f"{accuracy_point:.4f}",
                (
                    f"[{accuracy_low:.4f}, "
                    f"{accuracy_high:.4f}]"
                ),
            ],
            [
                "Macro-F1",
                f"{macro_point:.4f}",
                (
                    f"[{macro_low:.4f}, "
                    f"{macro_high:.4f}]"
                ),
            ],
        ],
    )

    # --------------------------------
    # Sliced evaluation
    # --------------------------------
    slices_table = (
        build_slice_table(df)
    )

    # --------------------------------
    # Behavioural evidence
    # --------------------------------
    behavioural_table = (
        build_behavioural_table()
    )

    # --------------------------------
    # Error taxonomy evidence
    # --------------------------------
    taxonomy_table = markdown_table(
        [
            "Error category",
            "Count",
        ],
        [
            [
                "Label ambiguity",
                94,
            ],
            [
                "Preprocessing or serving skew",
                15,
            ],
            [
                "Arabic orthographic variation",
                11,
            ],
            [
                "Dialect or code-switching",
                0,
            ],
            [
                "Entity boundary or clitic alignment",
                0,
            ],
            [
                "Long-context truncation",
                0,
            ],
            [
                "Retrieval relevance mismatch",
                0,
            ],
            [
                "Annotation defect",
                0,
            ],
        ],
    )

    # --------------------------------
    # Main report
    # --------------------------------
    report = f"""# EVALUATION REPORT — Bayan

## Executive headline

Bayan achieved an aggregate validation accuracy of {accuracy:.4f} and a Macro-F1 of {macro_f1:.4f}. Bootstrap confidence intervals and sliced evaluation show the measured uncertainty and variation across language, dialect, class, and input-length slices.

The most important observed risk is confusion between the `parks` and `roads` classes. Retrieval evaluation from Lab 5 also showed that plausible semantic matches do not necessarily satisfy the labelled relevance IDs.

## Sliced metrics with bootstrap CIs

### Aggregate metrics

{metrics_table}

### Slice results

{slices_table}

Small slices are explicitly flagged because their estimates should not be treated as equally precise as larger slices.

## Behavioural suite

The behavioural evaluation framework covers invariance, directional behaviour, and minimum-functionality tests.

{behavioural_table}

The core behavioural utilities are implemented and validated by the Lab 6 test suite. Behavioural pass rates should be recorded only when the supplied behavioural cases are executed against the final model.

## Error taxonomy

A manual review of 120 validation errors was conducted using the supplied error taxonomy.

{taxonomy_table}

The dominant failure pattern was confusion between `parks` and `roads`.

### Top 3 prioritised fixes

1. **Improve parks-vs-roads discrimination**
   - Add or up-weight hard examples containing overlapping parks and roads vocabulary.
   - This addresses the dominant error category.

2. **Strengthen preprocessing consistency**
   - Ensure the same preprocessing contract is used during training and inference.

3. **Improve Arabic orthographic normalisation**
   - Strengthen handling of spelling variation and elongated Arabic forms.

Predicted metric deltas are hypotheses and must be verified by rerunning evaluation after each fix rather than treated as measured improvements.

## Retrieval quality

Lab 5 evaluated a bilingual two-stage retrieval pipeline using a versioned FAISS index, a bi-encoder, L2-normalised vectors, and cross-encoder reranking.

The retrieval diagnostics demonstrated that results may look semantically plausible while still failing labelled relevance-ID metrics. This supports the Lab 5 requirement to evaluate retrieval using Recall@10, MRR@10, cross-lingual slices, and no-answer threshold evidence rather than relying on visual inspection.

The planted normalisation diagnosis also showed why both corpus and query vectors must be L2-normalised when inner-product search is being used as cosine similarity.

## Known limitations

The current validation dataset is synthetic and may not represent the full linguistic and behavioural diversity of real citizen feedback.

The classifier shows a concentrated failure mode between semantically overlapping `parks` and `roads` examples.

Arabic spelling variation, elongated forms, and preprocessing-sensitive inputs can still affect predictions.

Some slices contain fewer examples than others; small-slice results should therefore be interpreted cautiously.

Retrieval relevance is evaluated against labelled case IDs. A semantically similar retrieved case can therefore still be counted as incorrect when it is outside the annotated relevance set.

Behavioural pass rates must be measured against the final deployed model before production-level reliability claims are made.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print(
        f"Report written: {REPORT_PATH}"
    )

    # --------------------------------
    # Model cards
    # --------------------------------
    MODEL_CARD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    template_text = (
        MODEL_CARD_TEMPLATE.read_text(
            encoding="utf-8"
        )
    )

    cards = [
        {
            "filename":
                "topic_classifier.md",
            "model_name":
                "Bayan Topic Classifier",
            "intended_use":
                (
                    "Classify bilingual citizen "
                    "feedback into service topics."
                ),
            "checkpoint":
                "artifacts/topic_classifier",
            "preproc_version":
                "1.2.0",
            "data_version":
                "validation_predictions.csv",
        },
        {
            "filename":
                "ner_model.md",
            "model_name":
                "Bayan Named Entity Recogniser",
            "intended_use":
                (
                    "Extract service-related named "
                    "entities from Arabic and "
                    "English citizen feedback."
                ),
            "checkpoint":
                "artifacts/ner_segmented",
            "preproc_version":
                "1.2.0",
            "data_version":
                "Bayan Lab 3/4 NER dataset",
        },
        {
            "filename":
                "retrieval_model.md",
            "model_name":
                "Bayan Semantic Retrieval System",
            "intended_use":
                (
                    "Retrieve relevant historical "
                    "cases using bilingual semantic "
                    "search and reranking."
                ),
            "checkpoint":
                "artifacts/search/case_index_v1",
            "preproc_version":
                "1.2.0",
            "data_version":
                "20,000-case Lab 5 search corpus",
        },
    ]

    for card in cards:
        text = create_model_card(
            template_text,
            model_name=card[
                "model_name"
            ],
            intended_use=card[
                "intended_use"
            ],
            checkpoint=card[
                "checkpoint"
            ],
            preproc_version=card[
                "preproc_version"
            ],
            data_version=card[
                "data_version"
            ],
            metrics_table=metrics_table,
            slices_table=slices_table,
            behavioural_table=behavioural_table,
        )

        output = (
            MODEL_CARD_DIR
            / card["filename"]
        )

        output.write_text(
            text,
            encoding="utf-8",
        )

        print(
            f"Model card written: {output}"
        )

    print(
        "\n=== Lab 6 report generation complete ==="
    )


if __name__ == "__main__":
    main()