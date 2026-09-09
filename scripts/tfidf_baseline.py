"""Lab 3A: TF-IDF + LinearSVC baseline."""

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import LinearSVC

from bayan.models.data import build_topic_dataset


BENCHMARKS_PATH = Path("BENCHMARKS.md")


def main():
    ds = build_topic_dataset()

    train = ds["train"]
    validation = ds["validation"]
    test = ds["test"]

    # TF-IDF representation.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(train["text"])
    X_validation = vectorizer.transform(validation["text"])
    X_test = vectorizer.transform(test["text"])

    # Linear SVM classifier.
    classifier = LinearSVC(
        random_state=42,
    )

    classifier.fit(X_train, train["topic"])

    validation_predictions = classifier.predict(X_validation)
    test_predictions = classifier.predict(X_test)

    validation_f1 = f1_score(
        validation["topic"],
        validation_predictions,
        average="macro",
    )

    test_f1 = f1_score(
        test["topic"],
        test_predictions,
        average="macro",
    )

    print("\n=== TF-IDF + LinearSVC Baseline ===")
    print(f"Train samples:      {len(train)}")
    print(f"Validation samples: {len(validation)}")
    print(f"Test samples:       {len(test)}")
    print(f"Validation Macro-F1: {validation_f1:.4f}")
    print(f"Test Macro-F1:       {test_f1:.4f}")

    print("\n=== Test Classification Report ===")
    print(
        classification_report(
            test["topic"],
            test_predictions,
            zero_division=0,
        )
    )

    # Save benchmark information.
    benchmark_text = f"""
## Lab 3A — TF-IDF + LinearSVC Baseline

- Validation Macro-F1: `{validation_f1:.4f}`
- Test Macro-F1: `{test_f1:.4f}`
- Random state: `42`
- Features: TF-IDF word n-grams `(1, 2)`
- Classifier: LinearSVC
"""

    with BENCHMARKS_PATH.open("a", encoding="utf-8") as file:
        file.write("\n" + benchmark_text)

    print(f"Benchmark appended to {BENCHMARKS_PATH}")


if __name__ == "__main__":
    main()
