"""Lab 3A: TF-IDF + LinearSVC baseline."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.svm import LinearSVC


DATA_PATH = Path("data/raw/bayan_feedback.csv")


def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Use the supplied train/validation split
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"].isin(["validation", "val"])].copy()

    if train_df.empty:
        raise ValueError("No training rows found.")

    if val_df.empty:
        raise ValueError(
            f"No validation rows found. Available splits: "
            f"{df['split'].unique().tolist()}"
        )

    # TF-IDF converts text into numerical features
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
    )

    X_train = vectorizer.fit_transform(train_df["text"].fillna(""))
    X_val = vectorizer.transform(val_df["text"].fillna(""))

    y_train = train_df["topic"]
    y_val = val_df["topic"]

    # Train LinearSVC classifier
    model = LinearSVC()
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_val)

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
    )

    print("=== Lab 3A: TF-IDF + LinearSVC Baseline ===")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print()
    print(classification_report(y_val, predictions))

    # Append evidence to BENCHMARKS.md
    with open("BENCHMARKS.md", "a", encoding="utf-8") as f:
        f.write("\n## Lab 3A — TF-IDF + LinearSVC Baseline\n\n")
        f.write(f"- Train rows: {len(train_df)}\n")
        f.write(f"- Validation rows: {len(val_df)}\n")
        f.write(f"- Macro-F1: {macro_f1:.4f}\n")

    print("\nBaseline result appended to BENCHMARKS.md.")


if __name__ == "__main__":
    main()