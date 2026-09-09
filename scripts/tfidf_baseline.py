"""Lab 3A: TF-IDF + LinearSVC baseline using grouped splits."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.svm import LinearSVC

from bayan.models.data import build_topic_dataset


def main():
    ds = build_topic_dataset()

    train_df = ds["train"]
    val_df = ds["validation"]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
    )

    X_train = vectorizer.fit_transform(train_df["text"].fillna(""))
    X_val = vectorizer.transform(val_df["text"].fillna(""))

    y_train = train_df["topic"]
    y_val = val_df["topic"]

    model = LinearSVC()
    model.fit(X_train, y_train)

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

    with open("BENCHMARKS.md", "a", encoding="utf-8") as f:
        f.write("\n## Lab 3A — TF-IDF + LinearSVC Baseline (Grouped Split)\n\n")
        f.write(f"- Train rows: {len(train_df)}\n")
        f.write(f"- Validation rows: {len(val_df)}\n")
        f.write(f"- Macro-F1: {macro_f1:.4f}\n")

    print("\nGrouped baseline result appended to BENCHMARKS.md.")


if __name__ == "__main__":
    main()