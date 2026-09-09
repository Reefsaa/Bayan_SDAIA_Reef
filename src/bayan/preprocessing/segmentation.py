"""Lab 1: audit four tokenizer candidates on Bayan AR/EN text."""

from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}

DATA = Path("data/raw/bayan_feedback.csv")


def fertility(tokenizer, texts) -> float:
    """Calculate total subword pieces divided by whitespace words."""
    total_pieces = 0
    total_words = 0

    for text in texts:
        text = str(text)

        words = text.split()
        pieces = tokenizer.tokenize(text)

        total_words += len(words)
        total_pieces += len(pieces)

    if total_words == 0:
        return 0.0

    return total_pieces / total_words


def sequence_lengths(tokenizer, texts):
    """Return tokenized sequence lengths including special tokens."""
    lengths = []

    for text in texts:
        encoded = tokenizer.encode(
            str(text),
            add_special_tokens=True,
            truncation=False,
        )
        lengths.append(len(encoded))

    return lengths


def main():
    df = pd.read_csv(DATA)

    # Support either raw_text or text as the feedback column
    if "raw_text" in df.columns:
        text_column = "raw_text"
    elif "text" in df.columns:
        text_column = "text"
    else:
        raise ValueError(
            "Could not find a text column. Expected 'raw_text' or 'text'."
        )

    ar_texts = (
        df.loc[df["lang"].str.lower() == "ar", text_column]
        .dropna()
        .astype(str)
        .tolist()
    )

    en_texts = (
        df.loc[df["lang"].str.lower() == "en", text_column]
        .dropna()
        .astype(str)
        .tolist()
    )

    results = []

    for checkpoint, name in CANDIDATES.items():
        print(f"\nLoading {name}: {checkpoint}")

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        ar_fertility = fertility(tokenizer, ar_texts)
        en_fertility = fertility(tokenizer, en_texts)

        ar_lengths = sequence_lengths(tokenizer, ar_texts)
        en_lengths = sequence_lengths(tokenizer, en_texts)

        ar_p95 = float(np.percentile(ar_lengths, 95))
        en_p95 = float(np.percentile(en_lengths, 95))

        results.append(
            {
                "Tokenizer": name,
                "AR Fertility": round(ar_fertility, 3),
                "EN Fertility": round(en_fertility, 3),
                "AR p95 Length": round(ar_p95, 1),
                "EN p95 Length": round(en_p95, 1),
            }
        )

    results_df = pd.DataFrame(results)

    print("\nTokenizer Audit Results")
    print(results_df.to_string(index=False))

    benchmark_path = Path("BENCHMARKS.md")

    with benchmark_path.open("a", encoding="utf-8") as f:
        f.write("\n\n## Lab 1 — Tokenizer Audit\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n")

    print("\nResults added to BENCHMARKS.md")


if __name__ == "__main__":
    main()