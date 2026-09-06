"""Lab 1 starter: audit four tokenizer candidates on Bayan AR/EN text."""

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
    """Return total subword pieces divided by total whitespace words."""

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


def main():
    """Audit tokenizers on Arabic and English Bayan feedback."""

    df = pd.read_csv(DATA)

    ar_texts = df.loc[df["lang"].str.lower() == "ar", "text"].dropna().tolist()
    en_texts = df.loc[df["lang"].str.lower() == "en", "text"].dropna().tolist()

    results = []

    for checkpoint, name in CANDIDATES.items():
        print(f"\nLoading {name} ({checkpoint})...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        ar_fertility = fertility(tokenizer, ar_texts)
        en_fertility = fertility(tokenizer, en_texts)

        ar_lengths = [
            len(tokenizer(text, add_special_tokens=True)["input_ids"])
            for text in ar_texts
        ]

        en_lengths = [
            len(tokenizer(text, add_special_tokens=True)["input_ids"])
            for text in en_texts
        ]

        ar_p95 = float(np.percentile(ar_lengths, 95))
        en_p95 = float(np.percentile(en_lengths, 95))

        results.append(
            {
                "Tokenizer": name,
                "Checkpoint": checkpoint,
                "AR Fertility": ar_fertility,
                "EN Fertility": en_fertility,
                "AR Mean Length": np.mean(ar_lengths),
                "EN Mean Length": np.mean(en_lengths),
                "AR p95": ar_p95,
                "EN p95": en_p95,
            }
        )

    results_df = pd.DataFrame(results)

    print("\nTokenizer Audit Results")
    print(results_df.to_string(index=False))

    print("\nP95 sequence length per tokenizer/language:")
    for _, row in results_df.iterrows():
        print(
            f"{row['Tokenizer']}: "
            f"AR p95={row['AR p95']:.1f}, "
            f"EN p95={row['EN p95']:.1f}"
        )


if __name__ == "__main__":
    main()