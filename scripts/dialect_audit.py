"""Lab 4: audit dialect mix over the Arabic slice."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/bayan_feedback.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    # Arabic slice only
    arabic = df[df["lang"].str.lower() == "ar"].copy()

    counts = arabic["dialect_region"].value_counts(dropna=False)
    percentages = arabic["dialect_region"].value_counts(
        normalize=True, dropna=False
    ) * 100

    print("=== Lab 4: Dialect Audit ===")
    print(f"Total records: {len(df)}")
    print(f"Arabic records: {len(arabic)}")
    print()

    print("Dialect / region distribution:")
    for dialect, count in counts.items():
        pct = percentages.loc[dialect]
        print(f"{dialect}: {count} ({pct:.2f}%)")


if __name__ == "__main__":
    main()