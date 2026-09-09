"""Lab 3 starter: dataset construction and split integrity."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


DATA_PATH = Path("data/raw/bayan_feedback.csv")


def build_topic_dataset(*args, **kwargs):
    df = pd.read_csv(DATA_PATH)

    # Keep only rows needed for topic classification
    df = df.dropna(subset=["text", "topic", "citizen_group_id"]).copy()

    groups = df["citizen_group_id"]

    # First split: train 70%, temp 30%
    splitter1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=42,
    )

    train_idx, temp_idx = next(
        splitter1.split(df, groups=groups)
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # Second split: validation 15%, test 15%
    temp_groups = temp_df["citizen_group_id"]

    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42,
    )

    val_idx, test_idx = next(
        splitter2.split(temp_df, groups=temp_groups)
    )

    validation_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }