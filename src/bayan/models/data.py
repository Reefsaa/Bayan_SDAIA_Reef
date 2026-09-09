"""Lab 3A: dataset construction and grouped split integrity."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


DATA_PATH = Path("data/raw/bayan_feedback.csv")


def build_topic_dataset(
    data_path=DATA_PATH,
    test_size=0.20,
    validation_size=0.20,
    random_state=42,
):
    """Build grouped train/validation/test splits for topic classification."""

    df = pd.read_csv(data_path)

    required_columns = {
        "citizen_group_id",
        "topic",
        "text",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(
        subset=["citizen_group_id", "topic", "text"]
    ).copy()

    # Split off the test set by citizen group.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_valid_idx, test_idx = next(
        splitter.split(
            df,
            y=df["topic"],
            groups=df["citizen_group_id"],
        )
    )

    train_valid = df.iloc[train_valid_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    # Split the remaining data into train and validation.
    valid_relative_size = validation_size / (1 - test_size)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=valid_relative_size,
        random_state=random_state,
    )

    train_idx, valid_idx = next(
        splitter.split(
            train_valid,
            y=train_valid["topic"],
            groups=train_valid["citizen_group_id"],
        )
    )

    train = train_valid.iloc[train_idx].reset_index(drop=True)
    validation = train_valid.iloc[valid_idx].reset_index(drop=True)

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }
