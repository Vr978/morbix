import pandas as pd
from datasets import load_dataset


def load_disease_dataset(
    dataset_name: str,
    split: str = "train",
    revision: str | None = None,
) -> pd.DataFrame:
    print(f"Loading dataset: {dataset_name}")

    dataset = load_dataset(
        dataset_name,
        split=split,
        revision=revision,
    )

    dataframe = dataset.to_pandas()

    print("Dataset loaded successfully.")
    print(f"Rows: {len(dataframe):,}")
    print(f"Columns: {len(dataframe.columns):,}")

    return dataframe


def print_dataset_summary(
    dataframe: pd.DataFrame,
    target_column: str,
) -> None:
    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    symptom_columns = [
        column
        for column in dataframe.columns
        if column != target_column
    ]

    class_counts = dataframe[target_column].value_counts()

    print("\nDataset summary")
    print("-------------------------")
    print(f"Samples:          {len(dataframe):,}")
    print(f"Symptom columns:  {len(symptom_columns):,}")
    print(f"Disease classes:  {dataframe[target_column].nunique():,}")
    print(f"Smallest class:   {class_counts.min():,} samples")
    print(f"Largest class:    {class_counts.max():,} samples")