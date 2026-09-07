import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data_loader import (
    load_disease_dataset,
    print_dataset_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str) -> dict:
    path = PROJECT_ROOT / config_path

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def remove_rare_diseases(
    dataframe: pd.DataFrame,
    target_column: str,
    minimum_samples: int,
) -> tuple[pd.DataFrame, dict]:
    disease_counts = dataframe[target_column].value_counts()

    rare_diseases = disease_counts[
        disease_counts < minimum_samples
    ]

    if rare_diseases.empty:
        return dataframe, {}

    print("\nDiseases removed because of low sample count")
    print("--------------------------------------------")

    for disease, count in rare_diseases.sort_values().items():
        print(f"{disease}: {count}")

    dataframe = dataframe[
        ~dataframe[target_column].isin(rare_diseases.index)
    ].reset_index(drop=True)

    removed = {
        str(disease): int(count)
        for disease, count in rare_diseases.items()
    }

    return dataframe, removed


def validate_symptom_columns(
    dataframe: pd.DataFrame,
    symptom_columns: list[str],
) -> pd.DataFrame:
    invalid_columns = []

    for column in symptom_columns:
        try:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="raise",
            )
        except (ValueError, TypeError):
            invalid_columns.append(column)
            continue

        if dataframe[column].isna().any():
            invalid_columns.append(column)
            continue

        if not np.isin(
            dataframe[column].unique(),
            [0, 1],
        ).all():
            invalid_columns.append(column)

    if invalid_columns:
        raise ValueError(
            "Invalid symptom columns found: "
            f"{invalid_columns[:10]}"
        )

    dataframe[symptom_columns] = dataframe[
        symptom_columns
    ].astype("uint8")

    return dataframe


def clean_dataset(
    dataframe: pd.DataFrame,
    target_column: str,
    remove_exact_duplicates: bool,
    remove_empty_symptom_rows: bool,
    drop_constant_symptoms: bool,
    minimum_samples_per_disease: int,
) -> tuple[pd.DataFrame, list[str], dict]:
    dataframe = dataframe.copy()

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    if len(dataframe.columns) != len(set(dataframe.columns)):
        raise ValueError("Duplicate column names were found.")

    original_rows = len(dataframe)

    symptom_columns = [
        column
        for column in dataframe.columns
        if column != target_column
    ]

    original_symptom_count = len(symptom_columns)

    dataframe[target_column] = (
        dataframe[target_column]
        .astype("string")
        .str.strip()
    )

    missing_labels = (
        dataframe[target_column].isna()
        | dataframe[target_column].eq("")
    )

    missing_labels_removed = int(missing_labels.sum())

    if missing_labels_removed:
        dataframe = dataframe[
            ~missing_labels
        ].reset_index(drop=True)

    dataframe = validate_symptom_columns(
        dataframe,
        symptom_columns,
    )

    duplicates_removed = 0

    if remove_exact_duplicates:
        duplicates_removed = int(
            dataframe.duplicated().sum()
        )

        dataframe = (
            dataframe
            .drop_duplicates()
            .reset_index(drop=True)
        )

    empty_rows_removed = 0

    if remove_empty_symptom_rows:
        empty_mask = (
            dataframe[symptom_columns].sum(axis=1) == 0
        )

        empty_rows_removed = int(empty_mask.sum())

        dataframe = dataframe[
            ~empty_mask
        ].reset_index(drop=True)

    dataframe, rare_diseases_removed = remove_rare_diseases(
        dataframe=dataframe,
        target_column=target_column,
        minimum_samples=minimum_samples_per_disease,
    )

    if dataframe.empty:
        raise ValueError("No samples remain after cleaning.")

    constant_symptoms = []

    if drop_constant_symptoms:
        constant_symptoms = [
            column
            for column in symptom_columns
            if dataframe[column].nunique() <= 1
        ]

        dataframe = dataframe.drop(
            columns=constant_symptoms
        )

        symptom_columns = [
            column
            for column in symptom_columns
            if column not in constant_symptoms
        ]

    report = {
        "original_rows": original_rows,
        "cleaned_rows": len(dataframe),
        "missing_labels_removed": missing_labels_removed,
        "duplicates_removed": duplicates_removed,
        "empty_rows_removed": empty_rows_removed,
        "minimum_samples_per_disease": minimum_samples_per_disease,
        "rare_disease_classes_removed": len(rare_diseases_removed),
        "rare_diseases_removed": rare_diseases_removed,
        "original_symptom_columns": original_symptom_count,
        "final_symptom_columns": len(symptom_columns),
        "constant_symptoms_removed": constant_symptoms,
        "disease_classes": int(
            dataframe[target_column].nunique()
        ),
    }

    return dataframe, symptom_columns, report


def validate_split_sizes(
    train_size: float,
    validation_size: float,
    test_size: float,
) -> None:
    total = train_size + validation_size + test_size

    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Dataset split sizes must add up to 1.0. Current total: {total}"
        )


def split_dataset(
    dataframe: pd.DataFrame,
    target_column: str,
    train_size: float,
    validation_size: float,
    test_size: float,
    random_seed: int,
    minimum_samples_per_disease: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_split_sizes(
        train_size,
        validation_size,
        test_size,
    )

    smallest_class = int(
        dataframe[target_column]
        .value_counts()
        .min()
    )

    if smallest_class < minimum_samples_per_disease:
        raise ValueError(
            "At least one disease class has fewer than "
            f"{minimum_samples_per_disease} samples."
        )

    holdout_size = validation_size + test_size

    train_df, holdout_df = train_test_split(
        dataframe,
        test_size=holdout_size,
        random_state=random_seed,
        stratify=dataframe[target_column],
        shuffle=True,
    )

    test_fraction = test_size / holdout_size

    validation_df, test_df = train_test_split(
        holdout_df,
        test_size=test_fraction,
        random_state=random_seed,
        stratify=holdout_df[target_column],
        shuffle=True,
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def verify_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
) -> None:
    train_classes = set(
        train_df[target_column].unique()
    )

    validation_classes = set(
        validation_df[target_column].unique()
    )

    test_classes = set(
        test_df[target_column].unique()
    )

    if train_classes != validation_classes:
        raise ValueError(
            "Training and validation sets contain different disease classes."
        )

    if train_classes != test_classes:
        raise ValueError(
            "Training and test sets contain different disease classes."
        )

    print("\nSplit verification")
    print("-------------------------")
    print(f"Diseases in train:      {len(train_classes):,}")
    print(f"Diseases in validation: {len(validation_classes):,}")
    print(f"Diseases in test:       {len(test_classes):,}")


def save_processed_data(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    symptom_columns: list[str],
    target_column: str,
    cleaning_report: dict,
    output_directory: Path,
    dataset_config: dict,
    preprocessing_config: dict,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = output_directory / "train.parquet"
    validation_path = output_directory / "validation.parquet"
    test_path = output_directory / "test.parquet"

    train_df.to_parquet(
        train_path,
        index=False,
    )

    validation_df.to_parquet(
        validation_path,
        index=False,
    )

    test_df.to_parquet(
        test_path,
        index=False,
    )

    diseases = sorted(
        train_df[target_column]
        .astype(str)
        .unique()
        .tolist()
    )

    metadata = {
        "dataset": {
            "name": dataset_config["name"],
            "revision": dataset_config.get("revision"),
            "source_split": dataset_config["split"],
            "target_column": target_column,
        },
        "splits": {
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "test_rows": len(test_df),
            "train_fraction": preprocessing_config["train_size"],
            "validation_fraction": preprocessing_config[
                "validation_size"
            ],
            "test_fraction": preprocessing_config["test_size"],
            "random_seed": preprocessing_config["random_seed"],
        },
        "features": {
            "number_of_symptoms": len(symptom_columns),
            "symptoms": symptom_columns,
        },
        "labels": {
            "number_of_diseases": len(diseases),
            "diseases": diseases,
        },
        "cleaning": cleaning_report,
    }

    with open(
        output_directory / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    with open(
        output_directory / "symptoms.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            symptom_columns,
            file,
            indent=2,
        )

    with open(
        output_directory / "diseases.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            diseases,
            file,
            indent=2,
        )

    print("\nSaved processed data")
    print("-------------------------")
    print(f"Train:       {train_path}")
    print(f"Validation:  {validation_path}")
    print(f"Test:        {test_path}")
    print(f"Metadata:    {output_directory / 'metadata.json'}")


def print_cleaning_report(
    report: dict,
) -> None:
    print("\nCleaning report")
    print("-------------------------")

    print(f"Original rows:                {report['original_rows']:,}")
    print(f"Cleaned rows:                 {report['cleaned_rows']:,}")
    print(f"Missing labels removed:       {report['missing_labels_removed']:,}")
    print(f"Duplicates removed:           {report['duplicates_removed']:,}")
    print(f"Empty rows removed:           {report['empty_rows_removed']:,}")
    print(
        f"Rare disease classes removed: "
        f"{report['rare_disease_classes_removed']:,}"
    )
    print(
        f"Original symptom columns:     "
        f"{report['original_symptom_columns']:,}"
    )
    print(
        f"Final symptom columns:        "
        f"{report['final_symptom_columns']:,}"
    )
    print(
        f"Constant symptoms removed:    "
        f"{len(report['constant_symptoms_removed']):,}"
    )
    print(f"Disease classes:              {report['disease_classes']:,}")


def main(config_path: str) -> None:
    config = load_config(config_path)

    dataset_config = config["dataset"]
    preprocessing_config = config["preprocessing"]

    dataframe = load_disease_dataset(
        dataset_name=dataset_config["name"],
        split=dataset_config["split"],
        revision=dataset_config.get("revision"),
    )

    print_dataset_summary(
        dataframe,
        dataset_config["target_column"],
    )

    print("\nCleaning dataset...")

    cleaned_df, symptom_columns, report = clean_dataset(
        dataframe=dataframe,
        target_column=dataset_config["target_column"],
        remove_exact_duplicates=preprocessing_config[
            "remove_exact_duplicates"
        ],
        remove_empty_symptom_rows=preprocessing_config[
            "remove_empty_symptom_rows"
        ],
        drop_constant_symptoms=preprocessing_config[
            "drop_constant_symptoms"
        ],
        minimum_samples_per_disease=preprocessing_config[
            "minimum_samples_per_disease"
        ],
    )

    print_cleaning_report(report)

    print("\nCreating train/validation/test splits...")

    train_df, validation_df, test_df = split_dataset(
        dataframe=cleaned_df,
        target_column=dataset_config["target_column"],
        train_size=preprocessing_config["train_size"],
        validation_size=preprocessing_config["validation_size"],
        test_size=preprocessing_config["test_size"],
        random_seed=preprocessing_config["random_seed"],
        minimum_samples_per_disease=preprocessing_config[
            "minimum_samples_per_disease"
        ],
    )

    verify_splits(
        train_df,
        validation_df,
        test_df,
        dataset_config["target_column"],
    )

    print("\nSplit sizes")
    print("-------------------------")
    print(f"Train:       {len(train_df):,}")
    print(f"Validation:  {len(validation_df):,}")
    print(f"Test:        {len(test_df):,}")
    print(
        f"Total:       "
        f"{len(train_df) + len(validation_df) + len(test_df):,}"
    )

    output_directory = (
        PROJECT_ROOT
        / config["paths"]["processed_data_dir"]
    )

    save_processed_data(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        symptom_columns=symptom_columns,
        target_column=dataset_config["target_column"],
        cleaning_report=report,
        output_directory=output_directory,
        dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
    )

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config/config.yaml",
    )

    args = parser.parse_args()

    main(args.config)