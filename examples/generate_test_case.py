import argparse
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_symptoms(
    row: pd.Series,
    feature_names: list[str],
) -> list[str]:
    return [
        symptom
        for symptom in feature_names
        if row[symptom] == 1
    ]


def main(row_index: int, fraction: float) -> None:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")

    config = load_config()

    validation_path = (
        PROJECT_ROOT
        / config["paths"]["processed_data_dir"]
        / "validation.parquet"
    )

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation data not found: {validation_path}"
        )

    dataframe = pd.read_parquet(validation_path)

    if row_index < 0 or row_index >= len(dataframe):
        raise ValueError(
            f"row must be between 0 and {len(dataframe) - 1}"
        )

    target_column = config["dataset"]["target_column"]

    feature_names = [
        column
        for column in dataframe.columns
        if column != target_column
    ]

    row = dataframe.iloc[row_index]

    symptoms = get_symptoms(
        row=row,
        feature_names=feature_names,
    )

    number_to_show = max(
        1,
        round(len(symptoms) * fraction),
    )

    visible_symptoms = symptoms[:number_to_show]

    print("\nTest Case")
    print("=========")

    print(f"\nExpected disease: {row[target_column]}")
    print(f"Total symptoms: {len(symptoms)}")
    print(f"Symptoms used: {len(visible_symptoms)}")

    print("\nInput symptoms:")

    for symptom in visible_symptoms:
        print(f"  - {symptom}")

    command_symptoms = ", ".join(visible_symptoms)

    print("\nRun command:\n")

    print(
        "python3 run_solution.py "
        f'--symptoms "{command_symptoms}" '
        "--show-trace"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--row",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    main(
        row_index=args.row,
        fraction=args.fraction,
    )