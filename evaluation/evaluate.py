import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str) -> dict:
    path = PROJECT_ROOT / config_path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_model(model_path: Path) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)


def top_k_accuracy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    k: int,
) -> float:
    class_to_index = {
        disease: index
        for index, disease in enumerate(classes)
    }

    true_indices = np.array(
        [class_to_index[disease] for disease in y_true],
        dtype=np.int32,
    )

    top_indices = np.argpartition(
        probabilities,
        kth=-k,
        axis=1,
    )[:, -k:]

    correct = np.any(
        top_indices == true_indices[:, None],
        axis=1,
    )

    return float(np.mean(correct))


def evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    top_k_values: list[int],
) -> dict:
    probabilities = model.predict_proba(X)

    results = {}

    for k in top_k_values:
        results[f"top_{k}_accuracy"] = top_k_accuracy(
            y_true=y,
            probabilities=probabilities,
            classes=model.classes_,
            k=k,
        )

    return results


def print_results(
    results: dict,
    split_name: str,
    sample_count: int,
) -> None:
    print(f"\nEvaluation on {split_name} set")
    print("-------------------------")
    print(f"Samples: {sample_count:,}")

    for name, score in results.items():
        k = name.split("_")[1]

        print(
            f"Top-{k} accuracy: "
            f"{score:.4f} ({score * 100:.2f}%)"
        )


def main(
    config_path: str,
    split_name: str,
) -> None:
    config = load_config(config_path)

    processed_data_dir = (
        PROJECT_ROOT
        / config["paths"]["processed_data_dir"]
    )

    data_path = processed_data_dir / f"{split_name}.parquet"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data split not found: {data_path}"
        )

    model_path = (
        PROJECT_ROOT
        / config["paths"]["model_path"]
    )

    bundle = load_model(model_path)

    model = bundle["model"]
    feature_names = bundle["feature_names"]
    target_column = bundle["target_column"]

    dataframe = pd.read_parquet(data_path)

    X = dataframe[
        feature_names
    ].to_numpy(dtype=np.float32)

    y = dataframe[
        target_column
    ].astype(str).to_numpy()

    results = evaluate(
        model=model,
        X=X,
        y=y,
        top_k_values=config["evaluation"]["top_k"],
    )

    print_results(
        results=results,
        split_name=split_name,
        sample_count=len(dataframe),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config/config.yaml",
    )

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        split_name=args.split,
    )