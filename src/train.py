import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str) -> dict:
    path = PROJECT_ROOT / config_path

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_metadata(
    processed_data_dir: Path,
) -> dict:
    path = processed_data_dir / "metadata.json"

    if not path.exists():
        raise FileNotFoundError(
            "Processed dataset metadata was not found. "
            "Run python3 -m src.preprocess first."
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_data(
    processed_data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = processed_data_dir / "train.parquet"
    validation_path = processed_data_dir / "validation.parquet"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}"
        )

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation data not found: {validation_path}"
        )

    print("Loading processed datasets...")

    train_df = pd.read_parquet(train_path)
    validation_df = pd.read_parquet(validation_path)

    print("Processed datasets loaded.")

    return train_df, validation_df


def validate_data(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    symptom_columns: list[str],
    target_column: str,
) -> None:
    required_columns = set(symptom_columns + [target_column])

    missing_train = required_columns - set(train_df.columns)
    missing_validation = required_columns - set(validation_df.columns)

    if missing_train:
        raise ValueError(
            f"Training data is missing columns: "
            f"{sorted(missing_train)[:10]}"
        )

    if missing_validation:
        raise ValueError(
            f"Validation data is missing columns: "
            f"{sorted(missing_validation)[:10]}"
        )

    train_classes = set(
        train_df[target_column].unique()
    )

    validation_classes = set(
        validation_df[target_column].unique()
    )

    if train_classes != validation_classes:
        raise ValueError(
            "Training and validation sets contain different disease classes."
        )


def prepare_features(
    dataframe: pd.DataFrame,
    symptom_columns: list[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    X = dataframe[
        symptom_columns
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    y = dataframe[
        target_column
    ].astype(str).to_numpy()

    return X, y


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
        [
            class_to_index[disease]
            for disease in y_true
        ],
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


def evaluate_model(
    model: GaussianNB,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    top_k_values: list[int],
) -> dict:
    start_time = time.perf_counter()

    probabilities = model.predict_proba(
        X_validation
    )

    inference_time = time.perf_counter() - start_time

    predictions = model.classes_[
        np.argmax(probabilities, axis=1)
    ]

    metrics = {
        "top_1_accuracy": float(
            accuracy_score(
                y_validation,
                predictions,
            )
        )
    }

    for k in top_k_values:
        if k == 1:
            continue

        metrics[f"top_{k}_accuracy"] = top_k_accuracy(
            y_true=y_validation,
            probabilities=probabilities,
            classes=model.classes_,
            k=k,
        )

    metrics["validation_inference_seconds"] = float(
        inference_time
    )

    metrics["average_inference_ms_per_sample"] = float(
        inference_time
        / len(y_validation)
        * 1000
    )

    return metrics


def save_model(
    model: GaussianNB,
    symptom_columns: list[str],
    target_column: str,
    metadata: dict,
    metrics: dict,
    model_path: Path,
) -> None:
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bundle = {
        "model": model,
        "feature_names": symptom_columns,
        "target_column": target_column,
        "disease_classes": model.classes_.tolist(),
        "dataset": {
            "name": metadata["dataset"]["name"],
            "revision": metadata["dataset"]["revision"],
        },
        "metrics": metrics,
    }

    joblib.dump(
        bundle,
        model_path,
        compress=3,
    )


def save_metrics(
    metrics: dict,
    metrics_path: Path,
) -> None:
    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )


def print_metrics(
    metrics: dict,
) -> None:
    print("\nValidation results")
    print("-------------------------")

    for k in (1, 3, 5):
        name = f"top_{k}_accuracy"

        if name in metrics:
            score = metrics[name]

            print(
                f"Top-{k} accuracy: "
                f"{score:.4f} ({score * 100:.2f}%)"
            )

    print(
        f"\nValidation inference time: "
        f"{metrics['validation_inference_seconds']:.4f} seconds"
    )

    print(
        f"Average inference time: "
        f"{metrics['average_inference_ms_per_sample']:.4f} ms/sample"
    )


def main(config_path: str) -> None:
    config = load_config(config_path)

    dataset_config = config["dataset"]
    model_config = config["model"]
    evaluation_config = config["evaluation"]
    paths = config["paths"]

    processed_data_dir = (
        PROJECT_ROOT
        / paths["processed_data_dir"]
    )

    model_path = (
        PROJECT_ROOT
        / paths["model_path"]
    )

    metrics_path = (
        PROJECT_ROOT
        / paths["metrics_path"]
    )

    metadata = load_metadata(
        processed_data_dir
    )

    symptom_columns = metadata[
        "features"
    ]["symptoms"]

    target_column = dataset_config[
        "target_column"
    ]

    train_df, validation_df = load_data(
        processed_data_dir
    )

    validate_data(
        train_df=train_df,
        validation_df=validation_df,
        symptom_columns=symptom_columns,
        target_column=target_column,
    )

    print("\nTraining configuration")
    print("-------------------------")
    print(f"Model:               {model_config['name']}")
    print(f"Training samples:    {len(train_df):,}")
    print(f"Validation samples:  {len(validation_df):,}")
    print(f"Symptom features:    {len(symptom_columns):,}")
    print(
        f"Disease classes:     "
        f"{metadata['labels']['number_of_diseases']:,}"
    )

    X_train, y_train = prepare_features(
        train_df,
        symptom_columns,
        target_column,
    )

    X_validation, y_validation = prepare_features(
        validation_df,
        symptom_columns,
        target_column,
    )

    del train_df
    del validation_df

    model = GaussianNB(
        var_smoothing=float(
            model_config["var_smoothing"]
        )
    )

    print("\nTraining Gaussian Naive Bayes...")

    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    training_time = time.perf_counter() - start_time

    print("Training complete.")
    print(f"Training time: {training_time:.4f} seconds")
    print(f"Learned disease classes: {len(model.classes_):,}")

    expected_classes = metadata[
        "labels"
    ]["number_of_diseases"]

    if len(model.classes_) != expected_classes:
        raise ValueError(
            f"Expected {expected_classes} disease classes, "
            f"but the model learned {len(model.classes_)}."
        )

    print("\nEvaluating on validation data...")

    metrics = evaluate_model(
        model=model,
        X_validation=X_validation,
        y_validation=y_validation,
        top_k_values=evaluation_config["top_k"],
    )

    metrics.update(
        {
            "training_seconds": float(training_time),
            "training_samples": int(len(y_train)),
            "validation_samples": int(len(y_validation)),
            "number_of_symptoms": int(len(symptom_columns)),
            "number_of_diseases": int(len(model.classes_)),
            "model_name": model_config["name"],
            "var_smoothing": float(
                model_config["var_smoothing"]
            ),
        }
    )

    print_metrics(metrics)

    save_model(
        model=model,
        symptom_columns=symptom_columns,
        target_column=target_column,
        metadata=metadata,
        metrics=metrics,
        model_path=model_path,
    )

    model_size = (
        model_path.stat().st_size
        / (1024 * 1024)
    )

    metrics["model_size_mb"] = float(model_size)

    save_metrics(
        metrics,
        metrics_path,
    )

    print("\nSaved files")
    print("-------------------------")
    print(f"Model:   {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Model size: {model_size:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config/config.yaml",
    )

    args = parser.parse_args()

    main(args.config)