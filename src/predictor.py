from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DiseasePredictor:
    def __init__(
        self,
        model_path: str = "models/disease_model.joblib",
    ):
        self.model_path = PROJECT_ROOT / model_path

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Run python3 -m src.train first."
            )

        self.load_model()

    def load_model(self) -> None:
        bundle = joblib.load(self.model_path)

        required_fields = {
            "model",
            "feature_names",
            "disease_classes",
        }

        missing_fields = required_fields - bundle.keys()

        if missing_fields:
            raise ValueError(
                f"Model file is missing fields: {sorted(missing_fields)}"
            )

        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]
        self.disease_classes = bundle["disease_classes"]

        self.feature_index = {
            symptom: index
            for index, symptom in enumerate(self.feature_names)
        }

        self.normalized_features = {
            self.normalize_symptom(symptom): symptom
            for symptom in self.feature_names
        }

    @staticmethod
    def normalize_symptom(symptom: str) -> str:
        return symptom.strip().lower()

    def match_symptoms(
        self,
        symptoms: list[str],
    ) -> tuple[list[str], list[str]]:
        recognized = []
        unknown = []
        seen = set()

        for symptom in symptoms:
            normalized = self.normalize_symptom(symptom)

            if not normalized or normalized in seen:
                continue

            seen.add(normalized)

            if normalized in self.normalized_features:
                recognized.append(
                    self.normalized_features[normalized]
                )
            else:
                unknown.append(symptom.strip())

        return recognized, unknown

    def create_feature_vector(
        self,
        symptoms: list[str],
    ) -> np.ndarray:
        vector = np.zeros(
            len(self.feature_names),
            dtype=np.float32,
        )

        for symptom in symptoms:
            vector[self.feature_index[symptom]] = 1.0

        return vector.reshape(1, -1)

    def predict(
        self,
        symptoms: list[str],
        top_k: int = 5,
    ) -> dict:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if top_k > len(self.disease_classes):
            raise ValueError(
                "top_k cannot be larger than the number of disease classes."
            )

        recognized, unknown = self.match_symptoms(symptoms)

        if not recognized:
            return {
                "success": False,
                "recognized_symptoms": [],
                "unknown_symptoms": unknown,
                "predictions": [],
                "error": "None of the provided symptoms matched the model vocabulary.",
            }

        feature_vector = self.create_feature_vector(recognized)
        probabilities = self.model.predict_proba(feature_vector)[0]

        top_indices = np.argsort(probabilities)[::-1][:top_k]

        predictions = []

        for rank, index in enumerate(top_indices, start=1):
            predictions.append(
                {
                    "rank": rank,
                    "disease": str(self.model.classes_[index]),
                    "score": float(probabilities[index]),
                }
            )

        return {
            "success": True,
            "recognized_symptoms": recognized,
            "unknown_symptoms": unknown,
            "predictions": predictions,
            "error": None,
        }


def main() -> None:
    predictor = DiseasePredictor()

    print(
        f"Model loaded with {len(predictor.feature_names)} symptoms "
        f"and {len(predictor.disease_classes)} diseases."
    )


if __name__ == "__main__":
    main()