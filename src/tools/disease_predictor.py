from src.predictor import DiseasePredictor


class DiseasePredictorTool:
    name = "disease_predictor"
    description = "Ranks disease candidates from a list of symptoms."

    def __init__(
        self,
        model_path: str = "models/disease_model.joblib",
    ):
        self.predictor = DiseasePredictor(
            model_path=model_path
        )

    def run(
        self,
        symptoms: list[str],
        top_k: int = 5,
    ) -> dict:
        return self.predictor.predict(
            symptoms=symptoms,
            top_k=top_k,
        )