from src.tools.disease_predictor import DiseasePredictorTool


class SymptomAgent:
    def __init__(
        self,
        model_path: str = "models/disease_model.joblib",
        top_k: int = 5,
    ):
        self.top_k = top_k
        self.disease_tool = DiseasePredictorTool(
            model_path=model_path
        )

    @staticmethod
    def prepare_symptoms(
        symptoms: list[str],
    ) -> list[str]:
        cleaned = []

        for symptom in symptoms:
            symptom = symptom.strip()

            if symptom and symptom not in cleaned:
                cleaned.append(symptom)

        return cleaned

    def run(
        self,
        symptoms: list[str],
    ) -> dict:
        cleaned_symptoms = self.prepare_symptoms(symptoms)

        trace = [
            {
                "step": 1,
                "action": "receive_symptoms",
                "details": f"Received {len(cleaned_symptoms)} symptoms.",
            }
        ]

        if not cleaned_symptoms:
            trace.append(
                {
                    "step": 2,
                    "action": "stop",
                    "details": "No symptoms were provided.",
                }
            )

            return {
                "success": False,
                "agent": "symptom_agent",
                "trace": trace,
                "recognized_symptoms": [],
                "unknown_symptoms": [],
                "predictions": [],
                "message": "At least one symptom is required.",
            }

        trace.append(
            {
                "step": 2,
                "action": "call_tool",
                "tool": self.disease_tool.name,
                "details": "Rank disease candidates from the provided symptoms.",
            }
        )

        result = self.disease_tool.run(
            symptoms=cleaned_symptoms,
            top_k=self.top_k,
        )

        if not result["success"]:
            trace.append(
                {
                    "step": 3,
                    "action": "stop",
                    "details": result["error"],
                }
            )

            return {
                "success": False,
                "agent": "symptom_agent",
                "trace": trace,
                "recognized_symptoms": result["recognized_symptoms"],
                "unknown_symptoms": result["unknown_symptoms"],
                "predictions": [],
                "message": result["error"],
            }

        trace.append(
            {
                "step": 3,
                "action": "return_prediction",
                "details": "Return the ranked disease candidates.",
            }
        )

        return {
            "success": True,
            "agent": "symptom_agent",
            "trace": trace,
            "recognized_symptoms": result["recognized_symptoms"],
            "unknown_symptoms": result["unknown_symptoms"],
            "predictions": result["predictions"],
            "message": "Prediction completed.",
        }