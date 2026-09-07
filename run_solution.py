import argparse

from src.agents.symptom_agent import SymptomAgent


def parse_symptoms(symptom_text: str) -> list[str]:
    return [
        symptom.strip()
        for symptom in symptom_text.split(",")
        if symptom.strip()
    ]


def print_trace(trace: list[dict]) -> None:
    print("\nAgent trace:")

    for step in trace:
        step_number = step["step"]
        action = step["action"]
        details = step["details"]

        if "tool" in step:
            print(
                f"  Step {step_number}: "
                f"{action} [{step['tool']}]"
            )
        else:
            print(
                f"  Step {step_number}: {action}"
            )

        print(f"           {details}")


def print_result(
    result: dict,
    show_trace: bool = False,
    show_scores: bool = False,
) -> None:
    print("\nMorbix")
    print("======")

    if result["recognized_symptoms"]:
        print("\nRecognized symptoms:")

        for symptom in result["recognized_symptoms"]:
            print(f"  - {symptom}")

    if result["unknown_symptoms"]:
        print("\nUnrecognized symptoms:")

        for symptom in result["unknown_symptoms"]:
            print(f"  - {symptom}")

    if not result["success"]:
        print(f"\nResult: {result['message']}")

        if show_trace:
            print_trace(result["trace"])

        return

    print("\nRanked disease candidates:")

    for prediction in result["predictions"]:
        rank = prediction["rank"]
        disease = prediction["disease"]

        if show_scores:
            score = prediction["score"]

            print(
                f"  {rank}. {disease} "
                f"(score: {score:.6f})"
            )
        else:
            print(f"  {rank}. {disease}")

    if show_trace:
        print_trace(result["trace"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank disease candidates from a list of symptoms."
    )

    parser.add_argument(
        "--symptoms",
        required=True,
        help='Comma-separated symptoms, for example "cough, fever, fatigue"',
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of disease candidates to return",
    )

    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Show the agent execution steps",
    )

    parser.add_argument(
        "--show-scores",
        action="store_true",
        help="Show classifier scores",
    )

    args = parser.parse_args()

    agent = SymptomAgent(
        top_k=args.top_k
    )

    result = agent.run(
        parse_symptoms(args.symptoms)
    )

    print_result(
        result=result,
        show_trace=args.show_trace,
        show_scores=args.show_scores,
    )


if __name__ == "__main__":
    main()