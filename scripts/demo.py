"""Демонстрация на вымышленных JSON; не обращается к интернету или Ollama."""
import argparse
import sys
from pathlib import Path

# Работает и из архива без установки пакета.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resume_evaluator.candidate_assessment_pipeline import run_candidate_assessment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/demo"))
    args = parser.parse_args()
    fixtures = ROOT / "examples/precomputed"
    result = run_candidate_assessment(fixtures / "marketing_candidate.example.json",
        "Маркетолог / Middle", args.output,
        resume_evaluation_path=fixtures / "marketing_resume_evaluation.example.json",
        market_profile_path=fixtures / "marketing_market_profile.json",
        use_comparison_embeddings=False, use_recommendation_llm=False)
    print("ДЕМО: вымышленные резюме, оценка и вакансии; актуальный рынок не проверялся.")
    print(result.report_markdown)


if __name__ == "__main__":
    main()
