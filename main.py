import json
from pathlib import Path
import time

from rag.generate import SimpleRAG
from evaluation.judge import LLMJudge
from evaluation.metrics import calculate_summary


BENCHMARK_DATASET = "data/benchmark_dataset.json"

OUTPUT_FILE = "outputs/evaluation_results.json"


def load_dataset():

    with open(
        BENCHMARK_DATASET,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_results(results, summary):

    output = {
        "summary": summary,
        "results": results
    }

    Path("outputs").mkdir(
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
            ensure_ascii=False
        )


def main():

    print("\nLoading benchmark dataset...")

    dataset = load_dataset()

    print(
        f"Loaded {len(dataset)} questions"
    )

    print("\nLoading RAG...")

    rag = SimpleRAG()

    print("Loading Judge...")

    judge = LLMJudge()

    results = []

    for idx, item in enumerate(dataset, start=1):

        question = item["question"]

        expected_answer = item["expected_answer"]

        print(
            f"\n[{idx}/{len(dataset)}] Evaluating..."
        )

        generated_answer = (
            rag.generate_answer(question)
        )

        evaluation = judge.evaluate(
            question=question,
            expected_answer=expected_answer,
            generated_answer=generated_answer
        )

        result = {
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "correctness": evaluation.correctness,
            "completeness": evaluation.completeness,
            "relevance": evaluation.relevance,
            "hallucination_risk": evaluation.hallucination_risk,
            "reasoning": evaluation.reasoning
        }

        results.append(result)
        time.sleep(15)

    summary = calculate_summary(
        results
    )

    save_results(
        results,
        summary
    )

    print("\n===== SUMMARY =====\n")

    print(
        f"Questions: {summary['total_questions']}"
    )

    print(
        f"Average Correctness: {summary['avg_correctness']}"
    )

    print(
        f"Average Completeness: {summary['avg_completeness']}"
    )

    print(
        f"Average Relevance: {summary['avg_relevance']}"
    )

    print(
        f"Overall Score: {summary['overall_score']}"
    )

    print(
        "\nHallucination Distribution:"
    )

    print(
        summary[
            "hallucination_distribution"
        ]
    )

    print(
        f"\nDetailed report saved to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()