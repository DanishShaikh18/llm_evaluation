from collections import Counter


def calculate_summary(results):

    total = len(results)

    if total == 0:
        return {}

    avg_correctness = (
        sum(r["correctness"] for r in results) / total
    )

    avg_completeness = (
        sum(r["completeness"] for r in results) / total
    )

    avg_relevance = (
        sum(r["relevance"] for r in results) / total
    )

    overall_score = (
        avg_correctness +
        avg_completeness +
        avg_relevance
    ) / 3

    hallucination_counts = Counter(
        r["hallucination_risk"]
        for r in results
    )

    return {
        "total_questions": total,
        "avg_correctness": round(
            avg_correctness,
            2
        ),
        "avg_completeness": round(
            avg_completeness,
            2
        ),
        "avg_relevance": round(
            avg_relevance,
            2
        ),
        "overall_score": round(
            overall_score,
            2
        ),
        "hallucination_distribution": dict(
            hallucination_counts
        )
    }