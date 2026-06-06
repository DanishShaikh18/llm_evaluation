from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):

    correctness: int = Field(
        ge=1,
        le=10
    )

    completeness: int = Field(
        ge=1,
        le=10
    )

    relevance: int = Field(
        ge=1,
        le=10
    )

    hallucination_risk: str

    reasoning: str