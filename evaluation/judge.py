import json

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from models.schemas import EvaluationResult


load_dotenv()


class LLMJudge:

    def __init__(self):

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )

    def evaluate(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str
    ) -> EvaluationResult:

        prompt = f"""
You are an expert AI evaluator.

Your task is to compare a generated answer against a reference answer.

Question:
{question}

Reference Answer:
{expected_answer}

Generated Answer:
{generated_answer}

Evaluate the generated answer on:

1. Correctness (1-10)
2. Completeness (1-10)
3. Relevance (1-10)
4. Hallucination Risk (Low, Medium, High)

Scoring Guidelines:

Correctness:
- 10 = factually equivalent to reference
- 7-9 = mostly correct with minor omissions
- 4-6 = partially correct
- 1-3 = mostly incorrect

Completeness:
- 10 = covers all important points
- lower scores if key information is missing

Relevance:
- 10 = directly answers question
- lower scores if answer is off-topic

Hallucination Risk:
- Low = answer stays grounded
- Medium = minor unsupported claims
- High = significant unsupported claims

Return ONLY valid JSON.

Example:

{{
  "correctness": 9,
  "completeness": 8,
  "relevance": 10,
  "hallucination_risk": "Low",
  "reasoning": "Generated answer is correct but omits one detail from the reference."
}}
"""

        response = self.llm.invoke(prompt)

        content = response.content.strip()

        if content.startswith("```json"):
            content = content.replace("```json", "")
            content = content.replace("```", "")
            content = content.strip()

        data = json.loads(content)

        return EvaluationResult(**data)