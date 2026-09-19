from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

def evaluate_rag_response(question: str, actual_output: str, retrieval_context: list[str]) -> dict:
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )
    
    faithfulness_metric = FaithfulnessMetric(threshold=0.7)
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
    
    faithfulness_metric.measure(test_case)
    relevancy_metric.measure(test_case)

    return {
        "faithfulness_score": round(faithfulness_metric.score, 2),
        "faithfulness_passed": faithfulness_metric.is_successful(),
        "relevancy_score": round(relevancy_metric.score, 2),
        "relevancy_passed": relevancy_metric.is_successful(),
    }