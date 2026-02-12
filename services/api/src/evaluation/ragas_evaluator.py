"""
RAGAS evaluation framework for RAG system quality assessment.

Evaluates:
- Faithfulness: Does the answer contain only information from the context?
- Context Precision: Are the most relevant documents retrieved?
- Answer Relevancy: Does the answer address the query?
- Context Recall: Is all relevant information retrieved?
"""

from typing import Dict, List, Optional
import asyncio
import logging

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    RAGAS-based evaluation for RAG system quality.

    Medical safety requirement: Faithfulness score must be >0.95
    to ensure no hallucinations in clinical context.
    """

    def __init__(self):
        """Initialize RAGAS evaluator."""
        if not RAGAS_AVAILABLE:
            logger.warning("RAGAS not installed. Run: pip install ragas")
            raise ImportError("RAGAS library required for evaluation")

        self.metrics = [
            faithfulness,  # Most critical: no hallucinations
            answer_relevancy,  # Does answer address the question?
            context_precision,  # Are top results most relevant?
            context_recall,  # Is all relevant context retrieved?
        ]
        logger.info("Initialized RAGAS evaluator with all metrics")

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single query-answer pair.

        Args:
            question: User's query
            answer: Generated answer
            contexts: Retrieved context strings
            ground_truth: Expected answer (optional, for context_recall)

        Returns:
            Dictionary of metric scores (0-1)
        """
        # Prepare data in RAGAS format
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }

        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        # Evaluate
        try:
            # Select metrics based on available data
            metrics_to_use = self.metrics[:3]  # faithfulness, relevancy, precision
            if ground_truth:
                metrics_to_use = self.metrics  # Include recall

            result = evaluate(dataset, metrics=metrics_to_use)

            scores = {
                "faithfulness": result["faithfulness"],
                "answer_relevancy": result["answer_relevancy"],
                "context_precision": result["context_precision"],
            }

            if ground_truth:
                scores["context_recall"] = result["context_recall"]

            logger.info(f"Evaluation scores: {scores}")
            return scores

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            raise

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a batch of query-answer pairs.

        Args:
            questions: List of queries
            answers: List of generated answers
            contexts_list: List of context lists (one per query)
            ground_truths: Optional expected answers

        Returns:
            Dictionary of average metric scores
        """
        # Prepare dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        }

        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        # Evaluate
        try:
            metrics_to_use = self.metrics[:3]
            if ground_truths:
                metrics_to_use = self.metrics

            result = evaluate(dataset, metrics=metrics_to_use)

            scores = {
                "faithfulness": result["faithfulness"],
                "answer_relevancy": result["answer_relevancy"],
                "context_precision": result["context_precision"],
            }

            if ground_truths:
                scores["context_recall"] = result["context_recall"]

            logger.info(f"Batch evaluation (n={len(questions)}): {scores}")
            return scores

        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            raise

    def check_safety_threshold(self, scores: Dict[str, float]) -> bool:
        """
        Check if scores meet medical safety thresholds.

        Args:
            scores: Metric scores from evaluation

        Returns:
            True if safe for medical use, False otherwise
        """
        # Medical safety requirements
        FAITHFULNESS_THRESHOLD = 0.95  # >95% faithful to context
        RELEVANCY_THRESHOLD = 0.80  # >80% relevant to query
        PRECISION_THRESHOLD = 0.70  # >70% precision in retrieval

        faithfulness_ok = scores.get("faithfulness", 0) >= FAITHFULNESS_THRESHOLD
        relevancy_ok = scores.get("answer_relevancy", 0) >= RELEVANCY_THRESHOLD
        precision_ok = scores.get("context_precision", 0) >= PRECISION_THRESHOLD

        passed = faithfulness_ok and relevancy_ok and precision_ok

        if not passed:
            logger.warning(
                f"Safety threshold check FAILED: "
                f"faithfulness={scores.get('faithfulness', 0):.3f} (need {FAITHFULNESS_THRESHOLD}), "
                f"relevancy={scores.get('answer_relevancy', 0):.3f} (need {RELEVANCY_THRESHOLD}), "
                f"precision={scores.get('context_precision', 0):.3f} (need {PRECISION_THRESHOLD})"
            )
        else:
            logger.info("Safety threshold check PASSED")

        return passed

    def generate_report(
        self,
        scores: Dict[str, float],
        test_name: str = "RAG Evaluation",
    ) -> str:
        """
        Generate a human-readable evaluation report.

        Args:
            scores: Metric scores
            test_name: Name of the test run

        Returns:
            Formatted report string
        """
        report = f"""
{'=' * 60}
{test_name}
{'=' * 60}

Faithfulness:       {scores.get('faithfulness', 0):.3f}
  - Measures: Answer contains only information from context
  - Threshold: 0.95 (medical safety requirement)
  - Status: {'✅ PASS' if scores.get('faithfulness', 0) >= 0.95 else '❌ FAIL'}

Answer Relevancy:   {scores.get('answer_relevancy', 0):.3f}
  - Measures: Answer addresses the query
  - Threshold: 0.80
  - Status: {'✅ PASS' if scores.get('answer_relevancy', 0) >= 0.80 else '❌ FAIL'}

Context Precision:  {scores.get('context_precision', 0):.3f}
  - Measures: Top retrieved chunks are most relevant
  - Threshold: 0.70
  - Status: {'✅ PASS' if scores.get('context_precision', 0) >= 0.70 else '❌ FAIL'}
"""

        if "context_recall" in scores:
            report += f"""
Context Recall:     {scores.get('context_recall', 0):.3f}
  - Measures: All relevant information retrieved
  - Threshold: 0.70
  - Status: {'✅ PASS' if scores.get('context_recall', 0) >= 0.70 else '❌ FAIL'}
"""

        overall_pass = self.check_safety_threshold(scores)
        report += f"""
{'=' * 60}
Overall Status: {'✅ SAFE FOR MEDICAL USE' if overall_pass else '❌ NOT SAFE - REQUIRES IMPROVEMENT'}
{'=' * 60}
"""

        return report
