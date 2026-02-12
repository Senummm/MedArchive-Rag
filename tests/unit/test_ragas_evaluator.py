"""
Unit tests for RAGAS evaluator.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from services.api.src.evaluation import RAGASEvaluator


class TestRAGASEvaluator:
    """Test suite for RAGAS evaluation."""

    def test_evaluator_initialization(self):
        """Test RAGASEvaluator can be initialized."""
        evaluator = RAGASEvaluator()
        assert evaluator is not None

    def test_evaluator_graceful_degradation_without_ragas(self):
        """Test evaluator returns default scores if RAGAS not installed."""
        with patch("services.api.src.evaluation.ragas_evaluator.RAGAS_AVAILABLE", False):
            evaluator = RAGASEvaluator()

            scores = evaluator.evaluate_single(
                question="What is hypertension?",
                answer="High blood pressure",
                contexts=["Hypertension is defined as BP >140/90"],
            )

            # Should return default scores
            assert scores["faithfulness"] == 1.0
            assert scores["answer_relevancy"] == 1.0
            assert scores["context_precision"] == 1.0

    @patch("services.api.src.evaluation.ragas_evaluator.RAGAS_AVAILABLE", True)
    @patch("services.api.src.evaluation.ragas_evaluator.evaluate")
    def test_evaluate_single_success(self, mock_evaluate):
        """Test single evaluation with mocked RAGAS."""
        # Mock RAGAS evaluate function
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = MagicMock(
            to_dict=lambda orient: {
                "faithfulness": [0.96],
                "answer_relevancy": [0.85],
                "context_precision": [0.75],
                "context_recall": [0.90],
            }
        )
        mock_evaluate.return_value = mock_result

        evaluator = RAGASEvaluator()
        scores = evaluator.evaluate_single(
            question="What is the dose of amoxicillin?",
            answer="500mg three times daily",
            contexts=["Amoxicillin: 500mg TID"],
            ground_truth="500mg three times daily for 7 days",
        )

        assert scores["faithfulness"] == 0.96
        assert scores["answer_relevancy"] == 0.85
        assert scores["context_precision"] == 0.75
        assert scores["context_recall"] == 0.90

    def test_check_safety_threshold_pass(self):
        """Test safety threshold check passes with good scores."""
        evaluator = RAGASEvaluator()

        scores = {
            "faithfulness": 0.97,
            "answer_relevancy": 0.82,
            "context_precision": 0.73,
        }

        passed = evaluator.check_safety_threshold(scores)
        assert passed is True

    def test_check_safety_threshold_fail_faithfulness(self):
        """Test safety threshold check fails with low faithfulness."""
        evaluator = RAGASEvaluator()

        scores = {
            "faithfulness": 0.92,  # Below 0.95 threshold
            "answer_relevancy": 0.85,
            "context_precision": 0.75,
        }

        passed = evaluator.check_safety_threshold(scores)
        assert passed is False

    def test_check_safety_threshold_fail_relevancy(self):
        """Test safety threshold check fails with low relevancy."""
        evaluator = RAGASEvaluator()

        scores = {
            "faithfulness": 0.97,
            "answer_relevancy": 0.75,  # Below 0.80 threshold
            "context_precision": 0.75,
        }

        passed = evaluator.check_safety_threshold(scores)
        assert passed is False

    def test_generate_report(self):
        """Test report generation."""
        evaluator = RAGASEvaluator()

        scores = {
            "faithfulness": 0.96,
            "answer_relevancy": 0.83,
            "context_precision": 0.72,
            "context_recall": 0.88,
        }

        report = evaluator.generate_report(scores, test_name="Test Run")

        assert "Test Run" in report
        assert "PASSED" in report
        assert "0.96" in report
        assert "faithfulness" in report.lower()

    @patch("services.api.src.evaluation.ragas_evaluator.RAGAS_AVAILABLE", True)
    @patch("services.api.src.evaluation.ragas_evaluator.evaluate")
    def test_evaluate_batch(self, mock_evaluate):
        """Test batch evaluation."""
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = MagicMock(
            to_dict=lambda orient: {
                "faithfulness": [0.96, 0.94],
                "answer_relevancy": [0.85, 0.82],
                "context_precision": [0.75, 0.70],
                "context_recall": [0.90, 0.88],
            }
        )
        mock_evaluate.return_value = mock_result

        evaluator = RAGASEvaluator()

        questions = ["Q1?", "Q2?"]
        answers = ["A1", "A2"]
        contexts_list = [["C1"], ["C2"]]

        scores = evaluator.evaluate_batch(questions, answers, contexts_list)

        # Should return average scores
        assert "faithfulness" in scores
        assert "answer_relevancy" in scores
        assert isinstance(scores["faithfulness"], float)

    def test_evaluate_handles_exceptions(self):
        """Test evaluator handles exceptions gracefully."""
        evaluator = RAGASEvaluator()

        with patch("services.api.src.evaluation.ragas_evaluator.evaluate", side_effect=Exception("RAGAS error")):
            scores = evaluator.evaluate_single(
                question="Test?",
                answer="Test",
                contexts=["Context"],
            )

            # Should return default scores on error
            assert scores["faithfulness"] == 1.0
