"""
Pre-deployment evaluation script (Judge).

Runs RAGAS benchmarking on test queries to verify system safety
before deployment to production.

Usage:
    python scripts/evaluate_rag.py --test-file tests/evaluation/test_queries.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.api.src.evaluation import RAGASEvaluator
from services.api.src.retrieval import Retriever, Reranker
from services.api.src.llm import LLMService
from shared.utils import get_settings
import logging

logger = logging.getLogger(__name__)


async def evaluate_system(test_file: Path) -> Dict[str, any]:
    """
    Run comprehensive evaluation on test queries.

    Args:
        test_file: JSON file with test queries

    Returns:
        Evaluation results dictionary
    """
    logger.info(f"Loading test queries from {test_file}")

    # Load test data
    with open(test_file) as f:
        test_data = json.load(f)

    test_queries = test_data.get("queries", [])
    logger.info(f"Loaded {len(test_queries)} test queries")

    # Initialize services
    logger.info("Initializing RAG services...")
    settings = get_settings()

    retriever = Retriever()
    reranker = Reranker()
    llm_service = LLMService(api_key=settings.groq_api_key)
    evaluator = RAGASEvaluator()

    # Run queries and collect results
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for i, test_query in enumerate(test_queries, 1):
        query = test_query["query"]
        ground_truth = test_query.get("expected_answer")

        logger.info(f"[{i}/{len(test_queries)}] Processing: {query}")

        try:
            # Retrieve (50→5 pattern)
            search_results = retriever.search(query, top_k=50, score_threshold=0.3)
            search_results = reranker.rerank(query, search_results, top_k=5)

            # Generate answer
            answer = await llm_service.generate_answer(query, search_results, stream=False)

            # Collect for evaluation
            questions.append(query)
            answers.append(answer)
            contexts_list.append([r.text for r in search_results])
            if ground_truth:
                ground_truths.append(ground_truth)

            logger.info(f"  Answer: {answer[:100]}...")

        except Exception as e:
            logger.error(f"  Failed to process query: {e}")
            continue

    # Run RAGAS evaluation
    logger.info(f"\nEvaluating {len(questions)} queries with RAGAS...")

    scores = evaluator.evaluate_batch(
        questions=questions,
        answers=answers,
        contexts_list=contexts_list,
        ground_truths=ground_truths if ground_truths else None,
    )

    # Check safety thresholds
    passed = evaluator.check_safety_threshold(scores)

    # Generate report
    report = evaluator.generate_report(scores, test_name="Pre-Deployment Evaluation")
    print(report)

    # Save results
    results = {
        "timestamp": time.time(),
        "num_queries": len(questions),
        "scores": scores,
        "passed": passed,
        "details": [
            {
                "query": q,
                "answer": a,
                "num_contexts": len(c),
            }
            for q, a, c in zip(questions, answers, contexts_list)
        ],
    }

    output_file = Path("evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    return results


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Pre-deployment RAG evaluation")
    parser.add_argument(
        "--test-file",
        type=Path,
        default="tests/evaluation/test_queries.json",
        help="Path to test queries JSON file",
    )
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit with code 1 if safety thresholds not met",
    )

    args = parser.parse_args()

    # Check test file exists
    if not args.test_file.exists():
        logger.error(f"Test file not found: {args.test_file}")
        sys.exit(1)

    # Run evaluation
    try:
        results = asyncio.run(evaluate_system(args.test_file))

        # Exit with appropriate code
        if args.fail_on_threshold and not results["passed"]:
            logger.error("❌ EVALUATION FAILED: Safety thresholds not met")
            sys.exit(1)
        else:
            logger.info("✅ EVALUATION COMPLETE")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
