#!/usr/bin/env python
"""
Simple demonstration of how to evaluate a language‑model chain using LangSmith.

This script does not require any external API keys: it uses a dummy chain that
returns hard‑coded answers.  If you configure a `LANGSMITH_API_KEY` in your
environment, the script will upload the dataset and evaluation results to
your LangSmith project.

Usage:

    python scripts/test_llm_langsmith.py --dataset demo

You can customise the dataset name or extend the `load_examples` function to
include more questions and expected answers.
"""

import argparse
import os
import statistics
from typing import List, Dict

# Additional evaluation utilities
try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )
except ImportError:
    # Define fallbacks if scikit‑learn isn't installed; these do nothing.
    accuracy_score = None  # type: ignore
    precision_recall_fscore_support = None  # type: ignore
    confusion_matrix = None  # type: ignore

try:
    # Import here to avoid forcing installation when only running the dummy chain.
    from langsmith import Client
except ImportError:
    Client = None  # type: ignore


def load_examples() -> List[Dict[str, str]]:
    """Return a list of examples with a question and the expected answer."""
    return [
        {"question": "What is the capital of France?", "expected": "Paris"},
        {"question": "Who wrote 'Moby Dick'?", "expected": "Herman Melville"},
        {"question": "2 + 2 = ?", "expected": "4"},
        {"question": "What year did the first man land on the moon?", "expected": "1969"},
    ]


def dummy_chain(question: str) -> str:
    """A stand‑in for a real LLM chain.

    The function returns deterministic answers for known questions and
    falls back to a generic response otherwise.  Replace this function
    with your own LangChain chain or LLM call to test real models.
    """
    if "capital of France" in question.lower():
        return "Paris"
    if "moby dick" in question.lower():
        return "Herman Melville"
    if "2 + 2" in question or "2+2" in question:
        return "4"
    if "first man land on the moon" in question.lower():
        return "1969"
    # default fallback
    return "I don't know"


def evaluate(examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Run the dummy chain on each example and return results.

    Each result includes the question, expected answer, predicted answer,
    and whether the prediction was correct.
    """
    results = []
    for ex in examples:
        question = ex["question"]
        expected = ex["expected"]
        predicted = dummy_chain(question)
        results.append(
            {
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "correct": str(predicted.strip().lower() == expected.strip().lower()),
            }
        )
    return results


def print_report(results: List[Dict[str, str]]) -> None:
    """Print a detailed evaluation report to stdout.

    The report includes per‑example correctness, overall accuracy and
    additional statistics (precision, recall, F1 and confusion matrix)
    if scikit‑learn is available.
    """
    total = len(results)
    correct = sum(1 for r in results if r["correct"] == "True")
    print("\nEvaluation results (dummy chain):")
    for r in results:
        status = "✔" if r["correct"] == "True" else "✘"
        print(
            f"  {status} Q: {r['question']}\n"
            f"    Expected: {r['expected']}\n"
            f"    Predicted: {r['predicted']}"
        )

    # Compute accuracy
    acc = correct / total if total else 0.0
    print(f"\nAccuracy: {acc:.2%} ({correct}/{total})")

    # Compute additional metrics if sklearn is available
    if accuracy_score and precision_recall_fscore_support and confusion_matrix:
        labels = [r["expected"] for r in results]
        preds = [r["predicted"] for r in results]
        acc2 = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        cm = confusion_matrix(labels, preds)
        print("\nStatistical metrics:")
        print(f"  Accuracy (sklearn): {acc2:.3f}")
        print(f"  Precision (macro): {precision:.3f}")
        print(f"  Recall (macro): {recall:.3f}")
        print(f"  F1‑score (macro): {f1:.3f}")
        print("  Confusion matrix (rows=true labels, cols=predictions):")
        # Print confusion matrix with labels
        unique_labels = sorted(set(labels + preds))
        # Header
        header = "       " + "  ".join(f"{lbl[:7]:>7}" for lbl in unique_labels)
        print(header)
        for i, true_lbl in enumerate(unique_labels):
            row_vals = cm[i]
            row_str = "  ".join(f"{val:7d}" for val in row_vals)
            print(f"  {true_lbl[:7]:>7} {row_str}")
    else:
        print("\nInstall scikit‑learn to see precision, recall, F1 and confusion matrix.")
    print()


def upload_to_langsmith(dataset_name: str, results: List[Dict[str, str]]) -> None:
    """Upload a dataset and evaluation results to LangSmith if possible.

    Requires that the langsmith package is installed and that the user has
    configured their API key via the LANGSMITH_API_KEY environment variable.
    """
    if Client is None:
        print("langsmith package not installed; skipping upload.")
        return
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("LANGSMITH_API_KEY not set; skipping upload.")
        return
    client = Client()
    # Create or get dataset
    dataset = client.get_or_create_dataset(name=dataset_name, description="Demo dataset for LangSmith evaluation")
    # Prepare examples in LangSmith format
    examples = []
    for r in results:
        examples.append(
            {
                "inputs": {"question": r["question"]},
                "outputs": {"answer": r["predicted"]},
                "expected": r["expected"],
            }
        )
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Uploaded {len(examples)} examples to LangSmith dataset '{dataset_name}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a dummy LLM chain using LangSmith.")
    parser.add_argument("--dataset", default="demo_dataset", help="Name of the dataset to upload to LangSmith")
    args = parser.parse_args()
    examples = load_examples()
    results = evaluate(examples)
    print_report(results)
    upload_to_langsmith(args.dataset, results)


if __name__ == "__main__":
    main()