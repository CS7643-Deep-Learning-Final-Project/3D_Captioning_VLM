"""
metrics_test.py
---------------
Quick sanity check for metrics.py (BLEU, ROUGE-L, BERTScore).
Run:  python metrics_test.py
"""

from evaluation.metrics import CaptionEvaluator


def main():
    evaluator = CaptionEvaluator(metrics=["bleu", "rouge-l", "bertscore"])

    # Example 1: good match
    preds = ["a cat is sitting on the mat"]
    refs = [["the cat is sitting on the mat", "a cat sits on a mat"]]
    print("Example 1:")
    print(evaluator.evaluate(preds, refs))
    print("-" * 50)

    # Example 2: partial overlap
    preds = ["a dog is playing in the park"]
    refs = [["a puppy plays in the park"]]
    print("Example 2:")
    print(evaluator.evaluate(preds, refs))
    print("-" * 50)

    # Example 3: completely different
    preds = ["the car is driving on the road"]
    refs = [["a man is cooking in the kitchen"]]
    print("Example 3:")
    print(evaluator.evaluate(preds, refs))
    print("-" * 50)

    # Example 4: single quick test using evaluate_single
    result = evaluator.evaluate_single(
        "the child is eating an apple",
        "a boy eats an apple",
    )
    print("Example 4 (single):")
    print(result)


if __name__ == "__main__":
    main()