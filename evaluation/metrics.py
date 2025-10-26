"""
metrics.py
-----------
Defines the evaluation manager for assessing caption quality.
Supports standard metrics such as BLEU and CIDEr.
"""

from typing import Any, Dict, List, Optional


class CaptionEvaluator:
    """
    Evaluation manager for computing caption quality metrics.
    Supports standard captioning metrics: BLEU and CIDEr.
    """

    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize evaluator with specified metrics to compute.

        Args:
            metrics (Optional[List[str]]): List of metric names to compute.
                Default: ['bleu', 'cider']
        """
        # Responsibilities:
        # - Store selected metric names
        # - Optionally load external metric computation libraries
        #   (e.g., nltk, pycocoevalcap)
        pass

    def evaluate(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        Compute evaluation metrics for generated captions.

        Args:
            predictions (List[str]): List of generated captions.
            references (List[List[str]]): List of reference caption lists,
                where each prediction can have multiple reference captions.

        Returns:
            Dict[str, float]: Dictionary containing metric names and scores.
        """
        # Responsibilities:
        # - Iterate over all predictions and references
        # - Compute BLEU, CIDEr, or other metrics depending on config
        # - Aggregate and return average scores
        pass

    def evaluate_single(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Convenience method for evaluating a single prediction-reference pair.

        Args:
            prediction (str): Generated caption.
            reference (str): Ground-truth caption.

        Returns:
            Dict[str, float]: Metric scores for the given pair.
        """
        # Responsibilities:
        # - Compute quick metrics for a single example
        # - Useful for debugging or qualitative analysis
        pass


__all__ = ["CaptionEvaluator"]