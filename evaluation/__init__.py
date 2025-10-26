"""
Evaluation module for caption quality metrics.
Provides the CaptionEvaluator class for BLEU and CIDEr scoring.
"""

from .metrics import CaptionEvaluator

__all__ = ["CaptionEvaluator"]