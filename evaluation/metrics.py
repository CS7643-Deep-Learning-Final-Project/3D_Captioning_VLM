"""
metrics.py
-----------
Defines the evaluation manager for assessing caption quality.
Supports standard metrics such as BLEU, ROUGE-L and BERTScore.
"""
from typing import Any, Dict, List, Optional, Tuple
from pycocoevalcap.ciderD.ciderD import CiderD
_cider = CiderD()
import torch
try:
    from bert_score import score as bert_score  # noqa: F401
    _HAS_BERTSCORE = True
except Exception:
    _HAS_BERTSCORE = False

class _Config:
  bleu_max_n: int = 4
  rouge_beta: float = 1.2
  bertscore_model: Optional[str] = None   # e.g. "roberta-large"
  bertscore_lang: Optional[str] = "en"
  lowercase: bool = True
  strip: bool = True


def _normalize(s: str, lowercase: bool, strip: bool):
  if s is None:
    s = ""
  if strip:
    s = s.strip()
  if lowercase:
    s = s.lower()
  return s

def _simple_tokenize(s: str):
    """Whitespace tokenizer; uses nltk.word_tokenize if available (lazy import)."""
    if not s:
      return []
    try:
      from nltk import word_tokenize  # type: ignore
      return word_tokenize(s)
    except Exception:
      return s.split()

def _prepare_texts(
  predictions: List[str],
  references: List[List[str]],
  cfg: _Config
):
  preds_n = [_normalize(p, cfg.lowercase, cfg.strip) for p in predictions]
  refs_n = [[_normalize(r, cfg.lowercase, cfg.strip) for r in rs] for rs in references]
  return preds_n, refs_n

# ------------------------ CIDER ------------------------
def _compute_cider(preds, refs, cfg):
    # pycocoevalcap expects dicts: {i: [str]}, {i: [str, ...]}
    gts = {i: (r if r else [""]) for i, r in enumerate(refs)}
    res = {i: [p] for i, p in enumerate(preds)}
    score, _ = _cider.compute_score(gts, res)
    return float(score * 100.0)

# ------------------------ ROUGE-L ------------------------
def _lcs_length(a: List[str], b: List[str]):
  la, lb = len(a), len(b)
  if la == 0 or lb == 0:
    return 0
  dp = [0] * (lb + 1)
  for i in range(1, la + 1):
    prev = 0
    ai = a[i - 1]
    for j in range(1, lb + 1):
      tmp = dp[j]
      if ai == b[j - 1]:
        dp[j] = prev + 1
      else:
        dp[j] = max(dp[j], dp[j - 1])
      prev = tmp
  return dp[lb]

def _compute_rouge_l(preds: List[str], refs: List[List[str]], cfg: _Config):
  eps = 1e-8
  beta2 = cfg.rouge_beta ** 2
  per_item = []
  for p, rset in zip(preds, refs):
    p_tok = _simple_tokenize(p)
    if not rset:
      per_item.append(0.0)
      continue
    best_f = 0.0
    for r in rset:
      r_tok = _simple_tokenize(r)
      if not p_tok and not r_tok:
        best_f = max(best_f, 1.0)
        continue
      lcs = _lcs_length(p_tok, r_tok)
      prec = lcs / (len(p_tok) + eps)
      rec = lcs / (len(r_tok) + eps)
      if prec <= 0.0 and rec <= 0.0:
        f = 0.0
      else:
        f = (1 + beta2) * prec * rec / (rec + beta2 * prec + eps)
      if f > best_f:
        best_f = f
    per_item.append(best_f)
  return float(sum(per_item) / max(1, len(per_item)) * 100.0)


# ----------------------------- BLEU-4 ---------------------------------
def _compute_bleu(preds: List[str], refs: List[List[str]], cfg: _Config):
  """
  Try sacrebleu.
  Both return a 0-100 score here.
  """
  # SacreBLEU (standardized & reproducible)
  try:
      import sacrebleu  # type: ignore
      refs = [rset if len(rset) > 0 else [""] for rset in refs]
      # SacreBLEU expects detokenized strings.
      sys_hyp = preds  # already normalized by _prepare_texts
      # Transpose refs to shape: K lists of length N (K = max #refs)
      max_refs = max(len(rset) for rset in refs) if refs else 0
      refs_k: List[List[str]] = []
      for k in range(max_refs):
          refs_k.append([rset[k] if k < len(rset) else rset[-1] for rset in refs])

      bleu = sacrebleu.corpus_bleu(
          sys_hyp, refs_k,
          smooth_method="exp",
          smooth_value=0.0,
          force=True,
          use_effective_order=True,
      )
      # sacrebleu already returns 0–100
      return float(bleu.score)
  except Exception:
      return 0.0

# --------------------------- BERTScore --------------------------------
def _compute_bertscore(preds: List[str], refs: List[List[str]], cfg: _Config):
  if not _HAS_BERTSCORE:
    raise ImportError("bert-score is not installed. `pip install bert-score`")
  device = "cuda" if torch.cuda.is_available() else "cpu"
  refs_norm = [r if len(r) > 0 else [""] for r in refs]
  max_refs = max(len(r) for r in refs_norm)
  best_f = None
  for k in range(max_refs):
    cur_refs = [(rset[k] if k < len(rset) else rset[-1]) for rset in refs_norm]
    P, R, F1 = bert_score(
      preds,
      cur_refs,
      lang=cfg.bertscore_lang,
      model_type=cfg.bertscore_model,
      device=device,
      rescale_with_baseline=True,
    )
    best_f = F1 if best_f is None else torch.maximum(best_f, F1)
  return float(best_f.mean().item() * 100.0)


class CaptionEvaluator:
    """
    Evaluation manager for computing caption quality metrics.
    Supports standard captioning metrics: BLEU, ROUGE-L and BERTScore.
    """

    _VALID = {"bleu", "rouge", "rouge-l", "bertscore", "cider"}

    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize evaluator with specified metrics to compute.

        Args:
            metrics (Optional[List[str]]): List of metric names to compute.
        """
        # Responsibilities:
        # - Store selected metric names
        # - Optionally load external metric computation libraries

        if metrics is None:
          metrics = ["bleu", "rouge", "rouge-l", "bertscore", "cider"]
        self.metrics = [m.lower() for m in metrics]
        for m in self.metrics:
          if m not in self._VALID:
            raise ValueError(f"Unknown metric '{m}'. Valid: {sorted(self._VALID)}")
        self._cfg = _Config()


    def evaluate(self, predictions: List[str], references: List[List[str]]):
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
        # - Compute BLEU, ROUGE-L and BERTScore depending on config
        # - Aggregate and return average scores
        if not isinstance(predictions, list) or not isinstance(references, list):
          raise TypeError("predictions must be List[str] and references must be List[List[str]].")
        if len(predictions) != len(references):
          raise ValueError(
              f"Length mismatch: {len(predictions)} predictions vs {len(references)} reference sets."
          )

        preds, refs = _prepare_texts(predictions, references, self._cfg)
        out: Dict[str, float] = {}
        for name in self.metrics:
          if name == "bleu":
            out["bleu"] = _compute_bleu(preds, refs, self._cfg)
          elif name in ("rouge", "rouge-l"):
            out["rouge-l"] = _compute_rouge_l(preds, refs, self._cfg)
          elif name == "cider":
            out["cider"] = _compute_cider(preds, refs, self._cfg)
          elif name == "bertscore":
            out["bertscore"] = _compute_bertscore(preds, refs, self._cfg)
        return out

    def evaluate_single(self, prediction: str, reference: str):
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

        return self.evaluate([prediction], [[reference]])

__all__ = ["CaptionEvaluator"]