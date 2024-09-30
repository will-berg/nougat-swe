import re
import numpy as np
import nltk
from nltk import edit_distance
from jiwer import wer, cer


def compute_metrics(pred, gt):
    metrics = {}
    try:
        metrics["edit_dist"] = edit_distance(pred, gt) / max(len(pred), len(gt))
    except:
        metrics["edit_dist"] = np.nan
    reference = gt.split()
    hypothesis = pred.split()
    try:
        metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    except:
        metrics["bleu"] = np.nan
    try:
        metrics["meteor"] = nltk.translate.meteor([reference], hypothesis)
    except LookupError:
        metrics["meteor"] = np.nan
    reference = set(reference)
    hypothesis = set(hypothesis)
    try:
        metrics["precision"] = nltk.scores.precision(reference, hypothesis)
    except:
        metrics["precision"] = np.nan
    try:
        metrics["recall"] = nltk.scores.recall(reference, hypothesis)
    except:
        metrics["recall"] = np.nan
    try:
        metrics["f_measure"] = nltk.scores.f_measure(reference, hypothesis)
    except:
        metrics["f_measure"] = np.nan

    return metrics
