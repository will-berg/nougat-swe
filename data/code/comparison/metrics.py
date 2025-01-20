import nltk
import re
import numpy as np
from jiwer import wer, cer


def tokenize_latex(latex_string, ignore_curly_braces=True):
    """
    Tokenizes a LaTeX string into meaningful components.
    """
    latex_string = latex_string[2:-2]
    if ignore_curly_braces:
        token_pattern = r"\\[a-zA-Z]+|[\^\-\+\=\_\<\>\|\:]|[a-zA-Z]+|[0-9]+|[\(\)\[\]]|[^\s{}]"
    else:
        token_pattern = r"\\[a-zA-Z]+|[\^\-\+\=\_\<\>\|\:]|[a-zA-Z]+|[0-9]+|[\(\)\[\]]|[^\s]"
    tokens = re.findall(token_pattern, latex_string)
    return tokens

def compute_metrics(pred, gt):
    metrics = {}

    metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
    metrics["wer"] = wer(gt, pred)
    metrics["cer"] = cer(gt, pred)

    reference = gt.split()
    hypothesis = pred.split()

    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)

    reference = set(reference)
    hypothesis = set(hypothesis)

    metrics["precision"] = nltk.scores.precision(reference, hypothesis)
    metrics["recall"] = nltk.scores.recall(reference, hypothesis)
    metrics["f_measure"] = nltk.scores.f_measure(reference, hypothesis)

    return metrics

"""
def compute_metrics(predictions, ground_truths):
    assert len(predictions) == len(ground_truths), "Predictions and ground truths must have the same length."

    edit_distances = []
    bleu_scores = []
    exact_matches = []
    precisions = []
    recalls = []
    f_measures = []

    # smoother = SmoothingFunction().method1

    for pred, gt in zip(predictions, ground_truths):
        # Remove all whitespace for exact match
        ex_pred = re.sub(r"\s+", " ", pred)
        ex_gt = re.sub(r"\s+", " ", gt)

        edit_distance = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
        if edit_distance == 1:
            continue
        # Edit Distance (normalized by ground truth length)
        edit_distances.append(edit_distance)

        # Exact Match: transform all whitespace to single space
        exact_matches.append(int(ex_pred == ex_gt))

        # Splitting on whitespace does not work for the remaining metrics since the predictions contain
        # almost no whitespace in comparison to the ground truth. Instead use the tokenizer for the remaining metrics.
        gt_tokens = tokenize_latex(gt)
        pred_tokens = tokenize_latex(pred)

        # BLEU Score
        bleu_score = nltk.translate.bleu([gt_tokens], pred_tokens, smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4)
        bleu_scores.append(bleu_score)

        # Precision, recall, and F-measure
        reference = set(gt_tokens)
        hypothesis = set(pred_tokens)

        p = nltk.precision(reference, hypothesis)
        r = nltk.recall(reference, hypothesis)
        f_m = nltk.f_measure(reference, hypothesis)

        precisions.append(p if p is not None else np.nan)
        recalls.append(r if r is not None else np.nan)
        f_measures.append(f_m if f_m is not None else np.nan)

    # Aggregate metrics
    return {
        "edit_dist": np.mean(edit_distances),
        "bleu": np.mean(bleu_scores),
        "exact_match": np.mean(exact_matches),
        "precision": np.nanmean(precisions),
        "recall": np.nanmean(recalls),
        "f_measure": np.nanmean(f_measures),
    }
"""
