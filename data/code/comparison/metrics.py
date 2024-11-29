import nltk
import re
import numpy as np

def compute_metrics(predictions, ground_truths):
    for pred, gt in zip(predictions, ground_truths):
        print(pred)
        print(gt)
        print()
    exit()
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
        # Exact Match: transform all whitespace to single space
        exact_matches.append(int(ex_pred == ex_gt))

        # Edit Distance (normalized by ground truth length)
        edit_distances.append(nltk.edit_distance(pred, gt) / max(len(pred), len(gt)))

        # BLEU Score
        bleu_score = nltk.translate.bleu([gt.split()], pred.split(), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4)
        bleu_scores.append(bleu_score)

        # Precision, recall, and F-measure
        # Replace all whitespace with single space
        reference = set(gt.split())
        hypothesis = set(pred.split())

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

