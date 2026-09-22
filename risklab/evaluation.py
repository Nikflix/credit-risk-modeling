"""Probability quality, uncertainty, and explicit review-cost assumptions."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


def scores(y, p):
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
    }


def operating_point(y, p, threshold, missed_default_cost=5, false_alert_cost=1):
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "recall": float(tp / max(tp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "review_rate": float((tp + fp) / len(y)),
        "cost_per_1000": float(1000 * (missed_default_cost * fn + false_alert_cost * fp) / len(y)),
    }


def choose_threshold(y, p, ratio=5):
    candidates = [operating_point(y, p, t, ratio) for t in np.linspace(0.01, 0.99, 99)]
    # On a tie, prefer fewer reviews. Selection is done on validation only.
    return min(candidates, key=lambda r: (r["cost_per_1000"], r["review_rate"]))


def bootstrap_intervals(y, p, groups, repeats=200, seed=42):
    """Resample profiles rather than treating duplicate records as independent."""
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    draws = []
    for _ in range(repeats):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        draws.append(scores(np.asarray(y)[idx], np.asarray(p)[idx]))
    return {key: np.quantile([r[key] for r in draws], [0.025, 0.975]).tolist() for key in draws[0]}
