"""Run the fixed development protocol, then evaluate the frozen choice once."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from risklab.common import environment, save_bundle, write_json
from risklab.data import TARGET, features, load_data, split_indices
from risklab.evaluation import (
    bootstrap_intervals,
    choose_threshold,
    operating_point,
    scores,
)
from risklab.monitoring import drift_report, make_reference


def run(data_dir="data", output="artifacts", reports="reports"):
    reports, output = Path(reports), Path(output)
    reports.mkdir(parents=True, exist_ok=True)
    frame = load_data(data_dir)
    splits, groups = split_indices(frame)
    x, y = features(frame), frame[TARGET]
    tr, ca, va, te = [splits[k] for k in ["train", "calibration", "validation", "test"]]
    models = {
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=1500, random_state=42)
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=5,
            early_stopping=False,
            random_state=42,
        ),
    }
    for name, model in list(models.items()):
        model.fit(x.iloc[tr], y.iloc[tr])
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
        calibrated.fit(x.iloc[ca], y.iloc[ca])
        models[name + "_calibrated"] = calibrated
    validation = {
        name: scores(y.iloc[va], model.predict_proba(x.iloc[va])[:, 1])
        for name, model in models.items()
    }
    selected = min(validation, key=lambda name: validation[name]["brier"])
    model = models[selected]
    pv = model.predict_proba(x.iloc[va])[:, 1]
    threshold = choose_threshold(y.iloc[va], pv)["threshold"]
    # No refit on validation/test: the tested object is exactly what the API loads.
    predictions = {name: m.predict_proba(x.iloc[te])[:, 1] for name, m in models.items()}
    p = predictions[selected]
    test = {name: scores(y.iloc[te], pred) for name, pred in predictions.items()}
    report = {
        "environment": environment(),
        "selected_model": selected,
        "selection_rule": "minimum validation Brier score among four fixed candidates",
        "split_counts": {k: len(v) for k, v in splits.items()},
        "split_default_rates": {k: float(y.iloc[v].mean()) for k, v in splits.items()},
        "unique_input_profiles": len(np.unique(groups)),
        "validation": validation,
        "test": test,
        "test_95pct_cluster_bootstrap": bootstrap_intervals(y.iloc[te], p, groups[te]),
        "operating_point": operating_point(y.iloc[te], p, threshold),
        "threshold_0_5": operating_point(y.iloc[te], p, 0.5),
        "no_review_cost_per_1000": float(5000 * y.iloc[te].mean()),
        "review_everyone_cost_per_1000": float(1000 * (1 - y.iloc[te].mean())),
        "cost_assumption": "missed default = 5 units; false alert = 1; correct decisions = 0; illustrative, not dollars",
    }
    sensitivity = []
    for ratio in [2, 5, 10]:
        t = choose_threshold(y.iloc[va], pv, ratio)["threshold"]
        sensitivity.append(
            {"missed_default_cost": ratio, **operating_point(y.iloc[te], p, t, ratio)}
        )
    report["cost_sensitivity"] = sensitivity
    audits = []
    audit_frame = frame.iloc[te].copy()
    audit_frame["age_band"] = pd.cut(
        audit_frame["AGE"], [0, 30, 50, 200], labels=["under_31", "31_to_50", "over_50"]
    )
    for field in ["SEX", "age_band"]:
        for value in audit_frame[field].unique():
            mask = (audit_frame[field] == value).to_numpy()
            row = {
                "attribute": field,
                "group": str(value),
                "n": int(mask.sum()),
                "default_rate": float(y.iloc[te].to_numpy()[mask].mean()),
                **operating_point(y.iloc[te].to_numpy()[mask], p[mask], threshold),
            }
            audits.append(row)
    pd.DataFrame(audits).to_csv(reports / "subgroup_audit.csv", index=False)
    importance = permutation_importance(
        model, x.iloc[te], y.iloc[te], scoring="roc_auc", n_repeats=3, random_state=42
    )
    pd.DataFrame(
        {
            "feature": x.columns,
            "auc_drop": importance.importances_mean,
            "repeat_std": importance.importances_std,
        }
    ).sort_values("auc_drop", ascending=False).to_csv(
        reports / "feature_importance.csv", index=False
    )
    reference = make_reference(x.iloc[tr])
    shifted = frame.iloc[te].copy()
    shifted["LIMIT_BAL"] *= 1.5
    report["drift_stress_test"] = {
        "scenario": "synthetic 50% increase in credit limits; not real temporal drift",
        "unshifted": drift_report(x.iloc[te], reference),
        "shifted": drift_report(features(shifted), reference),
    }
    manifest = save_bundle(
        output,
        {
            "model": model,
            "threshold": threshold,
            "name": selected,
            "feature_names": x.columns.tolist(),
        },
    )
    write_json(output / "reference.json", reference)
    report["artifact"] = manifest
    write_json(reports / "metrics.json", report)
    split_frame = pd.DataFrame(
        {"customer_id": frame["ID"], "split": "", "profile_group": groups.astype(str)}
    )
    for name, idx in splits.items():
        split_frame.loc[idx, "split"] = name
    split_frame.to_csv(output / "split_manifest.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, pred in predictions.items():
        observed, predicted = calibration_curve(y.iloc[te], pred, n_bins=8, strategy="quantile")
        axes[0].plot(predicted, observed, marker="o", markersize=3, label=name.replace("_", " "))
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set(
        xlabel="Predicted default probability",
        ylabel="Observed default rate",
        title="Held-out calibration",
    )
    axes[0].legend(fontsize=7)
    thresholds = np.linspace(0.01, 0.99, 99)
    axes[1].plot(
        thresholds,
        [operating_point(y.iloc[va], pv, t)["cost_per_1000"] for t in thresholds],
    )
    axes[1].axvline(threshold, linestyle="--", color="#b45309", label=f"Chosen: {threshold:.2f}")
    axes[1].set(
        xlabel="Review threshold",
        ylabel="Illustrative cost / 1,000",
        title="Validation threshold selection",
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(reports / "validation.svg")
    plt.close(fig)
    # Save a synthetic example, not a customer record.
    write_json(
        reports / "example_request.json",
        {
            "limit_balance": 100000,
            "repayment_status": [0, 0, 0, 0, 0, 0],
            "bill_amounts": [30000] * 6,
            "payment_amounts": [4000] * 6,
        },
    )
    print(
        {"selected": selected, "test": test[selected], "threshold": threshold},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.data_dir, args.output, args.reports)
