"""Population stability uses fixed training bins, including overflow buckets."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from risklab.common import write_json
from risklab.data import features


def make_reference(frame):
    reference = {}
    for name in frame:
        cuts = np.unique(np.quantile(frame[name], np.linspace(0, 1, 11)[1:-1]))
        counts = np.histogram(frame[name], [-np.inf, *cuts, np.inf])[0]
        reference[name] = {
            "cuts": cuts.tolist(),
            "proportions": (counts / counts.sum()).tolist(),
        }
    return reference


def drift_report(frame, reference):
    result = []
    for name, spec in reference.items():
        values = frame[name].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {name}")
        counts = np.histogram(values, [-np.inf, *spec["cuts"], np.inf])[0]
        observed = np.maximum(counts / max(counts.sum(), 1), 1e-6)
        expected = np.maximum(spec["proportions"], 1e-6)
        observed, expected = observed / observed.sum(), expected / expected.sum()
        psi = float(np.sum((observed - expected) * np.log(observed / expected)))
        result.append({"feature": name, "psi": psi, "investigate": psi >= 0.2})
    return sorted(result, key=lambda r: r["psi"], reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--reference", default="artifacts/reference.json")
    parser.add_argument("--output", default="reports/batch_drift.json")
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    if frame.empty:
        raise ValueError("Batch is empty")
    write_json(
        args.output,
        drift_report(features(frame), json.loads(Path(args.reference).read_text())),
    )


if __name__ == "__main__":
    main()
