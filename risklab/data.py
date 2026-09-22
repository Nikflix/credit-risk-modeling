"""Acquire the UCI source and keep duplicate customer profiles in one split."""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from risklab.common import acquire

SOURCE = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
CHECKSUM = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]
RAW_FEATURES = ["LIMIT_BAL", *STATUS, *BILLS, *PAYMENTS]
TARGET = "default payment next month"


def load_data(directory="data"):
    archive = acquire(SOURCE, Path(directory) / "credit.zip", CHECKSUM)
    with ZipFile(archive) as zipped:
        with zipped.open("default of credit card clients.xls") as source:
            frame = pd.read_excel(source, header=1)
    if len(frame) != 30000 or not set(RAW_FEATURES + [TARGET]).issubset(frame):
        raise ValueError("Unexpected source schema")
    if frame.isna().any().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("Missing values or invalid target")
    return frame


def split_indices(frame, seed=42):
    # Group on the actual model inputs, not ID or outcome. Identical predictor
    # profiles must never appear in both development and evaluation sets.
    groups = pd.util.hash_pandas_object(frame[RAW_FEATURES], index=False).to_numpy()
    remaining = np.arange(len(frame))
    result = {}
    for name, fraction in [
        ("test", 0.2),
        ("validation", 0.125),
        ("calibration", 1 / 7),
    ]:
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        keep, take = next(splitter.split(remaining, groups=groups[remaining]))
        result[name], remaining = remaining[take], remaining[keep]
    result["train"] = remaining
    return result, groups


def features(frame):
    """Use only information available before the next-month outcome."""
    x = frame[RAW_FEATURES].astype(float).copy()
    x["utilization_latest"] = x["BILL_AMT1"] / x["LIMIT_BAL"].clip(lower=1)
    x["payment_to_bill"] = x["PAY_AMT1"] / x["BILL_AMT1"].abs().clip(lower=1)
    x["months_overdue"] = (x[STATUS] > 0).sum(axis=1)
    x["mean_bill"] = x[BILLS].mean(axis=1)
    x["bill_change"] = (x["BILL_AMT1"] - x["BILL_AMT6"]) / x["LIMIT_BAL"].clip(lower=1)
    return x
