"""Shared scoring contract for the API and local analyst demo."""

from typing import Annotated

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from risklab.common import load_bundle, sha256
from risklab.data import BILLS, PAYMENTS, STATUS, features

Status = Annotated[int, Field(ge=-2, le=9, strict=True)]
Bill = Annotated[float, Field(ge=-10000000, le=100000000, allow_inf_nan=False)]
Payment = Annotated[float, Field(ge=0, le=100000000, allow_inf_nan=False)]


class CreditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit_balance: float = Field(gt=0, le=10000000, allow_inf_nan=False)
    repayment_status: list[Status] = Field(min_length=6, max_length=6)
    bill_amounts: list[Bill] = Field(min_length=6, max_length=6)
    payment_amounts: list[Payment] = Field(min_length=6, max_length=6)

    def frame(self):
        record = {"LIMIT_BAL": self.limit_balance}
        record.update(zip(STATUS, self.repayment_status))
        record.update(zip(BILLS, self.bill_amounts))
        record.update(zip(PAYMENTS, self.payment_amounts))
        return pd.DataFrame([record])


class RiskService:
    def __init__(self, directory="artifacts"):
        from pathlib import Path

        self.bundle = load_bundle(directory)
        self.version = sha256(Path(directory) / "model.joblib")[:12]

    def predict(self, request):
        probability = float(self.bundle["model"].predict_proba(features(request.frame()))[0, 1])
        threshold = self.bundle["threshold"]
        return {
            "default_probability": probability,
            "review_threshold": threshold,
            "action": "review" if probability >= threshold else "monitor",
            "model": self.bundle["name"],
            "model_version": self.version,
            "notice": "Historical-data research model; not a lending decision or adverse-action explanation.",
        }
