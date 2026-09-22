# Explaining this project

**One-minute walkthrough:** Start with the business decision, not the algorithm:
"I wanted to estimate default probabilities and understand the review workload.
I compared logistic regression with gradient boosting, kept repeated profiles out
of different splits, and separated calibration from policy selection. Boosting
reached 0.782 AUC. The biggest practical finding was that capturing 81% of defaults
would flag 53% of the cohort under my illustrative cost assumptions."

Be able to demonstrate `risklab/data.py`, reproduce the four partitions, and show
where the outcome is excluded from features. Explain why Brier score measures
probability quality, why average precision matters with class imbalance, and why
0.5 is not automatically the correct threshold.

**Why not calibrate the final model?** Calibration was tested on a separate
partition, but the uncalibrated model had slightly better validation Brier score.
Do not claim a significant difference or force a technique into the final model.

**Why bootstrap groups?** Identical input profiles create correlated records;
sampling profiles avoids pretending every repeated record is independent.

**What is missing?** Time-based validation, contemporary US data, real costs,
intervention outcomes, and independent review. The drift example is synthetic.

**What was deployed?** A tested local FastAPI service and Streamlit demo. A Docker
configuration is included; no live bank integration or cloud production rollout
is claimed. Be precise about which parts you understand and have personally run.
