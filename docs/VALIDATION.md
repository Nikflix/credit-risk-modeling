# Validation findings

## Question and protocol

Can a nonlinear model improve probability estimates over logistic regression,
and how much review workload follows from a cost-sensitive operating point?

The fixed protocol uses seed 42. Training has 18,033 rows, calibration 3,028,
validation 2,957, and test 5,982. Duplicate predictor profiles are grouped before
splitting. Scaling is fitted only on training; post-hoc calibrators see only the
calibration partition. Four candidates are compared on validation Brier score.
The test partition evaluates the frozen choice and is not used for model or
threshold selection. Reported test results for other candidates are comparisons,
not a second selection stage.

## Decisions supported by the run

1. **Keep gradient boosting.** Its validation Brier score is 0.134737 versus
   0.143336 for logistic regression. The model uses 180 boosting iterations,
   learning rate 0.06, 15 maximum leaves, L2 penalty 5 and no early stopping.
   These are fixed modest settings, not an exhaustive search.
2. **Do not force sigmoid calibration.** The calibrated boosting model has
   validation Brier 0.134795, slightly worse than its raw counterpart. The
   difference is tiny and is not claimed to be statistically significant.
3. **Separate ranking quality from a decision policy.** A 0.14 threshold captures
   81.1% of test defaults, but creates 2,121 false alerts and
   flags 53.1% of the test cohort. At 0.5, recall falls to 35.6%.

With an illustrative missed-default cost of 5 and false-alert cost of 1, cost per
1,000 test records is 560.2 units at the selected threshold,
737.9 at 0.5, 1,086.6 for no review, and 782.7 for reviewing everyone. Correct
decisions have zero cost in this simplified model. This ignores review labor,
exposure at default, recoveries and intervention effectiveness. It is a scenario
comparison, not ROI or dollars saved. Cost ratios 2, 5 and 10 are also evaluated,
with each threshold chosen on validation.

## Uncertainty and error analysis

The 200-resample cluster bootstrap samples identical-input profile groups with
replacement to account for repeated profiles. Test intervals: ROC-AUC
0.765–0.795; average precision 0.520–0.577; Brier score 0.128–0.139. They do not
cover model-selection uncertainty, population shift or unseen time periods.

Sex-code groups have false-positive rates of about 47.8% and 43.7%. Customers over
50 have a 52.5% false-positive rate, compared with 42.4% for ages 31–50. These are
descriptive slice results, without a causal interpretation or a fairness pass/fail
claim; denominators are in `reports/subgroup_audit.csv`.

Recent repayment status and the count of overdue months produce the largest
permutation-importance drops. Correlated features can share importance; this is
global predictive dependence, not a causal explanation or customer-level reason
code. Three permutations give only a rough measure of variability.

## Monitoring experiment

Training quantiles define fixed histogram bins, with explicit underflow/overflow
buckets. Population Stability Index compares new feature distributions with that
reference. PSI >= 0.2 is a demonstration investigation trigger, not a universal
statistical or regulatory cutoff. The stress scenario increases credit limits by
50%; it is synthetic, and the report labels it as such. A PSI alert cannot prove
predictive deterioration. Once delayed labels arrive, also review Brier score,
calibration, recall and subgroup errors.

For a batch with the raw feature columns:

```bash
python -m risklab.monitoring batch.csv --output reports/batch_drift.json
```

## Release assessment

Suitable for a reproducible portfolio demo. Real deployment would require a
current representative dataset, time-based validation, an approved intervention
policy, capacity analysis and independent review. The largest remaining issues
are historical population mismatch and the high review burden.
