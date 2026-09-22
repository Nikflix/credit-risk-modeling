# Model card

**Owner:** Nikhil Shrivastava. **Version:** 1.0 portfolio experiment.

**Purpose:** Research next-month default probabilities and a hypothetical analyst
review policy. **Intended user:** Someone evaluating the experiment, not a lender
making an automated approval or rejection.

**Inputs:** Six monthly repayment statuses, statement balances and payments,
plus credit limit. The service accepts exactly six entries, most recent first.
**Output:** Probability, illustrative review/monitor action, threshold, model name
and hash. The API does not approve or deny credit.

**Model:** HistGradientBoostingClassifier. Logistic regression and sigmoid-calibrated
variants are challengers. The selected estimator is not post-hoc calibrated;
its probability quality was evaluated and the calibration alternative did not win.

**Evidence:** `reports/metrics.json`, subgroup audit, feature importance and
`docs/VALIDATION.md`. A held-out random cohort test is not a temporal backtest.

| Risk | Implemented control | Remaining limitation |
|---|---|---|
| Leakage | Excluded outcome/ID; grouped duplicate inputs; separate calibration and validation | Public-data provenance cannot establish production availability |
| Probability error | Brier/log-loss comparison; reliability curves | Future populations untested |
| Unequal error rates | Age/sex slice audit | No claim of fairness or causal discrimination |
| Data change | Fixed training bins and synthetic PSI stress test | No live feed or real temporal drift |
| Invalid requests | Typed request schema and bounded arrays | Semantically unusual but valid values can still extrapolate |
| Wrong artifact | Hash and sklearn version checks | Hashes do not authenticate an untrusted pickle |

**Lifecycle proposal:** Maintain source/version evidence, review new results before
promotion, monitor with labeled outcomes, retain a prior bundle for rollback,
and retire the model when its data or purpose is no longer appropriate. These
are project design controls, not proof of bank governance approval.
