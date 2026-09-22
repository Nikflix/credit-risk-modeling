# Data card

- Source: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
- Citation: Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C55S3H.
- License: CC BY 4.0. Code is MIT; the dataset is not relicensed by this repository.
- 30,000 records, 23 source predictors, binary next-month default outcome.
- Population: Taiwan credit-card customers; repayment history April–September
  2005. Monetary values are New Taiwan dollars. There is no sequence of outcome
  cohorts that would support a true out-of-time evaluation.
- Source ZIP SHA-256: `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`.

The ID, target, sex, education, marriage and age columns are excluded from the
predictor matrix. Age and sex are used only to examine error-rate differences.
Nonpositive repayment-status codes are retained as recorded. Negative bill
amounts are not automatically treated as bad data; they can represent credits.
No source rows are manually removed. Equal model-input profiles share a split,
even if their outcomes differ. Group allocation makes split counts approximate
60/10/10/20 rather than exactly those percentages.

This cohort cannot establish performance on modern US lending, changing economic
conditions, or populations missing from the source. Demographic exclusion alone
does not establish fairness: remaining variables may act as proxies. Public
availability also does not justify using real personal records in the demo.
