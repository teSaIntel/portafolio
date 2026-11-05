# 2 · Test-issue detector

![detector](fig-detector.png)

**Champion:** TF-IDF on log text plus one-hot categorical metadata, in one logistic regression. **Average precision .78**,
and **33% recall at precision ≥ 0.9**, vs hand-written rules at 23% recall and 34% precision.

## Why it matters
Many failures aren't product bugs: lab infrastructure, the test harness, timeouts. Catching them early saves human time
and keeps them out of the matcher's candidate pool.

## How it works
- Text features (log skeletons) and categorical features (test, stage, configuration) are trained together.
- Out-of-fold predictions use GroupKFold by test family, so the score is never optimistic.
- The threshold is set at precision ≥ 0.9: it only acts when it's sure.

| Features | AP |
|---|---|
| Text only | .63 |
| Categorical only | .75 |
| **Combined** | **.78** |

## Files
`detector.py` · `issue_detector_colab.ipynb`

---
[← Back to the overview](../README.md) · The notebook runs on synthetic data; the numbers above come from the real (private) evaluation.
