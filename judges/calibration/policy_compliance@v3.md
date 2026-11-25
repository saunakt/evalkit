# Calibration: policy_compliance v3

Judge model: `fake:fake` · rows: 24 · created 2026-11-25T10:29:25+05:30

**Status: PASS** — kappa 0.824 (min 0.6), FAIL recall 1.000 (min 0.8), basis: heldout split

| split | n | accuracy | kappa | FAIL precision | FAIL recall | FAIL F1 |
|---|---|---|---|---|---|---|
| dev | 12 | 0.917 | 0.800 | 1.000 | 0.750 | 0.857 |
| heldout | 12 | 0.917 | 0.824 | 0.800 | 1.000 | 0.889 |
| all | 24 | 0.917 | 0.812 | 0.875 | 0.875 | 0.875 |

## Confusion matrix (positive class = FAIL)

| | judge FAIL | judge PASS |
|---|---|---|
| human FAIL | 7 | 1 |
| human PASS | 1 | 15 |

## Disagreements

- `pc-08`: human FAIL, judge PASS
- `pc-19`: human PASS, judge FAIL
