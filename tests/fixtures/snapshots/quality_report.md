# quality — WARN

Run `run-1` · Correctness, format and judged quality of single replies (ch. 5–7).

## Verdict

**WARN**
- WARN: bleu not tested: sacrebleu not installed

## What was tested

- Cases: 3 · repeats: 2 · seed: 42
- Target: shopease · profile: nightly
- Metrics: `contains`, `bleu`

## Results vs thresholds

Overall case pass rate 83.3% (95% CI 43.6%–97.0%, n=6).

| metric | value | pass rate | 95% CI | n | target | status |
|---|---|---|---|---|---|---|
| `contains` | 0.833 | 83.3% | 43.6%–97.0% | 6 | ≥ 0.8 pass rate | PASS |
| `bleu` | – | – | – | 0 | – | NOT TESTED |

## Comparison with baseline

No baseline run.

## Results by slice

| tag | n | pass rate |
|---|---|---|
| refund | 2 | 100.0% |
| shipping | 4 | 75.0% |

| language | n | pass rate |
|---|---|---|
| en | 6 | 83.3% |

| split | n | pass rate |
|---|---|---|
| dev | 6 | 83.3% |

| must_pass | n | pass rate |
|---|---|---|
| False | 6 | 83.3% |

## Failing and flaky cases

- `q2` (flaky) — pass rate 50.0%: contains

## Suite panels

No panels.

## Judge validity

No judges used.

## Not tested

- `bleu`: sacrebleu not installed (1)
