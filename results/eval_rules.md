# Evaluation report - mode: rules

Claims evaluated: 36
Decision accuracy: 100.0% (95% bootstrap CI 100.0% - 100.0%)
Mean tool-call recall: 100.0%
Escalation rate: 16.7%

## Accuracy by denial code

| Code | n | Accuracy |
|---|---|---|
| CO-11 | 2 | 100% |
| CO-16 | 4 | 100% |
| CO-18 | 4 | 100% |
| CO-197 | 6 | 100% |
| CO-29 | 5 | 100% |
| CO-4 | 4 | 100% |
| CO-50 | 6 | 100% |
| CO-97 | 5 | 100% |

## Confusion matrix (rows = expected, columns = predicted)

| expected \ predicted | correct_and_resubmit | appeal | write_off | escalate |
|---|---|---|---|---|
| correct_and_resubmit | 8 | 0 | 0 | 0 |
| appeal | 0 | 13 | 0 | 0 |
| write_off | 0 | 0 | 9 | 0 |
| escalate | 0 | 0 | 0 | 6 |

## Errors (0)

- none
