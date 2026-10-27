# Evaluation report - mode: rules

Claims evaluated: 36
Decision accuracy: 88.9% (95% bootstrap CI 77.8% - 97.2%)
Mean tool-call recall: 100.0%
Escalation rate: 5.6%

## Accuracy by denial code

| Code | n | Accuracy |
|---|---|---|
| CO-11 | 2 | 100% |
| CO-16 | 4 | 100% |
| CO-18 | 4 | 100% |
| CO-197 | 6 | 83% |
| CO-29 | 5 | 80% |
| CO-4 | 4 | 100% |
| CO-50 | 6 | 83% |
| CO-97 | 5 | 80% |

## Confusion matrix (rows = expected, columns = predicted)

| expected \ predicted | correct_and_resubmit | appeal | write_off | escalate |
|---|---|---|---|---|
| correct_and_resubmit | 8 | 0 | 0 | 0 |
| appeal | 0 | 13 | 0 | 0 |
| write_off | 0 | 0 | 9 | 0 |
| escalate | 0 | 0 | 4 | 2 |

## Errors (4)

- CLM-1032 (CO-50): expected escalate, got write_off - No supporting clinical documentation.
- CLM-1033 (CO-197): expected escalate, got write_off - No authorization on file.
- CLM-1034 (CO-29): expected escalate, got write_off - No proof of timely filing.
- CLM-1035 (CO-97): expected escalate, got write_off - Service was not distinct; bundling is correct.
