# Descriptive results

No statistical significance or quantum speedup is claimed.

## Held-out median TV, k=2

| Method | Haar | Local 2 | Local 4 |
|---|---:|---:|---:|
| Conditional-MI parents | 0.10193460 | 0.00365249 | 0.03649139 |
| First-click calibrated | 0.07305845 | 0.00328775 | 0.02570774 |
| All zeros + recent clicks | 0.01805349 | 0.00053518 | 0.00334143 |
| All zeros + MI-selected clicks | 0.01400320 | 0.00049658 | 0.00283508 |

## Paired outcomes on held-out cases

| Comparison | Lower TV | Higher TV | Tied |
|---|---:|---:|---:|
| zero_memory_vs_greedy_cmi | 71 | 1 | 0 |
| zero_pairwise_vs_greedy_cmi | 72 | 0 | 0 |
| zero_pairwise_vs_zero_memory | 54 | 18 | 0 |
| calibrated_greedy_cmi_vs_greedy_cmi | 72 | 0 | 0 |

The 72 held-out configurations use four reserved seeds; squeezing and loss variants share their interferometers.
An equal k is not an equal information or computational budget.
