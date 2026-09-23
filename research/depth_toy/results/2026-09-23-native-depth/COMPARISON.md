# 100gaussians generator-depth comparison

Final step: 7000. One seed, CPU, matched initial D/prior/shared G tensors and monitor bank.

| Hidden layers | Seconds incl. diagnostics | Online modes | Online HQ | EMA modes | EMA HQ | EMA sliced W1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 74.5 | 100 | 98.70% | 100 | 98.94% | 0.1582 |
| 8 | 92.1 | 84 | 49.00% | 85 | 49.29% | 0.2252 |
| 16 | 119.8 | 37 | 36.86% | 35 | 28.49% | 0.4363 |

## Initial contraction and first actual G update

| Layers | Initial output spread | Requested shared energy | Actual shared energy | Movement cosine | Movement gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.0788763 | 46.82% | 88.29% | 0.6457 | 78.24 |
| 8 | 0.0110584 | 82.93% | 88.03% | 0.8524 | 44.48 |
| 16 | 0.00050493 | 99.39% | 98.82% | 0.9831 | 13.32 |

Shared energy is a fraction, not absolute shared movement. Gain includes the
native batch-mean gradient convention. A single first-step response is not a
causal diagnosis. HQ/coverage do not validate within-mode variance.
