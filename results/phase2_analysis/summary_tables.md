# Second-round descriptive results

## Median full-distribution TV on 24 held-out input configurations

k4 is a post-baseline exploratory extension. Six interferometers are reused across parameters/profiles.

| Method | Haar | Local 2 | Local 4 | Median core cold 128-draw seconds |
|---|---:|---:|---:|---:|
| Pairwise k=2 | 0.011497972 | 0.0024550376 | 0.0076663604 | 0.0541685 |
| Influence k=2 | 0.008573023 | 0.0010641559 | 0.0048150284 | 0.122547 |
| Pairwise k=3 | 0.0042480105 | 0.00024703953 | 0.001741627 | 0.0633556 |
| Influence k=3 | 0.0030163013 | 8.2790526e-05 | 0.00099730851 | 0.122578 |
| Pairwise k=4 (exploratory) | 0.0011899999 | 1.4359123e-05 | 0.00035091622 | 0.0776355 |
| Influence k=4 (exploratory) | 0.00087396874 | 3.7416709e-06 | 0.00020107417 | 0.110771 |
| Frugal K=2 | 0.027267679 | 0.019980553 | 0.021616465 | 0.000472086 |
| Frugal K=3 | 0.014878087 | 0.0058608217 | 0.010629697 | 0.000744853 |
| Frugal K=4 | 0.0054643001 | 0.00067679179 | 0.0023163218 | 0.00197436 |
| Frugal K=5 | 0.0022819211 | 3.4184884e-05 | 0.00074505005 | 0.00447069 |

## Bound interpretation

Twelve fixed bounds allocate alpha=0.05/12; the simultaneous statistical coverage is conditional on correct floating point point probabilities.
No outward-rounded certificate is claimed. Cutoff and numerical failures receive score 1 and are never dropped.

## Main evaluation count

879 full-law method evaluations on 63 distinct main inputs; full-law references remain at 10 or 12 modes.
