# Iteration 6 post hoc numerical diagnostic

This separate diagnostic was chosen after a numerical failure was observed in the primary run. It includes all six fixed m=12 inputs for pair_k4, without changing k, the completion rule, the Gaussian selector, guard threshold, or the primary archive. No new samples are drawn.

The saved full target P supplies event probabilities through TableEventOracle. This is an exponential reference and inherits the saved target's floating arithmetic and any recorded corrections. Its timings cannot support native or scalable speed claims.

Each case preserves both full laws and their failure symbols. Three distances are reported: primary to P, table reference to P, and primary to table reference. Their differences are not an additive decomposition of formula and numerical errors. A table improvement diagnoses oracle/guard sensitivity; it does not replace the primary result or establish a certified exact-arithmetic law. Primary failure reasons and masses remain in each report.
