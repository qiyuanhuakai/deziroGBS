# Iteration 16: layer checkpoints and rolling budgets

One exploratory run per method/input. Larger caps/resources than iteration 15; restart rerun as control. Soft-limit granularity differs, so equal per-attempt seconds is not equal total budget.

| m | Seed | Target | Method | Status | d | Bound | Search seconds (all attempts) | Final coefficients | Construction peak MiB | Actual small-state distance |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 4 | 150801 | 0.03 | checkpoint_rolling | accepted | 12 | 0.019844092482196173 | 0.26042239999515004 | 1872 | 94.76953125 | 0.002575951724330397 |
| 4 | 150801 | 0.03 | restart | accepted | 12 | 0.018568747715447684 | 0.687590999994427 | 2184 | 94.87109375 | 0.0026453082174965684 |
| 4 | 150801 | 0.1 | checkpoint_rolling | accepted | 10 | 0.06009271102554842 | 0.08343529999547172 | 900 | 92.578125 | 0.008446323363668302 |
| 4 | 150801 | 0.1 | restart | accepted | 10 | 0.08548348734707581 | 0.07509570001275279 | 1350 | 93.32421875 | 0.019111191032427245 |
| 4 | 150802 | 0.03 | restart | accepted | 12 | 0.02011419006050437 | 0.7685938999929931 | 4200 | 95.484375 | 0.0032848176982009903 |
| 4 | 150802 | 0.03 | checkpoint_rolling | accepted | 12 | 0.020972315869350472 | 0.3282854999997653 | 4032 | 94.703125 | 0.0032299321271636334 |
| 4 | 150802 | 0.1 | restart | accepted | 10 | 0.06593693838863197 | 0.17082359999767505 | 2040 | 93.55078125 | 0.011141697790298366 |
| 4 | 150802 | 0.1 | checkpoint_rolling | accepted | 10 | 0.0643541476859196 | 0.10559089999878779 | 1920 | 92.7109375 | 0.009724146283203219 |
| 32 | 150801 | 0.1 | restart | search_unaccepted |  |  | 165.15026730000682 |  |  |  |
| 32 | 150801 | 0.1 | checkpoint_equal | search_unaccepted |  |  | 88.02450460000546 |  |  |  |
| 32 | 150801 | 0.1 | checkpoint_rolling | search_unaccepted |  |  | 121.24679130000004 |  |  |  |
| 32 | 150802 | 0.1 | restart | search_unaccepted |  |  | 78.35130579999532 |  |  |  |
| 32 | 150802 | 0.1 | checkpoint_rolling | search_unaccepted |  |  | 68.16640899999766 |  |  |  |
| 32 | 150802 | 0.1 | checkpoint_equal | search_unaccepted |  |  | 82.5348745000083 |  |  |  |
| 128 | 150801 | 0.1 | restart | search_unaccepted |  |  | 98.39411660000042 |  |  |  |
| 128 | 150801 | 0.1 | checkpoint_rolling | search_unaccepted |  |  | 72.08184769999934 |  |  |  |
| 128 | 150802 | 0.1 | restart | search_unaccepted |  |  | 94.17587040000944 |  |  |  |
| 128 | 150802 | 0.1 | checkpoint_rolling | search_unaccepted |  |  | 90.74801009999646 |  |  |  |

## All attempt terminal states

| Run | Method | d | cap | State | Complete/partial ledger | Error |
|---:|---|---:|---:|---|---:|---|
| 0 | restart | 10 | 8 | budget_missed | 2.7583800088107004 |  |
| 0 | restart | 12 | 8 | budget_missed | 2.6750723682537405 |  |
| 0 | restart | 12 | 16 | budget_missed | 1.2690070115965204 |  |
| 0 | restart | 12 | 32 | resource_failed |  | TimeoutError('Attempt time cap reached between gates') |
| 1 | restart | 10 | 8 | budget_missed | 3.210402193380454 |  |
| 1 | restart | 12 | 8 | budget_missed | 3.13831649617393 |  |
| 1 | restart | 12 | 16 | budget_missed | 1.6536220282617053 |  |
| 1 | restart | 12 | 32 | resource_failed |  | TimeoutError('Attempt time cap reached between gates') |
| 2 | checkpoint_rolling | 10 | layerwise | projection_budget_exhausted | 0.03042071466602603 |  |
| 2 | checkpoint_rolling | 12 | layerwise | accepted | 0.019844092482196173 |  |
| 3 | restart | 10 | 8 | budget_missed | 0.15048365032854605 |  |
| 3 | restart | 10 | 16 | accepted | 0.06593693838863197 |  |
| 4 | checkpoint_rolling | 10 | layerwise | accepted | 0.0643541476859196 |  |
| 5 | restart | 10 | 8 | budget_missed | 0.143545286170687 |  |
| 5 | restart | 10 | 16 | budget_missed | 0.05170046142031485 |  |
| 5 | restart | 10 | 32 | budget_missed | 0.034293330862628923 |  |
| 5 | restart | 10 | 64 | budget_missed | 0.034293330862628923 |  |
| 5 | restart | 10 | 128 | budget_missed | 0.034293330862628923 |  |
| 5 | restart | 12 | 128 | accepted | 0.02011419006050437 |  |
| 6 | checkpoint_rolling | 10 | layerwise | accepted | 0.06009271102554842 |  |
| 7 | restart | 10 | 8 | budget_missed | 1.7250236191977408 |  |
| 7 | restart | 10 | 16 | budget_missed | 0.9282951350008601 |  |
| 7 | restart | 10 | 32 | budget_missed | 0.465754854790823 |  |
| 7 | restart | 10 | 64 | resource_failed |  | TimeoutError('Attempt time cap reached between gates') |
| 8 | checkpoint_rolling | 10 | layerwise | projection_budget_exhausted | 0.10690753277987099 |  |
| 8 | checkpoint_rolling | 12 | layerwise | resource_failed | 0.051482926494413286 | TimeoutError('Attempt time cap reached between operations') |
| 9 | checkpoint_rolling | 10 | layerwise | resource_failed | 0.09792409897772948 | TimeoutError('Attempt time cap reached between operations') |
| 10 | restart | 10 | 8 | budget_missed | 1.604594252709411 |  |
| 10 | restart | 10 | 16 | budget_missed | 0.833273665901407 |  |
| 10 | restart | 10 | 32 | budget_missed | 0.3987828699647776 |  |
| 10 | restart | 10 | 64 | budget_missed | 0.19805789054141043 |  |
| 10 | restart | 10 | 128 | resource_failed |  | TimeoutError('Attempt time cap reached between gates') |
| 11 | checkpoint_rolling | 10 | layerwise | projection_budget_exhausted | 0.03162415001153793 |  |
| 11 | checkpoint_rolling | 12 | layerwise | accepted | 0.020972315869350472 |  |
| 12 | checkpoint_rolling | 10 | layerwise | resource_failed | 0.06659749068203516 | TimeoutError('Attempt time cap reached between operations') |
| 13 | restart | 10 | 8 | accepted | 0.08548348734707581 |  |
| 14 | checkpoint_equal | 10 | layerwise | rank_budget_unmet | 0.05572828142986051 |  |
| 14 | checkpoint_equal | 12 | layerwise | resource_failed | 0.03364461701463986 | TimeoutError('Attempt time cap reached between operations') |
| 15 | checkpoint_equal | 10 | layerwise | rank_budget_unmet | 0.052890423776648654 |  |
| 15 | checkpoint_equal | 12 | layerwise | resource_failed | 0.035326416043791914 | TimeoutError('Attempt time cap reached between operations') |
| 16 | restart | 10 | 8 | budget_missed | 0.07657356400788787 |  |
| 16 | restart | 10 | 16 | budget_missed | 0.03812416921022808 |  |
| 16 | restart | 10 | 32 | budget_missed | 0.03152178073823316 |  |
| 16 | restart | 10 | 64 | budget_missed | 0.03152178073823316 |  |
| 16 | restart | 10 | 128 | budget_missed | 0.03152178073823316 |  |
| 16 | restart | 12 | 128 | accepted | 0.018568747715447684 |  |
| 17 | checkpoint_rolling | 10 | layerwise | rank_budget_unmet | 0.065687021231051 |  |
| 17 | checkpoint_rolling | 12 | layerwise | resource_failed | 0.03767607123576931 | TimeoutError('Attempt time cap reached between operations') |
