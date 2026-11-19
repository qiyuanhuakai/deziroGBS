"""Archive the analytic obstruction and every small numerical crosscheck."""
from pathlib import Path
import json
from datetime import datetime, timezone
from gbslab.paired_stress import recent_tv, sampler_crosscheck

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase2_paired"
OUT.mkdir(parents=True, exist_ok=True)
if (OUT / "results.json").exists():
    raise FileExistsError("Existing stress archive must not be overwritten")
analytic = [{"n_pairs": n, "modes": 2*n, "p": .5, "k": k,
             "blocked_recent_tv": recent_tv(n, k, .5),
             "blocked_pairwise_tv": 0.0, "interleaved_recent_tv": 0.0}
            for n in (2, 4, 8, 16, 32, 64, 128) for k in (1, 2, 3)]
checks = []
for n in (2, 3, 4):
    for k in (1, 2):
        for selector, order in (("recent", "blocked"), ("pairwise", "blocked"), ("recent", "interleaved")):
            for oracle in ("analytic", "covariance"):
                checks.append(sampler_crosscheck(n, k, .5, selection=selector,
                              order_kind=order, oracle_kind=oracle))
result = {"generated_utc": datetime.now(timezone.utc).isoformat(),
          "meaning": "Selection/order obstruction on a classically trivial physical TMSV-pair family",
          "large_sizes": "analytic formula only, not dense generic-GBS enumeration",
          "analytic_series": analytic, "crosschecks": checks,
          "guard_failures": sum(r["status"] != "ok" for r in checks)}
(OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"analytic_rows": len(analytic), "crosschecks":len(checks), "guard_failures":result["guard_failures"]}))
