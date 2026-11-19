"""Second-round GBS experiments: selection, exact kernel equivalence and cost.

All target probabilities are evaluator-only. Full law and chain diagnostics
remain exponential small-system tools, explicitly separate from sampling.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
from time import perf_counter

import numpy as np
import scipy
from scipy.special import rel_entr

from .cache import cached_reference
from .exact import ThresholdModel
from .experiment import digest, json_safe, source_digest, click_count_distribution
from .gaussian import haar_unitary, local_unitary, covariance_from_interferometer
from .surrogates import distribution_metrics
from .zero_memory import ZeroMemorySampler
from .schur_memory import SchurMemorySampler
from .conditional_memory import SchurConditionalMemorySampler


def instance(spec):
    m = int(spec["modes"])
    rng = np.random.default_rng(spec["seed"])
    topology = spec["topology"]
    U = haar_unitary(m, rng) if topology == "haar" else local_unitary(m, int(topology[5:]), rng)
    r, eta = np.full(m, spec["squeezing"]), np.full(m, spec["transmissivity"])
    if spec.get("profile", "uniform") == "heterogeneous":
        # The same interferometer is reused; independent profile RNG avoids
        # changing optical structure when changing squeezing/loss profiles.
        profile_rng = np.random.default_rng(spec["seed"] + 80123)
        r *= profile_rng.uniform(0.55, 1.45, m)
        eta = np.clip(eta + profile_rng.uniform(-.25, .25, m), .05, .95)
    V = covariance_from_interferometer(U, r, loss=eta)
    return U, r, eta, V


def candidate(model, method):
    name, k = method["strategy"], int(method["k"])
    if name.startswith("old_"):
        return ZeroMemorySampler(model, k, selection=name.removeprefix("old_"))
    if name.startswith("schur_"):
        return SchurMemorySampler(model, k, selection=name.removeprefix("schur_"),
            covariance_cache_size=128, covariance_cache_bytes=64*1024**2)
    if name in ("conditional_mi", "click_influence"):
        return SchurConditionalMemorySampler(model, k, selection=name,
            covariance_cache_size=128, covariance_cache_bytes=64*1024**2)
    raise ValueError(f"Unknown second-round method: {name}")


def chain_diagnostics(p, q):
    """Exact-law diagnostic, not a scalable or omitted-information bound.

TV(P,Q)<=sum_i E_{P(prefix)} TV(P_i(.|prefix), Q_i(.|prefix)).
Forward KL equals the analogous sum of conditional Bernoulli KL values.
Q-prefixes of zero mass are assigned a neutral kernel extension; this leaves
the law Q unchanged. Such prefixes are reported, not assumed impossible.
"""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.ndim != 1 or q.shape != p.shape or p.size < 2 or p.size & (p.size-1):
        raise ValueError("p and q must have the same nontrivial power-of-two shape")
    if any(np.min(v)<0 or not np.isclose(v.sum(),1,atol=1e-9,rtol=0) for v in (p,q)):
        raise ValueError("p and q must be normalized")
    # Match distribution_metrics' normalization convention after accepting
    # only small floating normalization residue.
    p, q = p / p.sum(), q / q.sum()
    m = p.size.bit_length()-1
    stage_rows = []
    total_l1, total_kl = 0.0, 0.0
    for i in range(m):
        # Lowest i bits are the sampled prefix in original mode order.
        jp = p.reshape(-1, 1 << (i+1)).sum(axis=0).reshape(2, 1 << i)
        jq = q.reshape(-1, 1 << (i+1)).sum(axis=0).reshape(2, 1 << i)
        pp, qp = jp.sum(axis=0), jq.sum(axis=0)
        pc = np.divide(jp[1], pp, out=np.full_like(pp,.5), where=pp>0)
        qc = np.divide(jq[1], qp, out=np.full_like(qp,.5), where=qp>0)
        weighted = pp * np.abs(pc-qc)
        active = pp>0
        # Divide both outcome masses directly. Taking 1-p(1) would erase a
        # tiny positive p(0) when p(1) rounds to one, creating false infinity.
        cp = np.divide(jp, pp[None,:], out=np.full_like(jp,.5), where=pp[None,:]>0)
        cq = np.divide(jq, qp[None,:], out=np.full_like(jq,.5), where=qp[None,:]>0)
        kl = rel_entr(cp[:,active],cq[:,active]).sum(axis=0)
        kl_value = float(np.dot(pp[active],kl))
        click_groups = {}
        for bits, value in enumerate(weighted):
            c = bits.bit_count()
            click_groups[c] = click_groups.get(c,0.0) + float(value)
        d = float(weighted.sum())
        stage_rows.append({"mode":i,"weighted_conditional_tv":d,
            "weighted_conditional_kl":kl_value,
            "p_mass_on_q_zero_prefixes":float(pp[qp==0].sum()),
            "conditional_tv_by_prior_click_count":click_groups})
        total_l1 += d; total_kl += kl_value
    return {"prefix_law":"P", "tv_hybrid_bound":min(1.0,total_l1),
        "sum_expected_conditional_tv":total_l1,"sum_expected_conditional_kl":total_kl,
        "stages":stage_rows,"scalable_certificate":False}


def snapshot(output):
    target = output / "source_snapshot" / "gbslab"
    target.mkdir(parents=True, exist_ok=True)
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path,target/path.name)


def run(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True,exist_ok=True)
    if any((output/name).exists() for name in ("metrics.csv","metadata.json","records.jsonl")):
        raise FileExistsError("Previous run exists; use a new output directory")
    snapshot(output)
    inputs = output/"instances";inputs.mkdir(exist_ok=True)
    metadata={"config":config,"config_sha256":digest(config),"source_sha256":source_digest(),
        "started_utc":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),
        "numpy":np.__version__,"scipy":scipy.__version__,"cpu_count":os.cpu_count(),
        "thread_environment":{k:os.environ.get(k) for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"]},
        "notes":["Fresh candidate oracle, sampling before dense evaluation.",
            "Dense TV/KL and prefix diagnostics are evaluation-only.",
            "Sample timing includes lazy selection; fit-only timings are insufficient.",
            "Profiles/parameters share interferometers; no iid case-population inference."],
        "status":"running"}
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    keys=["modes","topology","squeezing","transmissivity","profile","seed"]
    specs=[dict(zip(keys,v)) for v in itertools.product(config["modes"],config["topologies"],
        config["squeezing"],config["transmissivity"],config["profiles"],config["seeds"])]
    rows, failures, manifests=[],[],[]
    started=perf_counter()
    with (output/"records.jsonl").open("w") as journal:
        for ci,spec in enumerate(specs):
            case_id=digest(spec)[:16]
            U,r,eta,V=instance(spec)
            p,ref_time,ref_diag,hit,key=cached_reference(spec,V,config_path.parent.parent/".cache"/"targets")
            count_p=click_count_distribution(p,spec["modes"])
            input_path=inputs/f"{case_id}.npz"
            np.savez_compressed(input_path,U=U,squeezing=r,transmissivity=eta,covariance=V,target=p)
            manifests.append({"case_id":case_id,"spec":spec,"path":f"instances/{case_id}.npz",
                "sha256":hashlib.sha256(input_path.read_bytes()).hexdigest(),"reference_seconds":ref_time,
                "reference_diagnostics":ref_diag,"reference_cache_hit":hit,"reference_cache_key":key})
            qs={}
            for method in config["methods"]:
                stage="fit"
                row={"case_id":case_id,**spec,**method,"sample_count":config["samples_per_method"],
                    "target_mean_clicks":float(np.dot(np.arange(len(count_p)),count_p)),
                    "target_mean_photons":float((np.trace(V)-2*spec["modes"])/4)}
                try:
                    t=perf_counter(); oracle=ThresholdModel(V,vacuum_cache_size=8192)
                    fit=candidate(oracle,method); row["fit_seconds"]=perf_counter()-t
                    stage="sampling";t=perf_counter()
                    samples=fit.sample(config["samples_per_method"],np.random.default_rng(spec["seed"]+100000))
                    row["sample_seconds"]=perf_counter()-t
                    row["cold_batch_seconds"]=row["fit_seconds"]+row["sample_seconds"]
                    after_sampling={"oracle":oracle.cache_info(),"candidate":fit.cache_info()}
                    stage="enumeration";t=perf_counter();q=fit.full_distribution()
                    row["evaluation_seconds"]=perf_counter()-t
                    row.update(distribution_metrics(p,q))
                    row["click_count_tv"]=float(np.abs(count_p-click_count_distribution(q,spec["modes"])).sum()/2)
                    row["click_tail_bound"]=float(count_p[method["k"]+1:].sum())
                    if spec["profile"]=="uniform":
                        from .photon_tail import photon_number_tail
                        row["photon_tail_bound"]=photon_number_tail(r,float(eta[0]),method["k"])
                    else: row["photon_tail_bound"]=None
                    chain=chain_diagnostics(p,q)
                    row["hybrid_tv_bound_evaluation_only"]=chain["tv_hybrid_bound"]
                    row["chain_kl_identity_error"]=float(abs(row["kl"]-chain["sum_expected_conditional_kl"]))
                    row["reference_numerical_l1_correction"]=ref_diag["largest_l1_numerical_correction"]
                    if row["tv"]>chain["tv_hybrid_bound"]+1e-9:
                        raise ArithmeticError("TV exceeds chain diagnostic bound")
                    if np.isfinite(row["kl"]) and row["chain_kl_identity_error"]>1e-9:
                        raise ArithmeticError("KL chain identity failed")
                    qs[f"{method['strategy']}_k{method['k']}"]=q
                    rows.append(row)
                    record={"status":"ok","metrics":row,"after_sampling":after_sampling,
                        "after_evaluation":{"oracle":oracle.cache_info(),"candidate":fit.cache_info()},
                        "chain_diagnostic":chain}
                except (ArithmeticError,ValueError,FloatingPointError) as exc:
                    record={"status":"failed","case_id":case_id,"spec":spec,"method":method,
                        "stage":stage,"error_type":type(exc).__name__,"error":str(exc)}
                    failures.append(record)
                journal.write(json.dumps(json_safe(record))+"\n");journal.flush()
            np.savez_compressed(output/f"surrogates_{case_id}.npz",**qs)
            print(f"{ci+1}/{len(specs)} {spec['topology']} {spec['profile']} r={spec['squeezing']} eta={spec['transmissivity']} rows={len(rows)} failures={len(failures)}",flush=True)
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (output/"metrics.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    (output/"instances_manifest.json").write_text(json.dumps(manifests,indent=2)+"\n")
    (output/"failures.json").write_text(json.dumps(failures,indent=2)+"\n")
    metadata.update(completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=perf_counter()-started,
        cases=len(specs),successful_rows=len(rows),failed_rows=len(failures),status="complete" if not failures else "complete_with_failures")
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({k:metadata[k] for k in ["cases","successful_rows","failed_rows","elapsed_seconds","status"]}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    args=p.parse_args();run(args.config,args.output)


if __name__=="__main__": main()
