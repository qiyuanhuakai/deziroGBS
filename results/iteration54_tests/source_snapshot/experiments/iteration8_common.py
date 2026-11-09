"""Shared bookkeeping for the fixed round-eight experiments, not a sampler."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np

from gbslab.exact import NumericalProbabilityError
from gbslab.positive_completion import TableEventOracle
from iteration7_forest import extended_law

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def save(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sources():
    paths = sorted((ROOT / "src/gbslab").glob("*.py"))
    paths += sorted((ROOT / "tests").glob("test_*forest*.py"))
    paths += sorted((ROOT / "experiments").glob("*batched_forest*"))
    paths += sorted((ROOT / "experiments").glob("iteration8*.py"))
    paths += [ROOT / "experiments/click_sampler.cpp", ROOT / "experiments/build_click_sampler.py",
              ROOT / "experiments/iteration7_forest.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths)) if p.is_file()}


def verify_gate(path, library, retained, forest):
    gate = json.loads(Path(path).read_text())
    if gate.get("passed") is not True or gate.get("skipped", 0) != 0:
        raise ValueError("A passing, zero-skip round-eight gate is required")
    required = {"src/gbslab/batched_forest.py", "src/gbslab/cpp_forest.py",
                "experiments/batched_forest_kernel.cpp", "experiments/batched_forest_kernel.h",
                "experiments/batched_forest_sampler.cpp", "tests/test_batched_forest_numerics.py"}
    bound = gate.get("source_hashes", {})
    if not required.issubset(bound):
        raise ValueError("Validation gate does not bind all new scientific sources")
    for name, expected in bound.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Validation source changed: " + name)
    expected = {"library": sha(library), "retained": sha(retained), "forest": sha(forest)}
    if gate.get("binary_hashes") != expected:
        raise ValueError("Gate must bind the exact three candidate binaries")
    return gate


def start_run(output, config_path, gate_path, library, retained, forest, kind):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh output directory, including after a partial run")
    verify_gate(gate_path, library, retained, forest)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.json")
    shutil.copy2(gate_path, output / "validation_gate.json")
    bound = sources()
    for name in bound:
        dest = output / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    binaries = {"library": Path(library), "retained": Path(retained), "forest": Path(forest)}
    for name, path in binaries.items():
        build = path.parent / "build.json"
        if build.is_file():
            shutil.copy2(build, output / (name + "_build.json"))
    metadata = {"kind": kind, "started_utc": utc(), "status": "running",
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "config_sha256": sha(config_path), "validation_sha256": sha(gate_path),
                "source_hashes": bound, "binary_hashes": {n: sha(p) for n, p in binaries.items()},
                "python": sys.version, "numpy": np.__version__, "dedicated_host": False,
                "platform": platform.platform(), "machine": platform.machine(),
                "logical_cpu_count": os.cpu_count(),
                "allowed_cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                "thread_environment": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown") if Path("/proc/cpuinfo").is_file() else "unknown"}
    save(output / "metadata.json", metadata)
    return metadata, binaries


def finish_run(output, metadata, binaries, **updates):
    stable = all(sha(ROOT / name) == digest for name, digest in metadata["source_hashes"].items())
    stable = stable and all(sha(path) == metadata["binary_hashes"][name] for name, path in binaries.items())
    metadata.update(completed_utc=utc(), sources_and_binaries_stable=stable, **updates)
    if not stable:
        metadata["status"] = "investigation_required"
    save(Path(output) / "metadata.json", metadata)
    return stable


def validate_law(q, failure, modes):
    q = np.asarray(q, dtype=float)
    if (q.shape != (1 << modes,) or not np.all(np.isfinite(q)) or np.any(q < 0)
            or np.any(q > 1) or not np.isfinite(failure) or not 0 <= failure <= 1
            or abs(float(np.sum(q, dtype=np.longdouble)) + failure - 1) > 2e-11):
        raise ArithmeticError("Invalid extended probability law")
    return q


def total_variation(q, failure, p, p_failure=0):
    return float((np.abs(np.asarray(q) - p).sum() + abs(failure - p_failure)) / 2)


def make_python_sampler(V, spec, config, library):
    from gbslab.batched_forest import BatchedForestSampler
    from gbslab.conditional_memory import ConditionalMemorySampler
    from gbslab.exact import ThresholdModel
    model = ThresholdModel(V, max_exact_modes=config["full_law_max_modes"],
                           vacuum_cache_size=config["python_cache_size"])
    common = dict(selection=config["selector"], cache_size=config["python_cache_size"],
                  max_relative_roundoff=config["max_relative_roundoff"])
    if spec["completion"] == "none":
        return ConditionalMemorySampler(model, spec["k"], **common)
    return BatchedForestSampler(model, spec["k"], forest=spec["completion"],
        precision=config["precision"], library=library, **common)


def native_run(V, spec, config, retained, forest, *, samples=0, seed=0,
               enumerate_distribution=False, trace=False, timeout=600):
    from gbslab.cpp_click import run_cpp_click
    from gbslab.cpp_forest import run_cpp_forest
    common = dict(max_clicks=spec["k"], samples=samples, seed=seed,
                  enumerate_distribution=enumerate_distribution,
                  raw_samples=False, trace_conditionals=trace, timeout=timeout)
    if spec["completion"] == "none":
        return run_cpp_click(V, selection=config["selector"], executable=retained,
            vacuum_kernel="product", prefix_cache_entries=0, **common)
    return run_cpp_forest(V, forest=spec["completion"], precision=config["precision"],
                         executable=forest, **common)


class TracedTableForest:
    """Independent forest algebra using exact native-traced R/E and table sums.

    No Gaussian determinants or the batched score routine are used. This is
    an exponential reference, not the proposed sampler. Missing traces are
    an explicit coverage error; unobserved decisions are never manufactured.
    """
    def __init__(self, target, trace):
        self.oracle = TableEventOracle(target)
        self.trace = {}
        for row in trace:
            prefix = tuple(row["prefix"])
            if prefix in self.trace:
                raise ValueError("Duplicate native trace prefix")
            self.trace[prefix] = row

    def conditional_click_probability(self, prefix):
        if prefix not in self.trace:
            raise RuntimeError("Native trace does not cover the reference prefix")
        row = self.trace[prefix]
        mode = len(prefix)
        if row["mode"] != mode:
            raise ValueError("This formal trace oracle requires natural order")
        zeros = tuple(i for i, bit in enumerate(prefix) if bit == 0)
        clicks = {i for i, bit in enumerate(prefix) if bit == 1}
        retained = tuple(row["retained_clicks"])
        omitted = tuple(row["omitted_clicks"])
        edges = tuple(tuple(e) for e in row["forest_edges"])
        if set(retained) & set(omitted) or set(retained) | set(omitted) != clicks:
            raise ValueError("Traced R/F do not partition the observed clicks")
        if len(retained) != len(set(retained)) or len(omitted) != len(set(omitted)):
            raise ValueError("Duplicate traced mode")
        degree = dict.fromkeys(omitted, 0)
        parent = {j: j for j in omitted}

        def root(j):
            while parent[j] != j:
                j = parent[j]
            return j

        for a, b in edges:
            if a not in degree or b not in degree or root(a) == root(b):
                raise ValueError("Invalid or cyclic traced edge")
            parent[root(a)] = root(b)
            degree[a] += 1
            degree[b] += 1
        if len(edges) != max(0, len(omitted) - 1):
            raise ValueError("Traced tree is not spanning")
        log_scores = []
        for outcome in (0, 1):
            z = zeros + ((mode,) if outcome == 0 else ())
            r = retained + ((mode,) if outcome else ())
            w = self.oracle.event_probability(z, r).value
            nodes = {j: self.oracle.event_probability(z, r + (j,)).value for j in omitted}
            pairs = {e: self.oracle.event_probability(z, r + e).value for e in edges}
            if w == 0 or any(v == 0 for v in nodes.values()) or any(v == 0 for v in pairs.values()):
                log_scores.append(None)
                continue
            terms = [(1 - len(omitted) + len(edges), w)]
            terms.extend((1 - degree[j], nodes[j]) for j in omitted)
            terms.extend((1, pairs[e]) for e in edges)
            log_scores.append(sum((coefficient * np.log(mass) for coefficient, mass in terms if coefficient), np.longdouble(0)))
        if log_scores == [None, None]:
            raise NumericalProbabilityError("Table conditioning event has no support")
        if log_scores[0] is None:
            return 1.0
        if log_scores[1] is None:
            return 0.0
        delta = log_scores[0] - log_scores[1]
        if delta >= 0:
            t = np.exp(-delta)
            p = float(t / (1 + t))
        else:
            t = np.exp(delta)
            p = float(1 / (1 + t))
        if not 0 < p < 1:
            raise NumericalProbabilityError("Positive table branches reached an endpoint")
        return p


def compare_trace_selector(trace, sampler):
    mismatches = []
    for row in trace:
        if row.get("p_one", row.get("probability")) is None:
            continue
        detail = sampler.conditional_details(tuple(row["prefix"]))
        same = (tuple(sorted(row["retained_clicks"])) == tuple(sorted(detail["retained_clicks"]))
            and tuple(row["omitted_clicks"]) == tuple(detail["omitted_clicks"])
            and {tuple(sorted(e)) for e in row["forest_edges"]}
                == {tuple(sorted(e)) for e in detail["forest_edges"]})
        if not same:
            mismatches.append({"prefix": row["prefix"], "native_R": row["retained_clicks"],
                               "python_R": detail["retained_clicks"], "native_E": row["forest_edges"],
                               "python_E": detail["forest_edges"]})
    return {"checked_prefixes": len(trace), "mismatches": mismatches, "passed": not mismatches}
