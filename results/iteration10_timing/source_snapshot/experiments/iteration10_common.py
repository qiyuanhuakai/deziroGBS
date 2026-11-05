"""Bookkeeping for the fixed coherent-core experiment; not a sampling kernel."""
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    """JSON metadata conversion, preserving values outside binary64 range.

    Regular finite floating values become JSON numbers for readable metadata.
    A finite nonzero extended-precision value that would become zero or infinity
    is emitted as a decimal string instead.  Full state precision belongs in
    the accompanying NPZ/native decimal evidence, not this metadata conversion.
    In particular, longdouble.item() must not recurse: it is still longdouble.
    """
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            raise ValueError("Nonfinite floating metadata")
        ordinary = float(value)
        if not np.isfinite(ordinary) or (value != 0 and ordinary == 0):
            return np.format_float_scientific(value, unique=True, trim="k")
        return ordinary
    if isinstance(value, np.complexfloating):
        return [plain(value.real), plain(value.imag)]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, np.generic):
            raise TypeError("Unsupported NumPy metadata scalar: " + type(value).__name__)
        return plain(item)
    if isinstance(value, complex):
        return [plain(value.real), plain(value.imag)]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def save(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    paths = list((ROOT / "src/gbslab").glob("*.py"))
    paths += list((ROOT / "tests").glob("test_*.py"))
    paths += list((ROOT / "experiments").glob("*coherent_core*"))
    paths += list((ROOT / "experiments").glob("iteration10*.py"))
    paths += [ROOT / "experiments/verify_iteration10.py", ROOT / "experiments/click_sampler.cpp",
              ROOT / "experiments/build_click_sampler.py", ROOT / "configs/iteration10_coherent_core.json"]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths)) if p.is_file()}


def verify_frozen_config(config_path):
    plan = json.loads((ROOT / "results/iteration10_plan/plan.json").read_text())
    if plan["config_sha256"] != sha(config_path):
        raise ValueError("The predeclared configuration has changed")
    return plan


def verify_gate(path, core, retained):
    gate = json.loads(Path(path).read_text())
    if gate.get("passed") is not True or gate.get("skipped") != 0:
        raise ValueError("A passing complete zero-skip gate is required")
    for name, expected in gate["source_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("A gated source changed: " + name)
    for name, path in {"core": core, "retained": retained}.items():
        if gate["binary_hashes"][name] != sha(path):
            raise ValueError("The requested binary differs from the tested binary: " + name)
    required = {"experiments/coherent_core_sampler.cpp", "src/gbslab/coherent_core.py",
                "experiments/iteration10_reference.py", "tests/test_coherent_core_independent.py"}
    if not required.issubset(gate["source_hashes"]):
        raise ValueError("The gate does not bind the complete new implementation/reference")
    return gate


def start_run(output, config_path, gate_path, core, retained, kind):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory, also after a failed run")
    verify_frozen_config(config_path)
    verify_gate(gate_path, core, retained)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.json")
    shutil.copy2(gate_path, output / "validation_gate.json")
    bound = source_hashes()
    for name in bound:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    binaries = {"core": Path(core), "retained": Path(retained)}
    for name, path in binaries.items():
        shutil.copy2(path.parent / "build.json", output / (name + "_build.json"))
    metadata = {"kind": kind, "started_utc": utc(), "status": "running",
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": sha(config_path), "validation_sha256": sha(gate_path),
        "source_hashes": bound, "binary_hashes": {n: sha(p) for n, p in binaries.items()},
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "dedicated_host": False, "logical_cpu_count": os.cpu_count(),
        "allowed_cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown")}
    save(output / "metadata.json", metadata)
    return metadata, binaries


def finish_run(output, metadata, binaries, **updates):
    stable = all(sha(ROOT / name) == expected for name, expected in metadata["source_hashes"].items())
    stable = stable and all(sha(path) == metadata["binary_hashes"][name] for name, path in binaries.items())
    metadata.update(completed_utc=utc(), sources_and_binaries_stable=stable, **updates)
    if not stable:
        metadata["status"] = "investigation_required"
    save(Path(output) / "metadata.json", metadata)
    return stable


def total_variation(q, p, failure=0.0):
    q, p = np.asarray(q, dtype=float), np.asarray(p, dtype=float)
    if q.shape != p.shape or np.any(q < 0) or np.any(p < 0):
        raise ValueError("TV requires matching nonnegative probability arrays")
    return float((np.sum(np.abs(q - p), dtype=np.longdouble) + abs(failure)) / 2)


def validate_law(q, modes, failure=0.0):
    q = np.asarray(q, dtype=float)
    if q.shape != (1 << modes,) or not np.all(np.isfinite(q)) or np.any(q < 0):
        raise ArithmeticError("Invalid complete law")
    if not np.isfinite(failure) or failure < 0 or abs(float(np.sum(q, dtype=np.longdouble)) + failure - 1) > 2e-10:
        raise ArithmeticError("Invalid complete-law mass")
    return q
