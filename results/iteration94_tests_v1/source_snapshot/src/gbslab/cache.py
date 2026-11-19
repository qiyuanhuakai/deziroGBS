"""Content-addressed cache for synthetic exact reference distributions.

Only evaluator references are cached on disk. Every candidate fit still gets
a fresh oracle; reading cached exact targets cannot help candidate timings.
Hashes cover the physical specification and the exact/optics source files.
Use one writer at a time: simultaneous experiments sharing this directory are
not supported. All bundled reproduction commands run experiments sequentially.
No pickle, downloaded experimental records, or credentials are stored.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .exact import ThresholdModel


def cached_reference(spec, covariance, directory):
    directory = Path(directory)
    engine = hashlib.sha256()
    for name in ("gaussian.py", "exact.py"):
        engine.update(name.encode())
        engine.update((Path(__file__).parent / name).read_bytes())
    identity = {"schema": 1, "spec": spec, "engine_sha256": engine.hexdigest(),
                "numpy_version": np.__version__}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    payload = directory / f"{key}.npz"
    manifest = directory / f"{key}.json"
    if payload.exists() and manifest.exists():
        data = json.loads(manifest.read_text())
        actual = hashlib.sha256(payload.read_bytes()).hexdigest()
        if actual != data["payload_sha256"] or data["identity"] != identity:
            raise ValueError(f"Reference cache integrity failure: {key}")
        with np.load(payload, allow_pickle=False) as saved:
            if not np.array_equal(saved["covariance"], covariance):
                raise ValueError("Cached covariance differs despite matching key")
            p = saved["target"].copy()
        return p, data["compute_seconds"], data["diagnostics"], True, key
    start = perf_counter()
    model = ThresholdModel(covariance)
    p = model.full_distribution().copy()
    elapsed = perf_counter() - start
    directory.mkdir(parents=True, exist_ok=True)
    with payload.with_suffix(".tmp").open("wb") as handle:
        np.savez_compressed(handle, covariance=covariance, target=p)
    payload.with_suffix(".tmp").replace(payload)
    data = {"identity": identity, "payload_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            "compute_seconds": elapsed, "diagnostics": model.cache_info()}
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    return p, elapsed, data["diagnostics"], False, key
