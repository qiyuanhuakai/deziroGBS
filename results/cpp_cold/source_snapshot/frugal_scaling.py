"""Adapter for the separate native/portable official Frugal sampling build."""
from pathlib import Path
import json
import subprocess
import tempfile
from time import perf_counter
import numpy as np


def frugal_native_sample(covariance,cutoff,sample_count,seed,*,enumerate_distribution=False,
                         executable=None,timeout=300.0):
    V=np.asarray(getattr(covariance,"V",covariance),dtype=float)
    if V.ndim!=2 or V.shape[0]!=V.shape[1] or V.shape[0]%2 or not np.all(np.isfinite(V)):
        raise ValueError("finite even square covariance required")
    m=V.shape[0]//2
    if not 1<=cutoff<=min(5,m):raise ValueError("Frugal cutoff outside supported range")
    if sample_count<0 or not 0<=seed<2**32:raise ValueError("invalid count or seed")
    if enumerate_distribution and m>14:raise ValueError("full-law cap is14 modes")
    if executable is None:executable=Path(__file__).resolve().parents[2]/".cache/frugal-native/frugal_sampling_adapter"
    start=perf_counter()
    with tempfile.TemporaryDirectory(prefix="gbs-frugal-native-") as directory:
        path=Path(directory)/"covariance.csv";np.savetxt(path,V,delimiter=",",fmt="%.17g")
        cmd=[str(executable),"--input",str(path),"--modes",str(m),"--cutoff",str(cutoff),"--samples",str(sample_count),"--seed",str(seed),"--threads","1"]
        if enumerate_distribution:cmd.append("--enumerate")
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
    if result.returncode:raise RuntimeError(result.stderr.strip() or "Frugal native adapter failed")
    out=json.loads(result.stdout);out["wrapper_wall_seconds"]=perf_counter()-start
    if out["modes"]!=m or out["sample_count"]!=sample_count or sum(out["click_count_histogram"])!=sample_count:
        raise RuntimeError("Frugal metadata/sample count mismatch")
    if m<=14 and sum(out["histogram_counts"])!=sample_count:raise RuntimeError("histogram count mismatch")
    if enumerate_distribution:
        q=np.asarray(out["distribution"])
        if np.any(q<0) or not np.isfinite(q).all() or abs(q.sum()-1)>1e-11 or out["trace"]["nonfinite_threshold_path_steps"]:
            raise ArithmeticError("invalid Frugal traced law")
    return out
