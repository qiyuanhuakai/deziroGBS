"""Schmidt spectra of the displaced pure remainder on the measured circuit."""
import argparse
import ctypes
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.linalg import eigvals
from threadpoolctl import threadpool_limits

from iteration111_memory_v1 import realify
from iteration115_displacement_v1 import conditional
from iteration115_visible_v1 import ROOT, hardware


def compile_counter(directory):
    name = "schmidt.dll" if os.name == "nt" else "schmidt.so"
    library = Path(directory) / name
    subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC",
                    str(ROOT / "native/gaussian_schmidt116.cpp"), "-o", str(library)], check=True)
    if os.name == "nt":
        compile_counter.runtime = os.add_dll_directory(str(Path(shutil.which("g++")).parent))
    fn = ctypes.CDLL(str(library)).schmidt_mass116
    floats = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    integers = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    fn.argtypes = [floats, ctypes.c_int, integers, ctypes.c_int, floats]
    fn.restype = ctypes.c_int
    return fn


def masses(fn, ratios, ranks):
    ratios = np.sort(np.asarray(ratios, dtype=float))[::-1].copy()
    ranks = np.asarray(ranks, dtype=np.int32)
    result = np.empty(len(ranks))
    assert fn(ratios, len(ratios), ranks, len(ranks), result) == 0
    assert np.all(np.diff(result) >= -1e-13) and result.max() <= 1 + 1e-12
    return result


def check_counter(fn):
    for ratios in ([0.2, 0.2, 0.04], [0.5, 0.24, 0.03], [0., 0., 0.]):
        # Outside the box every individual configuration weighs at most max(q)^17
        # times the vacuum; check that none can enter the first 37 terms.
        weights = sorted((np.prod(np.power(ratios, n)) for n in itertools.product(range(17), repeat=3)),
                         reverse=True)
        assert weights[36] >= max(ratios)**17
        expect = np.cumsum(weights[:37]) * np.prod(1 - np.array(ratios))
        np.testing.assert_allclose(masses(fn, ratios, np.arange(1, 38)), expect, atol=3e-15, rtol=1e-14)
    np.testing.assert_allclose(masses(fn, [.8], [1, 3, 15, 150]), 1-.8**np.array([1, 3, 15, 150]))


def thermal_ratios(q, v, indices):
    # Vacuum directions outside this thin QR need no eigensolve.
    b = realify(np.linalg.qr(q[indices], mode="r"))
    local = np.eye(len(b)) + b @ (v - np.eye(len(v))) @ b.T
    d = len(b)//2
    om = np.block([[np.zeros((d,d)), np.eye(d)], [-np.eye(d), np.zeros((d,d))]])
    spectrum = eigvals(1j * local @ om)
    assert abs(spectrum.imag).max() < 1e-10
    nu = np.sort(spectrum.real)[d:]
    assert nu.min() > 1-2e-10
    return np.sort(np.maximum(0, (nu-1)/(nu+1)))[::-1]


def run(output):
    t, ss = hardware()
    models = [conditional(t, s)[:2] for s in ss]
    # Each pair of adjacent bank boundaries is 8 physical output modes apart.
    orders = {"physical": np.arange(144), "shuffle_116": np.random.default_rng(116).permutation(144)}
    ranks = [1, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
    result = {}
    with tempfile.TemporaryDirectory() as scratch:
        fn = compile_counter(scratch)
        check_counter(fn)
        for order_name, banks in orders.items():
            modes = (8*banks[:,None]+np.arange(8)).ravel()
            rows = {}
            for cut in [1, 9, 36, 72, 108, 135, 143]:
                left, right = modes[:8*cut], modes[8*cut:]
                spectra = [thermal_ratios(q, v, left) for q,v in models]
                complement = [thermal_ratios(q, v, right) for q,v in models]
                for a, b in zip(spectra, complement):
                    size = min(len(a), len(b))
                    np.testing.assert_allclose(a[:size], b[:size], atol=3e-12, rtol=0)
                    assert max(a[size:], default=0) < 3e-12 and max(b[size:], default=0) < 3e-12
                rows[cut] = {}
                for tag, ratios in zip(["spectrum_0","spectrum_1","product"], [*spectra,np.concatenate(spectra)]):
                    retained = masses(fn, ratios, ranks)
                    rows[cut][tag] = dict(ratios=ratios.tolist(), discarded=np.maximum(0,1-retained).tolist())
                print(order_name, cut, rows[cut]["product"]["discarded"][-1], flush=True)
            result[order_name] = rows
        if os.name == "nt":
            import _ctypes
            _ctypes.FreeLibrary(fn._objects["0"]._handle)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(ranks=ranks, orders=result), indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    with threadpool_limits(limits=2):
        run(parser.parse_args().output)
