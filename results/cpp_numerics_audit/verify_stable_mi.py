"""Independent, tiny numerical audit of Gaussian threshold pair mutual information.

The production implementation is not imported.  The reference enumerates the
2x2/4x4 determinant permutation formula using 80-digit Decimal arithmetic on
the exact stored float64 covariance entries.  The tested identity deliberately
uses NumPy float64 Cholesky solves plus long-double scalar arithmetic, a
different factorization path from the production scalar long-double routine.

Run from any directory:
    python /path/to/verify_stable_mi.py --output /new/path/results.json

An existing output is never overwritten.  Without --output, print full JSON.
"""

from argparse import ArgumentParser
from decimal import Decimal, getcontext
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
import platform

import numpy as np


HERE = Path(__file__).resolve().parent
LD = np.longdouble
getcontext().prec = 80


def decimal_vacuum(covariance, modes):
    m = covariance.shape[0] // 2
    indices = list(modes) + [j + m for j in modes]
    half = Decimal("0.5")
    matrix = [[Decimal.from_float(float(covariance[i, j])) * half
               + (half if i == j else Decimal(0))
               for j in indices] for i in indices]
    dimension = len(indices)
    determinant = Decimal(0)
    for permutation in permutations(range(dimension)):
        inversions = sum(permutation[i] > permutation[j]
                         for i in range(dimension)
                         for j in range(i + 1, dimension))
        term = Decimal(-1 if inversions % 2 else 1)
        for i, j in enumerate(permutation):
            term *= matrix[i][j]
        determinant += term
    if determinant <= 0:
        raise ArithmeticError("Reference determinant is not positive")
    return Decimal(1) / determinant.sqrt()


def decimal_reference(covariance, first, second):
    a = decimal_vacuum(covariance, (first,))
    b = decimal_vacuum(covariance, (second,))
    c = decimal_vacuum(covariance, (first, second))
    joint = [c, b-c, a-c, 1-a-b+c]
    product = [a*b, (1-a)*b, a*(1-b), (1-a)*(1-b)]
    valid = all(value >= 0 for value in joint)
    value = (sum(p * (p/r).ln() for p, r in zip(joint, product) if p > 0)
             if valid else None)
    return {"mi": value, "joint": joint, "delta": c-a*b,
            "stored_float_covariance_joint_nonnegative": valid}


def numpy_single_vacuum(covariance, mode):
    m = covariance.shape[0] // 2
    ids = [mode, mode+m]
    block = (covariance[np.ix_(ids, ids)] + np.eye(2))/2
    sign, logdet = np.linalg.slogdet(block)
    if sign <= 0:
        raise ArithmeticError("Single-mode determinant is not positive")
    result = float(np.exp(-0.5*logdet))
    if result > 1+2e-10:
        raise ArithmeticError("Vacuum probability exceeds one")
    return LD(min(1.0, result))


def stable_formula(covariance, first, second):
    m = covariance.shape[0] // 2
    ii, jj = [first, first+m], [second, second+m]
    A = covariance[np.ix_(ii, ii)] + np.eye(2)
    B = covariance[np.ix_(jj, jj)] + np.eye(2)
    C = covariance[np.ix_(ii, jj)]
    if np.all(C == 0):
        return {"mi": LD(0), "delta": LD(0), "structural_zero": True,
                "negative_cell_corrected": False}
    LA, LB = np.linalg.cholesky(A), np.linalg.cholesky(B)
    # T=LA^{-1} C LB^{-T}; right solve is implemented by transposition.
    T = np.linalg.solve(LB, np.linalg.solve(LA, C).T).T.astype(LD)
    determinant_T = T[0, 0]*T[1, 1] - T[0, 1]*T[1, 0]
    s = np.sum(T*T) - determinant_T*determinant_T
    if not np.isfinite(s) or not 0 <= s < 1:
        raise ArithmeticError("Whitened determinant ratio is invalid")
    a, b = numpy_single_vacuum(covariance, first), numpy_single_vacuum(covariance, second)
    delta = a*b*np.expm1(-LD("0.5")*np.log1p(-s))
    product = np.array([a*b, (1-a)*b, a*(1-b), (1-a)*(1-b)], dtype=LD)
    signed_delta = np.array([1, -1, -1, 1], dtype=LD)*delta
    raw = product + signed_delta
    if np.min(raw) < -LD("2e-10") or abs(np.sum(raw)-1) > LD("2e-10"):
        raise ArithmeticError("Pair joint exceeds the existing numerical guard")
    corrected = bool(np.any(raw < 0))
    if corrected:
        joint = np.maximum(raw, LD(0))
        joint /= np.sum(joint)
        a, b = joint[0]+joint[2], joint[0]+joint[1]
        product = np.array([a*b, (1-a)*b, a*(1-b), (1-a)*(1-b)], dtype=LD)
        signed_delta = joint-product
        if np.sum(np.abs(joint-raw)) > LD("2e-9"):
            raise ArithmeticError("Pair correction exceeds existing guard")
    result = LD(0)
    for mass, difference in zip(product, signed_delta):
        if mass == 0:
            if difference != 0:
                raise ArithmeticError("Positive joint with zero product mass")
            continue
        x = difference/mass
        if x < -1:
            raise ArithmeticError("Invalid binary-KL ratio")
        if x == -1:
            h = LD(1)
        elif abs(x) <= LD("0.001"):
            h = sum(((-1)**n*x**n/LD(n*(n-1)) for n in range(2, 9)), LD(0))
        else:
            h = (1+x)*np.log1p(x)-x
        result += mass*h
    if not np.isfinite(result) or result < 0:
        raise ArithmeticError("Stable MI is invalid")
    return {"mi": result, "delta": delta, "structural_zero": False,
            "negative_cell_corrected": corrected, "s": s,
            "raw_joint": [str(value) for value in raw]}


def main():
    parser = ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.output is not None and arguments.output.exists():
        raise FileExistsError("Choose a fresh audit output path")
    manifest_path, fixtures_path = HERE/"manifest.json", HERE/"fixtures.npz"
    manifest = json.loads(manifest_path.read_text())
    rows = []
    with np.load(fixtures_path) as fixtures:
        for item in manifest["fixtures"]:
            V = fixtures[item["array_key"]]
            if sha256(np.ascontiguousarray(V).tobytes()).hexdigest() != item["covariance_sha256"]:
                raise ArithmeticError("Fixture covariance hash changed")
            first, second = item["pair"]
            tested = stable_formula(V, first, second)
            reference = decimal_reference(V, first, second)
            if item.get("analytic_reference") == "binary_entropy_half":
                expected = Decimal(2).ln()
                reference_kind = "analytic intended TMSV p=1/2; ln(2)"
            else:
                if reference["mi"] is None:
                    raise ArithmeticError("Stored float covariance does not define a nonnegative pair")
                expected = reference["mi"]
                reference_kind = "80-digit full determinant/marginal calculation"
            observed = Decimal(str(tested["mi"]))
            error = abs(observed-expected)
            relative = error/expected if expected else None
            passed = (relative is not None and relative < Decimal("1e-11")) or error < Decimal("1e-30")
            if not passed:
                raise ArithmeticError("Independent numerical comparison failed")
            rows.append({**item, "passed": passed, "reference_kind": reference_kind,
                "mi_reference_80_digits": str(expected), "mi_stable_formula": str(tested["mi"]),
                "absolute_error": str(error), "relative_error": str(relative),
                "delta_stable_formula": str(tested["delta"]),
                "delta_reference_80_digits": str(reference["delta"]),
                "negative_cell_corrected": tested["negative_cell_corrected"],
                "raw_joint_stable_formula": tested.get("raw_joint"),
                "raw_joint_reference_80_digits": [str(value) for value in reference["joint"]],
                "stored_float_covariance_joint_nonnegative":
                    reference["stored_float_covariance_joint_nonnegative"]})
    result = {"passed": all(row["passed"] for row in rows), "case_count": len(rows),
        "python": platform.python_version(), "numpy": np.__version__,
        "decimal_precision": getcontext().prec,
        "numpy_longdouble_mantissa_bits": int(np.finfo(LD).nmant),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "fixtures_sha256": sha256(fixtures_path.read_bytes()).hexdigest(),
        "qualification": "Independent formula audit, not validation of production code or an interval proof. "
            "The TMSV stored float covariance has a tiny boundary inconsistency; its analytic reference "
            "is the intended p=1/2 state, and the recorded correction is retained explicitly.",
        "cases": rows}
    if arguments.output is None:
        print(json.dumps(result, indent=2))
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({"passed": result["passed"], "cases": len(rows),
                          "output": str(arguments.output)}))


if __name__ == "__main__":
    main()
