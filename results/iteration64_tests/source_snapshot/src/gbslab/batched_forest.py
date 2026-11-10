"""Shared-Schur native events for the frozen conditional forest rule.

The C ABI batches all retained-subset decompositions, sparse Schur blocks and
local vacuum determinants.  Python and the native CLI call the SAME native
merged-log scorer and precision state machine.  A long-double numerical
failure may trigger one independent quad batch, with unchanged R/E/guard.
This refinement is arithmetic work within a conditional, never a sample retry.

Only the new module installs a canonical zero conditioner in the unchanged
Gaussian selector.  It reuses the frozen float64 rank-two formula in fixed
measurement order; this removes history-dependent choice of cached parents.
The selector rule is unchanged, but bitwise reproduction of every old cache
path is not claimed.  The inherited failure-symbol enumeration and sampling
interfaces retain their original semantics.  No target table is consulted.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .exact import NumericalProbabilityError
from .positive_completion import EventEstimate
from .positive_forest import ForestCompletionSampler
from .schur_memory import GaussianVacuumConditioner


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIBRARY = ROOT / ".cache/batched-forest-native/libbatched_forest.so"
PRECISIONS = {"longdouble": 0, "quad": 1, "auto": 2}
PRECISION_NAMES = ("longdouble", "quad")
COUNTERS = (
    "queries", "query_successes", "numeric_failures", "state_rebuilds", "state_reuses",
    "state_appends", "zero_updates", "zero_factorizations", "zero_solve_columns",
    "zero_schur_blocks", "subsets", "subset_factorizations", "subset_dimension_sum",
    "subset_cubic_work", "shared_solve_columns", "shared_solve_dot_terms",
    "sparse_schur_blocks", "sparse_schur_dot_terms", "local_determinants",
    "local_dimension_sum", "local_cubic_work", "event_accumulations", "tracked_add_sub",
    "tracked_multiply", "tracked_divide", "tracked_sqrt", "output_conversions",
    "analytic_zero_events", "maximum_retained", "maximum_omitted", "maximum_zeros",
    "state_invalidations",
)
TIMES = ("query_seconds", "state_seconds", "event_seconds", "output_seconds")
ZERO_REASONS = (None, "analytic_zero_baseline_branch", "analytic_zero_forest_node", "analytic_zero_forest_edge")
_LD_PTR = ct.POINTER(ct.c_longdouble)
_INT_PTR = ct.POINTER(ct.c_int)


class _Query(ct.Structure):
    _fields_ = [(name, _INT_PTR) for name in ("zeros", "retained", "omitted", "edges")] + [
        (name, ct.c_int) for name in ("mode", "zero_count", "retained_count", "omitted_count", "edge_count")]


class _Output(ct.Structure):
    _fields_ = [("values", _LD_PTR), ("indicators", _LD_PTR),
                ("exact_zero", ct.POINTER(ct.c_uint8)), ("capacity", ct.c_size_t)]


class _Conditional(ct.Structure):
    _fields_ = [("probability", ct.c_double), ("precision_used", ct.c_int),
                ("refinement_attempted", ct.c_int), ("zero_score_reason", ct.c_int * 2),
                ("log_scores", ct.c_longdouble * 2), ("log_indicators", ct.c_longdouble * 2),
                ("maximum_event_relative_indicator", ct.c_longdouble),
                ("maximum_log_score_indicator", ct.c_longdouble), ("first_error", ct.c_char * 512)]


class _Diagnostics(ct.Structure):
    _fields_ = [("counters", (ct.c_uint64 * len(COUNTERS)) * 2),
                ("seconds", (ct.c_double * len(TIMES)) * 2)] + [
        (name, ct.c_uint64) for name in ("probability_calls", "auto_refinements",
            "auto_refinement_successes", "auto_refinement_failures", "reset_requests", "original_input_bytes")
    ] + [(name, ct.c_uint64 * 2) for name in ("state_payload_bytes", "peak_state_payload_bytes",
                                             "peak_workspace_payload_bytes")] + [
        ("peak_kernel_payload_bytes", ct.c_uint64), ("setup_seconds", ct.c_double)]


class _Formats(ct.Structure):
    _fields_ = [(name, ct.c_int) for name in ("abi_version", "long_double_bytes",
        "long_double_mantissa_bits", "quad_bytes", "quad_mantissa_bits", "maximum_modes", "maximum_retained")
    ] + [(name, ct.c_size_t) for name in ("query_bytes", "output_bytes", "conditional_bytes", "diagnostics_bytes")]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _indices(values, modes, name):
    values = tuple(values)
    if any(not isinstance(j, (int, np.integer)) for j in values):
        raise ValueError(f"{name} must contain integer mode indices")
    values = tuple(int(j) for j in values)
    if len(set(values)) != len(values) or any(j < 0 or j >= modes for j in values):
        raise ValueError(f"{name} must contain distinct valid mode indices")
    return values


def _order(values, modes):
    result = tuple(range(modes)) if values is None else _indices(values, modes, "order")
    if len(result) != modes:
        raise ValueError("order must be a full permutation")
    return result


def _raise_status(status, error):
    if status == 0:
        return
    message = error.value.decode("utf-8", errors="replace") or f"native status {status}"
    if status == 2:
        raise NumericalProbabilityError(message)
    if status == 1:
        raise ValueError(message)
    if status == 3:
        raise MemoryError(message)
    raise RuntimeError(message)


class BatchedForestKernel:
    """Single-threaded native handle with two independent precision states.

    ``query`` returns raw joint event estimates on the common P(event)/P(Z)
    scale.  ``probability`` additionally applies the fixed forest guards and
    AUTO precision rule.  Standalone callers must supply a physical zero-mean
    covariance; finite/exact-symmetry and ABI checks are performed here.  A
    BatchedForestSampler uses its already validated ThresholdModel.V.
    """

    def __init__(self, covariance, *, order=None, library=None):
        started = perf_counter()
        self._handle = None
        self._closed_info = None
        raw = np.asarray(covariance)
        if np.iscomplexobj(raw) and np.any(np.imag(raw) != 0):
            raise ValueError("covariance must be real")
        V = np.array(np.real(raw), dtype=np.float64, order="C", copy=True)
        if (V.ndim != 2 or V.shape[0] == 0 or V.shape[0] != V.shape[1]
                or V.shape[0] % 2 or not np.all(np.isfinite(V))
                or not np.array_equal(V, V.T)):
            raise ValueError("covariance must be finite, even-dimensional and exactly symmetric")
        self.m = V.shape[0] // 2
        self.order = _order(order, self.m)
        self._positions = {mode: j for j, mode in enumerate(self.order)}
        self.input_covariance_sha256 = hashlib.sha256(V.tobytes()).hexdigest()
        self.library_path = Path(library or os.environ.get("GBSLAB_BATCHED_FOREST_LIBRARY", DEFAULT_LIBRARY)).resolve()
        if not self.library_path.is_file():
            raise FileNotFoundError(f"Native forest library not found: {self.library_path}; run experiments/build_batched_forest.py explicitly")
        self.library_sha256 = _sha(self.library_path)
        metadata_path = self.library_path.parent / "build.json"
        self.build_metadata_sha256 = _sha(metadata_path) if metadata_path.is_file() else None
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("library_sha256") != self.library_sha256:
                raise RuntimeError("Native forest library differs from its build metadata")
            for name, digest in metadata.get("source_hashes", {}).items():
                path = Path(name)
                if path.is_absolute() or ".." in path.parts or not (ROOT / path).is_file() or _sha(ROOT / path) != digest:
                    raise RuntimeError(f"Native forest build source binding differs: {name}")
        self._lib = ct.CDLL(str(self.library_path))
        self._lib.bf_formats.argtypes = [ct.POINTER(_Formats)]
        self._lib.bf_formats.restype = ct.c_int
        self._lib.bf_create.argtypes = [ct.c_int, ct.POINTER(ct.c_double), _INT_PTR, ct.c_char_p, ct.c_size_t]
        self._lib.bf_create.restype = ct.c_void_p
        self._lib.bf_destroy.argtypes = [ct.c_void_p]
        self._lib.bf_destroy.restype = None
        self._lib.bf_reset.argtypes = [ct.c_void_p]
        self._lib.bf_reset.restype = None
        self._lib.bf_diagnostics.argtypes = [ct.c_void_p, ct.POINTER(_Diagnostics)]
        self._lib.bf_diagnostics.restype = ct.c_int
        self._lib.bf_query.argtypes = [ct.c_void_p, ct.POINTER(_Query), ct.c_int, ct.c_int,
                                     ct.POINTER(_Output), ct.c_char_p, ct.c_size_t]
        self._lib.bf_query.restype = ct.c_int
        self._lib.bf_probability.argtypes = [ct.c_void_p, ct.POINTER(_Query), ct.c_int, ct.c_longdouble,
            ct.POINTER(_Output), ct.POINTER(_Conditional), ct.c_char_p, ct.c_size_t]
        self._lib.bf_probability.restype = ct.c_int
        formats = _Formats()
        if self._lib.bf_formats(ct.byref(formats)) != 0:
            raise RuntimeError("Native forest format discovery failed")
        self.formats = {name: int(getattr(formats, name)) for name, _ in formats._fields_}
        sizes = {"query_bytes": ct.sizeof(_Query), "output_bytes": ct.sizeof(_Output),
                 "conditional_bytes": ct.sizeof(_Conditional), "diagnostics_bytes": ct.sizeof(_Diagnostics)}
        if (formats.abi_version != 1 or formats.long_double_bytes != np.dtype(np.longdouble).itemsize
                or formats.long_double_bytes != ct.sizeof(ct.c_longdouble)
                or formats.long_double_mantissa_bits != np.finfo(np.longdouble).nmant + 1
                or any(self.formats[k] != v for k, v in sizes.items())):
            raise RuntimeError("Native forest ABI / NumPy long-double format mismatch")
        order_buffer = np.asarray(self.order, dtype=np.int32)
        error = ct.create_string_buffer(512)
        self._handle = self._lib.bf_create(self.m, V.ctypes.data_as(ct.POINTER(ct.c_double)),
            order_buffer.ctypes.data_as(_INT_PTR), error, len(error))
        if not self._handle:
            raise ValueError(error.value.decode("utf-8", errors="replace") or "native constructor failed")
        self.initialization_seconds = perf_counter() - started

    def _require_open(self):
        if not self._handle:
            raise RuntimeError("native forest handle is closed")

    def _request(self, zeros, retained, mode, omitted, edges):
        self._require_open()
        zeros = tuple(sorted(_indices(zeros, self.m, "zeros"), key=self._positions.__getitem__))
        retained = _indices(retained, self.m, "retained")
        omitted = _indices(omitted, self.m, "omitted")
        mode = _indices((mode,), self.m, "mode")[0]
        edges = tuple(tuple(edge) for edge in edges)
        if any(len(edge) != 2 for edge in edges):
            raise ValueError("edges must be physical-mode pairs")
        for edge in edges:
            _indices(edge, self.m, "edge")
        buffers = [np.asarray(values, dtype=np.int32) for values in
                   (zeros, retained, omitted, tuple(j for edge in edges for j in edge))]
        query = _Query(*(b.ctypes.data_as(_INT_PTR) for b in buffers), mode,
                       len(zeros), len(retained), len(omitted), len(edges))
        count = 1 + 2 * (1 + len(omitted) + len(edges))
        values, indicators = np.zeros(count, dtype=np.longdouble), np.zeros(count, dtype=np.longdouble)
        exact_zero = np.zeros(count, dtype=np.uint8)
        output = _Output(values.ctypes.data_as(_LD_PTR), indicators.ctypes.data_as(_LD_PTR),
                         exact_zero.ctypes.data_as(ct.POINTER(ct.c_uint8)), count)
        return buffers, query, output, values, indicators, exact_zero

    @staticmethod
    def _batch(values, indicators, exact_zero, f, e, precision):
        for array in (values, indicators, exact_zero):
            array.setflags(write=False)
        estimates = tuple(EventEstimate(value, indicator, bool(zero),
            f"native_{precision}_shared_schur_scaled_by_PZ")
            for value, indicator, zero in zip(values, indicators, exact_zero))
        group = 1 + f + e
        return {"values": values, "indicators": indicators, "exact_zero": exact_zero,
            "baseline": estimates[0], "branches": (estimates[1], estimates[1+group]),
            "nodes": tuple(tuple(estimates[2+a*group+j] for j in range(f)) for a in (0, 1)),
            "edges": tuple(tuple(estimates[2+a*group+f+j] for j in range(e)) for a in (0, 1)),
            "precision_used": precision, "event_mass_scale": "P(original joint event) / P(all history zeros)",
            "roundoff_indicators_are_rigorous_bounds": False}

    def query(self, zeros, retained, mode, omitted=(), edges=(), *, precision="longdouble", force_rebuild=False):
        if precision not in ("longdouble", "quad"):
            raise ValueError("raw query precision must be 'longdouble' or 'quad'")
        buffers, query, output, values, indicators, exact_zero = self._request(zeros, retained, mode, omitted, edges)
        error = ct.create_string_buffer(512)
        status = self._lib.bf_query(self._handle, ct.byref(query), PRECISIONS[precision], bool(force_rebuild),
                                   ct.byref(output), error, len(error))
        _raise_status(status, error)
        return self._batch(values, indicators, exact_zero, query.omitted_count, query.edge_count, precision)

    def probability(self, zeros, retained, mode, omitted=(), edges=(), *, precision="auto", max_relative_roundoff=1e-6):
        if precision not in PRECISIONS:
            raise ValueError("precision must be 'auto', 'longdouble' or 'quad'")
        if not np.isfinite(max_relative_roundoff) or not 0 < max_relative_roundoff < 1:
            raise ValueError("max_relative_roundoff must be finite and in (0,1)")
        buffers, query, output, values, indicators, exact_zero = self._request(zeros, retained, mode, omitted, edges)
        result, error = _Conditional(), ct.create_string_buffer(512)
        status = self._lib.bf_probability(self._handle, ct.byref(query), PRECISIONS[precision],
            ct.c_longdouble(max_relative_roundoff), ct.byref(output), ct.byref(result), error, len(error))
        _raise_status(status, error)
        actual_precision = PRECISION_NAMES[result.precision_used]
        batch = self._batch(values, indicators, exact_zero, query.omitted_count, query.edge_count, actual_precision)
        # Reading a ctypes scalar converts through Python float.  Copy the
        # arrays as bytes to preserve the C long-double diagnostics themselves.
        logs = np.frombuffer(result.log_scores, dtype=np.longdouble).copy()
        log_indicators = np.frombuffer(result.log_indicators, dtype=np.longdouble).copy()
        reasons = tuple(ZERO_REASONS[value] for value in result.zero_score_reason)
        batch.update(click_probability=float(result.probability), precision_requested=precision,
            refinement_attempted=bool(result.refinement_attempted), first_error=bytes(result.first_error).decode(),
            log_scores=tuple(None if reasons[j] is not None else logs[j] for j in (0, 1)),
            log_score_indicators=tuple(float(v) for v in log_indicators), zero_score_reasons=reasons,
            maximum_passed_event_relative_indicator=float(result.maximum_event_relative_indicator),
            maximum_passed_log_score_indicator=float(result.maximum_log_score_indicator))
        return batch

    def reset(self):
        self._require_open()
        self._lib.bf_reset(self._handle)

    def cache_info(self):
        if not self._handle:
            if self._closed_info is not None:
                return {**self._closed_info, "closed": True}
            raise RuntimeError("native forest handle is closed")
        raw = _Diagnostics()
        if self._lib.bf_diagnostics(self._handle, ct.byref(raw)) != 0:
            raise RuntimeError("native forest diagnostics failed")
        info = {name: int(getattr(raw, name)) for name in ("probability_calls", "auto_refinements",
            "auto_refinement_successes", "auto_refinement_failures", "reset_requests", "original_input_bytes",
            "peak_kernel_payload_bytes")}
        for precision, name in enumerate(PRECISION_NAMES):
            info[name] = {key: int(raw.counters[precision][j]) for j, key in enumerate(COUNTERS)}
            info[name].update({key: float(raw.seconds[precision][j]) for j, key in enumerate(TIMES)})
            for key in ("state_payload_bytes", "peak_state_payload_bytes", "peak_workspace_payload_bytes"):
                info[name][key] = int(getattr(raw, key)[precision])
        return {**info, "closed": False, "kind": "shared_schur_batched_forest_native",
            "library_path": str(self.library_path), "library_sha256": self.library_sha256,
            "build_metadata_sha256": self.build_metadata_sha256, "formats": self.formats,
            "input_covariance_sha256": self.input_covariance_sha256,
            "input_covariance_order": "binary64 xxpp, hbar=2, zero mean",
            "setup_seconds": float(raw.setup_seconds), "python_glue_initialization_seconds": self.initialization_seconds,
            "indicator_policy": "absolute scalar propagation, mu=8*format epsilon; quad-to-LD conversion included",
            "roundoff_indicators_are_rigorous_bounds": False,
            "counter_semantics": "query_successes means raw event batch completed, including subsequent score failure; numeric_failures may include that score guard",
            "scalar_counter_scope": "tracked value/indicator operations; not total machine floating-point instruction counts",
            "memory_scope": "tracked major native numerical containers only; excludes query bookkeeping, scalar stack temporaries, allocator overhead, caller buffers and Python selector/model/caches; not RSS or a complete allocator peak",
            "state_policy": "at most one current canonical Z state in each precision; no subset-factorization cache",
            "auto_policy": "one LD batch, at most one whole-prefix quad refinement on numerical failure; fixed R/E; no samples redrawn"}

    def close(self):
        if self._handle:
            self._closed_info = self.cache_info()
            self._lib.bf_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *unused):
        self.close()

    def __del__(self):
        if getattr(self, "_handle", None):
            self._lib.bf_destroy(self._handle)
            self._handle = None


class CanonicalVacuumConditioner:
    """Reuse the frozen float64 rank-two update in a canonical zero order.

    The original covariance and one current conditional state are retained.
    Append only if the old canonical sequence is an exact prefix of the new
    sequence; otherwise replay from base.  This adapter intentionally does
    not use the old conditioner's arbitrary-subset-parent LRU selection.
    """

    def __init__(self, covariance, order=None):
        self._backend = GaussianVacuumConditioner(covariance, cache_size=0, max_cache_bytes=0)
        self.m, self.V = self._backend.m, self._backend.V
        self.order = _order(order, self.m)
        self._positions = {mode: j for j, mode in enumerate(self.order)}
        self._state = self._backend._base
        self._zeros = ()
        self.diagnostics = {"requests": 0, "state_reuses": 0, "state_appends": 0,
                            "state_replays": 0, "reset_requests": 0, "state_failures": 0}

    def reset(self):
        self._state, self._zeros = self._backend._base, ()
        self.diagnostics["reset_requests"] += 1

    def conditioned(self, zeros):
        requested = tuple(sorted(_indices(zeros, self.m, "zeros"), key=self._positions.__getitem__))
        self.diagnostics["requests"] += 1
        if self._zeros == requested:
            self.diagnostics["state_reuses"] += 1
        elif self._zeros == requested[:len(self._zeros)]:
            self.diagnostics["state_appends"] += 1
        else:
            self._state, self._zeros = self._backend._base, ()
            self.diagnostics["state_replays"] += 1
        try:
            while len(self._zeros) < len(requested):
                mode = requested[len(self._zeros)]
                self._state = self._backend._from_parent(self._state, mode)
                self._zeros += (mode,)
        except Exception:
            self._state, self._zeros = self._backend._base, ()
            self.diagnostics["state_failures"] += 1
            raise
        return self._state.modes, self._state.covariance

    def cache_info(self):
        return {**self.diagnostics, "kind": "canonical_float64_rank_two_selector_conditioner",
            "fixed_zero_order": self.order, "current_zeros": self._zeros,
            "payload_bytes": self._backend._base.payload_bytes + (
                0 if self._state is self._backend._base else self._state.payload_bytes),
            "rank_two_backend": self._backend.cache_info(), "additional_subset_states": 0,
            "finite_arithmetic_path": "fixed measurement-order replay/append; not bitwise old arbitrary cached-parent paths"}


def install_canonical_selector(sampler):
    """Install the same selector adapter on a fresh independent table reference.

    Example: create ForestCompletionSampler(..., event_oracle=TableEventOracle(P))
    and call this function before the first conditional.  Its forest formula,
    independent event oracle, selected-click rule and static MI stay untouched.
    """
    if sampler.diagnostics.get("conditional_evaluations", 0) or getattr(sampler._selector, "_selection_cache", {}):
        raise ValueError("canonical selector must be installed before evaluating conditionals")
    if not hasattr(sampler._selector, "conditioner"):
        return False
    sampler._selector.conditioner = CanonicalVacuumConditioner(sampler.model.V, sampler.order)
    return True


class BatchedForestSampler(ForestCompletionSampler):
    """Native batched event/score backend with the existing forest interfaces.

    Precision is 'auto' (default), 'longdouble', or 'quad'.  The selector,
    tree and its setup stay in Python; the separate native CLI is needed for
    an end-to-end native throughput comparison.  Each sample begins with
    invalidated native zero states; conditional caches remain bounded.
    """

    def __init__(self, model, max_clicks, *, forest="chain", selection="click_influence",
                 order=None, cache_size=8192, max_relative_roundoff=1e-6,
                 precision="auto", library=None):
        started = perf_counter()
        if precision not in PRECISIONS:
            raise ValueError("precision must be 'auto', 'longdouble' or 'quad'")
        if not hasattr(model, "V"):
            raise ValueError("native batched events require the original Gaussian covariance")
        super().__init__(model, max_clicks, forest=forest, selection=selection, order=order,
                         cache_size=cache_size, max_relative_roundoff=max_relative_roundoff)
        self.precision = precision
        install_canonical_selector(self)
        self.kernel = BatchedForestKernel(model.V, order=self.order, library=library)
        if self.max_clicks > self.kernel.formats["maximum_retained"]:
            self.kernel.close()
            raise ValueError("retained budget exceeds the native resource cap")
        self._event_backend = self.kernel
        self.oracle_kind = "native_shared_schur_batch_with_common_PZ_scale"
        self.diagnostics["initialization_seconds"] = perf_counter() - started

    def _conditional_record(self, prefix):
        prefix = tuple(prefix)
        if not prefix:
            self.kernel.reset()
            if isinstance(getattr(self._selector, "conditioner", None), CanonicalVacuumConditioner):
                self._selector.conditioner.reset()
        return super()._conditional_record(prefix)

    def _compute_conditional(self, prefix):
        mode = self.order[len(prefix)]
        zeros = tuple(self.order[j] for j, bit in enumerate(prefix) if not bit)
        clicks = tuple(self.order[j] for j, bit in enumerate(prefix) if bit)
        started = perf_counter()
        self.diagnostics["selector_calls"] += 1
        try:
            retained = self._selector._retained_clicks(clicks, mode)
        finally:
            self.diagnostics["selector_seconds"] += perf_counter() - started
        selected = set(retained)
        omitted = tuple(j for j in clicks if j not in selected)
        edges = self._choose_edges(omitted)
        degrees = {j: 0 for j in omitted}
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
        width = len(retained) + 1 + min(2, len(omitted))
        for name, value in (("forest_maximum_local_event_width", width),
                            ("forest_maximum_event_ones", width),
                            ("forest_maximum_event_constraints", len(zeros) + width),
                            ("maximum_factor_count", len(omitted) + len(edges))):
            self.diagnostics[name] = max(self.diagnostics[name], value)
        self.diagnostics["forest_singleton_requests"] += 2 * len(omitted)
        self.diagnostics["forest_edge_event_requests"] += 2 * len(edges)
        self.diagnostics["event_requests"] += 1 + 2 * (1 + len(omitted) + len(edges))
        self.diagnostics["event_oracle_calls"] += 1
        started = perf_counter()
        try:
            batch = self.kernel.probability(zeros, retained, mode, omitted, edges,
                precision=self.precision, max_relative_roundoff=self.max_relative_roundoff)
        finally:
            self.diagnostics["event_oracle_seconds"] += perf_counter() - started
        baseline, branches = batch["baseline"], batch["branches"]
        self.diagnostics["maximum_event_relative_indicator"] = max(
            self.diagnostics["maximum_event_relative_indicator"], batch["maximum_passed_event_relative_indicator"])
        self.diagnostics["maximum_log_score_indicator"] = max(
            self.diagnostics["maximum_log_score_indicator"], batch["maximum_passed_log_score_indicator"])
        self.diagnostics["event_exact_zero_returns"] += int(np.count_nonzero(batch["exact_zero"]))
        self.diagnostics["analytic_zero_score_branches"] += sum(reason is not None for reason in batch["zero_score_reasons"])
        node_probabilities, edge_probabilities = [[], []], [[], []]
        for outcome, branch in enumerate(branches):
            if branch.value == 0:
                continue
            for node, estimate in zip(omitted, batch["nodes"][outcome]):
                node_probabilities[outcome].append((node, float(estimate.value / branch.value)))
                if estimate.value == 0:
                    break
            if batch["zero_score_reasons"][outcome] == "analytic_zero_forest_node":
                continue
            for edge, estimate in zip(edges, batch["edges"][outcome]):
                edge_probabilities[outcome].append((edge, float(estimate.value / branch.value)))
                if estimate.value == 0:
                    break
        return {"prefix": prefix, "mode": mode, "zeros": zeros, "clicks": clicks,
            "retained_clicks": retained, "omitted_clicks": omitted, "forest": self.forest,
            "forest_edges": edges, "node_degrees": degrees,
            "branch_log_coefficient": 1-len(omitted)+len(edges),
            "node_log_coefficients": {j: 1-degrees[j] for j in omitted},
            "baseline_probability": float(baseline.value),
            "branch_masses": tuple(float(e.value) for e in branches),
            "baseline_probabilities": tuple(float(e.value/baseline.value) for e in branches),
            "node_probabilities": tuple(tuple(v) for v in node_probabilities),
            "edge_probabilities": tuple(tuple(v) for v in edge_probabilities),
            "log_scores_up_to_common_baseline": tuple(None if v is None else float(v) for v in batch["log_scores"]),
            "log_score_indicators": batch["log_score_indicators"],
            "zero_score_reasons": batch["zero_score_reasons"], "click_probability": batch["click_probability"],
            "log_score_common_factor_omitted": "P(history zeros)/P(B); all native events are divided by P(history zeros)",
            "event_mass_scale": batch["event_mass_scale"],
            "precision_requested": self.precision, "precision_used": batch["precision_used"],
            "refinement_attempted": batch["refinement_attempted"], "first_error": batch["first_error"]}

    def cache_info(self):
        info = super().cache_info()
        info.update(precision=self.precision,
            event_evaluator_scope="shared retained-subset Schur batches; <=3-mode local determinants; common PZ factor removed",
            scoring_backend="same bf_probability C ABI as end-to-end native CLI",
            timing_scope="Python selector/tree plus native events/scoring; measure the full call, not kernel-only time",
            numerical_policy="mu=8*format epsilon propagated at each operation; fixed relative guard; at most one quad refinement",
            maximum_indicator_scope="maxima over events/scores passing their guards; failures separately counted by native backend",
            selector_covariance_policy="canonical fixed-order float64 rank-two replay/append, using frozen formula",
            event_request_counter_scope="requested mass outputs per Python conditional; native backend counts all refinement work")
        return info

    def close(self):
        self.kernel.close()

    def __enter__(self):
        self.kernel._require_open()
        return self

    def __exit__(self, *unused):
        self.close()
