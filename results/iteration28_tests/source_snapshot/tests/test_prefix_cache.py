"""Correctness gates for batch-local conditional-prefix reuse.

These tests never compile. Build .cache/click-prefix-native explicitly, or set
GBSLAB_PREFIX_EXECUTABLE to the already-built executable being verified.
Exhaustive injected trajectories exercise cache hits at every small-system
prefix; comparing the unchanged enumeration alone would not establish that.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from gbslab.cpp_click import (
    CppClickError, MAX_PREFIX_CACHE_ENTRIES, _check_prefix_cache,
    find_cpp_executable, run_cpp_click,
)
from gbslab.gaussian import covariance_from_interferometer, haar_unitary, local_unitary
from gbslab.paired_stress import paired_covariance


ROOT = Path(__file__).resolve().parents[1]
BINARY = find_cpp_executable(os.environ.get(
    "GBSLAB_PREFIX_EXECUTABLE", str(ROOT / ".cache/click-prefix-native/click_sampler")))
ARCHIVED = find_cpp_executable(ROOT / ".cache/click-iteration4-native/click_sampler")
VALIDATION_CHECKS = []


def record(name, **details):
    VALIDATION_CHECKS.append(dict(name=name, passed=True, details=details))


def haar_covariance(modes=6):
    return covariance_from_interferometer(
        haar_unitary(modes, np.random.default_rng(551001+modes)),
        np.linspace(.65, .95, modes), np.linspace(.45, .8, modes))


def every_trajectory(modes, order):
    bits = (np.arange(1 << modes)[:, None] >> np.arange(modes)) & 1
    uniforms = np.where(bits[:, order], 0.0, np.nextafter(1.0, 0.0))
    # Every node is revisited after the full first traversal populated it.
    return np.concatenate((uniforms, uniforms[::-1])), np.concatenate((bits, bits[::-1]))


class PrefixCacheWrapperTests(unittest.TestCase):
    def test_invalid_cache_budgets_never_start_process(self):
        for budget in (-1, True, 1.5, MAX_PREFIX_CACHE_ENTRIES+1):
            with self.subTest(budget=budget), patch("gbslab.cpp_click.subprocess.run") as run:
                with self.assertRaisesRegex(ValueError, "prefix_cache_entries"):
                    run_cpp_click(np.eye(4), prefix_cache_entries=budget)
                run.assert_not_called()
        record("invalid_cache_budget_rejected_before_process")

    def test_missing_forged_and_unrequested_metadata_are_rejected(self):
        _check_prefix_cache(None, 0, 3, 2)  # Archived binaries remain usable.
        with self.assertRaisesRegex(CppClickError, "missing"):
            _check_prefix_cache(None, 16, 3, 2)
        correct = dict(enabled=True, requested_max_entries=16,
            hard_max_entries=1048576, hard_payload_byte_cap=67108864,
            entry_bytes=48, capacity_entries=16, capacity_bytes=768,
            entries_used=3, hits=3, misses=3, capacity_skips=0,
            state_reconstructions=2, replayed_zero_updates=1,
            samples_without_state=1, enumeration_uses_cache=False)
        _check_prefix_cache(correct, 16, 3, 2)
        for change in (dict(enabled=False), dict(enumeration_uses_cache=True),
                       dict(requested_max_entries=8), dict(entry_bytes=65),
                       dict(capacity_bytes=767), dict(entries_used=17),
                       dict(hits=True), dict(hits=4), dict(capacity_skips=1),
                       dict(state_reconstructions=3), dict(replayed_zero_updates=5)):
            with self.subTest(change=change), self.assertRaises(CppClickError):
                _check_prefix_cache({**correct, **change}, 16, 3, 2)
        with self.assertRaises(CppClickError):
            _check_prefix_cache(correct, 0, 3, 2)
        record("prefix_cache_metadata_fail_closed")


@unittest.skipIf(BINARY is None, "build .cache/click-prefix-native explicitly first")
class PrefixCacheBinaryTests(unittest.TestCase):
    def compare(self, name, covariance, *, uniforms=None, samples=None,
                budget=65536, enumerate_law=True, **kwargs):
        modes = len(covariance)//2
        number = len(uniforms) if uniforms is not None else samples
        common = dict(samples=number, uniforms=uniforms,
            executable=BINARY, raw_samples=True, trace_conditionals=True,
            record_log_probabilities=True, enumerate_distribution=enumerate_law,
            **kwargs)
        baseline = run_cpp_click(covariance, prefix_cache_entries=0, **common)
        cached = run_cpp_click(covariance, prefix_cache_entries=budget, **common)
        for key in ("raw_samples", "sample_log_probabilities", "conditional_trace",
                    "distribution", "histogram_counts", "click_count_histogram",
                    "mode_click_counts"):
            self.assertEqual(cached[key], baseline[key], (name, key))
        meta = cached["prefix_cache"]
        self.assertGreater(meta["hits"], 0)
        self.assertEqual(meta["hits"]+meta["misses"], number*modes)
        self.assertEqual(cached["sampling_diagnostics"]["conditional_evaluations"], meta["misses"])
        self.assertEqual(baseline["sampling_diagnostics"]["conditional_evaluations"], number*modes)
        self.assertLessEqual(meta["entries_used"], budget)
        self.assertLessEqual(meta["capacity_bytes"], 64*1024*1024)
        self.assertIs(meta["enumeration_uses_cache"], False)
        record(name, modes=modes, samples=number, prefix_cache_entries=budget,
            selector=kwargs.get("selection", "pairwise"), k=kwargs.get("max_clicks", 2),
            vacuum_kernel=kwargs.get("vacuum_kernel", "log"),
            explicit_uniforms=uniforms is not None,
            samples_bitwise_equal=True, conditionals_exactly_equal=True,
            retained_sets_exactly_equal=True, log_q_exactly_equal=True,
            full_q_exactly_equal=True if enumerate_law else None,
            maximum_absolute_conditional_error=0.0, maximum_absolute_log_q_error=0.0,
            prefix_cache=meta)
        return baseline, cached

    def test_all_small_prefixes_both_selectors_and_k_zero(self):
        covariance = haar_covariance()
        order = (4, 0, 5, 2, 1, 3)
        uniforms, expected = every_trajectory(6, order)
        for selector in ("pairwise", "click_influence"):
            for k in (0, 2):
                with self.subTest(selector=selector, k=k):
                    _, cached = self.compare(f"exhaustive_prefixes_{selector}_k{k}", covariance,
                        uniforms=uniforms, order=order, selection=selector, max_clicks=k,
                        vacuum_kernel="product")
                    np.testing.assert_array_equal(cached["raw_samples"], expected)
                    self.assertEqual(cached["prefix_cache"]["entries_used"], (1 << 6)-1)
                    self.assertGreaterEqual(cached["prefix_cache"]["samples_without_state"], 1 << 6)

    def test_capacity_saturation_retains_the_exact_law(self):
        covariance = haar_covariance()
        order = tuple(range(6))
        uniforms, _ = every_trajectory(6, order)
        for budget in (1, 3, 17):
            with self.subTest(budget=budget):
                _, cached = self.compare(f"saturated_capacity_{budget}", covariance,
                    uniforms=uniforms, budget=budget, selection="click_influence",
                    max_clicks=2, vacuum_kernel="product")
                self.assertEqual(cached["prefix_cache"]["entries_used"], budget)
                self.assertGreater(cached["prefix_cache"]["capacity_skips"], 0)
                self.assertGreater(cached["prefix_cache"]["replayed_zero_updates"], 0)

    def test_log_and_product_kernels_and_sparse_optics(self):
        covariance = covariance_from_interferometer(
            local_unitary(8, 2, np.random.default_rng(551101)), .8, .6)
        uniforms = np.random.default_rng(551102).random((48, 8))
        uniforms[24:] = uniforms[:24]
        uniforms[0] = uniforms[24] = 0.0
        for kernel in ("log", "product"):
            with self.subTest(kernel=kernel):
                self.compare(f"sparse_optics_{kernel}", covariance, uniforms=uniforms,
                    selection="click_influence", max_clicks=4, vacuum_kernel=kernel)

    def test_cache_preserves_mt19937_uniform_consumption(self):
        self.compare("internal_prng_same_stream", haar_covariance(), samples=128,
            selection="click_influence", max_clicks=2, vacuum_kernel="product",
            seed=(1 << 60)+551103)

    def test_single_mode_and_maximum_retained_vector(self):
        self.compare("single_mode_cached_root", np.eye(2)*2.0,
            uniforms=np.array([[0.0], [.9], [0.0], [.9]]), max_clicks=0,
            vacuum_kernel="product")
        _, cached = self.compare("maximum_eight_retained_clicks", np.eye(20)*3.0,
            uniforms=np.zeros((2, 10)), selection="pairwise", max_clicks=8,
            order=tuple(reversed(range(10))), enumerate_law=False,
            vacuum_kernel="product")
        self.assertEqual(max(len(step["retained"]) for step in cached["conditional_trace"]), 8)

    def test_tmsv_support_and_nontrivial_orders(self):
        covariance = paired_covariance(2, .5)
        uniforms = np.random.default_rng(551104).random((32, 4))
        uniforms[0] = [np.nextafter(1.0, 0.0)]*2+[0.0]*2
        uniforms[1] = [0.0]*2+[np.nextafter(1.0, 0.0)]*2
        uniforms[16:] = uniforms[:16]
        for order in ((0, 1, 2, 3), (0, 2, 1, 3), (3, 1, 2, 0)):
            with self.subTest(order=order):
                _, cached = self.compare(f"tmsv_support_order_{''.join(map(str, order))}", covariance,
                    uniforms=uniforms, order=order, enumerate_law=False,
                    selection="click_influence", max_clicks=2, vacuum_kernel="product")
                values = np.asarray(cached["raw_samples"])
                np.testing.assert_array_equal(values[:, :2], values[:, 2:])

    def test_physical_bits_and_cache_ids_at_64_100_256_modes(self):
        for modes in (64, 100, 256):
            with self.subTest(modes=modes):
                covariance = np.eye(2*modes)*1.6
                order = tuple(reversed(range(modes)))
                bits = np.zeros((4, modes), dtype=int)
                bits[:, modes-1] = 1
                bits[:, min(63, modes-1)] = 1
                bits[1::2, modes//2] = 1
                uniforms = np.where(bits[:, order], 0.0, np.nextafter(1.0, 0.0))
                _, cached = self.compare(f"physical_bits_{modes}", covariance,
                    uniforms=uniforms, order=order, enumerate_law=False,
                    selection="pairwise", max_clicks=2, vacuum_kernel="product")
                np.testing.assert_array_equal(cached["raw_samples"], bits)
                self.assertEqual(cached["prefix_cache"]["samples_without_state"], 2)
                self.assertTrue(any(modes-1 in step["retained"]
                                    for step in cached["conditional_trace"]))

    def test_enumeration_neither_populates_nor_reads_prefix_cache(self):
        covariance = haar_covariance(5)
        options = dict(samples=0, enumerate_distribution=True, executable=BINARY,
                       selection="click_influence", max_clicks=2, vacuum_kernel="product")
        plain = run_cpp_click(covariance, **options)
        cached = run_cpp_click(covariance, prefix_cache_entries=65536, **options)
        self.assertEqual(plain["distribution"], cached["distribution"])
        for key in ("entries_used", "capacity_bytes", "hits", "misses", "state_reconstructions"):
            self.assertEqual(cached["prefix_cache"][key], 0)
        self.assertGreater(cached["enumeration_diagnostics"]["conditional_evaluations"], 0)
        record("enumeration_independent_and_zero_sample_no_allocation", full_q_exactly_equal=True)

    def test_warmed_cache_does_not_hide_numerical_failure(self):
        # First draw warms an all-zero path. The second shares its first zero,
        # then forces a rare click; replay must preserve the unresolved-event
        # guard, its sample number and original error context, with no redraw.
        covariance = np.eye(10)*(1.0+2e-12)
        uniforms = np.full((2, 5), np.nextafter(1.0, 0.0))
        uniforms[1, 1] = 0.0
        errors = []
        for budget in (0, 65536):
            with self.assertRaises(CppClickError) as caught:
                run_cpp_click(covariance, samples=2, uniforms=uniforms,
                    selection="click_influence", max_clicks=2, vacuum_kernel="product",
                    prefix_cache_entries=budget, record_log_probabilities=True,
                    executable=BINARY)
            errors.append(caught.exception)
        before, after = (error.payload for error in errors)
        for key in ("error", "error_type", "context", "completed_samples"):
            self.assertEqual(before[key], after[key], key)
        self.assertEqual(before["error_type"], "numerical_probability")
        self.assertEqual(before["completed_samples"], 1)
        self.assertEqual(before["context"]["sample_index"], 1)
        self.assertGreater(after["prefix_cache"]["hits"], 0)
        self.assertGreater(after["prefix_cache"]["replayed_zero_updates"], 0)
        self.assertEqual(errors[0].returncode, errors[1].returncode)
        record("warmed_cache_guard_and_failure_symbol_preserved", failure_type=after["error_type"],
            completed_samples=after["completed_samples"], context=after["context"],
            error=after["error"], returncode=errors[1].returncode,
            prefix_cache=after["prefix_cache"])

    def test_default_off_matches_archived_iteration4_binary(self):
        self.assertIsNotNone(ARCHIVED, "archived iteration4 binary is required by this gate")
        covariance = haar_covariance()
        uniforms = np.random.default_rng(551105).random((24, 6))
        options = dict(samples=24, uniforms=uniforms, selection="click_influence",
            max_clicks=2, vacuum_kernel="product", enumerate_distribution=True,
            trace_conditionals=True, record_log_probabilities=True)
        archived = run_cpp_click(covariance, executable=ARCHIVED, **options)
        current = run_cpp_click(covariance, executable=BINARY, **options)
        for key in ("raw_samples", "conditional_trace", "sample_log_probabilities", "distribution"):
            self.assertEqual(archived[key], current[key], key)
        self.assertIs(current["prefix_cache"]["enabled"], False)
        self.assertEqual(current["prefix_cache"]["capacity_bytes"], 0)
        record("default_off_matches_archived_iteration4",
            archived_executable_sha256=hashlib.sha256(ARCHIVED.read_bytes()).hexdigest(),
            samples_bitwise_equal=True, conditionals_exactly_equal=True,
            log_q_exactly_equal=True, full_q_exactly_equal=True)

    def test_native_entry_ceiling_rejects_before_input_or_allocation(self):
        completed = subprocess.run([str(BINARY), "--input", "unused.csv", "--modes", "2",
            "--prefix-cache-entries", str(MAX_PREFIX_CACHE_ENTRIES+1)],
            capture_output=True, text=True, check=False)
        self.assertNotEqual(completed.returncode, 0)
        result = json.loads(completed.stdout)
        self.assertEqual(result["error_type"], "invalid_argument")
        self.assertEqual(result["context"]["phase"], "arguments")
        record("native_cache_ceiling_rejected_before_allocation")


if __name__ == "__main__":
    unittest.main()
