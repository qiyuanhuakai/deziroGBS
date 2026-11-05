"""Analytic/resource/transport gates for the capped coherent-core prototype.

No test compiles a binary.  Independent amplitude-pair/averaged-Gaussian
references live in test_coherent_core_independent.py, owned by a separate audit.
"""
from copy import deepcopy
import hashlib
from math import comb
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from gbslab.coherent_core import (
    CoherentCoreError, coherent_core_fixed_law, coherent_core_resources,
    coherent_core_state, coherent_measure_channels, run_coherent_core,
)


ROOT = Path(__file__).resolve().parents[1]


class CoherentCoreResourceTests(unittest.TestCase):
    def test_full_basis_and_all_stage_preflight_at_the_declared_boundary(self):
        small = coherent_core_resources(32, 4)
        self.assertTrue(small["feasible"])
        self.assertEqual(small["full_basis"], comb(36, 4))
        self.assertEqual(small["full_basis"], 58905)
        self.assertEqual(small["even_initial_basis"], 52889)
        self.assertEqual(small["all_stage_basis"], sum(comb(m + 4, 4) for m in range(33)))
        self.assertGreater(small["full_basis"], small["even_initial_basis"])
        rejected = coherent_core_resources(64, 6)
        self.assertFalse(rejected["feasible"])
        self.assertIn("full_basis", rejected["rejection_reasons"])
        self.assertIn("all_stage_basis", rejected["rejection_reasons"])
        self.assertIn("fiber_index_capacity", rejected["rejection_reasons"])
        # A capacity is not permission to silently reduce K or omit odd states.
        limited = coherent_core_resources(32, 4, max_stage_basis=400000)
        self.assertFalse(limited["feasible"])
        self.assertEqual(limited["full_basis"], small["full_basis"])
        self.assertEqual(limited["cutoff"], 4)

    def test_size_rejection_precedes_subprocess_and_missing_binary_lookup(self):
        with mock.patch("gbslab.coherent_core.subprocess.run") as invoked:
            with self.assertRaisesRegex(ValueError, "resource cap"):
                run_coherent_core(np.eye(64), .6, .35, 6, samples=1,
                                  executable="/nonexistent/coherent-core")
            invoked.assert_not_called()

    def test_missing_binary_does_not_compile_or_invoke_a_process(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("gbslab.coherent_core.subprocess.run") as invoked:
                with self.assertRaisesRegex(FileNotFoundError, "explicitly"):
                    coherent_core_state(np.eye(1), .3, .5, 2,
                                        executable=Path(directory) / "absent")
                invoked.assert_not_called()

    def test_invalid_parameters_are_rejected_before_native_work(self):
        for arguments in ((0, 2), (65, 2), (2, -1), (2, 9), (True, 2), (2, 2.0)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                coherent_core_resources(*arguments)
        with mock.patch("gbslab.coherent_core.subprocess.run") as invoked:
            for kwargs in ({"samples": -1}, {"seed": -1}, {"samples": True},
                           {"numeric_guard": 0}, {"max_enumeration_nodes": 0}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    run_coherent_core(np.eye(2), .3, .5, 2, **kwargs)
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                run_coherent_core(np.eye(2), .3, .5, 2, samples=1,
                                  fixed_alpha=[0, 0], normals=np.zeros((1, 2)))
            invoked.assert_not_called()


class CoherentCoreNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        explicit = os.environ.get("GBSLAB_COHERENT_CORE_EXECUTABLE")
        cls.binary = Path(explicit) if explicit else ROOT / ".cache/coherent-core-native/coherent_core_sampler"
        if not cls.binary.is_file():
            if explicit:
                raise FileNotFoundError(cls.binary)
            raise unittest.SkipTest("Build the coherent-core native executable explicitly")
        cls.digest = hashlib.sha256(cls.binary.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        if hashlib.sha256(cls.binary.read_bytes()).hexdigest() != cls.digest:
            raise AssertionError("Coherent-core executable changed during tests")

    def test_two_mode_core_has_geometric_pair_tail_and_coherent_parity(self):
        # Two identically squeezed modes: total pair count is geometric.
        # tau_K=(tanh(s)^2)^(floor(K/2)+1), independent of passive U.
        s = np.arctanh(.5)
        for K in (0, 1, 2, 4, 8):
            with self.subTest(cutoff=K):
                result = coherent_core_state(np.eye(2), s, 1, K, executable=self.binary)
                expected_tail = np.longdouble(.25) ** (K // 2 + 1)
                self.assertAlmostEqual(float(result["tail_probability"]), float(expected_tail), delta=3e-16)
                self.assertAlmostEqual(float(result["retained_weight"]), float(1 - expected_tail), delta=4e-16)
                self.assertAlmostEqual(float(result["trace_distance_bound"]), float(np.sqrt(expected_tail)), delta=4e-16)
                self.assertEqual(result["occupations"].shape, (comb(2 + K, K), 2))
                odd = result["occupations"].sum(axis=1) % 2 == 1
                np.testing.assert_array_equal(result["amplitudes"][odd], np.zeros(odd.sum()))
                # Coherence is exported as a single normalized amplitude vector.
                norm = np.sum(np.abs(result["amplitudes"]) ** 2, dtype=np.longdouble)
                self.assertLess(abs(norm - 1), np.longdouble("2e-16"))
                self.assertEqual(result["parameters"]["displacement_variance"], 0)

    def test_exact_vacuum_and_zero_noise_boundaries(self):
        for m, r, eta, K in ((3, .6, 0, 4), (3, 0, .7, 4), (2, .6, 1, 0)):
            with self.subTest(m=m, r=r, eta=eta, K=K):
                result = run_coherent_core(np.eye(m), r, eta, K, samples=3,
                    seed=91, trace=True, executable=self.binary)
                np.testing.assert_array_equal(result["raw_samples"], np.zeros((3, m), dtype=np.uint8))
                self.assertEqual(result["parameters"]["displacement_variance"], 0)
                self.assertEqual(result["diagnostics"]["normal_variates"], 0)
                self.assertEqual(result["diagnostics"]["gaussian_uniforms"], 0)
                self.assertEqual(result["diagnostics"]["quantum_uniforms"], 6 * m)
                self.assertTrue(all(step["channel"] == 0 and step["q"] == 0 for step in result["trace"]))
                if r == 0 or eta == 0:
                    self.assertEqual(result["tail_probability"], 0)
                else:
                    self.assertGreater(result["tail_probability"], 0)

    def test_cutoff_zero_uses_the_full_displaced_vacuum_povm(self):
        alpha = np.array([0, .2 + .1j, 2j])
        split = np.array([[0., 0., 0.], [.5, .5, .5], [.9, .01, .01], [.1, .99, .99]])
        uniforms = np.empty((4, 6)); uniforms[:, ::2] = .37; uniforms[:, 1::2] = split
        result = run_coherent_core(np.eye(3), .6, .35, 0, samples=4,
            fixed_alpha=alpha, uniforms=uniforms, trace=True, executable=self.binary)
        t = np.exp(-np.abs(alpha) ** 2)
        q = -np.expm1(-np.abs(alpha) ** 2)
        expected = np.zeros((4, 3), dtype=np.uint8)
        for sample in range(4):
            for j in range(3):
                expected[sample, j] = (split[sample, j] < q[j] if q[j] <= t[j]
                                       else not split[sample, j] < t[j])
        np.testing.assert_array_equal(result["raw_samples"], expected)
        self.assertTrue(all(step["channel"] == 0 for step in result["trace"]))
        self.assertEqual(result["diagnostics"]["normal_variates"], 0)
        full = coherent_core_fixed_law(np.eye(3), .6, .35, 0, alpha,
                                       method="refined", executable=self.binary)
        expected_q = [np.prod([q[j] if mask >> j & 1 else t[j] for j in range(3)]) for mask in range(8)]
        np.testing.assert_allclose(full["distribution"], expected_q, atol=2e-16, rtol=2e-15)
        self.assertEqual(full["failure_mass"], 0)

    def test_finite_gaussian_random_stream_is_reproducible_and_counted(self):
        labels = np.arange(3)
        U = np.exp(2j * np.pi * labels[:, None] * labels[None, :] / 3) / np.sqrt(3)
        first = run_coherent_core(U, .6, .35, 2, samples=8, seed=1927,
                                 trace=True, executable=self.binary)
        second = run_coherent_core(U, .6, .35, 2, samples=8, seed=1927,
                                  trace=True, executable=self.binary)
        np.testing.assert_array_equal(first["raw_samples"], second["raw_samples"])
        for a, b in zip(first["trace"], second["trace"]):
            for key in ("sample", "mode", "prefix_mask", "channel", "bit", "alpha", "uniform_channel", "uniform_split"):
                self.assertEqual(a[key], b[key])
        self.assertEqual(first["diagnostics"]["normal_variates"], 24)
        self.assertEqual(first["diagnostics"]["gaussian_uniforms"], 32)
        self.assertEqual(first["diagnostics"]["quantum_uniforms"], 48)
        self.assertEqual(first["diagnostics"]["displacement_evaluations"], 8)
        self.assertAlmostEqual(first["timings_seconds"]["cold_batch"],
            first["timings_seconds"]["setup"] + first["timings_seconds"]["sampling"], delta=1e-12)
        self.assertEqual(first["timings_seconds"]["reference"], 0)
        self.assertNotIn("sample_log_probabilities", first)
        self.assertIn("no trajectory logQ", first["law_semantics"])

    def test_longdouble_transport_preserves_tail_below_binary64_range(self):
        # Pois(1e-80) tail >4 is ~1e-400/120, genuinely positive but
        # unrepresentable as a binary64 JSON number.  This is a transport test.
        result = coherent_measure_channels([[0]], [1], 1e-40, cutoff=4, executable=self.binary)
        q = result["q"]
        self.assertGreater(q, 0)
        self.assertEqual(float(q), 0.0)
        expected = np.longdouble("1e-80") ** 5 / 120
        self.assertLess(abs(q / expected - 1), np.longdouble("2e-14"))
        self.assertGreater(result["flattened_probabilities"][1], 0)
        self.assertEqual(result["t"], 1)

    def test_traversal_resource_error_is_not_reported_as_a_successful_failed_law(self):
        with self.assertRaises(CoherentCoreError) as caught:
            coherent_core_fixed_law(np.eye(2), .6, .35, 2, [.3, .4j],
                method="refined", max_enumeration_nodes=1, executable=self.binary)
        payload = caught.exception.payload
        self.assertFalse(caught.exception.numerical_failure)
        self.assertEqual(payload["failed_phase"], "enumeration")
        self.assertFalse(payload["enumeration_complete"])
        self.assertIsNone(payload["distribution"])
        self.assertEqual(payload["failure_mass"], 0)
        self.assertIn("node cap", payload["error"])

    def test_wrapper_rejects_trace_transport_corruption(self):
        from gbslab.coherent_core import _decode_payload
        uniforms = np.full((1, 4), .4)
        result = run_coherent_core(np.eye(2), .3, .5, 0, samples=1,
            fixed_alpha=[.2, .3j], uniforms=uniforms, trace=True, executable=self.binary)
        mutations = [
            lambda row: row.update(mode=1),
            lambda row: row.update(prefix_mask=1),
            lambda row: row.update(channel=1),
            lambda row: row.update(bit=1 - row["bit"]),
            lambda row: row.update(uniform_channel=np.longdouble(".3")),
        ]
        for change in mutations:
            bad = deepcopy(result)
            change(bad["trace"][0])
            with self.assertRaises(CoherentCoreError):
                _decode_payload(bad, 2, 0, 1, True, True, uniforms)


if __name__ == "__main__":
    unittest.main()
