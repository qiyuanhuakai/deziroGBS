"""Adapter checks and optional, already-built C++ integration checks.

These tests never compile.  Binary-dependent checks explicitly skip if the
default executable has not been built by experiments/build_click_sampler.py.
The larger archived validation suite is an explicit gbslab.cpp_click command.
"""

import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gbslab.cpp_click import (
    CppClickError, _UniformSequence, find_cpp_executable, run_cpp_click, validate_cpp_click,
)
from gbslab.conditional_memory import SchurConditionalMemorySampler
from gbslab.exact import ThresholdModel
from gbslab.gaussian import covariance_from_interferometer, haar_unitary, local_unitary
from gbslab.paired_stress import PairedThresholdOracle, dense_target, paired_covariance
from gbslab.schur_memory import SchurMemorySampler
from gbslab.zero_memory import ZeroMemorySampler, _gaussian_pairwise_mi


class StableGaussianPairwiseTests(unittest.TestCase):
    def test_structural_zero_local_pair_has_exact_score_and_physical_tie_break(self):
        V = covariance_from_interferometer(local_unitary(10, 2, np.random.default_rng(5101)), .8, .5)
        sampler = SchurMemorySampler(ThresholdModel(V), 4, selection="pairwise")
        for past in (1, 4, 6):
            np.testing.assert_array_equal(V[np.ix_([9, 19], [past, past+10])], np.zeros((2, 2)))
            self.assertEqual(sampler.pairwise_mi[9, past], 0.0)
        self.assertEqual(sampler._retained_clicks((1, 4, 6, 7, 8), 9), (8, 7, 1, 4))
        self.assertGreater(sampler.cache_info()["pairwise_structural_zero_pairs"], 0)

    def test_weak_nonzero_mi_matches_independent_high_precision_analytic_pair(self):
        # Vaa=(1+2n1)I, Vbb=(1+2n2)I, C=cI is a classical physical
        # Gaussian state here.  Its exact scalar determinant ratio is known.
        V = np.diag([2.4, 3.2, 2.4, 3.2])
        V[0, 1] = V[1, 0] = V[2, 3] = V[3, 2] = 1e-4
        actual, info = _gaussian_pairwise_mi(V, 0, 1)
        with localcontext() as context:
            context.prec = 80
            one = Decimal(1)
            n1, n2, cross = Decimal('.7'), Decimal('1.1'), Decimal('0.0001')
            a, b = one/(one+n1), one/(one+n2)
            rho_squared = cross*cross/((2+2*n1)*(2+2*n2))
            delta = a*b*rho_squared/(one-rho_squared)
            products = [a*b, (one-a)*b, a*(one-b), (one-a)*(one-b)]
            probabilities = [r+sign*delta for r, sign in zip(products, (1, -1, -1, 1))]
            expected = float(sum(p*(p/r).ln() for p, r in zip(probabilities, products)))
        self.assertGreater(actual, 0)
        self.assertLess(actual, np.finfo(float).eps)
        self.assertLess(abs(actual/expected-1), 2e-13)
        self.assertFalse(info["structural_zero"])

    def test_tmsv_boundary_and_no_covariance_oracle_remain_valid(self):
        information, info = _gaussian_pairwise_mi(paired_covariance(1, .5), 0, 1)
        self.assertAlmostEqual(information, np.log(2.0), 14)
        self.assertLessEqual(info["l1_probability_correction"], 2e-9)
        sampler = ZeroMemorySampler(PairedThresholdOracle(4, .5), 2, selection="pairwise")
        np.testing.assert_array_equal(sampler.full_distribution(), dense_target(4, .5))
        self.assertEqual(sampler.cache_info()["pairwise_mi_policy"], "legacy_joint_marginal_v1")

    def test_local4_nonzero_weak_scores_are_not_floored(self):
        V = covariance_from_interferometer(local_unitary(10, 4, np.random.default_rng(5101)), .4, .5)
        sampler = SchurMemorySampler(ThresholdModel(V), 4, selection="pairwise")
        for past in (1, 2):
            self.assertGreater(np.max(np.abs(V[np.ix_([8, 18], [past, past+10])])), 0)
            self.assertGreater(sampler.pairwise_mi[8, past], 0)
        self.assertLess(sampler.pairwise_mi[8, 2], np.finfo(float).eps)
        self.assertGreater(sampler.pairwise_mi[8, 1], sampler.pairwise_mi[8, 2])


class CppClickWrapperTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="test-cpp-click-")
        self.addCleanup(self.directory.cleanup)
        self.executable = Path(self.directory.name) / "click_sampler"
        self.executable.write_bytes(b"test fixture, never executed")
        self.executable.chmod(0o700)

    @staticmethod
    def payload(**updates):
        result = dict(schema_version=1, status="ok", modes=2, selector="pairwise",
            k=2, sample_count=3, seed=7, order=[0, 1], raw_samples=[[0, 0], [1, 0], [1, 1]],
            distribution=None, histogram_counts=[1, 1, 0, 1], click_count_histogram=[1, 1, 1],
            timings_seconds={"setup": .01, "sampling": .02, "enumeration": 0.0},
            diagnostics={"minimum_p_one": 0.0, "maximum_p_one": 1.0})
        result.update(updates)
        return result

    def invoke(self, payload, **kwargs):
        completed = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
        options = dict(samples=3, seed=7, executable=self.executable)
        options.update(kwargs)
        with patch("gbslab.cpp_click.subprocess.run", return_value=completed) as run:
            result = run_cpp_click(np.eye(4), **options)
        self.assertEqual(run.call_count, 1)
        return result

    def test_missing_binary_is_explicit_and_never_compiles(self):
        missing = Path(self.directory.name) / "absent"
        self.assertIsNone(find_cpp_executable(missing))
        with patch("gbslab.cpp_click.subprocess.run") as run:
            with self.assertRaisesRegex(FileNotFoundError, "build_click_sampler.py"):
                run_cpp_click(np.eye(4), executable=missing)
        run.assert_not_called()

    def test_input_validation_catches_nonphysical_and_invalid_requests(self):
        invalid_covariances = (np.eye(3), np.eye(4)*.5, np.eye(4).astype(complex),
                               np.array([[1, np.nan], [np.nan, 1]]),
                               np.array([[1, .1], [0, 1]]))
        for covariance in invalid_covariances:
            with self.subTest(covariance=covariance), self.assertRaises(ValueError):
                run_cpp_click(covariance, executable=self.executable)
        invalid = [dict(max_clicks=True), dict(max_clicks=-1), dict(max_clicks=9),
            dict(samples=-1), dict(samples=1.5), dict(seed=-1), dict(seed=2**64),
            dict(selection="recent"), dict(order=[0, 0]), dict(order=[0, True]),
            dict(timeout=0), dict(enumerate_distribution=1),
            dict(samples=1, uniforms=np.ones((1, 2))),
            dict(samples=1, uniforms=np.zeros((2, 1))),
            dict(samples=1, uniforms=np.array([[0, np.nan]]))]
        with patch("gbslab.cpp_click.subprocess.run") as run:
            for options in invalid:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    run_cpp_click(np.eye(4), executable=self.executable, **options)
        run.assert_not_called()
        with self.assertRaisesRegex(ValueError, "14"):
            run_cpp_click(np.eye(30), executable=self.executable, enumerate_distribution=True)

    def test_csv_order_and_uniforms_round_trip_without_input_mutation(self):
        covariance = np.diag([2.0, 3.0, 2.0, 3.0])
        original = covariance.copy()
        uniforms = np.array([[.1, .2], [.3, .4], [.5, .6]])
        payload = self.payload(order=[1, 0])

        def fake_run(command, **kwargs):
            self.assertEqual(command[command.index("--order")+1], "1,0")
            np.testing.assert_array_equal(np.loadtxt(command[command.index("--input")+1],
                delimiter=","), covariance)
            np.testing.assert_array_equal(np.loadtxt(command[command.index("--uniforms")+1],
                delimiter=","), uniforms)
            self.assertIn("--raw-samples", command)
            self.assertNotIn("--enumerate", command)
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

        with patch("gbslab.cpp_click.subprocess.run", side_effect=fake_run) as run:
            result = run_cpp_click(ThresholdModel(covariance), samples=3, seed=7,
                order=[1, 0], uniforms=uniforms, executable=self.executable)
        np.testing.assert_array_equal(covariance, original)
        self.assertEqual(result["uniforms_sha256"], hashlib.sha256(uniforms.tobytes()).hexdigest())
        self.assertEqual(result["executable_sha256"], hashlib.sha256(self.executable.read_bytes()).hexdigest())
        self.assertFalse(result["build_metadata_matches_binary"])
        self.assertEqual(run.call_count, 1)

    def test_only_matching_build_metadata_is_bound_to_binary(self):
        metadata = self.executable.parent / "build.json"
        metadata.write_text(json.dumps({"executable_sha256": "different", "cache_key": "stale"}))
        result = self.invoke(self.payload())
        self.assertIsNone(result["build_cache_key"])
        digest = hashlib.sha256(self.executable.read_bytes()).hexdigest()
        metadata.write_text(json.dumps({"executable_sha256": digest, "cache_key": "matching"}))
        result = self.invoke(self.payload())
        self.assertEqual(result["build_cache_key"], "matching")

    def test_rejects_invalid_laws_metadata_and_conditional_probabilities(self):
        for distribution in ([1, 0, 0, -1e-15], [0, 0, 0, .99], [np.nan, 0, 0, 0],
                             [np.inf, 0, 0, 0], [1, 0, 0]):
            with self.subTest(distribution=distribution), self.assertRaises(CppClickError):
                self.invoke(self.payload(distribution=distribution), enumerate_distribution=True)
        for changes in (dict(k=3), dict(order=[1, 0]), dict(status="error"),
                        dict(diagnostics={"minimum_p_one": -1e-15}),
                        dict(diagnostics={"maximum_p_one": 1.0000001})):
            with self.subTest(changes=changes), self.assertRaises(CppClickError):
                self.invoke(self.payload(**changes))

    def test_rejects_nonbinary_samples_and_inconsistent_histograms(self):
        for changes in (dict(raw_samples=[[0, 0], [1, 0], [2, 1]]),
                        dict(raw_samples=[[0, 0], [1, 0]]), dict(raw_samples=None),
                        dict(histogram_counts=[0, 2, 0, 1]),
                        dict(histogram_counts=[1, .5, .5, 1]),
                        dict(click_count_histogram=[0, 2, 1]),
                        dict(mode_click_counts=[0, 3])):
            with self.subTest(changes=changes), self.assertRaises(CppClickError):
                self.invoke(self.payload(**changes))

    def test_sample_trace_checks_uniform_rule_and_retained_history(self):
        uniforms = np.array([[.9, .8], [.1, .8], [.1, .2]])
        trace = []
        bits = self.payload()["raw_samples"]
        for sample_index in range(3):
            for position in range(2):
                trace.append(dict(sample_index=sample_index, position=position, mode=position,
                    retained=[0] if position == 1 and bits[sample_index][0] else [],
                    p_one=.5, cancellation_indicator=1e-14))
        self.invoke(self.payload(conditional_trace=trace), uniforms=uniforms,
                    trace_conditionals=True)
        bad = json.loads(json.dumps(trace)); bad[1]["retained"] = [0]
        with self.assertRaisesRegex(CppClickError, "unclicked"):
            self.invoke(self.payload(conditional_trace=bad), uniforms=uniforms,
                        trace_conditionals=True)
        bad = json.loads(json.dumps(trace)); bad[0]["p_one"] = .95
        with self.assertRaisesRegex(CppClickError, "threshold"):
            self.invoke(self.payload(conditional_trace=bad), uniforms=uniforms,
                        trace_conditionals=True)

    def test_process_failure_preserves_payload_and_never_retries(self):
        payload = {"status": "error", "error_type": "numerical_probability",
                   "error": "conditioning event unresolved"}
        completed = subprocess.CompletedProcess([], 2, json.dumps(payload), "guard diagnostic")
        with patch("gbslab.cpp_click.subprocess.run", return_value=completed) as run:
            with self.assertRaises(CppClickError) as caught:
                run_cpp_click(np.eye(4), executable=self.executable)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(caught.exception.payload, payload)
        self.assertEqual(caught.exception.returncode, 2)
        self.assertEqual(caught.exception.stderr, "guard diagnostic")

    def test_malformed_json_and_timeout_are_errors_without_retries(self):
        with patch("gbslab.cpp_click.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 0, "not json", "")) as run:
            with self.assertRaises(CppClickError):
                run_cpp_click(np.eye(4), executable=self.executable)
        self.assertEqual(run.call_count, 1)
        with patch("gbslab.cpp_click.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("test", 1)) as run:
            with self.assertRaisesRegex(CppClickError, "no retry"):
                run_cpp_click(np.eye(4), executable=self.executable)
        self.assertEqual(run.call_count, 1)

    def test_validation_refuses_to_overwrite_an_interrupted_archive(self):
        archive = Path(self.directory.name) / "archive"
        archive.mkdir()
        saved = archive / "existing_result.txt"
        saved.write_text("keep the interrupted record")
        with patch("gbslab.cpp_click.subprocess.run") as run:
            with self.assertRaisesRegex(FileExistsError, "nonempty"):
                validate_cpp_click(executable=self.executable, archive_directory=archive)
        run.assert_not_called()
        self.assertEqual(saved.read_text(), "keep the interrupted record")


BINARY = find_cpp_executable()


@unittest.skipIf(BINARY is None,
    "C++ executable absent; build explicitly with python experiments/build_click_sampler.py")
class CppClickBinaryTests(unittest.TestCase):
    def test_sparse_local2_and_weak_local4_regressions(self):
        for depth, squeezing in ((2, .4), (2, .8), (4, .4)):
            V = covariance_from_interferometer(local_unitary(10, depth, np.random.default_rng(5101)),
                                               squeezing, .5)
            uniforms = np.random.default_rng(9400+depth).random((16, 10))
            prefix = (0, 1, 0, 0, 1, 0, 1, 1, 1)
            uniforms[0, :9] = [0.0 if bit else np.nextafter(1.0, 0.0) for bit in prefix]
            uniforms[0, 9] = .5
            for selection, cls in (("pairwise", SchurMemorySampler),
                                   ("click_influence", SchurConditionalMemorySampler)):
                with self.subTest(depth=depth, squeezing=squeezing, selection=selection):
                    sampler = cls(ThresholdModel(V), 4, selection=selection)
                    expected = sampler.sample(16, _UniformSequence(uniforms))
                    expected_q = sampler.full_distribution()
                    result = run_cpp_click(V, selection=selection, max_clicks=4,
                        samples=16, uniforms=uniforms, enumerate_distribution=True, executable=BINARY)
                    np.testing.assert_array_equal(result["raw_samples"], expected)
                    np.testing.assert_allclose(result["distribution"], expected_q, atol=2e-10, rtol=0)

    def test_vacuum_zero_samples_and_deterministic_endpoint_uniforms(self):
        result = run_cpp_click(np.eye(8), max_clicks=2, samples=0,
                              enumerate_distribution=True, executable=BINARY)
        expected = np.zeros(16); expected[0] = 1
        np.testing.assert_array_equal(result["distribution"], expected)
        self.assertEqual(result["raw_samples"], [])
        uniforms = np.array([[0.0]*4, [np.nextafter(1.0, 0.0)]*4])
        result = run_cpp_click(np.eye(8), max_clicks=4, samples=2,
                              uniforms=uniforms, executable=BINARY)
        np.testing.assert_array_equal(result["raw_samples"], np.zeros((2, 4)))

    def test_independent_thermal_analytic_law_and_physical_bit_order(self):
        photons = np.array([.25, .5, 1.0, 1.5])
        covariance = np.diag(np.tile(1+2*photons, 2))
        order = (2, 0, 3, 1)
        uniforms = np.random.default_rng(9301).random((32, 4))
        probabilities = photons/(1+photons)
        bits = (np.arange(16)[:, None] >> np.arange(4)) & 1
        expected_q = np.prod(np.where(bits, probabilities, 1-probabilities), axis=1)
        expected_samples = np.zeros((32, 4), dtype=np.uint8)
        for position, mode in enumerate(order):
            expected_samples[:, mode] = uniforms[:, position] < probabilities[mode]
        for selection in ("pairwise", "click_influence"):
            with self.subTest(selection=selection):
                result = run_cpp_click(covariance, selection=selection, max_clicks=2,
                    samples=32, uniforms=uniforms, order=order, enumerate_distribution=True,
                    executable=BINARY)
                np.testing.assert_array_equal(result["raw_samples"], expected_samples)
                np.testing.assert_allclose(result["distribution"], expected_q, atol=2e-12, rtol=0)

    def test_full_q_and_identical_uniforms_match_python_both_selectors(self):
        covariance = covariance_from_interferometer(
            haar_unitary(6, np.random.default_rng(9302)),
            np.linspace(.6, .9, 6), np.linspace(.45, .85, 6))
        order = (4, 0, 5, 1, 3, 2)
        uniforms = np.random.default_rng(9303).random((32, 6))
        for selection, cls in (("pairwise", SchurMemorySampler),
                               ("click_influence", SchurConditionalMemorySampler)):
            for k in (2, 3, 4):
                with self.subTest(selection=selection, k=k):
                    sampler = cls(ThresholdModel(covariance), k, order=order, selection=selection)
                    expected_samples = sampler.sample(32, _UniformSequence(uniforms))
                    expected_q = sampler.full_distribution()
                    result = run_cpp_click(covariance, selection=selection, max_clicks=k,
                        samples=32, uniforms=uniforms, order=order,
                        enumerate_distribution=True, executable=BINARY)
                    np.testing.assert_array_equal(result["raw_samples"], expected_samples)
                    np.testing.assert_allclose(result["distribution"], expected_q, atol=2e-10, rtol=0)

    def test_fully_conditioned_law_matches_physical_target(self):
        covariance = covariance_from_interferometer(
            haar_unitary(5, np.random.default_rng(9304)), .8, .65)
        expected = ThresholdModel(covariance).full_distribution()
        for selection in ("pairwise", "click_influence"):
            result = run_cpp_click(covariance, selection=selection, max_clicks=4,
                order=(4, 3, 2, 1, 0), enumerate_distribution=True, executable=BINARY)
            np.testing.assert_allclose(result["distribution"], expected, atol=2e-11, rtol=0)

    def test_cpp_seed_reproducibility_and_large_mode_sampling(self):
        covariance = np.eye(130)*2.0
        options = dict(samples=8, seed=2**40+17, max_clicks=2, executable=BINARY)
        first = run_cpp_click(covariance, **options)
        second = run_cpp_click(covariance, **options)
        self.assertIsNone(first["distribution"])
        self.assertIsNone(first["histogram_counts"])
        np.testing.assert_array_equal(first["raw_samples"], second["raw_samples"])
        self.assertEqual(np.asarray(first["raw_samples"]).shape, (8, 65))


if __name__ == "__main__":
    unittest.main()
