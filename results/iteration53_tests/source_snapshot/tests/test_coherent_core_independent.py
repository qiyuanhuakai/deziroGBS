"""Independent finite-Fock/analytic-integration gates for coherent-core sampling.

Native tests use an existing, explicitly built executable.  No test compiles or
downloads an implementation.  Reference-only tests do not import native code.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
try:
    from iteration10_reference import (
        ReferenceNumericalError, ReferenceResourceError,
        _law_from_marginals, core_state_reference, fixed_displacement_direct_patterns,
        fixed_displacement_law, gaussian_averaged_law, hafnian_by_pairings,
        occupation_basis, reference_work_estimate,
    )
finally:
    sys.path.pop(0)


def _state(occupations, amplitudes, cutoff):
    return {"occupations": np.asarray(occupations, dtype=int),
            "amplitudes": np.asarray(amplitudes, dtype=complex), "cutoff": cutoff}


def _coefficient_map(state):
    return {tuple(n): complex(c) for n, c in zip(state["occupations"], state["amplitudes"])}


class CoherentCoreIndependentReferenceTests(unittest.TestCase):
    def test_coefficient_recurrence_against_direct_small_hafnian_pairing(self):
        from gbslab.gaussian import haar_unitary
        U = haar_unitary(3, np.random.default_rng(101103))
        state = core_state_reference(U, .7, .6, 6)
        relative = []
        for n in state["occupations"]:
            repeated = [i for i, count in enumerate(n) for _ in range(int(count))]
            matrix = state["B"][np.ix_(repeated, repeated)]
            factorial = math.prod(math.factorial(int(v)) for v in n)
            relative.append(hafnian_by_pairings(matrix) / math.sqrt(factorial))
        relative = np.asarray(relative)
        relative /= np.linalg.norm(relative)
        np.testing.assert_allclose(state["amplitudes"], relative, rtol=2e-14, atol=2e-15)
        self.assertLess(state["diagnostics"]["retained_weight_absolute_residual"], 2e-14)
        self.assertLess(state["diagnostics"]["covariance_split_max_absolute_residual"], 2e-14)

    def test_two_mode_complex_interference_and_total_sector_coherence(self):
        U = np.array([[1, 1j], [1j, 1]]) / math.sqrt(2)
        state = core_state_reference(U, .6, 1., 4)
        t = math.tanh(.6)
        norm = math.sqrt(1 + t * t + t**4)
        expected = {(0, 0): 1 / norm, (1, 1): -1j * t / norm, (2, 2): -t*t / norm}
        for n, value in _coefficient_map(state).items():
            self.assertAlmostEqual(abs(value - expected.get(n, 0)), 0, delta=2e-15)
        # A passive phase rotation must not discard the N=0/N=2 phase.
        self.assertGreater(abs(_coefficient_map(state)[(0, 0)] * _coefficient_map(state)[(1, 1)].conjugate()), .1)

    def test_complex_bra_and_dephasing_have_distinct_closed_form_probabilities(self):
        state = _state([[0], [2]], [2 * math.sqrt(2) / 3, 1j / 3], 2)
        alpha = np.array([(1 + 1j) / math.sqrt(2)])
        coherent = fixed_displacement_law(state, alpha)["distribution"]
        direct = fixed_displacement_direct_patterns(state, alpha)
        dephased = sum(abs(c)**2 * fixed_displacement_law(_state([n], [1], 2), alpha)["distribution"]
                       for n, c in zip(state["occupations"], state["amplitudes"]))
        self.assertAlmostEqual(coherent[0], 25 / (18 * math.e), delta=2e-15)
        self.assertAlmostEqual(dephased[0], 17 / (18 * math.e), delta=2e-15)
        self.assertAlmostEqual(coherent[0] - dephased[0], 4 / (9 * math.e), delta=2e-15)
        self.assertGreater(abs(coherent[0] - 1 / (2 * math.e)), .3)
        np.testing.assert_allclose(coherent, direct, rtol=0, atol=2e-15)

    def test_direct_tensor_threshold_effects_agree_with_vacuum_mobius(self):
        from gbslab.gaussian import haar_unitary
        state = core_state_reference(haar_unitary(3, np.random.default_rng(101104)), .5, .45, 4)
        alpha = np.array([.2 + .4j, -.3 + .1j, .05 - .2j])
        direct = fixed_displacement_direct_patterns(state, alpha)
        transformed = fixed_displacement_law(state, alpha)["distribution"]
        np.testing.assert_allclose(transformed, direct, rtol=0, atol=3e-15)
        phase = dict(state, amplitudes=state["amplitudes"] * np.exp(.781j))
        np.testing.assert_allclose(fixed_displacement_law(phase, alpha)["distribution"], direct,
                                   rtol=0, atol=3e-15)

    def test_singular_gaussian_average_keeps_coherence_and_degree_eight(self):
        # A one-mode residual is a rank-one p-quadrature shift, alpha=i*y.
        state = core_state_reference(np.eye(1), .8, .6, 4)
        averaged = gaussian_averaged_law(state)
        t = math.tanh(state["parameters"]["core_squeezing"])
        v = state["displacement_covariance"][1, 1] / 4
        u = v / (1 + 2*v)
        moments = [1., u, 3*u**2, 15*u**3, 105*u**4]
        polynomial = moments[0] + t*moments[1] + t*t/2*moments[2] + t**3/8*moments[3] + t**4/64*moments[4]
        expected = polynomial / math.sqrt(1 + 2*v) / (1 + t*t/2 + 3*t**4/8)
        self.assertAlmostEqual(averaged["distribution"][0], expected, delta=3e-15)
        self.assertEqual(averaged["diagnostics"]["maximum_moment_degree"], 8)
        self.assertGreater(averaged["diagnostics"]["wick_cache_peak"], 1)
        state2 = core_state_reference(np.eye(1), .6, .35, 2)
        p_coherent = gaussian_averaged_law(state2)["distribution"][0]
        p_dephased = sum(abs(c)**2 * gaussian_averaged_law(_state([n], [1], 2),
                            state2["displacement_covariance"])["distribution"][0]
                         for n, c in zip(state2["occupations"], state2["amplitudes"]) if c != 0)
        t = math.tanh(state2["parameters"]["core_squeezing"])
        v = state2["displacement_covariance"][1, 1] / 4
        expected_difference = t*v / (1+2*v)**1.5 / (1 + t*t/2)
        self.assertAlmostEqual(p_coherent - p_dephased, expected_difference, delta=3e-15)
        self.assertGreater(expected_difference, 0)

    def test_correlated_improper_gaussian_tilt_against_fixed_gauss_hermite(self):
        from gbslab.gaussian import haar_unitary
        U = haar_unitary(2, np.random.default_rng(101105))
        # Deterministic quadrature is only an independent convergence check.
        # It is not used as an exact full-law value or as a formal certificate.
        def quadrature(state, order):
            factor = state["displacement_real_factor"]
            nodes, weights = np.polynomial.hermite.hermgauss(order)
            answer = np.zeros(4, dtype=np.longdouble)
            for i, zi in enumerate(nodes):
                for j, zj in enumerate(nodes):
                    d = factor @ (math.sqrt(2) * np.array([zi, zj]))
                    alpha = (d[:2] + 1j*d[2:]) / 2
                    p = fixed_displacement_direct_patterns(state, alpha)
                    answer += weights[i]*weights[j]/math.pi * p
            return np.asarray(answer, dtype=float)
        diagnostics = []
        for cutoff in (2, 4):
            state = core_state_reference(U, .4, .4, cutoff)
            averaged = gaussian_averaged_law(state)["distribution"]
            low, middle, high = (quadrature(state, n) for n in (12, 20, 28))
            delta = state["parameters"]["residual_variance"]
            pseudo = -delta/4 * (U @ U.T)
            self.assertGreater(np.max(np.abs(pseudo)), 1e-3)
            differences = {"cutoff": cutoff, "orders": [12, 20, 28],
                "max_abs_12_vs_20": float(np.max(np.abs(low - middle))),
                "max_abs_20_vs_28": float(np.max(np.abs(middle - high))),
                "max_abs_28_vs_analytic_wick": float(np.max(np.abs(high - averaged)))}
            diagnostics.append(differences)
            self.assertLessEqual(differences["max_abs_20_vs_28"], 1e-12)
            self.assertLessEqual(differences["max_abs_28_vs_analytic_wick"], 1e-12)
        print("coherent_core_gh_gate " + json.dumps({"seed": 101105, "modes": 2,
            "squeezing": .4, "loss": .4, "diagnostics": diagnostics,
            "development_check": {"cutoff": 4, "orders": [8, 14],
                "max_abs_difference": 2.5127122604828855e-11,
                "previous_threshold": 2e-11, "passed": False},
            "outward_rounded_certificate": False}, sort_keys=True))

    def test_zero_cutoff_matches_gaussian_classical_covariance_determinants(self):
        from gbslab.gaussian import haar_unitary
        state = core_state_reference(haar_unitary(4, np.random.default_rng(101106)), .7, .5, 0)
        result = gaussian_averaged_law(state)
        C = state["displacement_covariance"]
        for mask, actual in enumerate(result["vacuum_marginals"]):
            modes = [i for i in range(4) if mask >> i & 1]
            indices = modes + [4+i for i in modes]
            expected = 1.0 / math.sqrt(np.linalg.det(np.eye(len(indices)) + C[np.ix_(indices, indices)]/2))
            self.assertAlmostEqual(actual, expected, delta=2e-15)

    def test_vacuum_and_lossless_boundaries_match_direct_fock_threshold_bins(self):
        from gbslab.gaussian import haar_unitary
        U = haar_unitary(3, np.random.default_rng(101107))
        for r, eta in ((0., .6), (.7, 0.), (.7, 1.)):
            with self.subTest(r=r, eta=eta):
                state = core_state_reference(U, r, eta, 4)
                law = gaussian_averaged_law(state)["distribution"]
                direct = np.zeros(8)
                for n, c in zip(state["occupations"], state["amplitudes"]):
                    mask = sum((int(nj) > 0) << j for j, nj in enumerate(n))
                    direct[mask] += abs(c)**2
                np.testing.assert_allclose(law, direct, rtol=0, atol=2e-15)
                np.testing.assert_array_equal(state["displacement_covariance"], np.zeros((6, 6)))

    def test_small_complete_law_obeys_state_budget_against_independent_gaussian_target(self):
        from gbslab.exact import ThresholdModel
        from gbslab.gaussian import covariance_from_interferometer, haar_unitary
        U = haar_unitary(4, np.random.default_rng(101108))
        V = covariance_from_interferometer(U, .6, loss=.35)
        target = ThresholdModel(V, max_exact_modes=4).full_distribution()
        for cutoff in (0, 2, 4):
            with self.subTest(cutoff=cutoff):
                state = core_state_reference(U, .6, .35, cutoff)
                reference = gaussian_averaged_law(state)
                distance = .5 * float(np.abs(target - reference["distribution"]).sum())
                self.assertLessEqual(distance, state["trace_distance_bound"] + 1e-10)
                self.assertFalse(reference["diagnostics"]["outward_rounded_certificate"])

    def test_resource_counts_and_caps_are_explicit_without_panel_computation(self):
        for m, cutoff, expected_even, expected_full, expected_pairs in (
                (8, 4, 367, 495, 1_501_508), (12, 2, 79, 91, 2_012_337)):
            with self.subTest(m=m, cutoff=cutoff):
                # Merely a combinatorial basis, with a normalized vacuum vector.
                occupations = occupation_basis(m, cutoff)
                amplitudes = np.zeros(len(occupations), dtype=complex)
                amplitudes[0] = 1
                state = dict(_state(occupations, amplitudes, cutoff), even_initial_support=True)
                work = reference_work_estimate(state)
                self.assertEqual(len(occupations), expected_full)
                self.assertEqual(work["basis_size_for_count"], expected_even)
                self.assertEqual(work["compatible_unordered_pairs"], expected_pairs)
                with self.assertRaises(ReferenceResourceError):
                    reference_work_estimate(state, max_pairs=expected_pairs - 1)
        with self.assertRaises(ReferenceResourceError):
            occupation_basis(32, 8, max_basis=1000)
        state = core_state_reference(np.eye(1), .8, .6, 4)
        with self.assertRaises(ReferenceResourceError):
            gaussian_averaged_law(state, max_moments=1)

    def test_reference_repairs_are_visible_and_out_of_budget_values_fail(self):
        accepted = _law_from_marginals([1, 1 + 1e-13])
        self.assertLess(accepted["raw_distribution"][1], 0)
        self.assertEqual(accepted["distribution"][1], 0)
        self.assertGreater(accepted["diagnostics"]["correction_l1"], 0)
        with self.assertRaises(ReferenceNumericalError):
            _law_from_marginals([1, 1 + 2e-11])
        with self.assertRaises(ReferenceNumericalError):
            _law_from_marginals([1 + 2e-11, .5])
        with self.assertRaises(ReferenceNumericalError):
            _law_from_marginals([1, 1 + 1e-13], correction_l1_tolerance=1e-14)


class CoherentCoreIndependentNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        explicit = os.environ.get("GBSLAB_COHERENT_CORE_EXECUTABLE")
        cls.binary = Path(explicit) if explicit else ROOT / ".cache/coherent-core-native/coherent_core_sampler"
        if not cls.binary.is_file():
            if explicit:
                raise FileNotFoundError(cls.binary)
            raise unittest.SkipTest("Build the coherent-core executable explicitly; tests never compile")
        from gbslab.coherent_core import (
            CoherentCoreError, coherent_core_state, coherent_core_fixed_law,
            coherent_measure_channels, run_coherent_core,
        )
        cls.native_state = staticmethod(coherent_core_state)
        cls.native_law = staticmethod(coherent_core_fixed_law)
        cls.native_channels = staticmethod(coherent_measure_channels)
        cls.native_sample = staticmethod(run_coherent_core)
        cls.Error = CoherentCoreError
        cls.binary_digest = hashlib.sha256(cls.binary.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        if hashlib.sha256(cls.binary.read_bytes()).hexdigest() != cls.binary_digest:
            raise AssertionError("The native coherent-core executable changed during independent tests")

    def test_native_initial_coefficients_match_independent_maps_at_multiple_sizes(self):
        from gbslab.gaussian import haar_unitary
        cases = [(np.array([[np.exp(-1j*math.pi/4)]]), math.atanh(.5), 1., 2),
                 (np.array([[1, 1j], [1j, 1]]) / math.sqrt(2), .6, 1., 4)]
        cases += [(haar_unitary(m, np.random.default_rng(101110+m)), .6, .35, 4 if m == 4 else 2)
                  for m in (4, 6, 8)]
        for U, r, eta, cutoff in cases:
            with self.subTest(m=len(U), cutoff=cutoff):
                native = self.native_state(U, r, eta, cutoff, executable=self.binary)
                reference = core_state_reference(U, r, eta, cutoff)
                native_map, independent_map = _coefficient_map(native), _coefficient_map(reference)
                self.assertEqual(set(native_map), set(independent_map))
                self.assertLess(max(abs(native_map[n]-independent_map[n]) for n in native_map), 2e-13)
                self.assertEqual(len(native_map), math.comb(len(U)+cutoff, cutoff))
                self.assertAlmostEqual(float(native["tail_probability"]), reference["tail_probability"], delta=2e-13)
                self.assertAlmostEqual(float(native["retained_weight"]), reference["retained_weight"], delta=2e-13)
                self.assertTrue(all(c == 0 for n, c in native_map.items() if sum(n) % 2))
                self.assertEqual(native["diagnostics"]["quantum_uniforms"], 0)

    def test_native_complex_bra_fixture_has_correct_binary_probability(self):
        occupations = np.array([[0], [2]])
        amplitudes = np.array([2*math.sqrt(2)/3, 1j/3])
        channels = self.native_channels(occupations, amplitudes, (1+1j)/math.sqrt(2),
                                        cutoff=2, executable=self.binary)
        self.assertAlmostEqual(float(channels["flattened_probabilities"][0]), 25/(18*math.e), delta=2e-14)
        self.assertGreater(abs(float(channels["flattened_probabilities"][0])-1/(2*math.e)), .3)
        self.assertAlmostEqual(float(channels["projective_probabilities"].sum()), 1., delta=2e-14)

    def test_all_hidden_channels_survive_the_total_photon_budget(self):
        channels = self.native_channels([[0, 2]], [1+0j], 1., cutoff=2, executable=self.binary)
        probabilities = channels["projective_probabilities"]
        self.assertAlmostEqual(float(probabilities[0]), .4, delta=2e-14)
        self.assertAlmostEqual(float(probabilities[1:].sum()), .6, delta=2e-14)
        self.assertTrue(np.all(probabilities > 0))
        self.assertAlmostEqual(float(channels["flattened_probabilities"][0]), 1/math.e, delta=2e-14)
        remaining = channels["remaining_occupations"]
        self.assertEqual(set(map(tuple, remaining)), {(0,), (1,), (2,)})
        for vector in channels["remaining_amplitudes"]:
            self.assertIsNotNone(vector)
            for n, amplitude in zip(remaining, vector):
                self.assertAlmostEqual(float(abs(amplitude)**2), 1. if tuple(n) == (2,) else 0., delta=2e-14)

    def test_forgetting_channels_reproduces_the_full_conditional_density_and_odd_support(self):
        U = np.array([[1, 1j], [1j, 1]]) / math.sqrt(2)
        state = core_state_reference(U, .6, .55, 4)
        alpha = .31 + .47j
        channels = self.native_channels(state["occupations"], state["amplitudes"], alpha,
                                        cutoff=4, executable=self.binary)
        remaining = channels["remaining_occupations"]
        index = {tuple(n): i for i, n in enumerate(remaining)}
        psi = np.zeros((5, len(remaining)), dtype=complex)
        for n, amplitude in zip(state["occupations"], state["amplitudes"]):
            psi[n[0], index[tuple(n[1:])]] = amplitude
        u = np.array([math.exp(-abs(alpha)**2/2)*(-alpha)**n/math.sqrt(math.factorial(n)) for n in range(5)])
        b = u.conj() @ psi
        sigma_zero = np.outer(b, b.conj())
        sigma_all = psi.T @ psi.conj()
        native_zero = np.zeros_like(sigma_all)
        native_click = np.zeros_like(sigma_all)
        for channel, (probability, vector) in enumerate(zip(channels["projective_probabilities"],
                                                           channels["remaining_amplitudes"])):
            if vector is None:
                self.assertEqual(probability, 0)
                continue
            density = np.asarray(probability * np.outer(vector, vector.conj()), dtype=complex)
            if channel == 0:
                native_zero += float(channels["t"]) * density
                native_click += float(channels["q"]) * density
            else:
                native_click += density
        np.testing.assert_allclose(native_zero, sigma_zero, rtol=0, atol=2e-14)
        np.testing.assert_allclose(native_click, sigma_all-sigma_zero, rtol=0, atol=2e-14)
        odd = np.asarray(remaining).sum(axis=1) % 2 == 1
        self.assertGreater(float(np.sum(np.abs(channels["remaining_amplitudes"][0][odd])**2)), 1e-4)
        self.assertGreater(abs(native_zero[index[(0,)], index[(2,)]]), 1e-5)

    def test_fixed_displacement_complete_native_laws_against_independent_povm(self):
        from gbslab.gaussian import haar_unitary
        for modes, cutoff in ((2, 4), (4, 2), (6, 2), (8, 2)):
            U = haar_unitary(modes, np.random.default_rng(101130+modes))
            state = core_state_reference(U, .55, .5, cutoff)
            alpha = np.linspace(.2, .45, modes) + 1j*np.linspace(-.3, .15, modes)
            expected = fixed_displacement_law(state, alpha)["distribution"]
            for method in (("vacuum", "refined") if modes <= 4 else ("vacuum",)):
                with self.subTest(m=modes, cutoff=cutoff, method=method):
                    native = self.native_law(U, .55, .5, cutoff, alpha, method=method,
                                             executable=self.binary)
                    self.assertTrue(native["enumeration_complete"])
                    self.assertEqual(native["failure_mass"], 0)
                    np.testing.assert_allclose(native["distribution"], expected, rtol=0, atol=2e-13)
                    self.assertEqual(native["diagnostics"]["quantum_uniforms"], 0)
                    self.assertEqual(native["diagnostics"]["normal_variates"], 0)

    def test_one_joint_normal_vector_and_injected_quantum_paths_match_fixed_displacement(self):
        from gbslab.gaussian import haar_unitary
        U = haar_unitary(3, np.random.default_rng(101140))
        state = core_state_reference(U, .6, .35, 2)
        normals = np.array([[0., 1., -2.], [.5, -.3, .8], [-1., -.6, .2]])
        uniforms = np.random.default_rng(101141).random((3, 6))
        native = self.native_sample(U, .6, .35, 2, samples=3, normals=normals,
                                    uniforms=uniforms, trace=True, executable=self.binary)
        self.assertEqual(native["completed_samples"], 3)
        self.assertEqual(native["diagnostics"]["normal_variates"], 9)
        self.assertEqual(native["diagnostics"]["gaussian_uniforms"], 0)
        self.assertEqual(native["diagnostics"]["quantum_uniforms"], 18)
        self.assertNotIn("sample_log_probabilities", native)
        self.assertAlmostEqual(native["timings_seconds"]["cold_batch"],
            native["timings_seconds"]["setup"] + native["timings_seconds"]["sampling"], delta=1e-12)
        for sample, z in enumerate(normals):
            alpha = .5j * math.sqrt(state["parameters"]["residual_variance"]) * (U @ z)
            fixed = self.native_sample(U, .6, .35, 2, samples=1, fixed_alpha=alpha,
                uniforms=uniforms[sample:sample+1], trace=True, executable=self.binary)
            np.testing.assert_array_equal(native["raw_samples"][sample], fixed["raw_samples"][0])
            for mode, reference_step in enumerate(fixed["trace"]):
                actual = native["trace"][sample*3+mode]
                self.assertEqual((actual["sample"], actual["mode"]), (sample, mode))
                self.assertAlmostEqual(abs(complex(actual["alpha"])-alpha[mode]), 0, delta=2e-15)
                self.assertEqual(actual["channel"], reference_step["channel"])
                self.assertEqual(actual["bit"], int(native["raw_samples"][sample, mode]))
                np.testing.assert_allclose(actual["projective_probabilities"],
                    reference_step["projective_probabilities"], rtol=0, atol=2e-14)

    def test_positive_poisson_tail_below_machine_epsilon_is_not_subtracted_away(self):
        from scipy.special import gammainc
        result = self.native_channels([[0]], [1.], 1e-3, cutoff=4, executable=self.binary)
        expected = gammainc(5, 1e-6)
        self.assertGreater(result["q"], 0)
        self.assertLess(result["q"], np.finfo(float).eps)
        self.assertAlmostEqual(float(result["q"]) / expected, 1., delta=2e-13)
        self.assertGreater(result["flattened_probabilities"][1], 0)
        zero = self.native_channels([[0]], [1.], 0., cutoff=4, executable=self.binary)
        self.assertEqual(zero["q"], 0)
        self.assertEqual(zero["t"], 1)

    def test_failure_preserves_checkpoint_and_refined_enumeration_retains_failed_mass(self):
        normals = np.array([[0., 0.], [1e6, 0.], [0., 0.]])
        with self.assertRaises(self.Error) as caught:
            self.native_sample(np.eye(2), .6, .35, 2, samples=3, normals=normals,
                uniforms=np.full((3, 4), .5), trace=True, executable=self.binary)
        result = caught.exception.payload
        self.assertTrue(result["numerical_failure"])
        self.assertEqual(result["completed_samples"], 1)
        self.assertEqual(result["failed_sample"], 1)
        self.assertEqual(result["partial_prefix_length"], 0)
        self.assertEqual(result["raw_samples"].shape, (1, 2))
        self.assertEqual(result["diagnostics"]["normal_variates"], 4)
        self.assertEqual(result["diagnostics"]["quantum_uniforms"], 4)
        self.assertEqual(len(result["trace"]), 2)
        failed_law = self.native_law(np.eye(2), .6, .35, 2, [1e6, 0.], method="refined",
                                     executable=self.binary)
        self.assertTrue(failed_law["enumeration_complete"])
        self.assertEqual(failed_law["failure_mass"], 1)
        np.testing.assert_array_equal(failed_law["distribution"], np.zeros(4))
        self.assertAlmostEqual(sum(float(row["mass"]) for row in failed_law["enumeration_failures"]), 1., delta=1e-15)
        with self.assertRaisesRegex(ValueError, "resource cap"):
            self.native_state(np.eye(64), .6, .35, 6, executable=self.binary)


if __name__ == "__main__":
    unittest.main()
