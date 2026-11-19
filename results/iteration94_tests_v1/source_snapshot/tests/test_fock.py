"""Independent analytic and determinant/Fock cross-checks (stdlib unittest)."""

import math
import unittest

import numpy as np

from gbslab.fock import pure_fock_amplitudes, threshold_from_fock


class FockReferenceTests(unittest.TestCase):
    def test_single_mode_squeezed_photon_formula(self):
        r = 0.61
        amplitudes = pure_fock_amplitudes(np.eye(1), r, 12)
        for pair_number in range(7):
            occupation = (2 * pair_number,)
            expected = (
                math.factorial(2 * pair_number)
                / (2 ** (2 * pair_number) * math.factorial(pair_number) ** 2)
                * math.tanh(r) ** (2 * pair_number)
                / math.cosh(r)
            )
            self.assertAlmostEqual(abs(amplitudes[occupation]) ** 2, expected, places=14)
        self.assertTrue(all(sum(n) % 2 == 0 for n in amplitudes))

    def test_two_mode_squeezed_analytic_occupations(self):
        r = 0.42
        beam_splitter = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        amplitudes = pure_fock_amplitudes(beam_splitter, np.array([r, -r]), 12)
        for (left, right), amplitude in amplitudes.items():
            expected = (
                math.tanh(r) ** (2 * left) / math.cosh(r) ** 2 if left == right else 0.0
            )
            self.assertAlmostEqual(abs(amplitude) ** 2, expected, places=14)
        result = threshold_from_fock(beam_splitter, np.array([r, -r]), 1.0, 12)
        self.assertLess(result.probabilities[1] + result.probabilities[2], 1e-25)
        self.assertAlmostEqual(result.probabilities[0], 1 / math.cosh(r) ** 2, places=14)
        expected_tail = math.tanh(r) ** 14
        self.assertAlmostEqual(result.omitted_mass, expected_tail, places=14)

    def test_vacuum_and_complete_loss(self):
        vacuum = threshold_from_fock(np.eye(3), 0.0, np.array([0.2, 0.4, 0.9]), 4)
        np.testing.assert_allclose(vacuum.probabilities, [1, 0, 0, 0, 0, 0, 0, 0], atol=1e-14)
        lost = threshold_from_fock(np.eye(2), [0.4, 0.7], 0.0, 10)
        np.testing.assert_allclose(lost.normalized_distribution, [1, 0, 0, 0], atol=1e-14)
        self.assertGreater(lost.omitted_mass, 0)

    def test_single_mode_loss_against_closed_form_and_tail_bound(self):
        r = 0.69
        eta = 0.37
        result = threshold_from_fock(np.eye(1), r, eta, 14)
        p0 = 1 / np.sqrt(1 + eta * (2 - eta) * np.sinh(r) ** 2)
        exact = np.array([p0, 1 - p0])
        tv = 0.5 * np.abs(exact - result.normalized_distribution).sum()
        self.assertLessEqual(tv, result.omitted_mass + 5e-14)
        self.assertGreater(result.omitted_mass, 0)
        self.assertLess(result.omitted_mass, 1e-4)

    def test_passive_unitary_preserves_each_total_photon_sector(self):
        m = 3
        fourier = np.fft.fft(np.eye(m)) / np.sqrt(m)
        squeezing = np.array([0.25, 0.41, 0.52])
        unmixed = threshold_from_fock(np.eye(m), squeezing, 1.0, 10)
        mixed = threshold_from_fock(fourier, squeezing, 1.0, 10)
        for total in unmixed.retained_mass_by_photon_number:
            self.assertAlmostEqual(
                unmixed.retained_mass_by_photon_number[total],
                mixed.retained_mass_by_photon_number[total],
                places=14,
            )

    def test_untruncated_probability_is_componentwise_above_reference(self):
        # Truncation discards nonnegative Fock contributions to this diagonal
        # measurement. This is stronger than merely matching normalization.
        lower = threshold_from_fock(np.eye(2), [0.31, 0.47], [0.45, 0.83], 6)
        upper = threshold_from_fock(np.eye(2), [0.31, 0.47], [0.45, 0.83], 12)
        self.assertTrue(np.all(upper.probabilities >= lower.probabilities - 1e-14))
        self.assertGreater(upper.retained_mass, lower.retained_mass)

    def test_against_covariance_determinant_reference(self):
        try:
            from gbslab.exact import ThresholdModel
            from gbslab.gaussian import covariance_from_interferometer
        except ImportError as exc:
            self.skipTest(f"Exact reference module has not been integrated yet: {exc}")
        # Complex mode mixing, nonuniform input squeezing, nonuniform loss.
        # This comparison uses completely distinct probability calculations.
        for m in (2, 3, 4):
            with self.subTest(m=m):
                fourier = np.fft.fft(np.eye(m)) / np.sqrt(m)
                r = np.linspace(0.19, 0.43, m)
                eta = np.linspace(0.37, 0.91, m)
                V = covariance_from_interferometer(fourier, r, eta)
                exact = ThresholdModel(V).full_distribution()
                fock = threshold_from_fock(fourier, r, eta, 14)
                tv = 0.5 * np.abs(exact - fock.normalized_distribution).sum()
                self.assertLessEqual(tv, fock.omitted_mass + 2e-12)
                self.assertTrue(np.all(exact >= fock.probabilities - 2e-12))
                self.assertLess(fock.omitted_mass, 1e-5)

    def test_input_validation_and_resource_guard(self):
        for eta in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                threshold_from_fock(np.eye(2), 0.1, eta, 2)
        with self.assertRaises(ValueError):
            pure_fock_amplitudes(2 * np.eye(2), 0.1, 2)
        with self.assertRaises(ValueError):
            pure_fock_amplitudes(np.eye(3), 0.1, 100, max_states=1000)
        with self.assertRaises(ValueError):
            pure_fock_amplitudes(np.eye(1), 0.1, -1)

    def test_threshold_array_guard_even_for_vacuum_only_cutoff(self):
        # There is just one Fock occupation at cutoff zero, but 2**40
        # threshold entries would otherwise be allocated before returning it.
        with self.assertRaisesRegex(ValueError, "Threshold pattern array"):
            threshold_from_fock(np.eye(40), 0.0, max_photons=0)


if __name__ == "__main__":
    unittest.main()
