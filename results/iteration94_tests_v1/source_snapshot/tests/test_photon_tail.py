"""Independent Fock-sector and analytic checks of photon-tail calculations."""

from math import comb, cosh, fsum, tanh
import unittest

import numpy as np

from gbslab.exact import ThresholdModel
from gbslab.fock import pure_fock_amplitudes
from gbslab.gaussian import covariance_from_interferometer, haar_unitary
from gbslab.photon_tail import photon_number_tail, photon_probabilities_prefix


class PhotonTailTests(unittest.TestCase):
    def test_lossless_single_mode_coefficients_and_exact_odd_parity(self):
        r = 0.73
        prefix = photon_probabilities_prefix([r], 1.0, 16)
        expected = np.zeros(17)
        for k in range(9):
            expected[2 * k] = comb(2 * k, k) / (4 ** k) * tanh(r) ** (2 * k) / cosh(r)
        np.testing.assert_allclose(prefix, expected, atol=2e-16, rtol=2e-14)
        np.testing.assert_array_equal(prefix[1::2], np.zeros(8))
        multi = photon_probabilities_prefix([0.2, -0.4, 0.7], 1.0, 17)
        np.testing.assert_array_equal(multi[1::2], np.zeros(9))

    def test_complete_loss_and_vacuum(self):
        expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(photon_probabilities_prefix([0.4, 2.0, -0.7], 0.0, 4), expected)
        np.testing.assert_array_equal(photon_probabilities_prefix([0.0, 0.0], 0.7, 4), expected)
        np.testing.assert_array_equal(photon_probabilities_prefix([], 0.7, 4), expected)
        self.assertEqual(photon_number_tail([3.0, 5.0], 0.0, 0), 0.0)

    def test_against_independent_fock_sector_thinning_with_omitted_mass(self):
        rng = np.random.default_rng(201)
        U = haar_unitary(3, rng)
        squeezing = np.array([0.25, -0.4, 0.55])
        amplitudes = pure_fock_amplitudes(U, squeezing, 18)
        by_sector = {}
        for occupation, amplitude in amplitudes.items():
            by_sector.setdefault(sum(occupation), []).append(float(abs(amplitude) ** 2))
        sector_mass = {n: fsum(values) for n, values in by_sector.items()}
        omitted = max(0.0, 1.0 - fsum(sector_mass.values()))
        self.assertLess(omitted, 5e-7)
        cutoff = 6
        for eta in (0.35, 0.8, 1.0):
            with self.subTest(eta=eta):
                truncated_prefix = np.zeros(cutoff + 1)
                truncated_tail = 0.0
                for photons, mass in sector_mass.items():
                    for detected in range(photons + 1):
                        probability = mass * comb(photons, detected) * eta**detected * (1.0 - eta)**(photons - detected)
                        if detected <= cutoff:
                            truncated_prefix[detected] += probability
                        else:
                            truncated_tail += probability
                exact_prefix = photon_probabilities_prefix(squeezing, eta, cutoff)
                difference = exact_prefix - truncated_prefix
                self.assertGreaterEqual(difference.min(), -3e-15)
                self.assertLessEqual(difference.sum(), omitted + 3e-14)
                tail = photon_number_tail(squeezing, eta, cutoff)
                self.assertGreaterEqual(tail, truncated_tail - 3e-14)
                self.assertLessEqual(tail, truncated_tail + omitted + 3e-14)
                if eta == 1.0:
                    np.testing.assert_allclose(exact_prefix, truncated_prefix, atol=3e-15, rtol=3e-14)

    def test_photon_tail_dominates_exact_threshold_click_tail(self):
        U = haar_unitary(4, np.random.default_rng(982))
        squeezing = np.array([0.4, 0.6, 0.25, 0.55])
        eta = 0.45
        exact = ThresholdModel(covariance_from_interferometer(U, squeezing, loss=eta)).full_distribution()
        for cutoff in (0, 1, 2, 3):
            clicks_tail = sum(p for mask, p in enumerate(exact) if mask.bit_count() > cutoff)
            photons_tail = photon_number_tail(squeezing, eta, cutoff)
            self.assertLessEqual(clicks_tail, photons_tail + 3e-13)
        np.testing.assert_allclose(
            photon_probabilities_prefix(squeezing, eta, 5),
            photon_probabilities_prefix(-squeezing, eta, 5),
            atol=0,
            rtol=0,
        )

    def test_rejects_loss_vectors_and_invalid_parameters(self):
        for loss in ([0.2, 0.5], -0.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                photon_number_tail([0.3, 0.4], loss, 2)
        for cutoff in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                photon_number_tail([0.3], 0.5, cutoff)
        with self.assertRaises(ValueError):
            photon_number_tail([np.inf], 0.5, 2)


if __name__ == "__main__":
    unittest.main()
