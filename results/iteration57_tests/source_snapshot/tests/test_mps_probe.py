import itertools
import math
from pathlib import Path
import sys
import unittest

import numpy as np

from gbslab.gaussian import covariance_from_interferometer, haar_unitary
from gbslab.mps_probe import (symplectic_spectrum, schmidt_parameters,
    geometric_top_weights, dense_to_mps, mps_to_dense, threshold_channels,
    fixed_threshold_law, sample_threshold)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from iteration10_reference import core_state_reference, fixed_displacement_law
sys.path.pop(0)


class MPSProbeTests(unittest.TestCase):
    def test_vacuum_and_product_schmidt(self):
        v = covariance_from_interferometer(np.eye(4), [.2, .3, .4, .5])
        np.testing.assert_allclose(symplectic_spectrum(v), 1, atol=2e-14)
        np.testing.assert_allclose(schmidt_parameters(v, [0, 2]), 0, atol=2e-14)

    def test_two_mode_squeezed_analytic(self):
        u = np.array([[1, 1j], [1j, 1]]) / math.sqrt(2)
        v = covariance_from_interferometer(u, .7)
        q = schmidt_parameters(v, [0])
        np.testing.assert_allclose(q, [math.tanh(.7) ** 2], atol=3e-14)
        np.testing.assert_allclose(geometric_top_weights(q, 12), (1-q[0])*q[0]**np.arange(12))

    def test_heap_against_rectangular_enumeration(self):
        q = np.array([.12, .27, .41])
        brute = sorted((np.prod((1-q) * q**np.array(n)) for n in itertools.product(range(20), repeat=3)), reverse=True)
        np.testing.assert_allclose(geometric_top_weights(q, 100), brute[:100], atol=1e-16)

    def test_heap_degeneracy_and_zero(self):
        np.testing.assert_allclose(geometric_top_weights([0, 0], 10), [1.])
        p = geometric_top_weights([.2, .2], 6)
        np.testing.assert_allclose(p, [.64, .128, .128, .0256, .0256, .0256])

    def test_reject_mixed_and_invalid_covariances(self):
        for v in [np.eye(4)*.5, np.ones((3, 3)), np.eye(4)*np.nan]:
            with self.assertRaises(ValueError):
                symplectic_spectrum(v)
        with self.assertRaises(ValueError):
            schmidt_parameters(np.eye(4)*2, [0])

    def test_schmidt_complement_and_permutation(self):
        v = covariance_from_interferometer(haar_unitary(6, np.random.default_rng(11)), .6)
        np.testing.assert_allclose(schmidt_parameters(v, [0, 2]), schmidt_parameters(v, [1, 3, 4, 5]), atol=2e-14)

    def test_full_rank_reconstruction_complex(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(3, 3, 3, 3)) + 1j*rng.normal(size=(3, 3, 3, 3))
        x /= np.linalg.norm(x)
        a, info = dense_to_mps(x, 9)
        np.testing.assert_allclose(mps_to_dense(a), x, atol=3e-15)
        self.assertLess(info['state_trace_distance'], 1e-13)

    def test_discarded_sum_tracks_fidelity(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(3,)*5) + 1j*rng.normal(size=(3,)*5)
        x /= np.linalg.norm(x)
        _, info = dense_to_mps(x, 2)
        self.assertAlmostEqual(info['discarded_sum'], 1-info['fidelity'], delta=2e-14)
        self.assertAlmostEqual(info['state_trace_distance']**2, info['discarded_sum'], delta=2e-14)

    def test_resource_and_norm_guards(self):
        with self.assertRaises(ValueError):
            dense_to_mps(np.ones((2, 2)), 2)
        with self.assertRaises(ValueError):
            dense_to_mps(np.ones((2, 2))/2, 2, max_elements=3)

    def test_channel_completeness_including_large_displacement(self):
        for d in [1, 2, 5, 10]:
            for z in [0, .3+.7j, 3+4j, 30.]:
                k = threshold_channels(z, d)
                np.testing.assert_allclose(k.conj().T @ k, np.eye(d), atol=3e-15)

    def test_vacuum_coherent_clicks(self):
        a, _ = dense_to_mps(np.ones((1, 1, 1)), 1)
        z = np.array([.2+.1j, .5j, .4])
        p = fixed_threshold_law(a, z)
        p0 = np.exp(-np.abs(z)**2)
        expected = [np.prod([1-p0[j] if mask>>j&1 else p0[j] for j in range(3)]) for mask in range(8)]
        np.testing.assert_allclose(p, expected, atol=2e-15)

    def test_law_against_independent_fock_reference_and_contraction(self):
        state = core_state_reference(haar_unitary(4, np.random.default_rng(14)), .9, .65, 4)
        x = np.zeros((5,)*4, complex)
        x[tuple(state['occupations'].T)] = state['amplitudes']
        z = np.array([.1+.4j, -.2j, .3-.1j, -.4+.2j])
        exact = fixed_displacement_law(state, z)['distribution']
        full, _ = dense_to_mps(x, 25)
        np.testing.assert_allclose(fixed_threshold_law(full, z), exact, atol=2e-14)
        compressed, info = dense_to_mps(x, 2)
        self.assertLessEqual(.5*np.abs(fixed_threshold_law(compressed, z)-exact).sum(), info['state_trace_distance']+1e-12)

    def test_sampler_matches_complete_law(self):
        x = np.zeros((2, 2), complex)
        x[0, 0], x[1, 1] = np.sqrt(.7), 1j*np.sqrt(.3)
        a, _ = dense_to_mps(x, 2)
        z = np.array([.3+.2j, -.1+.4j])
        p = fixed_threshold_law(a, z)
        samples = sample_threshold(a, np.tile(z, (12000, 1)), np.random.default_rng(15))
        hist = np.bincount(samples @ np.array([1, 2]), minlength=4)/len(samples)
        # Fixed seed, generous finite-sample check; not a global sampler certificate.
        self.assertLess(.5*np.abs(hist-p).sum(), .025)

    def test_sampler_rejects_noncanonical_input(self):
        with self.assertRaises(ValueError):
            sample_threshold([np.ones((1, 2, 2)), np.ones((2, 2, 1))], [[0, 0]], np.random.default_rng(16))


if __name__ == '__main__':
    unittest.main()
