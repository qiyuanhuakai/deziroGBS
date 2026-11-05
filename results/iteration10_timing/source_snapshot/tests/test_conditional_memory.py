"""Independent probability checks for selectors conditioned on observed zeros."""

import unittest

import numpy as np

from gbslab.conditional_memory import ConditionalMemorySampler, SchurConditionalMemorySampler
from gbslab.exact import ThresholdModel
from gbslab.gaussian import covariance_from_interferometer, haar_unitary
from gbslab.zero_memory import ZeroMemorySampler, click_tail_coupling_bound


def marginal_from_dense(p, modes, zeros=(), ones=()):
    result = np.zeros(1 << len(modes))
    for mask, probability in enumerate(p):
        if any(mask & (1 << mode) for mode in zeros):
            continue
        if any(not mask & (1 << mode) for mode in ones):
            continue
        index = sum(((mask >> mode) & 1) << bit for bit, mode in enumerate(modes))
        result[index] += probability
    return result / result.sum()


def binary_pair_mi(joint):
    table = np.asarray(joint).reshape(2, 2)
    product = table.sum(axis=1)[:, None] * table.sum(axis=0)[None, :]
    positive = table > 0
    return float(np.sum(table[positive] * np.log(table[positive] / product[positive])))


class ConditionalMemoryTests(unittest.TestCase):
    def model(self, m=5):
        rng = np.random.default_rng(809)
        U = haar_unitary(m, rng)
        # Nonuniform loss and squeezing exercise general covariance handling.
        return ThresholdModel(covariance_from_interferometer(U,
            np.linspace(.3, .65, m), np.linspace(.45, .85, m)))

    def test_zero_conditioned_covariance_and_mi_match_full_reference(self):
        model = self.model()
        p = model.full_distribution()
        sampler = ConditionalMemorySampler(model, 1)
        for zeros in ((), (1,), (3, 0)):
            remaining, W = sampler.conditional_covariance(zeros)
            conditioned_model = ThresholdModel(W)
            reference = marginal_from_dense(p, remaining, zeros=zeros)
            np.testing.assert_allclose(conditioned_model.full_distribution(),
                reference, atol=3e-13, rtol=2e-10)
            target = remaining[-1]
            candidates = remaining[:-1]
            scores = sampler.conditional_pairwise_mi(zeros, target, candidates)
            for candidate in candidates:
                joint = marginal_from_dense(p, (target, candidate), zeros=zeros)
                self.assertAlmostEqual(scores[candidate], binary_pair_mi(joint), 11)

    def test_conditioning_changes_selected_mode_in_physical_gaussian_fixture(self):
        # Classical Gaussian latent components: conditioning bright mode 3 to
        # vacuum suppresses the first component shared by target 0 and mode 1.
        # Mode 2 shares an independent component with target 0.
        amplitudes = np.array([[1., .6], [1., 0.], [0., .6], [3., 0.]])
        covariance_block = np.eye(4) + 2 * amplitudes @ amplitudes.T
        V = np.block([[covariance_block, np.zeros((4, 4))],
                      [np.zeros((4, 4)), covariance_block]])
        model = ThresholdModel(V)
        order = (3, 1, 2, 0)
        ordinary = ZeroMemorySampler(model, 1, order=order, selection="pairwise")
        conditional = ConditionalMemorySampler(model, 1, order=order)
        self.assertEqual(ordinary._retained_clicks((1, 2), 0), (1,))
        scores = conditional.conditional_pairwise_mi((3,), 0, (1, 2))
        self.assertGreater(scores[2], scores[1])
        self.assertEqual(conditional._retained_clicks((1, 2), 0), (2,))

    def test_influence_scores_match_dense_observed_one_conditionals(self):
        model = self.model(6)
        p = model.full_distribution()
        sampler = ConditionalMemorySampler(model, 2, selection="click_influence")
        zeros, retained, mode, candidates = (1,), (0,), 5, (2, 3, 4)
        baseline, scores, probabilities = sampler.click_influence_scores(
            zeros, mode, candidates, retained)
        reference_base = marginal_from_dense(p, (mode,), zeros, retained)[1]
        self.assertAlmostEqual(baseline, reference_base, 10)
        for candidate in candidates:
            reference = marginal_from_dense(p, (mode,), zeros, retained+(candidate,))[1]
            self.assertAlmostEqual(probabilities[candidate], reference, 10)
            self.assertAlmostEqual(scores[candidate], abs(reference-reference_base), 10)

    def test_full_budget_is_exact_and_low_click_sector_is_preserved(self):
        model = self.model()
        p = model.full_distribution()
        order = (4, 1, 3, 0, 2)
        for cls in (ConditionalMemorySampler, SchurConditionalMemorySampler):
            for selection in ("conditional_mi", "click_influence"):
                for budget in (0, 1, 2, 5):
                    sampler = cls(model, budget, order=order, selection=selection)
                    q = sampler.full_distribution()
                    self.assertAlmostEqual(float(q.sum()), 1., 12)
                    self.assertGreaterEqual(float(q.min()), 0.)
                    low_masks = [mask for mask in range(len(p)) if mask.bit_count() <= budget]
                    np.testing.assert_allclose(q[low_masks], p[low_masks], atol=3e-13, rtol=2e-10)
                    self.assertLessEqual(.5*np.sum(abs(p-q)), click_tail_coupling_bound(p,budget)+1e-12)
                    if budget >= model.m-1:
                        np.testing.assert_allclose(q, p, atol=3e-13, rtol=2e-10)
                        self.assertEqual(sampler.diagnostics["selection_computations"], 0)

    def test_direct_and_schur_classes_agree_for_both_selectors(self):
        model = self.model(6)
        for selection in ("conditional_mi", "click_influence"):
            direct = ConditionalMemorySampler(model, 2, selection=selection)
            schur = SchurConditionalMemorySampler(model, 2, selection=selection)
            np.testing.assert_allclose(direct.full_distribution(), schur.full_distribution(),
                atol=3e-13, rtol=1e-9)

    def test_selection_and_covariance_caches_are_bounded(self):
        model = self.model(6)
        sampler = SchurConditionalMemorySampler(model, 1, cache_size=3,
            covariance_cache_size=2, covariance_cache_bytes=100000)
        q = sampler.full_distribution()
        self.assertAlmostEqual(float(q.sum()), 1., 12)
        self.assertLessEqual(len(sampler._selection_cache), 3)
        self.assertGreater(sampler.diagnostics["selection_cache_evictions"], 0)
        self.assertLessEqual(sampler.conditioner.cache_info()["cache_entries"], 2)
        self.assertLessEqual(sampler.conditioner.cache_info()["cache_payload_bytes"], 100000)
        reference = SchurConditionalMemorySampler(model, 1, cache_size=0,
            covariance_cache_size=0).full_distribution()
        np.testing.assert_allclose(q, reference, atol=3e-13, rtol=1e-9)

    def test_sampling_matches_enumerated_surrogate(self):
        sampler = SchurConditionalMemorySampler(self.model(4), 1,
            order=(2, 0, 3, 1), selection="conditional_mi")
        q = sampler.full_distribution()
        samples = sampler.sample(15000, np.random.default_rng(113))
        masks = samples @ (1 << np.arange(4))
        frequency = np.bincount(masks, minlength=16) / len(samples)
        self.assertLess(.5*np.sum(abs(frequency-q)), .025)
        self.assertEqual(samples.dtype, np.uint8)

    def test_invalid_inputs_and_vacuum(self):
        model = self.model(3)
        with self.assertRaises(ValueError):
            ConditionalMemorySampler(model, 1, selection="not_implemented")
        sampler = ConditionalMemorySampler(model, 1)
        with self.assertRaises(ValueError):
            sampler.conditional_pairwise_mi((0,), 0, (1,))
        with self.assertRaises(ValueError):
            sampler.click_influence_scores((), 0, (1,), retained=(1,))
        with self.assertRaises(ValueError):
            sampler.conditional_covariance((0, 0))
        vacuum = SchurConditionalMemorySampler(ThresholdModel(np.eye(8)), 1)
        expected = np.zeros(16); expected[0] = 1.
        np.testing.assert_array_equal(vacuum.full_distribution(), expected)


if __name__ == "__main__":
    unittest.main()
