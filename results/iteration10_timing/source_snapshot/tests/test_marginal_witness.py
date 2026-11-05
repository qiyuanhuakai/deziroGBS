"""Independent-split, support, and simultaneous-calibration regression tests."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from gbslab.exact import NumericalProbabilityError, ThresholdModel
from gbslab.marginal_witness import (
    assess_marginal_witness, default_marginal_windows, fixed_split_protocol,
    hoeffding_event_tv_lower,
)
from gbslab.paired_stress import paired_covariance


class MarginalWitnessTests(unittest.TestCase):
    def assess(self, V, samples, *, windows, pilot_size=32, alpha=.05, **kwargs):
        protocol = fixed_split_protocol(len(samples), pilot_size, windows=windows,
                                        source="deterministic_unit_fixture_not_an_experiment")
        return assess_marginal_witness(V, samples, windows=windows,
                                       pilot_size=pilot_size, alpha=alpha,
                                       protocol=protocol, **kwargs)

    def test_known_thermal_marginal_gives_analytic_lower_bound(self):
        # hbar=2 thermal covariance 3I has no-click and click probability 1/2.
        samples = np.ones((1056, 1), dtype=np.uint8)
        out = self.assess(np.eye(2)*3, samples, windows=((0,),))
        row = out["rows"][0]
        self.assertEqual(row["event_masks"], [1])
        self.assertAlmostEqual(row["target_event_probability"], .5, places=14)
        self.assertAlmostEqual(out["tv_lower"], .5-math.sqrt(math.log(20)/2048), places=14)
        self.assertEqual(out["decision"], "rejected")
        self.assertFalse(out["strict_floating_point_certificate"])
        self.assertFalse(out["global_tv_acceptance_supported"])

    def test_disjoint_suffix_alone_is_used_and_sign_never_flips(self):
        pilot = np.ones((32, 1), dtype=np.uint8)
        first = np.vstack((pilot, np.zeros((1024, 1), dtype=np.uint8)))
        second = np.vstack((pilot, np.ones((1024, 1), dtype=np.uint8)))
        a = self.assess(np.eye(2)*3, first, windows=((0,),))
        b = self.assess(np.eye(2)*3, second, windows=((0,),))
        ra, rb = a["rows"][0], b["rows"][0]
        self.assertEqual(ra["event_masks"], [1])
        self.assertEqual(ra["event_sha256"], rb["event_sha256"])
        self.assertEqual(a["pilot_samples_sha256"], b["pilot_samples_sha256"])
        self.assertNotEqual(a["heldout_samples_sha256"], b["heldout_samples_sha256"])
        self.assertEqual(ra["heldout_successes"], 0)  # pilot successes not reused
        self.assertEqual(ra["heldout_draws"], 1024)
        self.assertAlmostEqual(ra["signed_discrepancy"], -.5, places=14)
        self.assertEqual(a["tv_lower"], 0)
        self.assertEqual(a["decision"], "insufficient_evidence")
        self.assertGreater(b["tv_lower"], .4)

    def test_empty_event_and_full_event_do_not_reject(self):
        samples = np.zeros((1056, 1), dtype=np.uint8)
        out = self.assess(np.eye(2), samples, windows=((0,),))
        self.assertEqual(out["rows"][0]["event_masks"], [])
        self.assertEqual(out["tv_lower"], 0)
        self.assertEqual(out["decision"], "insufficient_evidence")
        self.assertEqual(hoeffding_event_tv_lower(0, 100, 0, alpha=.05)["tv_lower"], 0)
        self.assertEqual(hoeffding_event_tv_lower(100, 100, 1, alpha=.05)["tv_lower"], 0)

    def test_zero_target_support_is_detected_without_logq(self):
        samples = np.ones((1056, 1), dtype=np.uint8)
        out = self.assess(np.eye(2), samples, windows=((0,),))
        self.assertEqual(out["rows"][0]["target_event_probability"], 0)
        self.assertGreater(out["tv_lower"], .9)
        self.assertEqual(out["decision"], "rejected")

    def test_xxpp_and_ordered_sparse_window_select_the_correct_modes(self):
        # Distinct thermal means distinguish x/p indexing and mode ordering.
        # Marginal ((2,0)) has no-click probabilities 1/4 and 1/2.
        V = np.diag([3., 5., 7., 3., 5., 7.])
        samples = np.tile([[0, 0, 1]], (1056, 1)).astype(np.uint8)
        out = self.assess(V, samples, windows=((2, 0),))
        row = out["rows"][0]
        np.testing.assert_allclose(row["target_probabilities"], [.125, .375, .125, .375],
                                   atol=1e-14, rtol=0)
        self.assertEqual(row["event_masks"], [1])
        self.assertEqual(row["heldout_successes"], 1024)
        self.assertAlmostEqual(row["target_event_probability"], .375, places=14)

    def test_matched_singles_wrong_pair_structure_is_detectable(self):
        # The paired fixture at parameter .5 has P(00)=P(11)=.5.
        # Independent uniform bits match both single-mode means but TV=.5.
        patterns = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
        samples = np.tile(patterns, (264, 1))
        out = self.assess(paired_covariance(1, .5), samples, windows=((0, 1),))
        self.assertEqual(out["rows"][0]["event_masks"], [1, 2])
        self.assertAlmostEqual(out["rows"][0]["target_event_probability"], 0, places=13)
        self.assertAlmostEqual(out["rows"][0]["heldout_event_probability"], .5, places=14)
        self.assertGreater(out["tv_lower"], .45)

    def test_multiple_windows_receive_full_bonferroni_penalty(self):
        samples = np.ones((1056, 2), dtype=np.uint8)
        out = self.assess(np.eye(4)*3, samples, windows=((0,), (1,)))
        self.assertEqual(out["planned_windows"], 2)
        self.assertAlmostEqual(out["confidence_allocation"]["per_window_alpha"], .025)
        for row in out["rows"]:
            self.assertAlmostEqual(row["one_sided_hoeffding_radius"],
                                   math.sqrt(math.log(40)/2048), places=14)
        self.assertFalse(out["confidence_allocation"]["unused_alpha_reallocated"])

    def test_failed_target_window_does_not_reallocate_its_alpha(self):
        original = ThresholdModel.marginal
        def fail_one(model, modes):
            if tuple(modes) == (1,):
                raise NumericalProbabilityError("intentional determinant failure")
            return original(model, modes)
        samples = np.ones((1056, 2), dtype=np.uint8)
        with patch.object(ThresholdModel, "marginal", fail_one):
            out = self.assess(np.eye(4)*3, samples, windows=((0,), (1,)))
        self.assertEqual(out["status"], "partial_numerical_failure")
        self.assertEqual(out["evaluated_windows"], 1)
        self.assertEqual(len(out["numerical_failures"]), 1)
        self.assertEqual(out["rows"][0]["alpha"], .025)
        self.assertEqual(out["decision"], "rejected")

    def test_finite_sample_coverage_under_pilot_adaptation_by_exact_enumeration(self):
        # Enumerate the complete binomial pilot/test experiment, not a Monte
        # Carlo test that might itself fail randomly.  The event can be either
        # {0}, {1}, or empty, selected using only pilot frequency minus P.
        alpha, npilot, ntest = .1, 2, 10
        for target, candidate in ((.5, .5), (.2, .2), (.4, .7), (.8, .3)):
            true_tv = abs(target-candidate)
            false_lower_mass = 0.0
            for kp in range(npilot+1):
                pilot_mass = math.comb(npilot, kp)*candidate**kp*(1-candidate)**(npilot-kp)
                frequency = kp/npilot
                if frequency == target:
                    continue
                choose_one = frequency > target
                target_event = target if choose_one else 1-target
                for kt in range(ntest+1):
                    test_mass = math.comb(ntest, kt)*candidate**kt*(1-candidate)**(ntest-kt)
                    successes = kt if choose_one else ntest-kt
                    lower = hoeffding_event_tv_lower(successes, ntest, target_event,
                                                     alpha=alpha)["tv_lower"]
                    if lower > true_tv+1e-14:
                        false_lower_mass += pilot_mass*test_mass
            self.assertLessEqual(false_lower_mass, alpha+1e-14)

    def test_no_protocol_means_no_statistical_bound(self):
        out = assess_marginal_witness(np.eye(2), np.ones((1056, 1)),
                                      windows=((0,),), pilot_size=32)
        self.assertFalse(out["statistical_protocol_valid"])
        self.assertEqual(out["decision"], "insufficient_evidence")
        self.assertIsNone(out["tv_lower"])
        self.assertIsNone(out["rows"][0]["tv_lower"])

    def test_protocol_blocks_changed_split_windows_retries_and_false_counts(self):
        samples = np.ones((1056, 2), dtype=np.uint8)
        windows = ((0,),)
        base = fixed_split_protocol(len(samples), 32, windows=windows)
        for key, value in (("pilot_size", 64), ("split_fixed_before_samples", False),
                           ("windows_sha256", "wrong"), ("first_batch_no_retry", False),
                           ("iid_attempts", False), ("pilot_not_used_to_modify_sampler", False),
                           ("requested_draws", float(len(samples))),
                           ("split_rule", "overlapping_pilot_and_test")):
            protocol = {**base, key: value}
            out = assess_marginal_witness(np.eye(4), samples, windows=windows,
                                          pilot_size=32, protocol=protocol)
            self.assertEqual(out["decision"], "invalid_evidence", key)
            self.assertIsNone(out["tv_lower"], key)
            self.assertFalse(out["statistical_protocol_valid"], key)

    def test_incomplete_batch_and_observed_failure_never_get_a_bound(self):
        samples = np.ones((1056, 1), dtype=np.uint8)
        protocol = fixed_split_protocol(2048, 32, windows=((0,),))
        out = assess_marginal_witness(np.eye(2), samples, windows=((0,),),
                                      pilot_size=32, protocol=protocol)
        self.assertEqual(out["decision"], "evaluation_failed")
        self.assertEqual(out["status"], "incomplete_batch")
        self.assertIsNone(out["tv_lower"])
        failed = assess_marginal_witness(np.eye(2), samples, windows=((0,),),
                                         pilot_size=32, sampling_failure={"reason": "numerical abort"})
        self.assertEqual(failed["decision"], "evaluation_failed")
        self.assertIsNone(failed["tv_lower"])
        self.assertFalse(failed["conditional_on_success_coverage_claimed"])
        self.assertFalse(failed["unseen_failure_mass_assumed_zero"])

    def test_invalid_samples_windows_models_and_statistical_arguments(self):
        samples = np.ones((64, 2), dtype=np.uint8)
        for windows in ((), ((0, 0),), ((2,),), ((0, 1), (1, 0)), ((True,),)):
            out = assess_marginal_witness(np.eye(4), samples, windows=windows, pilot_size=16)
            self.assertEqual(out["decision"], "invalid_evidence")
        for bad in (np.ones((64, 2))*2, np.ones((64, 1)), np.ones((16, 2))):
            out = assess_marginal_witness(np.eye(4), bad, windows=((0,),), pilot_size=16)
            self.assertEqual(out["decision"], "invalid_evidence")
        invalid = assess_marginal_witness(np.eye(4)*.5, samples, windows=((0,),), pilot_size=16)
        self.assertEqual(invalid["decision"], "invalid_model")
        for alpha in (0, 1, True, float("nan"), -1):
            with self.assertRaises(ValueError):
                hoeffding_event_tv_lower(0, 1, .5, alpha=alpha)
        for arguments in ((True, 10, .5), (11, 10, .5), (0, 0, .5), (0, 1, 1.1)):
            with self.assertRaises(ValueError):
                hoeffding_event_tv_lower(*arguments, alpha=.05)
        with self.assertRaises(ValueError):
            fixed_split_protocol(10, 10, windows=((0,),))
        with self.assertRaises(ValueError):
            fixed_split_protocol(10, True, windows=((0,),))

    def test_deterministic_windows_cover_contiguous_and_sparse_coordinates(self):
        self.assertEqual(default_marginal_windows(8), (tuple(range(8)),))
        self.assertEqual(default_marginal_windows(1), ((0,),))
        windows = default_marginal_windows(32)
        self.assertEqual(windows[:3], (tuple(range(8)), tuple(range(12, 20)), tuple(range(24, 32))))
        self.assertEqual(windows[3], (0, 4, 8, 13, 17, 22, 26, 31))
        self.assertEqual(len(set(windows)), len(windows))
        for window in windows:
            self.assertEqual(len(set(window)), 8)


if __name__ == "__main__":
    unittest.main()
