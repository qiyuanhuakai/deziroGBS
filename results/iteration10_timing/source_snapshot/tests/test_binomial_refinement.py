"""Exact finite-binomial and frozen-evidence checks for secondary bounds."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import binom, binomtest

from gbslab.binomial_refinement import (
    RefinementError, clopper_pearson_event_tv_lower, refine_frozen_marginal_report,
    refine_archive,
)
from gbslab.marginal_witness import assess_marginal_witness, fixed_split_protocol


class ExactBinomialBoundTests(unittest.TestCase):
    def test_boundary_counts_match_analytic_formulas(self):
        for n in (1, 7, 768):
            for alpha in (.05, .025/14/4):
                empty = clopper_pearson_event_tv_lower(0, n, 0, alpha=alpha)
                self.assertEqual(empty["q_event_lower"], 0)
                self.assertEqual(empty["tv_lower"], 0)
                self.assertIsNone(empty["binomial_tail_at_quantile"])
                full = clopper_pearson_event_tv_lower(n, n, 1, alpha=alpha)
                self.assertAlmostEqual(full["q_event_lower"], alpha**(1/n), 14)
                self.assertEqual(full["tv_lower"], 0)

    def test_quantiles_match_independent_exact_binomial_inversion(self):
        for k, n, alpha in ((1, 20, .05), (19, 20, .001),
                             (448, 768, .025/14/4), (0, 100, .05), (100, 100, .05)):
            result = clopper_pearson_event_tv_lower(k, n, .25, alpha=alpha)
            reference = binomtest(k, n, alternative="greater").proportion_ci(
                confidence_level=1-alpha, method="exact").low
            self.assertLessEqual(abs(result["q_event_lower"]-reference), 3e-12)

    def test_cp_dominates_hoeffding_at_every_count(self):
        # Counts are enumerated, not sampled. This checks implementation
        # against the analytic pointwise-order argument in the module.
        for n in (1, 2, 5, 13, 50, 768):
            for alpha in (.2, .05, .025/14/4, 1e-10):
                for k in range(n+1):
                    result = clopper_pearson_event_tv_lower(k, n, .37, alpha=alpha)
                    self.assertGreaterEqual(result["q_event_lower"], result["primary_hoeffding_q_lower"])
                    self.assertGreaterEqual(result["tv_lower"], result["primary_hoeffding_tv_lower"])

    def test_exact_binomial_miss_probability_is_within_alpha(self):
        for n in (1, 2, 7, 30):
            for alpha in (.05, .001):
                lower = np.array([clopper_pearson_event_tv_lower(k, n, 0, alpha=alpha)["q_event_lower"]
                                  for k in range(n+1)])
                for true_q in np.linspace(0, 1, 101):
                    mass = binom.pmf(np.arange(n+1), n, true_q)
                    error = float(mass[lower > true_q].sum())
                    self.assertLessEqual(error, alpha+3e-14)

    def test_invalid_arguments_and_failed_inverse_are_rejected(self):
        for k, n, p, alpha in ((True, 10, .2, .05), (11, 10, .2, .05),
                                (0, 0, .2, .05), (1, 10, 1.1, .05),
                                (1, 10, .2, 0), (1, 10, .2, True)):
            with self.assertRaises(RefinementError):
                clopper_pearson_event_tv_lower(k, n, p, alpha=alpha)
        with patch("gbslab.binomial_refinement.beta.ppf", return_value=float("nan")):
            with self.assertRaises(RefinementError):
                clopper_pearson_event_tv_lower(3, 10, .2, alpha=.05)
        with patch("gbslab.binomial_refinement.beta.ppf", return_value=1.0):
            with self.assertRaisesRegex(RefinementError, "unresolved"):
                clopper_pearson_event_tv_lower(10, 10, .2, alpha=np.nextafter(1.0, 0.0))
        with patch("gbslab.binomial_refinement.binom.sf", return_value=.9):
            with self.assertRaisesRegex(RefinementError, "inversion"):
                clopper_pearson_event_tv_lower(3, 10, .2, alpha=.05)


class FrozenMarginalRefinementTests(unittest.TestCase):
    def setUp(self):
        self.covariance = np.eye(4)*2.0
        self.windows = [[0], [1], [0, 1]]
        # Deterministic validation fixture, not a statistical experiment.
        self.samples = np.zeros((72, 2), dtype=np.uint8)
        self.samples[:8] = 1
        self.samples[8:40] = 1
        self.samples[40:56, 0] = 1
        self.protocol = fixed_split_protocol(72, 8, windows=self.windows, source="unit_test_fixture")
        self.report = assess_marginal_witness(self.covariance, self.samples,
            windows=self.windows, pilot_size=8, alpha=.015, epsilon=.1,
            protocol=self.protocol, target_id="fixture")

    def refine(self, report=None, samples=None, covariance=None, **overrides):
        options = dict(expected_windows=self.windows, expected_protocol=self.protocol,
            expected_alpha=.015, expected_epsilon=.1, expected_target_id="fixture")
        options.update(overrides)
        return refine_frozen_marginal_report(self.report if report is None else report,
            self.covariance if covariance is None else covariance,
            self.samples if samples is None else samples, **options)

    def test_frozen_event_target_counts_and_primary_are_preserved(self):
        before = copy.deepcopy(self.report)
        refined = self.refine()
        self.assertEqual(self.report, before)
        self.assertGreaterEqual(refined["tv_lower"], self.report["tv_lower"])
        self.assertFalse(refined["method_pre_registered"])
        self.assertTrue(refined["method_chosen_after_primary_results"])
        self.assertFalse(refined["strict_floating_point_certificate"])
        for old, new in zip(self.report["rows"], refined["rows"]):
            for field in ("modes", "event_masks", "event_sha256", "heldout_successes",
                          "heldout_draws", "alpha", "target_event_probability"):
                self.assertEqual(old[field], new[field], field)
            self.assertEqual(new["target_recheck_max_abs_error"], 0.0)

    def test_sample_covariance_event_and_count_tampering_are_rejected(self):
        changed = self.samples.copy()
        changed[-1, 0] ^= 1
        with self.assertRaisesRegex(RefinementError, "samples_sha256"):
            self.refine(samples=changed)
        with self.assertRaisesRegex(RefinementError, "covariance_sha256"):
            self.refine(covariance=self.covariance*1.01)
        for change in (dict(event_sha256="wrong"), dict(event_masks=[0]),
                       dict(heldout_successes=1), dict(heldout_draws=63),
                       dict(alpha=.0051), dict(pilot_counts=[8, 0]),
                       dict(target_probabilities=[.5, .5])):
            report = copy.deepcopy(self.report)
            report["rows"][0].update(change)
            with self.subTest(change=change), self.assertRaises(RefinementError):
                self.refine(report=report)

    def test_failed_or_incomplete_batches_and_protocol_changes_do_not_gain_bounds(self):
        for change in (dict(status="sampling_failed"), dict(statistical_protocol_valid=False),
                       dict(sampling_failure={"reason": "numerical abort"}),
                       dict(n_observed=71), dict(pilot_size=9),
                       dict(conditional_on_success_coverage_claimed=True),
                       dict(evidence_scope="success_only_events"),
                       dict(failure_symbol_observable_value=1)):
            report = copy.deepcopy(self.report)
            report.update(change)
            with self.subTest(change=change), self.assertRaises(RefinementError):
                self.refine(report=report)
        with self.assertRaises(RefinementError):
            self.refine(samples=self.samples[:-1])
        with self.assertRaises(RefinementError):
            self.refine(expected_alpha=.03)
        protocol = {**self.protocol, "first_batch_no_retry": False}
        with self.assertRaises(RefinementError):
            self.refine(expected_protocol=protocol)

    def test_failed_windows_keep_original_multiplicity_without_replacement(self):
        report = copy.deepcopy(self.report)
        dropped = report["rows"].pop(0)
        report["numerical_failures"] = [dict(window_index=0, modes=[0],
            error_type="NumericalProbabilityError", error="frozen failure")]
        report["evaluated_windows"] = 2
        report["status"] = "partial_numerical_failure"
        report["tv_lower"] = max(row["tv_lower"] for row in report["rows"])
        report["decision"] = "rejected" if report["tv_lower"] > report["epsilon"] else "insufficient_evidence"
        refined = self.refine(report=report)
        self.assertEqual(refined["planned_windows"], 3)
        self.assertEqual(refined["evaluated_windows"], 2)
        self.assertEqual(refined["per_window_alpha"], dropped["alpha"])
        self.assertEqual(refined["unchanged_numerical_failures"], report["numerical_failures"])
        self.assertNotIn(0, [row["window_index"] for row in refined["rows"]])
        report["numerical_failures"].append(copy.deepcopy(report["numerical_failures"][0]))
        with self.assertRaisesRegex(RefinementError, "partition"):
            self.refine(report=report)

    def test_archive_keeps_missing_comparison_budgets_and_primary_bytes(self):
        with tempfile.TemporaryDirectory(prefix="binomial-refinement-") as temporary:
            base = Path(temporary)
            primary = base/"primary"
            primary.mkdir()

            def save(path, data):
                path.write_text(json.dumps(data, indent=2)+"\n")

            def digest(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            profile = dict(name="fixture", squeezing=.5, transmissivity=.5)
            config = dict(synthetic_modes=[2], profiles=[profile], planned_comparisons=3,
                sample_count_by_modes={"2": 72}, pilot_size_by_modes={"2": 8}, sample_seed_base=123,
                familywise_alpha=.05, old_identity_suite_alpha=.005, marginal_suite_alpha=.045,
                epsilon=.1, jiuzhang1=dict(sample_count=72, pilot_size=8, sample_seed=456))
            plan = [dict(id="m2_fixture", modes=2, profile=profile, synthetic=True,
                         method="candidate", count=72, pilot=8, seed=123)]
            plan.extend(dict(id="jiuzhang1_"+method, modes=100, synthetic=False,
                method=method, count=72, pilot=8, seed=456) for method in ("candidate", "frugal"))
            ids = [case["id"] for case in plan]
            save(primary/"config.json", config)
            save(primary/"comparison_plan.json", plan)
            save(primary/"witness_schedules.json", {name:dict(marginal_windows=self.windows,
                marginal_comparison_alpha=.045/3) for name in ids})
            save(primary/"split_protocols.json", {name:dict(marginal=self.protocol) for name in ids})
            save(primary/"confidence_plan.json", dict(comparison_ids=ids, planned_comparisons=3,
                                                       familywise_alpha=.005))
            save(primary/"witness_schedule_manifest.json", dict(config_sha256=digest(primary/"config.json"),
                artifact_sha256={name:digest(primary/name) for name in
                    ("witness_schedules.json", "split_protocols.json", "confidence_plan.json")}))
            case_dir = primary/"m2_fixture"
            case_dir.mkdir()
            report = copy.deepcopy(self.report)
            report["target"]["target_id"] = "m2_fixture"
            save(case_dir/"marginal_witness.json", report)
            np.savez(case_dir/"input.npz", covariance=self.covariance)
            covariance_hash = report["target"]["covariance_sha256"]
            native = dict(status="ok", executable_sha256="a"*64, covariance_sha256=covariance_hash,
                modes=2, sample_count=72, seed=123, raw_samples=self.samples.tolist())
            save(case_dir/"native.json", native)
            save(case_dir/"completion.json", dict(status="ok", native_sha256=digest(case_dir/"native.json")))
            save(case_dir/"attempt.json", dict(case=plan[0], covariance_sha256=covariance_hash,
                executable_sha256="a"*64, config_sha256=digest(primary/"config.json")))
            save(case_dir/"build.json", dict(executable_sha256="a"*64))
            before = {p:digest(p) for p in primary.rglob("*") if p.is_file()}
            result = refine_archive(primary, base/"secondary")
            self.assertEqual(result["planned_comparisons"], 3)
            self.assertEqual(result["rows"][0]["status"], "refined")
            self.assertTrue(all(row["status"] == "primary_report_not_available" for row in result["rows"][1:]))
            self.assertEqual(result["per_comparison_alpha"], .045/3)
            self.assertEqual(result["new_draws"], 0)
            self.assertEqual(before, {p:digest(p) for p in before})
            with self.assertRaises(FileExistsError):
                refine_archive(primary, base/"secondary")
            with self.assertRaises(RefinementError):
                refine_archive(primary, primary/"not_permitted")
            report["samples_sha256"] = "mismatched"
            save(case_dir/"marginal_witness.json", report)
            rejected = refine_archive(primary, base/"tamper_rejection")
            self.assertEqual(rejected["rows"][0]["status"], "not_refined")
            self.assertIsNone(rejected["rows"][0]["tv_lower"])


if __name__ == "__main__":
    unittest.main()
