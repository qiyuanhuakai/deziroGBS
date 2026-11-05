"""Meaningful acceptance, rejection, provenance, and failure-space checks."""
from pathlib import Path
import math
import unittest

import numpy as np

from gbslab.cpp_click import CppClickError, find_cpp_executable, run_cpp_click
from gbslab.exact import NumericalProbabilityError, ThresholdModel
from gbslab.fock import threshold_from_fock
from gbslab.gaussian import covariance_from_interferometer, haar_unitary
from gbslab.gbs_identity import (
    ConfidencePlan, assess_gbs_identity, assess_native_samples,
    default_witness_subsets, fixed_batch_protocol, sample_tv_upper,
)
from gbslab.paired_stress import dense_target, paired_covariance
from gbslab.tv_validation import tv_score_from_log_probabilities


ROOT = Path(__file__).resolve().parents[1]


class IdentityEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.plan = ConfidencePlan(("positive", "impostor", "unresolved", "witness_only"), .05)
        self.V = paired_covariance(1, .5)
        self.p = dense_target(1, .5)

    def test_target_physicality_and_detector_scope_are_separate(self):
        invalid = assess_gbs_identity(np.eye(4)*.5)
        self.assertEqual(invalid["decision"], "invalid_model")
        pnr = assess_gbs_identity(self.V, detector_model="pnr")
        self.assertEqual(pnr["decision"], "out_of_scope")
        self.assertIsNone(pnr["model_check"]["valid"])
        displaced = assess_gbs_identity(self.V, displacement=[.1, 0, 0, 0])
        self.assertEqual(displaced["decision"], "out_of_scope")
        supplied = assess_gbs_identity(self.V)
        self.assertTrue(supplied["model_check"]["valid"])
        self.assertEqual(supplied["decision"], "insufficient_evidence")

    def test_complete_physical_law_accepts_but_matched_mean_impostor_rejects(self):
        positive = assess_gbs_identity(self.V, candidate_distribution=self.p, epsilon=.01)
        independent = np.full(4, .25)  # also a different, thermal Gaussian target
        negative = assess_gbs_identity(self.V, candidate_distribution=independent, epsilon=.1)
        self.assertEqual(positive["decision"], "accepted_within_tolerance_conditional")
        self.assertLess(positive["full_law"]["tv"], 1e-12)
        self.assertEqual(negative["decision"], "rejected")
        self.assertAlmostEqual(negative["full_law"]["tv"], .5, places=12)
        for mode in range(2):
            indicator = (np.arange(4) >> mode) & 1
            self.assertEqual(float(self.p@indicator), float(independent@indicator))

    def test_pair_witness_rejects_impostor_with_correct_single_means(self):
        # Balanced, deterministic fixture used only for computing the statistic;
        # its iid protocol is a declared mathematical assumption in this test.
        samples = np.tile([[0,0],[1,0],[0,1],[1,1]], (1024,1))
        result = assess_gbs_identity(self.V, samples=samples, epsilon=.1, plan=self.plan,
            comparison_id="impostor", protocol=fixed_batch_protocol(len(samples)),
            witness_subsets=((0,), (1,), (0,1)))
        self.assertEqual(result["decision"], "rejected")
        self.assertEqual(result["evidence_scope"], "witnesses_only")
        singles = [r for r in result["witnesses"]["rows"] if r.get("modes") in ([0],[1])]
        self.assertTrue(all(abs(r["absolute_difference"]) < 1e-12 for r in singles))
        self.assertGreater(result["witnesses"]["tv_lower"], .1)

    def test_passing_witnesses_never_promotes_to_global_acceptance(self):
        samples = np.tile([[0,0],[1,1]], (1024,1))
        result = assess_gbs_identity(self.V, samples=samples, epsilon=.1,
            plan=self.plan, comparison_id="witness_only", protocol=fixed_batch_protocol(len(samples)))
        self.assertEqual(result["decision"], "insufficient_evidence")
        self.assertEqual(result["witnesses"]["tv_lower"], 0)
        self.assertIsNone(result["global_tv"])

    def test_support_outside_q_is_included_by_q_sample_identity(self):
        # P=(.2,.8), Q=(1,0), so TV=.8 even though Q never visits target bit1.
        samples = np.zeros((2048,1), dtype=np.uint8)
        result = sample_tv_upper(samples, np.zeros(len(samples)), lambda x: math.log(.2),
                                 alpha=.01, max_exact_clicks=1)
        self.assertAlmostEqual(result["mean_score"], .8, places=12)
        self.assertGreaterEqual(result["upper"], .8)

    def test_failure_inclusive_identity_allows_subnormalized_success_mass(self):
        p = np.array([.2,.8])
        q_success = np.array([.4,.1])
        failure = 1-float(q_success.sum())
        direct = .5*(float(np.abs(p-q_success).sum())+failure)
        expected = failure+sum(q*tv_score_from_log_probabilities(math.log(pp), math.log(q))
                               for pp,q in zip(p,q_success))
        self.assertAlmostEqual(direct, expected, places=14)
        conditional_tv = .5*float(np.abs(p-q_success/(1-failure)).sum())
        self.assertLessEqual(conditional_tv, direct)
        # A false acceptance after a first all-success batch is rare under the
        # unconditional process; conditioning/retrying until success is invalid.
        n, alpha, failure = 256, .01, .3
        bound = sample_tv_upper(np.zeros((n,1),dtype=np.uint8), np.full(n,math.log(1-failure)),
                                lambda x: 0.0, alpha=alpha, max_exact_clicks=0)
        self.assertLess(bound["upper"], failure)
        self.assertLess((1-failure)**n, alpha)
        self.assertFalse(bound["conditional_on_success_coverage_claimed"])
        self.assertFalse(bound["unseen_failure_mass_assumed_zero"])

    def test_high_click_and_unresolved_target_records_stay_in_denominator(self):
        samples = np.array([[1,1],[1,0],[0,1],[0,0]],dtype=np.uint8)
        def target(sample):
            if sample[0] and not sample[1]:
                raise NumericalProbabilityError("deliberate unresolved point")
            return math.log(.25)
        result = sample_tv_upper(samples, np.full(4,math.log(.25)), target,
                                 alpha=.05,max_exact_clicks=1,keep_scores=True)
        self.assertEqual(result["scores"], [1,1,0,0])
        self.assertEqual(result["cutoff_count"], 1)
        self.assertEqual(result["target_probability_failure_count"], 1)
        self.assertEqual(result["mean_score"], .5)
        self.assertEqual(result["n_observed"], 4)

    def test_complete_law_and_partial_density_normalization_are_checked(self):
        bad = assess_gbs_identity(self.V, candidate_distribution=np.full(4,.2))
        self.assertEqual(bad["decision"], "invalid_evidence")
        patterns = np.array([[0,0],[1,0],[0,1],[1,1]])
        with self.assertRaisesRegex(ValueError, "exceed one"):
            sample_tv_upper(patterns,np.full(4,math.log(.4)),lambda x:math.log(.25),alpha=.05)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            sample_tv_upper(np.zeros((2,1)),[-.1,-.2],lambda x:-.1,alpha=.05)

    def test_logged_density_must_match_declared_full_q(self):
        samples = np.array([[0,0],[1,1]])
        result = assess_gbs_identity(self.V, candidate_distribution=self.p, samples=samples,
            sample_log_probabilities=[math.log(.4),math.log(.4)])
        self.assertEqual(result["decision"], "invalid_evidence")

    def test_witness_conflict_prevents_declared_q_from_masking_wrong_samples(self):
        samples = np.ones((2048,2),dtype=np.uint8)
        result = assess_gbs_identity(self.V,candidate_distribution=self.p,samples=samples,
            sample_log_probabilities=np.full(len(samples),math.log(.5)),epsilon=.1,
            plan=self.plan,comparison_id="impostor",protocol=fixed_batch_protocol(len(samples)),max_exact_clicks=0)
        self.assertEqual(result["decision"],"invalid_evidence")
        self.assertIn("sample_law_consistency_warning",result)
        self.assertEqual(result["global_tv"]["upper"],1)

    def test_missing_or_retry_protocol_cannot_produce_statistical_acceptance(self):
        samples = np.tile([[0,0],[1,1]],(1024,1))
        kwargs = dict(samples=samples,sample_log_probabilities=np.full(len(samples),math.log(.5)),
                      epsilon=.1,plan=self.plan,comparison_id="positive")
        absent = assess_gbs_identity(self.V,**kwargs)
        self.assertEqual(absent["decision"],"insufficient_evidence")
        protocol=fixed_batch_protocol(len(samples));protocol["first_batch_no_retry"]=False
        retried=assess_gbs_identity(self.V,protocol=protocol,**kwargs)
        self.assertEqual(retried["decision"],"insufficient_evidence")
        self.assertIsNone(retried["global_tv"])

    def test_sampling_failure_aborts_even_when_a_good_full_q_is_supplied(self):
        result=assess_gbs_identity(self.V,candidate_distribution=self.p,
            sampling_failure={"error":"native numerical guard after13 attempts"})
        self.assertEqual(result["decision"],"evaluation_failed")
        self.assertIsNone(result["global_tv"])

    def test_familywise_schedule_and_witness_budget_are_fixed(self):
        allocation=self.plan.allocation("positive")
        self.assertEqual(allocation["alpha_global_tv"],.05/4/2)
        self.assertEqual(allocation["alpha_witnesses"],.05/4/2)
        self.assertEqual(self.plan.as_dict()["plan_sha256"],self.plan.as_dict()["plan_sha256"])
        with self.assertRaises(ValueError):self.plan.allocation("chosen_after_results")
        with self.assertRaises(ValueError):ConfidencePlan(("repeat","repeat"))
        self.assertEqual(default_witness_subsets(8),default_witness_subsets(8))
        self.assertIn((0,4),default_witness_subsets(8))

    def test_independent_fock_reference_preserves_truncation_scope(self):
        U=haar_unitary(2,np.random.default_rng(12203));r=np.array([.25,.35]);eta=.7
        V=covariance_from_interferometer(U,r,eta)
        reference=threshold_from_fock(U,r,eta,max_photons=12)
        result=assess_gbs_identity(V,candidate_distribution=reference.normalized_distribution,epsilon=1e-5)
        self.assertEqual(result["decision"],"accepted_within_tolerance_conditional")
        self.assertLessEqual(result["full_law"]["tv"],reference.omitted_mass+1e-12)
        self.assertGreater(reference.omitted_mass,0)


class NativeIdentityFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executable=find_cpp_executable(ROOT/".cache/click-iteration4-native/click_sampler")
        if cls.executable is None:raise unittest.SkipTest("iteration4 executable not built; tests do not compile")

    def test_native_log_q_matches_full_law_trace_and_binds_target(self):
        V=covariance_from_interferometer(haar_unitary(3,np.random.default_rng(12204)),.55,.65)
        plan=ConfidencePlan(("native_log","native_product"),.05)
        for kernel in ["log","product"]:
            out=run_cpp_click(V,selection="click_influence",max_clicks=2,samples=128,seed=12205,
                enumerate_distribution=True,record_log_probabilities=True,vacuum_kernel=kernel,
                trace_conditionals=True,executable=self.executable)
            result=assess_native_samples(V,out,epsilon=.01,plan=plan,comparison_id="native_"+kernel,
                protocol=fixed_batch_protocol(128),max_exact_clicks=3)
            self.assertEqual(result["decision"],"accepted_within_tolerance_conditional")
            self.assertLess(result["full_law"]["tv"],1e-11)
            self.assertEqual(result["native_provenance"]["vacuum_kernel"],kernel)
            with self.assertRaisesRegex(ValueError,"covariance"):
                assess_native_samples(V+np.eye(6)*.01,out)

    def test_native_actual_failure_has_no_quality_acceptance(self):
        result=assess_native_samples(np.eye(4),{"status":"error","error":"numerical probability"})
        self.assertEqual(result["decision"],"evaluation_failed")
        self.assertIsNone(result["global_tv"])


if __name__=="__main__":unittest.main()
