"""Prevent false experimental equivalence and unearned speedup claims."""

import copy
import unittest

from gbslab.jiuzhang_gap import assess_jiuzhang_gap, load_reference_catalog
from gbslab.gbs_identity import ConfidencePlan, fixed_batch_protocol


def bridge_fixture():
    catalog = load_reference_catalog()
    instance = catalog["targets"]["jiuzhang1"]["instances"]["jiuzhang1_author_cov0"]
    digest = instance["covariance_sha256"]
    provenance = {"executable_sha256": "fixture-binary", "selector": "fixture-exact",
                  "k": 4, "order": list(range(100)), "vacuum_kernel": "log",
                  "covariance_sha256": digest}
    run = {"run_id": "test-fixture-not-measured", "status": "ok", "synthetic": False,
           "instance_id": "jiuzhang1_author_cov0", "covariance_sha256": digest,
           "modes": 100, "output_bins": 100, "source_count": 50,
           "detector_model": "threshold", "output_encoding": "binary_threshold_bins",
           "noise_features": ["mode_dependent_loss"], "implementation": {"maximum_modes": 256},
           "sampling": {"count_requested": 1024, "count_observed": 1024,
                        "iid": True, "no_retries": True, "postselection": "none"},
           "timings": {"setup_seconds": 1., "sampling_seconds": 9., "wall_seconds": 11.,
                       "scope": "fixture-complete-wall"},
           "resources": {"cpu": "fixture", "threads": 1, "gpus": 0, "peak_memory_bytes": 1000000},
           "native_provenance": provenance}
    protocol = fixed_batch_protocol(1024, source="unit-test fixture")
    evidence = {"target": {"covariance_sha256": digest, "modes": 100, "detector_model": "threshold"},
                "model_check": {"valid": True}, "decision": "accepted_within_tolerance_conditional",
                "evidence_scope": "global_tv_upper",
                "statistical_protocol_valid": True, "sampling_protocol": protocol,
                "confidence_plan": ConfidencePlan(("fixture",), .02).allocation("fixture"),
                "global_tv": {"upper": 0.08, "alpha": 0.01, "n_requested": 1024, "n_observed": 1024,
                              "output_space": "binary_patterns_plus_failure",
                              "fixed_iid_no_retry_assumptions": copy.deepcopy(protocol),
                              "strict_floating_point_certificate": False},
                "native_provenance": copy.deepcopy(provenance)}
    return run, evidence, catalog


def matched_timing_fixture():
    run, evidence, catalog = bridge_fixture()
    reference = catalog["targets"]["jiuzhang1"]["instances"][run["instance_id"]]
    reference["matched_timing_reference"] = {
        "sample_count": 1024, "timing_scope": "fixture-complete-wall", "wall_seconds": 22.,
        "postselection": "none", "target_kind": "published_gaussian_model",
        "covariance_sha256": run["covariance_sha256"],
        "note": "unit-test fixture, not an experimental result"}
    return run, evidence, catalog


class JiuzhangGapTests(unittest.TestCase):
    def assess(self, run, evidence=None, **kwargs):
        return assess_jiuzhang_gap(run, evidence, reference_id="jiuzhang1", **kwargs)

    def test_catalog_tracks_latest_and_detector_units_separately(self):
        c = load_reference_catalog()
        self.assertEqual(c["latest_verified_published_reference"], "jiuzhang4_l1024")
        j4 = c["targets"]["jiuzhang4_l1024"]
        self.assertEqual((j4["squeezed_input_states"], j4["optical_output_modes"],
                          j4["measurement"]["hardware_detector_channels"]), (1024, 8176, 16))
        j3 = c["targets"]["jiuzhang3"]
        self.assertEqual(j3["measurement"]["binary_detection_bins"], 1152)
        self.assertFalse(j3["reported_detection_statistics"]["true_PNR_photons"])

    def test_equal_144_mode_threshold_is_not_jiuzhang3_ppnr(self):
        run, evidence, _ = bridge_fixture()
        run.update(modes=144, output_bins=144)
        r = assess_jiuzhang_gap(run, evidence, reference_id="jiuzhang3")
        self.assertEqual(r["gates"]["measurement_alphabet"]["status"], "fail")
        self.assertEqual(r["gates"]["full_output_extent"]["status"], "fail")
        self.assertIsNone(r["timing"]["classical_wall_over_reference_wall"])

    def test_expanded_ppnr_bins_need_explicit_grouping_and_capacity(self):
        run, _, _ = bridge_fixture()
        run.update(modes=1152, output_bins=1152, fanout_grouping="8 binary bins per main optical mode")
        r = assess_jiuzhang_gap(run, reference_id="jiuzhang3")
        self.assertEqual(r["gates"]["measurement_alphabet"]["status"], "pass")
        self.assertEqual(r["gates"]["full_output_extent"]["status"], "pass")
        self.assertEqual(r["gates"]["implementation_capacity"]["status"], "fail")
        run.pop("fanout_grouping")
        r = assess_jiuzhang_gap(run, reference_id="jiuzhang3")
        self.assertEqual(r["gates"]["measurement_alphabet"]["status"], "fail")

    def test_synthetic_circuit_fails_even_with_same_counts_and_accepted_quality(self):
        run, evidence, _ = bridge_fixture()
        run["synthetic"] = True
        r = self.assess(run, evidence)
        self.assertEqual(r["gates"]["physical_instance"]["status"], "fail")
        self.assertEqual(r["local_distribution_quality"]["status"], "pass")
        self.assertEqual(r["gates"]["distribution_quality"]["status"], "unknown")
        self.assertEqual(r["verdict"], "different_or_incomplete_task")
        self.assertIsNone(r["timing"]["classical_wall_over_reference_wall"])

    def test_registered_covariance_is_partial_model_progress_not_hardware_proof(self):
        run, evidence, _ = bridge_fixture()
        model = self.assess(run, evidence)
        hardware = self.assess(run, evidence, target_kind="empirical_hardware_samples")
        self.assertEqual(model["gates"]["physical_instance"]["status"], "pass")
        self.assertEqual(model["gates"]["distribution_quality"]["status"], "pass")
        self.assertEqual(hardware["gates"]["physical_instance"]["status"], "unknown")
        self.assertEqual(hardware["gates"]["distribution_quality"]["status"], "unknown")
        self.assertFalse(model["timing"]["comparison_authorized_by_evidence"])

    def test_wrong_covariance_cannot_transfer_quality(self):
        run, evidence, _ = bridge_fixture()
        evidence["target"]["covariance_sha256"] = "another-target"
        self.assertEqual(self.assess(run, evidence)["gates"]["distribution_quality"]["status"], "fail")

    def test_different_numerical_kernel_cannot_transfer_quality(self):
        run, evidence, _ = bridge_fixture()
        evidence["native_provenance"]["vacuum_kernel"] = "product"
        self.assertEqual(self.assess(run, evidence)["gates"]["distribution_quality"]["status"], "fail")
        evidence.pop("native_provenance")
        self.assertEqual(self.assess(run, evidence)["gates"]["distribution_quality"]["status"], "unknown")

    def test_witnesses_do_not_accept_global_closeness(self):
        run, evidence, _ = bridge_fixture()
        evidence.update(evidence_scope="witnesses_only", decision="insufficient_evidence")
        evidence["witnesses"] = {"all_passed": True}
        self.assertEqual(self.assess(run, evidence)["gates"]["distribution_quality"]["status"], "unknown")

    def test_loose_upper_bound_is_inconclusive_not_a_rejection(self):
        run, evidence, _ = bridge_fixture()
        evidence["global_tv"]["upper"] = 0.5
        evidence["decision"] = "insufficient_evidence"
        self.assertEqual(self.assess(run, evidence)["gates"]["distribution_quality"]["status"], "unknown")
        evidence.update(evidence_scope="full_distribution", full_law={"tv": 0.5}, decision="rejected")
        self.assertEqual(self.assess(run, evidence)["gates"]["distribution_quality"]["status"], "fail")

    def test_partial_distinguishability_and_frontier_capacity_are_explicit(self):
        run, _, _ = bridge_fixture()
        r = assess_jiuzhang_gap(run)
        self.assertEqual(r["reference_id"], "jiuzhang4_l1024")
        self.assertIn("partial_distinguishability", r["gates"]["imperfections"]["required"])
        self.assertEqual(r["gates"]["implementation_capacity"]["status"], "fail")
        self.assertEqual(r["scale"]["dense_float64_covariance_bytes_at_target_extent"], 32*8176**2)

    def test_incomplete_batches_and_failures_are_not_dropped(self):
        run, evidence, _ = bridge_fixture()
        run["sampling"]["count_observed"] = 1023
        evidence["global_tv"]["n_observed"] = 1023
        r = self.assess(run, evidence)
        self.assertEqual(r["gates"]["sampling_protocol"]["status"], "fail")
        self.assertEqual(r["gates"]["distribution_quality"]["status"], "fail")
        evidence = {"decision": "evaluation_failed"}
        self.assertEqual(self.assess(run, evidence)["gates"]["distribution_quality"]["status"], "fail")

    def test_no_speed_ratio_until_every_gate_passes(self):
        run, evidence, catalog = matched_timing_fixture()
        r = self.assess(run, evidence, catalog=catalog)
        self.assertEqual(r["verdict"], "comparable_for_declared_target_conditionally")
        self.assertEqual(r["timing"]["classical_wall_over_reference_wall"], 0.5)
        run["sampling"]["postselection"] = "43-clicks-only"
        r = self.assess(run, evidence, catalog=catalog)
        self.assertIsNone(r["timing"]["classical_wall_over_reference_wall"])

    def test_invalid_or_missing_statistical_attestation_cannot_transfer_a_global_bound(self):
        for field in ["statistical_protocol_valid", "iid_attempts", "first_batch_no_retry",
                      "no_postselection", "density_is_unconditional_path_mass"]:
            run, evidence, catalog = matched_timing_fixture()
            if field == "statistical_protocol_valid":
                evidence[field] = False
            else:
                evidence["sampling_protocol"][field] = False
            with self.subTest(field=field):
                r = self.assess(run, evidence, catalog=catalog)
                self.assertEqual(r["gates"]["distribution_quality"]["status"], "fail")
                self.assertIsNone(r["timing"]["classical_wall_over_reference_wall"])
        run, evidence, catalog = matched_timing_fixture()
        evidence.pop("statistical_protocol_valid")
        self.assertEqual(self.assess(run, evidence, catalog=catalog)["gates"]["distribution_quality"]["status"], "unknown")

    def test_inconsistent_alpha_or_plan_cannot_transfer_a_global_bound(self):
        for field, value in [("familywise_alpha", .001), ("alpha_global_tv", .00001),
                             ("planned_comparisons", True), ("plan_sha256", "wrong")]:
            run, evidence, catalog = matched_timing_fixture()
            evidence["confidence_plan"][field] = value
            with self.subTest(field=field):
                r = self.assess(run, evidence, catalog=catalog)
                self.assertEqual(r["gates"]["distribution_quality"]["status"], "fail")
                self.assertIsNone(r["timing"]["classical_wall_over_reference_wall"])

    def test_full_enumeration_does_not_require_an_iid_statistical_protocol(self):
        run, evidence, catalog = matched_timing_fixture()
        evidence.update(evidence_scope="full_distribution", full_law={"tv": .08})
        for key in ["statistical_protocol_valid", "sampling_protocol", "confidence_plan", "global_tv"]:
            evidence.pop(key)
        self.assertEqual(self.assess(run, evidence, catalog=catalog)["gates"]["distribution_quality"]["status"], "pass")

    def test_complete_wall_must_cover_serial_stages_and_scope_must_exist(self):
        for wall in [.01, 9.5]:
            run, evidence, catalog = matched_timing_fixture()
            run["timings"]["wall_seconds"] = wall
            with self.subTest(wall=wall):
                r = self.assess(run, evidence, catalog=catalog)
                self.assertEqual(r["gates"]["resource_timing_comparability"]["status"], "fail")
                self.assertIsNone(r["timing"]["classical_wall_over_reference_wall"])
        run, evidence, catalog = matched_timing_fixture()
        del run["timings"]["scope"]
        del catalog["targets"]["jiuzhang1"]["instances"][run["instance_id"]]["matched_timing_reference"]["timing_scope"]
        r = self.assess(run, evidence, catalog=catalog)
        self.assertEqual(r["gates"]["resource_timing_comparability"]["status"], "unknown")
        self.assertIsNone(r["timing"]["classical_wall_over_reference_wall"])

    def test_boolean_or_float_counts_cannot_equal_integer_counts(self):
        for value in [True, 1.0]:
            run, evidence, catalog = matched_timing_fixture()
            run["sampling"].update(count_requested=1, count_observed=value)
            reference = catalog["targets"]["jiuzhang1"]["instances"][run["instance_id"]]["matched_timing_reference"]
            reference["sample_count"] = 1
            with self.subTest(value=value):
                r = self.assess(run, evidence, catalog=catalog)
                self.assertEqual(r["gates"]["sampling_protocol"]["status"], "fail")
                self.assertIsNone(r["timing"]["classical_wall_over_reference_wall"])
        run, evidence, catalog = matched_timing_fixture()
        evidence["global_tv"].update(n_requested=1, n_observed=True)
        self.assertEqual(self.assess(run, evidence, catalog=catalog)["gates"]["distribution_quality"]["status"], "fail")

    def test_missing_resources_and_nonfinite_numbers_cannot_unlock_ratio(self):
        run, evidence, _ = bridge_fixture()
        run["resources"]["peak_memory_bytes"] = None
        r = self.assess(run, evidence)
        self.assertEqual(r["gates"]["resource_timing_comparability"]["status"], "unknown")
        for value in [float("nan"), float("inf"), -0.1, True]:
            with self.subTest(epsilon=value), self.assertRaises(ValueError):
                self.assess(run, evidence, epsilon=value)

    def test_missing_metadata_remains_unknown_not_a_physical_mismatch(self):
        run, evidence, _ = bridge_fixture()
        del run["output_bins"]
        del run["sampling"]["count_requested"]
        del evidence["target"]["covariance_sha256"]
        r = self.assess(run, evidence)
        for gate in ["full_output_extent", "sampling_protocol", "distribution_quality"]:
            self.assertEqual(r["gates"][gate]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
