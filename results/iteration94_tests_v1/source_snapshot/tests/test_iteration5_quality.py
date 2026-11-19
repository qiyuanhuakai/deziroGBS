"""Regression checks for frozen quality evidence, not sampler benchmarks."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import iteration5_quality as quality

from gbslab.frugal_evidence import assess_frugal_samples
from gbslab.gbs_identity import ConfidencePlan, covariance_sha256, fixed_batch_protocol
from gbslab.jiuzhang_gap import assess_jiuzhang_gap


def configuration():
    return json.loads((ROOT / "configs/iteration5_quality.json").read_text())


def generation_only_fixture(modes=8, count=128, bit=0):
    """A supplied-law fixture, never a claim of running the official sampler."""
    V = np.eye(2 * modes)
    native = {
        "status": "ok", "modes": modes, "sample_count": count, "seed": 901,
        "covariance_sha256": covariance_sha256(V), "executable_sha256": "unit-fixture-binary",
        "selector": "official_frugal_double_elision", "k": 5,
        "vacuum_kernel": "upstream", "order": list(range(modes)),
        "raw_samples": [[bit] * modes for _ in range(count)],
        "sample_log_probabilities": None, "log_probability_enabled": False,
        "distribution": None,
        "timings_seconds": {"layout": .1, "target": .2, "sampling": .3, "density": 0},
        "wrapper_wall_seconds": .7, "real_precision": 2, "precision": "float64",
        "sampling_api": "unit fixture, not a timed execution",
    }
    return V, native


def compose_fixture(config):
    target = {"target_id": "m12_lossy", "covariance_sha256": "target-fixture", "modes": 12,
        "detector_model": "threshold", "state_model": "zero_mean_gaussian", "quadrature_order": "xxpp", "hbar": 2}
    original = {"decision": "insufficient_evidence", "epsilon": config["epsilon"],
        "target": target, "samples_sha256": "sample-fixture", "reasons": [],
        "global_tv": {"upper": .5}, "full_law": None,
        "evidence_scope": "global_tv_upper"}
    marginal = {"decision": "insufficient_evidence", "target": deepcopy(target),
        "epsilon": config["epsilon"], "samples_sha256": "sample-fixture",
        "statistical_protocol_valid": True, "tv_lower": .2,
        "alpha": config["marginal_suite_alpha"] / config["planned_comparisons"]}
    return original, marginal


class GenerationOnlyFrugalAssessmentTests(unittest.TestCase):
    def test_default_remains_strict_and_opt_in_never_accepts_from_witnesses(self):
        V, native = generation_only_fixture()
        with self.assertRaisesRegex(ValueError, "recorded log probabilities"):
            assess_frugal_samples(V, native)
        protocol = fixed_batch_protocol(native["sample_count"])
        report = assess_frugal_samples(V, native, require_log_probabilities=False,
            protocol=protocol, plan=ConfidencePlan(("fixture",)), comparison_id="fixture")
        self.assertEqual(report["decision"], "insufficient_evidence")
        self.assertEqual(report["evidence_scope"], "witnesses_only")
        self.assertIsNone(report["global_tv"])
        self.assertFalse(report["density_available"])
        self.assertFalse(report["sampling_protocol"]["density_is_unconditional_path_mass"])
        self.assertTrue(protocol["density_is_unconditional_path_mass"], "Caller protocol was mutated")
        self.assertEqual(report["native_provenance"]["selector"], native["selector"])
        self.assertEqual(report["native_provenance"]["executable_sha256"], native["executable_sha256"])

    def test_opt_in_does_not_mask_inconsistent_density_flags(self):
        V, native = generation_only_fixture()
        for changes in ({"log_probability_enabled": True},
                        {"sample_log_probabilities": [0.] * native["sample_count"]},
                        {"log_probability_enabled": None}):
            with self.subTest(changes=tuple(changes)):
                with self.assertRaises(ValueError):
                    assess_frugal_samples(V, {**native, **changes}, require_log_probabilities=False)
        with self.assertRaisesRegex(ValueError, "boolean"):
            assess_frugal_samples(V, native, require_log_probabilities="no")

    def test_opt_in_retains_target_count_and_failed_attempt_guards(self):
        V, native = generation_only_fixture()
        with self.assertRaisesRegex(ValueError, "covariance"):
            assess_frugal_samples(V * 2, native, require_log_probabilities=False)
        short = deepcopy(native); short["raw_samples"].pop()
        with self.assertRaisesRegex(ValueError, "complete batch"):
            assess_frugal_samples(V, short, require_log_probabilities=False)
        report = assess_frugal_samples(V, native, require_log_probabilities=False,
            protocol=fixed_batch_protocol(native["sample_count"] + 1))
        self.assertEqual(report["decision"], "evaluation_failed")
        report = assess_frugal_samples(V, {**native, "status": "generation_complete"},
            require_log_probabilities=False)
        self.assertEqual(report["decision"], "evaluation_failed")
        self.assertIsNone(report["global_tv"])

    def test_density_free_rejection_reaches_d2_with_matching_sampler_identity(self):
        # Deliberately incorrect supplied output law for a vacuum target.
        V, native = generation_only_fixture(bit=1)
        report = assess_frugal_samples(V, native, require_log_probabilities=False,
            protocol=fixed_batch_protocol(native["sample_count"]),
            plan=ConfidencePlan(("fixture",)), comparison_id="fixture")
        self.assertEqual(report["decision"], "rejected")
        manifest = quality.normalize_native_run("fixture", {"modes": 8}, native,
            native["sample_count"], synthetic=True, maximum_modes=144)
        gap = assess_jiuzhang_gap(manifest, report, reference_id="jiuzhang1",
            target_kind="published_gaussian_model")
        self.assertEqual(gap["local_distribution_quality"]["status"], "fail")
        self.assertIn("rejected", gap["local_distribution_quality"]["reason"])
        self.assertEqual(gap["gates"]["physical_instance"]["status"], "fail")
        self.assertIsNone(gap["timing"]["classical_wall_over_reference_wall"])


class CombinedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.config = configuration()
        self.original, self.marginal = compose_fixture(self.config)

    def test_union_budget_covers_fourteen_comparisons_and_both_channels(self):
        plan, per_marginal = quality.confidence_allocation(self.config)
        self.assertEqual(len(plan.comparison_ids), 14)
        allocation = plan.allocation(plan.comparison_ids[0])
        self.assertAlmostEqual(allocation["alpha_global_tv"], .025 / 28)
        self.assertAlmostEqual(allocation["alpha_witnesses"], .025 / 28)
        self.assertAlmostEqual(per_marginal, .025 / 14)
        self.assertAlmostEqual(14 * (allocation["alpha_global_tv"] + allocation["alpha_witnesses"] + per_marginal), .05)
        excessive = deepcopy(self.config); excessive["marginal_suite_alpha"] = .04
        with self.assertRaisesRegex(ValueError, "familywise alpha"):
            quality.confidence_allocation(excessive)

    def test_marginal_rejects_but_cannot_accept_alone(self):
        result = quality.combine_identity(self.original, self.marginal, self.config)
        self.assertEqual(result["decision"], "rejected")
        self.assertEqual(result["decision_basis"], "held_out_marginal_tv_lower_bound")
        self.marginal["tv_lower"] = 0
        result = quality.combine_identity(self.original, self.marginal, self.config)
        self.assertEqual(result["decision"], "insufficient_evidence")
        self.assertEqual(self.original["decision"], "insufficient_evidence", "Original evidence was mutated")

    def test_wrong_target_sample_protocol_or_alpha_cannot_be_attached(self):
        variants = []
        for key, value in (("covariance_sha256", "other"), ("modes", 8), ("target_id", "other")):
            bad = deepcopy(self.marginal); bad["target"][key] = value; variants.append(bad)
        for key, value in (("samples_sha256", "other"), ("statistical_protocol_valid", False),
                           ("epsilon", .2), ("alpha", .05), ("alpha", None), ("tv_lower", True)):
            bad = deepcopy(self.marginal); bad[key] = value; variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad):
                self.assertEqual(quality.combine_identity(self.original, bad, self.config)["decision"], "invalid_evidence")

    def test_conflicting_lower_bound_cannot_override_global_or_full_evidence(self):
        for updates in ({"global_tv": {"upper": .1}}, {"full_law": {"tv": .1}}):
            result = quality.combine_identity({**self.original, **updates}, self.marginal, self.config)
            self.assertEqual(result["decision"], "invalid_evidence")
            self.assertTrue(result["statistical_evidence_conflict"])

    def test_observed_sampling_failure_is_never_rehabilitated(self):
        result = quality.combine_identity({**self.original, "decision": "evaluation_failed"},
                                          self.marginal, self.config)
        self.assertEqual(result["decision"], "evaluation_failed")


class FrozenQualityRunnerTests(unittest.TestCase):
    def test_materialized_schedule_is_reused_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "quality"
            config = quality.initialize(quality.DEFAULT_CONFIG, output)
            manifest = json.loads((output / "witness_schedule_manifest.json").read_text())
            self.assertTrue(manifest["explicit_indices_materialized_before_all_sampling"])
            self.assertFalse(manifest["sample_values_read_to_choose_schedule"])
            before = (output / "witness_schedule_manifest.json").read_bytes()
            quality.initialize(quality.DEFAULT_CONFIG, output)
            self.assertEqual((output / "witness_schedule_manifest.json").read_bytes(), before)
            schedules = json.loads((output / "witness_schedules.json").read_text())
            self.assertEqual(len(schedules), config["planned_comparisons"])
            schedules["m12_lossy"]["marginal_windows"][0][0] = 11
            quality.save(output / "witness_schedules.json", schedules)
            with self.assertRaisesRegex(ValueError, "Frozen evidence plan"):
                quality.initialize(quality.DEFAULT_CONFIG, output)

    def test_late_materialization_records_existing_attempt_without_backdating(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = configuration()
            (output / "config.json").write_bytes(quality.DEFAULT_CONFIG.read_bytes())
            quality.save(output / "comparison_plan.json", quality.cases(config))
            attempt = output / "jiuzhang1_frugal"
            attempt.mkdir()
            quality.save(attempt / "attempt.json", {"fixture": True})
            quality.initialize(quality.DEFAULT_CONFIG, output)
            manifest = json.loads((output / "witness_schedule_manifest.json").read_text())
            self.assertFalse(manifest["explicit_indices_materialized_before_all_sampling"])
            self.assertEqual(manifest["sampling_attempts_already_started"], ["jiuzhang1_frugal"])
            self.assertEqual(manifest["completed_batch_artifacts_present"], [])

    def test_synthetic_evaluation_uses_supported_d2_kind_and_preserves_real_instance_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "quality"
            config = quality.initialize(quality.DEFAULT_CONFIG, output)
            case = quality.cases(config)[0]
            V, native = generation_only_fixture(case["modes"], case["count"])
            native.update(seed=case["seed"], selector="click_influence", k=4, vacuum_kernel="product",
                log_probability_enabled=True, sample_log_probabilities=[0.] * case["count"],
                timings_seconds={"setup": .1, "sampling": .3})
            destination = output / case["id"]; destination.mkdir()
            quality.save(destination / "native.json", native)
            quality.save(destination / "attempt.json", {"case": case,
                "config_sha256": quality.digest(output / "config.json"),
                "covariance_sha256": covariance_sha256(V), "executable_sha256": native["executable_sha256"]})
            quality.save(destination / "build.json", {"executable_sha256": native["executable_sha256"]})
            quality.save(destination / "completion.json", {"status": "ok",
                "native_sha256": quality.digest(destination / "native.json")})
            with patch.object(quality, "covariance", return_value=V), redirect_stdout(io.StringIO()):
                quality.evaluate(config, output)
            gap = json.loads((destination / "gap_jiuzhang1.json").read_text())
            self.assertEqual(gap["target_kind"], "published_gaussian_model")
            self.assertEqual(gap["gates"]["physical_instance"]["status"], "fail")
            self.assertEqual(gap["verdict"], "different_or_incomplete_task")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summary["rows"]), 14)
            native["seed"] += 1
            quality.save(destination / "native.json", native)
            with self.assertRaisesRegex(ValueError, "Archived native batch"):
                quality.load_completed_attempt(config, case, output, V)
            quality.save(destination / "completion.json", {"status": "ok",
                "native_sha256": quality.digest(destination / "native.json")})
            with self.assertRaisesRegex(ValueError, "fixed executable, model, count, or seed"):
                quality.load_completed_attempt(config, case, output, V)

    def test_wrong_binary_validation_is_rejected_before_any_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            binary = directory / "sampler"; binary.write_bytes(b"unit fixture, not executable")
            sha = hashlib.sha256(binary.read_bytes()).hexdigest()
            quality.save(directory / "build.json", {"executable_sha256": sha})
            gate = directory / "validation.json"
            quality.save(gate, {"passed": True, "executable_sha256": "different-binary"})
            with patch.object(quality, "run_cpp_click") as run:
                with self.assertRaisesRegex(ValueError, "different executable"):
                    quality.validate_binary_gate(binary, "candidate", validation_path=gate)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
