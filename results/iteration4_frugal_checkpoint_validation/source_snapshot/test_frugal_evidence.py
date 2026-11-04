"""Official-sampler identity, clipped-law density, and evidence-binding checks."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gbslab.frugal_evidence import (
    CHECKPOINT_EXECUTABLE, DEFAULT_EXECUTABLE, FrugalEvidenceError, assess_frugal_samples,
    frugal_evidence_sample, frugal_storage_lower_bound,
)
from gbslab.frugal_scaling import frugal_native_sample
from gbslab.gaussian import covariance_from_interferometer, haar_unitary
from gbslab.gbs_identity import ConfidencePlan, fixed_batch_protocol

ROOT = Path(__file__).resolve().parents[1]


class FrugalEvidenceInputTests(unittest.TestCase):
    def test_invalid_inputs_fail_before_invoking_process(self):
        with patch("gbslab.frugal_evidence.subprocess.run") as run:
            for kwargs in [{"cutoff": 0}, {"cutoff": 6}, {"sample_count": -1},
                           {"sample_count": True}, {"seed": 2**32}, {"threads": 0},
                           {"timeout": 0}, {"raw_samples": "yes"}]:
                with self.assertRaises(ValueError):
                    frugal_evidence_sample(np.eye(16), **kwargs)
            with self.assertRaises(ValueError):
                frugal_evidence_sample(np.eye(32), enumerate_distribution=True)
            with self.assertRaises(ValueError):
                frugal_evidence_sample(np.eye(16) * .5)
            run.assert_not_called()

    def test_failure_evidence_cannot_be_accepted_as_gbs(self):
        result = assess_frugal_samples(np.eye(16), {"status": "error", "error": "trace failed"})
        self.assertEqual(result["decision"], "evaluation_failed")
        self.assertIsNone(result["global_tv"])

    def test_storage_payload_is_explicitly_not_total_memory(self):
        estimate = frugal_storage_lower_bound(144, 5)
        self.assertFalse(estimate["is_peak_rss_estimate"])
        self.assertEqual(estimate["layout_entries"], sum(math.comb(144, k) for k in range(6)))
        self.assertGreater(estimate["persistent_payload_plus_one_trace_workspace_bytes"], 15 * 2**30)


class FrugalEvidenceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DEFAULT_EXECUTABLE.is_file():
            raise unittest.SkipTest("Evidence binary is absent; unit tests do not compile")
        cls.old = ROOT / ".cache/frugal-native/frugal_sampling_adapter"
        if not cls.old.is_file():
            raise unittest.SkipTest("Frozen canonical native reference is absent")

    def test_nonmultiple_simd_batch_logs_full_law_and_unchanged_official_samples(self):
        V = covariance_from_interferometer(haar_unitary(8, np.random.default_rng(20032)), .8, .7)
        for k in (2, 5):
            evidence = frugal_evidence_sample(V, k, 17, 99001 + k, enumerate_distribution=True)
            untraced = frugal_evidence_sample(V, k, 17, 99001 + k, record_log_probabilities=False)
            old = frugal_native_sample(V, k, 17, 99001 + k, enumerate_distribution=True, executable=self.old)
            self.assertEqual(evidence["raw_samples"], untraced["raw_samples"])
            for name in ("histogram_counts", "per_mode_click_counts", "click_count_histogram"):
                self.assertEqual(evidence[name], old[name])
            self.assertEqual(evidence["sample_count"], 17)
            self.assertEqual(len(evidence["sample_log_probabilities"]), 17)
            self.assertTrue(evidence["build_metadata_matches_binary"])
            self.assertFalse(evidence["density_in_sampling_timer"])
            self.assertIsNone(untraced["sample_log_probabilities"])
            q = np.asarray(evidence["distribution"])
            np.testing.assert_allclose(q, old["distribution"], rtol=0, atol=2e-12)
            masks = np.asarray(evidence["raw_samples"]) @ (1 << np.arange(8))
            np.testing.assert_allclose(evidence["sample_log_probabilities"], np.log(q[masks]), rtol=0, atol=2e-12)

    def test_clipping_boundary_uses_actual_normalized_threshold_law(self):
        V = covariance_from_interferometer(haar_unitary(8, np.random.default_rng(4001)), 1., 1.)
        out = frugal_evidence_sample(V, 5, 129, 99003, enumerate_distribution=True)
        old = frugal_native_sample(V, 5, 129, 99003, enumerate_distribution=True, executable=self.old)
        self.assertGreater(out["trace"]["saturated_threshold_path_steps"], 0)
        self.assertEqual(out["trace"]["nonfinite_threshold_path_steps"], 0)
        self.assertEqual(out["histogram_counts"], old["histogram_counts"])
        self.assertEqual(out["trace"]["saturated_threshold_path_steps"], old["trace"]["saturated_threshold_path_steps"])
        q = np.asarray(out["distribution"])
        self.assertAlmostEqual(float(q.sum()), 1, places=12)
        self.assertEqual(int(np.asarray(out["histogram_counts"])[q == 0].sum()), 0)
        np.testing.assert_allclose(q, old["distribution"], rtol=0, atol=2e-12)

    def test_physical_bits_above_64_and_analytic_density(self):
        # Only the last three modes are thermal. This tests wide bitstrings and
        # trace workspace scaling, not quality on a correlated hard instance.
        for m in (100, 144):
            nbar = np.zeros(m); nbar[-3:] = [.5, 1., 2.]
            V = np.diag(np.tile(1 + 2 * nbar, 2))
            out = frugal_evidence_sample(V, 2, 17, 99004 + m)
            samples = np.asarray(out["raw_samples"])
            self.assertEqual(samples[:, :-3].sum(), 0)
            self.assertGreater(samples[:, -3:].sum(), 0)
            p = nbar[-3:] / (1 + nbar[-3:])
            expected = np.where(samples[:, -3:], np.log(p), np.log1p(-p)).sum(axis=1)
            np.testing.assert_allclose(out["sample_log_probabilities"], expected, rtol=0, atol=2e-9)
            self.assertEqual(out["density_trace"]["scored_samples"], 17)
            self.assertEqual(out["histogram_counts"], [])

    def test_generic_identity_helper_binds_frugal_target_and_full_batch(self):
        V = np.diag(np.tile(1 + 2 * np.linspace(.1, .7, 8), 2))
        out = frugal_evidence_sample(V, 2, 129, 99005, enumerate_distribution=True)
        report = assess_frugal_samples(V, out, epsilon=.01,
            plan=ConfidencePlan(("frugal",)), comparison_id="frugal",
            protocol=fixed_batch_protocol(129, source="unit fixture predeclared"))
        self.assertEqual(report["decision"], "accepted_within_tolerance_conditional")
        self.assertLess(report["full_law"]["tv"], 1e-10)
        self.assertEqual(report["native_provenance"]["selector"], "official_frugal_double_elision")
        with self.assertRaisesRegex(ValueError, "covariance"):
            assess_frugal_samples(V + np.eye(16) * .01, out)
        incomplete = deepcopy(out); incomplete["raw_samples"].pop()
        with self.assertRaisesRegex(ValueError, "complete batch"):
            assess_frugal_samples(V, incomplete)
        retry_protocol = fixed_batch_protocol(129); retry_protocol["first_batch_no_retry"] = False
        no_full_q = deepcopy(out); no_full_q["distribution"] = None
        rejected_protocol = assess_frugal_samples(V, no_full_q, epsilon=.5,
            plan=ConfidencePlan(("frugal",)), comparison_id="frugal", protocol=retry_protocol)
        self.assertEqual(rejected_protocol["decision"], "insufficient_evidence")

    def test_zero_draws_and_raw_opt_out_keep_exact_shapes(self):
        out = frugal_evidence_sample(np.eye(16), 1, 0, 99006)
        self.assertEqual(out["raw_samples"], [])
        self.assertEqual(out["sample_log_probabilities"], [])
        self.assertEqual(out["sample_count"], 0)
        no_raw = frugal_evidence_sample(np.eye(16), 1, 17, 99006, raw_samples=False)
        self.assertIsNone(no_raw["raw_samples"])
        self.assertEqual(len(no_raw["sample_log_probabilities"]), 17)
        with self.assertRaisesRegex(ValueError, "raw samples"):
            assess_frugal_samples(np.eye(16), no_raw)


class FrugalCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT_EXECUTABLE.is_file() or not DEFAULT_EXECUTABLE.is_file():
            raise unittest.SkipTest("Checkpoint and frozen v1 binaries must be built explicitly")

    def test_checkpoint_precedes_density_and_preserves_frozen_samples(self):
        V = covariance_from_interferometer(haar_unitary(8, np.random.default_rng(4001)), 1., 1.)
        old = frugal_evidence_sample(V, 5, 17, 99201, executable=DEFAULT_EXECUTABLE,
                                     enumerate_distribution=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            out = frugal_evidence_sample(V, 5, 17, 99201, executable=CHECKPOINT_EXECUTABLE,
                sample_checkpoint_path=path, enumerate_distribution=True)
            checkpoint = json.loads(path.read_text())
            self.assertEqual(checkpoint["raw_samples"], old["raw_samples"])
            self.assertEqual(out["raw_samples"], old["raw_samples"])
            self.assertEqual(out["sample_log_probabilities"], old["sample_log_probabilities"])
            self.assertEqual(out["distribution"], old["distribution"])
            self.assertTrue(out["completed_generation"])
            self.assertEqual(out["sample_checkpoint_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(checkpoint["covariance_sha256"], out["covariance_sha256"])
            self.assertEqual(checkpoint["executable_sha256"], out["executable_sha256"])
            self.assertEqual(checkpoint["timings_seconds_so_far"]["sampling"], out["timings_seconds"]["sampling"])
            self.assertFalse(checkpoint["density_complete"])
            self.assertIsNone(checkpoint["sample_log_probabilities"])
            stages = [(item["phase"], item["status"]) for item in out["stage_diagnostics"]]
            self.assertLess(stages.index(("sample_checkpoint", "completed")), stages.index(("density", "starting")))
            self.assertEqual(list(path.parent.glob("*.partial.*")), [])
            # A generation checkpoint is a distinct artifact, never a valid
            # complete-density result for automatic global acceptance.
            report = assess_frugal_samples(V, checkpoint)
            self.assertEqual(report["decision"], "evaluation_failed")

    def test_mocked_wrapper_timeout_recovers_verified_complete_checkpoint(self):
        # Execute one small real native batch, then inject TimeoutExpired at
        # the subprocess boundary. This tests the failure recovery path without
        # introducing timing races, long sleeps, or another large experiment.
        real_run = subprocess.run
        V = np.eye(16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            def timeout_after_native_checkpoint(command, **kwargs):
                result = real_run(command, **kwargs)
                self.assertEqual(result.returncode, 0)
                checkpoint = json.loads(path.read_text())
                self.assertEqual(checkpoint["sample_count"], 17)
                stages = [json.loads(line) for line in result.stderr.splitlines()]
                events = [(item["phase"], item["status"]) for item in stages]
                self.assertLess(events.index(("sample_checkpoint", "completed")), events.index(("density", "starting")))
                raise subprocess.TimeoutExpired(command, kwargs["timeout"], stderr=result.stderr)
            with patch("gbslab.frugal_evidence.subprocess.run", side_effect=timeout_after_native_checkpoint) as run:
                with self.assertRaises(FrugalEvidenceError) as context:
                    frugal_evidence_sample(V, 2, 17, 99202, executable=CHECKPOINT_EXECUTABLE,
                        sample_checkpoint_path=path, timeout=10)
                self.assertEqual(run.call_count, 1)
            payload = context.exception.payload
            self.assertEqual(payload["status"], "error")
            self.assertEqual(payload["error"], "timeout")
            self.assertTrue(payload["completed_generation"])
            self.assertEqual(payload["completed_generation_sample_count"], 17)
            self.assertEqual(payload["sample_checkpoint"]["status"], "verified_complete_samples")
            self.assertEqual(payload["sample_checkpoint_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertFalse(payload["sample_checkpoint"]["density_available"])
            self.assertFalse(payload["sample_checkpoint"]["quality_acceptance_claimed"])
            self.assertTrue(payload["no_retry"])
            self.assertEqual(assess_frugal_samples(V, payload)["decision"], "evaluation_failed")

    def test_existing_checkpoint_stops_before_sampling_and_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            path.write_text("previous evidence\n")
            with patch("gbslab.frugal_evidence.subprocess.run") as run:
                with self.assertRaises(FileExistsError):
                    frugal_evidence_sample(np.eye(16), 2, 17, 99203,
                        executable=CHECKPOINT_EXECUTABLE, sample_checkpoint_path=path)
                run.assert_not_called()
            self.assertEqual(path.read_text(), "previous evidence\n")

    def test_timeout_does_not_promote_a_malformed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            def incomplete_checkpoint(command, **kwargs):
                path.write_text(json.dumps({"status": "generation_complete", "raw_samples": [[0] * 8]}))
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            with patch("gbslab.frugal_evidence.subprocess.run", side_effect=incomplete_checkpoint) as run:
                with self.assertRaises(FrugalEvidenceError) as context:
                    frugal_evidence_sample(np.eye(16), 2, 17, 99204,
                        executable=CHECKPOINT_EXECUTABLE, sample_checkpoint_path=path)
                self.assertEqual(run.call_count, 1)
            self.assertEqual(context.exception.payload["sample_checkpoint"]["status"], "invalid")
            self.assertFalse(context.exception.payload["completed_generation"])
            self.assertEqual(context.exception.payload["error"], "timeout")


if __name__ == "__main__":
    unittest.main()
