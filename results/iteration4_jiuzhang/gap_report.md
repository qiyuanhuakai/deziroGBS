# Jiuzhang gap: archived_synthetic_m64_seed7201_cpp_click_influence_k4_repeat0

Target: **Jiuzhang 4.0 L1024 (2026)**, `published_gaussian_model`.

Verdict: **different_or_incomplete_task**. No readiness percentage is assigned.

| Axis | Status | Finding |
|---|---|---|
| physical_instance | fail | The run uses a synthetic circuit, not this experimental instance. |
| full_output_extent | fail | The full output extent differs from the target task. |
| measurement_alphabet | pass | Declared detector alphabet matches; calibrated response is a separate gate. |
| imperfections | fail | Required model/response features are omitted or unsupported. |
| implementation_capacity | fail | The current implementation cannot represent the full target extent. |
| distribution_quality | unknown | Local validation does not establish quality for an unbound or different Jiuzhang target. |
| sampling_protocol | pass | A complete declared IID batch uses the target selection rule without retries. |
| resource_timing_comparability | unknown | Complete timing boundaries and resource measurements are missing or invalid. |

## Scale and timing

```json
{
  "scale": {
    "local_modes": 64,
    "target_optical_modes": 8176,
    "local_squeezed_input_states": 64,
    "target_squeezed_input_states": 1024,
    "target_binary_detection_bins": 8176,
    "target_hardware_detector_channels": 16,
    "local_sample_count": 512,
    "local_click_statistics": {
      "observed_mean": 7.87890625,
      "observed_maximum": 17,
      "sample_size": 512,
      "source": "archived observed histogram; not target-law mean"
    },
    "target_click_statistics": {
      "observed_maximum_detection_events": 3050,
      "mean_detection_events": null,
      "maximum_is_not_typical": true,
      "true_PNR_photons": false,
      "note": "3050 is a reported maximum detection-event count in the largest configuration; subgroup means and maxima need their specific data."
    },
    "dense_float64_covariance_bytes_at_target_extent": 2139103232,
    "dense_memory_note": "One dense real 2m-by-2m covariance only; excludes copies, complex validation, workspaces and sampling state."
  },
  "timing": {
    "local_sampling_draws_per_second": 981.7246021278228,
    "published_hardware_sample_period_seconds": 2.56e-05,
    "classical_wall_over_reference_wall": null,
    "comparison_authorized_by_evidence": false,
    "historical_advantage_claim_is_not_a_current_classical_lower_bound": true
  }
}
```

All source definitions and unknowns remain in the versioned reference catalog. 
A null timing ratio means that matched experimental comparison is not supported.
