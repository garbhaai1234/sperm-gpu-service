# Uploaded-video grid diagnostics

Implementation date: 2026-10-08. This feature is a read-only visualization and
measurement layer over existing canonical tracking. It does not establish
biological identity correctness or fix switches.

## Implementation and changed functions

| File | Functions/classes changed or added | Purpose |
|---|---|---|
| `sperm_pipeline/sperm_pipeline/grid_tracking.py` (new) | `GridConfig`, `bbox_center`, `column_name`, `grid_cell`; `GridDiagnostics.__init__`, `_row`, `_event`, `capture`, `_diagnose`, `write`, `render`, `draw`; `coordinate_history` | Coordinate/grid mapping, read-only frame snapshots, advisory events, CSV/JSON, separate video, ID lookup |
| `sperm_pipeline/sperm_pipeline/pipeline.py` | `SpermAnalysisPipeline.__init__`, `_reset_run_state`, `_generate_inference_video`, `_generate_csrt_inference_video`, `_save_results`; new `_csrt_grid_snapshot`, `_finish_grid_diagnostics` | Default-off configuration, hooks after existing assignment, CSRT prediction provenance, additive artifact paths |
| `main.py` | Grid configuration constants; `initialize_pipeline`, `process_video_async`, `get_results`; new `get_grid_coordinate_history` | Wire configuration, add result URLs and protected coordinate-history endpoint |
| `tests/test_grid_tracking.py` (new) | Coordinate, boundary/resolution, canonical-ID, overlap/prediction, advisory, export/render and API tests | Regression coverage for the requested diagnostic contract |
| `tests/validate_grid_videos.py` (new) | `digest`, `main` | Reproducible three-source-video enabled/disabled comparisons |
| `README.md` | Coordinate grid tracking diagnostics section | Configuration, schema/provenance, endpoints and limitations |
| `docs/GRID_TRACKING_VALIDATION.md` (new) | This report | Change inventory and validation results |

Pre-existing working-tree edits in pipeline/linker tests and earlier investigation
scripts were preserved. Source hashes compared against the prior audit confirm
that detection, segmentation, tracking, morphology, motility/CASA, configuration
and existing tests were not modified by this task. Pipeline additions surround
existing decisions; they do not alter identity gates or coordinate calculations.

## Regression tests

**117 passed**, including all 94 pre-existing tests and 23 added parametrized test
cases. Runtime: 7.81 seconds. Two existing FastAPI startup-hook deprecation
warnings surfaced when importing the API for the new endpoint test.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sperm_pipeline \
  /home/user_garbha/sperm/bin/python3 -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/grid_regression_complete
```

Coverage includes bbox centers, fractional/boundary coordinates, four resolutions,
columns beyond Z, crossings and shared cells without ID reassignment, merged
observations with null IDs and preserved occluded identities, dashed predictions,
expired group exclusion, default-off rendering, CSV/JSON accuracy, canonical labels,
bounded trails, advisory flags, CSRT prediction provenance, authenticated lookup,
missing-ID responses and traversal rejection.

## Full-video comparison method

For each of three existing original uploads, the validation runner processes every
source frame through Mask R-CNN, ByteTrack, the unchanged canonical linker,
morphology and CASA twice: grid video disabled and enabled. It compares:

- Every assigned grid coordinate/ID against `inference_tracking.csv`.
- Full analysis dictionaries (including morphology and CASA) and trajectories.
- SHA-256 of the standard inference video, not only its identity count.
- Grid-video width, height, FPS and decoded frame-count metadata.
- Every source frame represented in the log, including empty frames.
- Explicitly marked predictions and imported occlusion-recovery lifecycle counts.

This is a complete inference/analysis-pass comparison, not a rerun of the separate
YOLO diagnostic processed-video pass. CSRT provenance has unit coverage; the three
real-video comparisons use the default ByteTrack upload mode. The grid itself
cannot prove that a canonical assignment refers to the correct biological sperm.

## Runtime contract and limits

Default: logs on for every upload; separate grid video off. Enable with
`SPERM_PIPELINE_SHOW_TRACKING_GRID=true`. Geometry, trail length, coordinate
provenance, advisory thresholds and API usage are documented in README.

A predicted point outside the source frame is retained with `in_frame=false` and
null cell, and is not drawn; no clamping changes scientific coordinates. Numeric
overlay labels use two decimal places; files retain full precision. Predictions
never update observed diagnostic history. Existing CSRT CASA trajectories are not
changed by labeling their tracker-only coordinates as predictions in diagnostics.

The service returns AVI fallback if existing FFmpeg conversion fails; validation
requires the requested MP4 artifact. No external uploaded-results dashboard source
is present, so this change supplies additive result URLs and a lookup endpoint
without modifying the unrelated live mobile page. Rendering text can overlap in
dense scenes; CSV/JSON and coordinate lookup remain the precise inspection tools.
