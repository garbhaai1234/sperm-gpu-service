"""Uploaded analysis, API rows, and trajectories share the rendered application ID."""

from sperm_pipeline.pipeline import SpermAnalysisPipeline


def test_uploaded_trajectory_builder_uses_application_id_and_skips_identity_only_rows():
    rows = [
        {"frame_index": 0, "application_id": 42, "x1": 10, "y1": 20,
         "x2": 30, "y2": 40, "morphology_valid": True},
        {"frame_index": 1, "application_id": 42, "x1": 12, "y1": 20,
         "x2": 32, "y2": 40, "morphology_valid": False},
        {"frame_index": 2, "application_id": 42, "x1": 14, "y1": 20,
         "x2": 34, "y2": 40, "morphology_valid": True},
    ]
    assert SpermAnalysisPipeline._trajectories_from_inference_rows(rows) == {
        42: [(0, 20.0, 30.0), (2, 24.0, 30.0)]
    }


def test_csrt_tracker_ids_map_to_a_separate_application_namespace():
    state = {"next_id": 1, "application_by_tracker": {}}
    assert SpermAnalysisPipeline._csrt_application_id(77, state) == 1
    assert SpermAnalysisPipeline._csrt_application_id(12, state) == 2
    assert SpermAnalysisPipeline._csrt_application_id(77, state) == 1


def test_final_analysis_and_casa_use_the_canonical_application_id():
    pipeline = object.__new__(SpermAnalysisPipeline)
    pipeline.application_trajectories = {42: [(0, 10.0, 20.0), (1, 12.0, 20.0)]}
    pipeline._canonical_identity_available = True
    pipeline.micrometers_per_pixel = None
    pipeline.morphology_results = {}

    class OldAnalysisTracker:
        @staticmethod
        def get_all_trajectories():
            return {7: [(0, 500.0, 500.0)]}

    class Motility:
        seen = None

        def analyze_batch(self, trajectories, fps):
            self.seen = trajectories
            return {track_id: {"label": "immotile"} for track_id in trajectories}

    pipeline.tracker = OldAnalysisTracker()
    pipeline.motility_analyzer = Motility()
    result = pipeline._compute_final_analysis(16.0)

    assert list(pipeline.motility_analyzer.seen) == [42]
    assert [item["sperm_id"] for item in result] == [42]
