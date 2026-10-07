"""Deterministic identity-safety checks for the default analysis tracker."""

import json

from sperm_pipeline.tracking import SpermTracker
from sperm_pipeline.pipeline import SpermAnalysisPipeline


def detection(x, y=50.0, *, width=12.0, height=8.0, confidence=0.9):
    box = [x - width / 2, y - height / 2, x + width / 2, y + height / 2]
    return {"center": (float(x), float(y)), "bbox": box,
            "confidence": confidence, "track_id": None}


def ids(observations):
    return [item.get("track_id") for item in observations]


def test_normal_motion_and_two_sperm_keep_distinct_ids():
    tracker = SpermTracker()
    assert ids(tracker.update([detection(20), detection(120)], 0)) == [1, 2]
    assert ids(tracker.update([detection(27), detection(112)], 1)) == [1, 2]
    assert ids(tracker.update([detection(34), detection(104)], 2)) == [1, 2]


def test_crossing_tracks_follow_predicted_motion_without_swapping():
    tracker = SpermTracker()
    tracker.update([detection(20), detection(100)], 0)
    tracker.update([detection(35), detection(85)], 1)
    assert ids(tracker.update([detection(50), detection(70)], 2)) == [1, 2]
    assert ids(tracker.update([detection(65), detection(55)], 3)) == [1, 2]


def test_exact_overlap_is_left_unassigned_then_tracks_recover():
    tracker = SpermTracker()
    tracker.update([detection(20), detection(100)], 0)
    tracker.update([detection(35), detection(85)], 1)
    assert ids(tracker.update([detection(60), detection(60)], 2)) == [None, None]
    assert ids(tracker.update([detection(65), detection(55)], 3)) == [1, 2]


def test_short_detection_gap_uses_actual_frame_indices_and_recovers():
    tracker = SpermTracker()
    tracker.update([detection(10)], 0)
    tracker.update([detection(20)], 2)
    recovered = tracker.update([detection(50)], 8)
    assert ids(recovered) == [1]


def test_fast_motion_reconnects_when_prediction_is_within_gate():
    tracker = SpermTracker()
    tracker.update([detection(0)], 0)
    tracker.update([detection(30)], 1)
    assert ids(tracker.update([detection(60)], 2)) == [1]


def test_ambiguous_single_observation_does_not_create_or_steal_id():
    tracker = SpermTracker()
    tracker.update([detection(0), detection(20)], 0)
    result = tracker.update([detection(10)], 1)
    assert ids(result) == [None]
    assert tracker.next_id == 3
    assert tracker.association_diagnostics[-1]["reason"] == "AMBIGUOUS"


def test_new_sperm_entering_gets_new_id_without_renumbering():
    tracker = SpermTracker()
    tracker.update([detection(20), detection(100)], 0)
    assert ids(tracker.update([detection(25), detection(95), detection(250)], 1)) == [1, 2, 3]


def test_one_to_one_assignment_never_reuses_track_or_detection():
    tracker = SpermTracker()
    tracker.update([detection(0), detection(100)], 0)
    updated = tracker.update([detection(5), detection(95), detection(200)], 1)
    assigned = [item["track_id"] for item in updated if item["track_id"] is not None]
    assert len(assigned) == len(set(assigned)) == 3
    assert len({item["track_id"] for item in updated[:2]}) == 2


def test_lost_track_expires_and_retired_id_is_never_reused():
    tracker = SpermTracker()
    tracker.update([detection(0)], 0)
    tracker.update([], tracker.max_lost_frames + 1)
    assert tracker.track_status[1] == "TERMINATED_REPORTED"
    assert ids(tracker.update([detection(0)], tracker.max_lost_frames + 2)) == [2]
    assert tracker.diagnostics_counts["TRACK_EXPIRED"] == 1


def test_single_sperm_does_not_fragment_across_recoverable_gaps():
    tracker = SpermTracker()
    for frame, x in [(0, 0), (1, 8), (4, 32), (5, 40), (9, 72)]:
        assert ids(tracker.update([detection(x)], frame)) == [1]
    assert tracker.next_id == 2


def test_missed_frame_crossing_recovery_keeps_pre_crossing_ids():
    tracker = SpermTracker()
    tracker.update([detection(20), detection(100)], 0)
    tracker.update([detection(35), detection(85)], 1)
    tracker.update([detection(60)], 2)
    after = tracker.update([detection(65), detection(55)], 3)
    assert ids(after) == [1, 2]


def test_association_diagnostics_include_candidate_evidence_and_reasons():
    tracker = SpermTracker()
    tracker.update([detection(10)], 0)
    tracker.update([detection(15)], 1)
    row = tracker.association_diagnostics[-1]
    assert row["reason"] == "ACTIVE_CONTINUATION"
    assert row["accepted"] is True
    assert row["track_id"] == 1
    assert row["candidate_evaluations"][0]["predicted_center"] == [10.0, 50.0]


def test_trajectory_export_uses_source_fps_and_writes_analysis_diagnostics(tmp_path):
    pipeline = object.__new__(SpermAnalysisPipeline)
    pipeline.source_fps = 24.0
    pipeline.tracker = SpermTracker()
    pipeline.tracker.update([detection(10)], 0)
    pipeline._convert_to_mp4 = lambda path: path
    output_paths = pipeline._save_results(str(tmp_path), [])
    with open(tmp_path / "meta" / "trajectories.json", encoding="utf-8") as handle:
        trajectories = json.load(handle)
    with open(output_paths["analysis_id_diagnostics_path"], encoding="utf-8") as handle:
        diagnostics = json.load(handle)
    assert trajectories["fps"] == 24.0
    assert diagnostics["counts"]["NEW_TRACK"] == 1
    assert diagnostics["decisions"][0]["reason"] == "NEW_TRACK"
