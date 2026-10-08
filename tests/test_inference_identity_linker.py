"""Deterministic identity-safety checks for inference application IDs."""

import numpy as np
import torch

from sperm_pipeline.pipeline import SpermAnalysisPipeline


def _pipeline():
    return object.__new__(SpermAnalysisPipeline)


def _box(center_x, center_y=10.0, width=20.0, height=20.0):
    return (center_x - width / 2, center_y - height / 2,
            center_x + width / 2, center_y + height / 2)


def _observation(score, center_x=10.0, tracker_id=1, width=20.0):
    return {
        "tracker_id": tracker_id,
        "confidence": score,
        "bbox": _box(center_x, width=width),
        "center": (center_x, 10.0),
    }


def _state():
    return {"next_id": 1, "identities": {}, "byte_owner": {}}


def _seed_track(pipeline, state, center_x=10.0, tracker_id=1, width=20.0, frame=0):
    item = _observation(0.90, center_x, tracker_id, width)
    result = pipeline._link_inference_application_ids([item], state, frame)
    return result[0][0]


def _stub_geometry(pipeline, distance, iou, *, anchor_distance=None,
                   anchor_iou=None, predicted_distance=None, speed=0.0,
                   motion_opposes=False):
    anchor_distance = distance if anchor_distance is None else anchor_distance
    anchor_iou = iou if anchor_iou is None else anchor_iou
    predicted_distance = distance if predicted_distance is None else predicted_distance
    pipeline._low_identity_geometry = lambda item, identity, frame: {
        "gap": 1,
        "anchor_gap": 1,
        "dist_last": distance,
        "iou": iou,
        "dist_anchor": anchor_distance,
        "iou_anchor": anchor_iou,
        "dist_pred": predicted_distance,
        "speed": speed,
        "motion_opposes": motion_opposes,
        "side_ratio": 1.0,
        "previous_center": identity.get("last_observation_center"),
        "previous_bbox": identity.get("last_observation_bbox"),
        "anchor_center": identity.get("last_high_conf_center"),
        "anchor_bbox": identity.get("last_high_conf_bbox"),
        "predicted_center": identity.get("last_high_conf_center"),
    }


def test_high_confidence_wins_competition_with_low_support():
    pipeline, state = _pipeline(), _state()
    track_id = _seed_track(pipeline, state)
    high = _observation(0.90, tracker_id=1)
    low = _observation(0.50, tracker_id=None)
    _stub_geometry(pipeline, 5.0, 0.80)
    links = pipeline._link_inference_application_ids([low, high], state, 1)
    assert links[1][0] == track_id
    assert links[1][1] == "KEEP"
    assert links[0] is None


def test_two_low_detections_do_not_update_one_track_twice():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state)
    _stub_geometry(pipeline, 5.0, 0.80)
    links = pipeline._link_inference_application_ids(
        [_observation(0.50, 10.0, None), _observation(0.55, 11.0, None)], state, 1
    )
    assert links == [None, None]
    assert state["identities"][1]["identity_support_count"] == 0


def test_low_observation_matching_two_tracks_is_ambiguous():
    pipeline, state = _pipeline(), _state()
    pipeline._link_inference_application_ids(
        [_observation(0.90, 0.0, 1), _observation(0.90, 20.0, 2)], state, 0
    )
    low = _observation(0.50, 10.0, None)
    _stub_geometry(pipeline, 30.0, 0.50)
    link = pipeline._link_inference_application_ids([low], state, 1)[0]
    assert link is None
    assert any(row["ambiguous"] for row in state["identity_assignment_diagnostics"])


def test_near_equal_candidates_at_31_034_and_27_036_are_ambiguous():
    pipeline, state = _pipeline(), _state()
    pipeline._link_inference_application_ids(
        [_observation(0.90, 0.0, 1), _observation(0.90, 20.0, 2)], state, 0
    )

    def geometry(item, identity, _frame):
        distance, iou = (31.0, 0.34) if identity is state["identities"][1] else (27.0, 0.36)
        return {
            "gap": 1, "anchor_gap": 1, "dist_last": distance, "iou": iou,
            "dist_anchor": distance, "iou_anchor": iou, "dist_pred": distance,
            "speed": 0.0, "motion_opposes": False, "side_ratio": 1.0,
            "previous_center": identity["last_observation_center"],
            "previous_bbox": identity["last_observation_bbox"],
            "anchor_center": identity["last_high_conf_center"],
            "anchor_bbox": identity["last_high_conf_bbox"],
            "predicted_center": identity["last_high_conf_center"],
        }

    pipeline._low_identity_geometry = geometry
    assert pipeline._link_inference_application_ids(
        [_observation(0.50, 10.0, None)], state, 1
    ) == [None]
    ambiguous = [row for row in state["identity_assignment_diagnostics"] if row["ambiguous"]]
    assert len(ambiguous) == 2


def _gate_case(distance, iou):
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state)
    _stub_geometry(pipeline, distance, iou)
    item = _observation(0.50, tracker_id=None)
    link = pipeline._link_inference_application_ids([item], state, 1)[0]
    return link, item, state


def test_gate_rejects_distance_70_iou_005():
    assert _gate_case(70, 0.05)[0] is None


def test_gate_rejects_distance_35_iou_010():
    assert _gate_case(35, 0.10)[0] is None


def test_gate_rejects_distance_60_iou_060():
    assert _gate_case(60, 0.60)[0] is None


def test_gate_accepts_distance_35_iou_035_as_identity_only():
    link, item, _ = _gate_case(35, 0.35)
    assert link[0] == 1
    assert link[1] == "IDENTITY_ONLY"
    assert item["identity_only"] is True
    assert item["morphology_valid"] is False


def test_gate_rejects_distance_41_iou_090():
    assert _gate_case(41, 0.90)[0] is None


def test_three_low_support_frames_are_allowed():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state)
    for frame in (1, 2, 3):
        item = _observation(0.50, tracker_id=None)
        link = pipeline._link_inference_application_ids([item], state, frame)[0]
        assert link[0] == 1 and link[1] == "IDENTITY_ONLY"
        assert state["identities"][1]["identity_support_count"] == frame


def test_fourth_consecutive_low_support_frame_is_rejected():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state)
    for frame in (1, 2, 3):
        pipeline._link_inference_application_ids([_observation(0.50, tracker_id=None)], state, frame)
    assert pipeline._link_inference_application_ids(
        [_observation(0.50, tracker_id=None)], state, 4
    ) == [None]


def test_high_confidence_recovery_resets_low_support_count():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state)
    for frame in (1, 2):
        pipeline._link_inference_application_ids([_observation(0.50)], state, frame)
    recovered = pipeline._link_inference_application_ids([_observation(0.90)], state, 3)[0]
    assert recovered[0] == 1
    assert state["identities"][1]["identity_support_count"] == 0
    low = pipeline._link_inference_application_ids([_observation(0.50)], state, 4)[0]
    assert low[0] == 1
    assert state["identities"][1]["identity_support_count"] == 1


def test_low_score_without_existing_track_does_not_create_id():
    pipeline, state = _pipeline(), _state()
    low = _observation(0.50, tracker_id=None)
    assert pipeline._link_inference_application_ids([low], state, 0) == [None]
    assert state["next_id"] == 1
    assert state["identities"] == {}


def test_low_score_drift_is_bounded_by_high_confidence_anchor():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state, center_x=50.0, width=100.0)
    # First 25px weak step passes; the next is 50px from the trusted anchor.
    first = pipeline._link_inference_application_ids(
        [_observation(0.50, center_x=75.0, tracker_id=None, width=100.0)], state, 1
    )[0]
    second = pipeline._link_inference_application_ids(
        [_observation(0.50, center_x=100.0, tracker_id=None, width=100.0)], state, 2
    )[0]
    assert first[0] == 1
    assert second is None
    assert state["identities"][1]["last_high_conf_center"] == (50.0, 10.0)


def test_same_bytetrack_id_with_motion_opposed_jump_is_not_kept():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state, center_x=0.0, tracker_id=1)
    pipeline._link_inference_application_ids([_observation(0.90, 10.0, 1)], state, 1)
    # Same ByteTrack ID, but a reversal against established velocity.
    result = pipeline._link_inference_application_ids([_observation(0.90, -20.0, 1)], state, 2)[0]
    assert result[1] == "NEW"
    assert any(
        row["rejection_reason"] == "MOTION_INCONSISTENT"
        for row in state["identity_assignment_diagnostics"]
    )


def test_same_bytetrack_id_keeps_near_identical_observation_when_motion_turns():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state, center_x=0.0, tracker_id=1)
    pipeline._link_inference_application_ids([_observation(0.90, 4.0, 1)], state, 1)
    result = pipeline._link_inference_application_ids(
        [_observation(0.90, 3.0, 1)], state, 2
    )[0]
    assert result[0] == 1
    assert result[1] == "KEEP"


def test_same_bytetrack_near_identical_box_overrides_direction_noise():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state, center_x=0.0, tracker_id=10, width=100.0)
    pipeline._link_inference_application_ids(
        [_observation(0.90, 2.29, 10, width=100.0)], state, 1
    )
    result = pipeline._link_inference_application_ids(
        [_observation(0.90, 1.82, 10, width=100.0)], state, 2
    )[0]
    assert result[0] == 1
    assert result[1] == "KEEP"
    assert state["identities"][1]["state"] == "CONFIRMED"


def test_same_bytetrack_frame_470_geometry_survives_direction_disagreement():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state, center_x=426.7, tracker_id=1014, frame=469)
    identity = state["identities"][1]
    identity["last_frame"] = 469
    identity["last_seen_frame"] = 469
    identity["last_high_conf_frame"] = 469
    identity["last_high_conf_center"] = (426.7, 123.435358)
    identity["last_center"] = (426.7, 123.435358)
    identity["last_bbox"] = (384.956665, 50.099781, 468.443359, 196.770935)
    identity["trusted_center"] = identity["last_high_conf_center"]
    identity["trusted_bbox"] = identity["last_bbox"]
    identity["velocity"] = (-9.0022, -4.5119)
    state["byte_owner"] = {1014: 1}
    detection = {
        "tracker_id": 1014,
        "confidence": 0.974,
        "center": (429.492386, 182.379906),
        "bbox": (386.441437, 106.547043, 472.543335, 258.212769),
    }

    result = pipeline._link_inference_application_ids([detection], state, 470)[0]

    assert result[:2] == (1, "KEEP")
    assert state["next_id"] == 2


def test_byte_track_id_fragments_are_absorbed_by_one_application_identity():
    pipeline, state = _pipeline(), _state()
    assert _seed_track(pipeline, state, center_x=0.0, tracker_id=10) == 1
    for frame, byte_id, x in ((1, 30, 1.0), (2, 71, 2.0)):
        linked = pipeline._link_inference_application_ids(
            [_observation(0.92, x, byte_id)], state, frame
        )[0]
        assert linked[0] == 1
    identity = state["identities"][1]
    assert [row["tracker_id"] for row in identity["byte_id_history"]] == [10, 30, 71]
    assert state["next_id"] == 2


def test_high_confidence_reassociation_ambiguity_stays_unassigned():
    pipeline, state = _pipeline(), _state()
    pipeline._link_inference_application_ids(
        [_observation(0.90, 0.0, 1), _observation(0.90, 30.0, 2)], state, 0
    )
    pipeline._link_inference_application_ids([], state, 1)
    observations = [
        _observation(0.90, 15.0, 101),
        _observation(0.90, 15.0, 102),
    ]
    links = pipeline._link_inference_application_ids(observations, state, 1)
    assert links == [None, None]
    assert state["next_id"] == 3
    assert any(row["rejection_reason"] == "AMBIGUOUS_HIGH_REASSOCIATION"
               for row in state["identity_assignment_diagnostics"])


def test_lost_identity_window_is_measured_in_source_video_seconds():
    pipeline = _pipeline()
    assert pipeline._identity_recovery_window_frames(16.129) == 49
    assert pipeline._identity_recovery_window_frames(30.0) == 90


def test_crossing_low_tracks_are_left_unassigned_when_ambiguous():
    pipeline, state = _pipeline(), _state()
    pipeline._link_inference_application_ids(
        [_observation(0.90, 0.0, 1), _observation(0.90, 20.0, 2)], state, 0
    )
    _stub_geometry(pipeline, 20.0, 0.60)
    item = _observation(0.50, 10.0, None)
    assert pipeline._link_inference_application_ids([item], state, 1) == [None]
    assert any(row["ambiguous"] for row in state["identity_assignment_diagnostics"])


def test_crossing_trajectories_use_motion_and_leave_crossing_frame_unassigned():
    pipeline, state = _pipeline(), _state()
    pipeline._link_inference_application_ids(
        [_observation(0.90, 0.0, 1, 30.0), _observation(0.90, 40.0, 2, 30.0)], state, 0
    )
    pipeline._link_inference_application_ids(
        [_observation(0.90, 15.0, 1, 30.0), _observation(0.90, 25.0, 2, 30.0)], state, 1
    )
    # Both tracks reach the crossing area; the weak midpoint is equally
    # consistent with either trajectory and must not inherit either ID.
    result = pipeline._link_inference_application_ids(
        [_observation(0.50, 20.0, None, 30.0)], state, 2
    )
    assert result == [None]
    assert any(row["ambiguous"] for row in state["identity_assignment_diagnostics"])


def test_skipped_frames_reset_consecutive_low_support_count():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state)
    assert pipeline._link_inference_application_ids([_observation(0.50)], state, 1)[0][0] == 1
    assert pipeline._link_inference_application_ids([_observation(0.50)], state, 3)[0][0] == 1
    assert state["identities"][1]["identity_support_count"] == 1


def test_low_support_never_changes_trusted_high_anchor():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state)
    pipeline._link_inference_application_ids([_observation(0.50)], state, 1)
    identity = state["identities"][1]
    assert identity["last_high_conf_center"] == (10.0, 10.0)
    assert identity["last_observation_center"] == (10.0, 10.0)


def test_merged_detection_preserves_ids_and_reidentifies_after_separation():
    pipeline, state = _pipeline(), _state()
    first = _observation(0.90, 10.0, 1)
    second = _observation(0.90, 90.0, 2)
    seeded = pipeline._link_inference_application_ids([first, second], state, 1)
    assert [row[0] for row in seeded] == [1, 2]

    # One merged box covers both trusted centers. It is not assigned to either ID
    # and must not create a third identity or overwrite either anchor.
    merged = _observation(0.92, 50.0, 3, width=100.0)
    assert pipeline._link_inference_application_ids([merged], state, 2) == [None]
    assert state["next_id"] == 3
    assert state["identities"][1]["status"] == "OCCLUDED"
    assert state["identities"][2]["status"] == "OCCLUDED"
    assert state["identities"][1]["last_high_conf_center"] == (10.0, 10.0)
    assert state["identities"][2]["last_high_conf_center"] == (90.0, 10.0)
    assert state["overlap_groups"]
    group = next(iter(state["overlap_groups"].values()))
    assert group["identity_ids"] == [1, 2]
    assert group["pre_overlap_state"]["1"]["center"] == (10.0, 10.0)

    # A single merged/occluded interval, followed by detections with new
    # ByteTrack IDs, recovers the original application IDs by prediction/geometry.
    recovered = pipeline._link_inference_application_ids(
        [_observation(0.91, 14.0, 31), _observation(0.93, 86.0, 32)], state, 7
    )
    assert [row[0] for row in recovered] == [1, 2]
    assert {row[1] for row in recovered} == {"REIDENTIFIED"}
    assert state["identities"][1]["current_byte_id"] == 31
    assert state["identities"][2]["current_byte_id"] == 32
    assert len([row for row in state["lifecycle_events"] if row["event"] == "ID_REIDENTIFIED"]) == 2


def test_per_video_identity_state_is_fresh_and_uncapped():
    pipeline = _pipeline()
    first = pipeline._initialize_application_identity_state(640, 480, 16.0)
    second = pipeline._initialize_application_identity_state(1280, 1024, 50.0)
    assert first["identity_capacity"] is None
    assert first["identities"] == {}
    assert first["next_id"] == second["next_id"] == 1
    assert first["overlap_groups"] is not second["overlap_groups"]
    assert pipeline._identity_recovery_window_frames(16.0) == 48
    assert pipeline._identity_recovery_window_frames(30.0) == 90
    assert pipeline._identity_recovery_window_frames(50.0) == 150
    assert first.allocate_id() == 1
    assert first.allocate_id() == 2
    assert second.allocate_id() == 1


def test_registry_optional_capacity_is_monotonic_and_never_reuses_ids():
    pipeline = _pipeline()
    registry = pipeline._initialize_application_identity_state(640, 480, 30.0, identity_capacity=2)
    assert [registry.allocate_id(), registry.allocate_id(), registry.allocate_id()] == [1, 2, None]
    assert registry["next_id"] == 3


def test_plausible_new_detection_is_held_as_unresolved_candidate():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state, center_x=10.0, tracker_id=10)
    pipeline._link_inference_application_ids([], state, 1)
    item = _observation(0.92, center_x=55.0, tracker_id=44)
    links = pipeline._link_inference_application_ids([item], state, 2)
    assert links == [None]
    assert state["next_id"] == 2
    assert item["identity_state"] == "UNRESOLVED"
    assert state["unresolved_candidates"]
    assert any(event["event"] == "UNRESOLVED_CANDIDATE"
               for event in state["quality_events"])


def test_quality_summary_counts_ids_and_keeps_switch_fragmentation_diagnostic_only():
    pipeline = _pipeline()
    rows = [
        {"frame_index": 0, "application_id": 1, "tracker_id": 10},
        {"frame_index": 1, "application_id": 1, "tracker_id": 30},
        {"frame_index": 0, "application_id": 2, "tracker_id": 20},
    ]
    state = {
        "identities": {},
        "lifecycle_events": [{"event": "ID_CREATED"}, {"event": "ID_CREATED"}],
        "quality_events": [
            {"event": "POSSIBLE_ID_FRAGMENTATION"},
            {"event": "POSSIBLE_ID_SWITCH"},
            {"event": "UNRESOLVED_CANDIDATE"},
        ],
        "identity_assignment_diagnostics": [{"ambiguous": True}],
        "unresolved_candidates": {},
    }
    summary = pipeline._build_tracking_quality_summary(2, 16.0, 640, 480, rows, state)
    assert summary["application_ids_created"] == 2
    assert summary["maximum_simultaneous_ids"] == 2
    assert summary["possible_fragmentations"] == 1
    assert summary["possible_id_switches"] == 1
    assert summary["ambiguous_events"] == 1


def test_low_support_changes_observation_state_but_not_trusted_anchor_or_history():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state, center_x=50.0, width=100.0)
    result = pipeline._link_inference_application_ids(
        [_observation(0.50, center_x=60.0, tracker_id=None, width=100.0)], state, 1
    )[0]
    identity = state["identities"][1]
    assert result[0] == 1
    assert identity["state"] == "LOW_SUPPORT"
    assert identity["trusted_center"] == (50.0, 10.0)
    assert identity["trusted_bbox"] == _box(50.0, width=100.0)
    assert identity["last_observation_center"] == (60.0, 10.0)


def test_ambiguous_occlusion_reappearance_stays_unassigned_without_new_id():
    pipeline, state = _pipeline(), _state()
    pipeline._link_inference_application_ids(
        [_observation(0.90, 10.0, 1), _observation(0.90, 30.0, 2)], state, 1
    )
    merged = _observation(0.92, 20.0, 3, width=40.0)
    assert pipeline._link_inference_application_ids([merged], state, 2) == [None]
    before = state["next_id"]
    # Both boxes are equally plausible for each old identity. Ambiguity is
    # unresolved rather than swapping the two application IDs.
    results = pipeline._link_inference_application_ids(
        [_observation(0.92, 20.0, 31), _observation(0.92, 20.0, 32)], state, 3
    )
    assert results == [None, None]
    assert state["next_id"] == before
    assert all(row["event"] != "ID_REIDENTIFIED" for row in state["lifecycle_events"])


def test_overlap_group_does_not_partially_recover_when_one_member_is_visible():
    pipeline, state = _pipeline(), _state()
    pipeline._link_inference_application_ids(
        [_observation(0.90, 10.0, 1), _observation(0.90, 90.0, 2)], state, 1
    )
    assert pipeline._link_inference_application_ids(
        [_observation(0.92, 50.0, 3, width=100.0)], state, 2
    ) == [None]
    assert pipeline._link_inference_application_ids(
        [_observation(0.92, 14.0, 31)], state, 3
    ) == [None]
    assert state["next_id"] == 3
    assert state["identities"][1]["state"] == "UNRESOLVED"
    assert state["identities"][2]["state"] == "UNRESOLVED"
    assert not any(event["event"] == "ID_REIDENTIFIED" for event in state["lifecycle_events"])


def test_overlap_group_retires_terminated_member_when_other_member_is_confirmed():
    pipeline = _pipeline()
    state = {
        "identities": {
            1: {"state": "CONFIRMED", "status": "CONFIRMED", "last_high_conf_frame": 20,
                "last_seen_frame": 20},
            2: {"state": "TERMINATED", "status": "TERMINATED", "last_high_conf_frame": 5,
                "last_seen_frame": 5},
        },
        "overlap_groups": {
            7: {"overlap_group_id": 7, "identity_ids": [1, 2], "start_frame": 10,
                "last_frame": 12, "state": "UNRESOLVED", "unresolved_frames": [13]},
        },
    }

    pipeline._update_overlap_group_states(state, 20)

    group = state["overlap_groups"][7]
    assert group["state"] == "RESOLVED"
    assert group["identity_ids"] == [1]
    assert group["all_identity_ids"] == [1, 2]
    assert group["retired_identity_ids"] == [2]


def test_expired_overlap_group_is_retired_and_stays_out_of_future_recovery():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state, center_x=10.0, tracker_id=1)
    _seed_track(pipeline, state, center_x=90.0, tracker_id=2)
    state["identities"][1]["state"] = state["identities"][1]["status"] = "TERMINATED"
    state["identities"][2]["state"] = state["identities"][2]["status"] = "TERMINATED"
    group = {"overlap_group_id": 1, "identity_ids": [1, 2], "start_frame": 1,
             "last_frame": 2, "state": "UNRESOLVED", "unresolved_frames": []}
    state["overlap_groups"] = {1: group}

    pipeline._update_overlap_group_states(state, 3)
    assert group["state"] == "EXPIRED"
    assert group["identity_ids"] == []
    before = len(group["unresolved_frames"])
    pipeline._update_overlap_group_states(state, 4)
    assert len(group["unresolved_frames"]) == before


def test_video_lifecycle_windows_are_fps_based_in_seconds():
    pipeline = _pipeline()
    for fps, expiry_frame in ((16.0, 49), (30.0, 91), (49.0, 148)):
        state = {"source_fps": fps, "identities": {1: {
            "state": "LOST", "status": "LOST", "last_high_conf_frame": 0,
            "last_high_conf_center": (10.0, 10.0), "last_high_conf_bbox": _box(10.0),
            "last_center": (10.0, 10.0), "last_bbox": _box(10.0),
            "current_byte_id": None,
        }}}
        pipeline._update_occlusion_states([], state, expiry_frame)
        assert state["identities"][1]["state"] == "TERMINATED"


def test_unresolved_candidate_trajectory_evidence_recovers_original_identity():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state, center_x=10.0, tracker_id=10, frame=0)
    identity = state["identities"][1]
    identity["state"] = identity["status"] = "UNRESOLVED"
    identity["velocity"] = (0.0, 0.0)
    candidate = {
        "candidate_id": 1, "possible_identities": [1], "last_frame": 2,
        "observations": [
            {"frame": 1, "center": (35.0, 10.0), "bbox": _box(35.0), "tracker_id": 31},
            {"frame": 2, "center": (50.0, 10.0), "bbox": _box(50.0), "tracker_id": 32},
        ],
        "byte_track_ids": [31, 32],
    }
    state["unresolved_candidates"] = {1: candidate}
    detection = _observation(0.95, center_x=65.0, tracker_id=33)
    recovered = pipeline._reidentify_occluded_tracks([detection], state, 3, set(), set())

    assert len(recovered) == 1
    assert recovered[0][1] == 1
    assert recovered[0][2]["unresolved_candidate_evidence"]["direction"] == (15.0, 0.0)


def test_multi_member_overlap_recovers_with_global_one_to_one_assignment():
    pipeline, state = _pipeline(), _state()
    seeded = pipeline._link_inference_application_ids(
        [_observation(0.92, 10.0, 1), _observation(0.92, 50.0, 2),
         _observation(0.92, 90.0, 3)], state, 1
    )
    assert [row[0] for row in seeded] == [1, 2, 3]
    merged = {"tracker_id": 4, "confidence": 0.95, "center": (50.0, 10.0),
              "bbox": (0.0, 0.0, 100.0, 20.0)}
    assert pipeline._link_inference_application_ids([merged], state, 2) == [None]
    recovered = pipeline._link_inference_application_ids(
        [_observation(0.94, 14.0, 11), _observation(0.94, 54.0, 12),
         _observation(0.94, 86.0, 13)], state, 6
    )
    assert [row[0] for row in recovered] == [1, 2, 3]
    assert {row[1] for row in recovered} == {"REIDENTIFIED"}


def test_occluded_crossing_reidentification_follows_velocity_not_nearest_anchor():
    pipeline, state = _pipeline(), _state()
    pipeline._link_inference_application_ids(
        [_observation(0.90, 0.0, 1), _observation(0.90, 40.0, 2)], state, 0
    )
    pipeline._link_inference_application_ids(
        [_observation(0.90, 10.0, 1), _observation(0.90, 30.0, 2)], state, 1
    )
    # The merged box covers both anchors. After the crossing, the detection
    # nearer ID 2's old location belongs to ID 1's predicted path, and vice versa.
    assert pipeline._link_inference_application_ids(
        [_observation(0.92, 20.0, 9, width=40.0)], state, 2
    ) == [None]
    recovered = pipeline._link_inference_application_ids(
        [_observation(0.92, 60.0, 31), _observation(0.92, -20.0, 32)], state, 6
    )
    assert [row[0] for row in recovered] == [1, 2]
    assert [row[1] for row in recovered] == ["REIDENTIFIED", "REIDENTIFIED"]


def test_new_high_detection_records_true_new_entrant_reason_at_frame_edge():
    pipeline, state = _pipeline(), _state()
    state["frame_size"] = (100, 100)
    item = _observation(0.95, 4.0, 77)
    link = pipeline._link_inference_application_ids([item], state, 0)[0]
    assert link[1] == "NEW"
    assert state["identities"][1]["last_new_id_reason"] == "TRUE_NEW_ENTRANT"
    assert any(e["event"] == "ID_CREATED" and e["new_id_reason"] == "TRUE_NEW_ENTRANT"
               for e in state["lifecycle_events"])


def test_maskrcnn_low_scores_are_separate_from_bytetrack_input():
    class FakeModel:
        def __call__(self, _images):
            return [{
                "boxes": torch.tensor([[1., 1., 8., 8.], [10., 10., 18., 18.], [20., 20., 29., 29.]]),
                "scores": torch.tensor([0.50, 0.70, 0.90]),
                "labels": torch.tensor([1, 1, 1]),
                "masks": torch.ones((3, 1, 32, 32)),
            }]

    pipeline = _pipeline()
    pipeline.device = "cpu"
    pipeline.segmenter = type("Segmenter", (), {
        "maskrcnn_model": FakeModel(), "sperm_class_ids": {1},
    })()
    result = pipeline._run_maskrcnn_inference(np.zeros((32, 32, 3), dtype=np.uint8))
    assert np.allclose(result["scores"], [0.90])
    assert np.allclose(result["identity_scores"], [0.50, 0.70])


def test_unresolved_chains_do_not_mix_simultaneous_detections():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state)
    state["identities"][1]["state"] = state["identities"][1]["status"] = "LOST"
    for frame in (1, 2):
        for index, x in enumerate((12.0, 22.0)):
            assert pipeline._defer_unresolved_new_detection(
                _observation(0.95, x + frame, 30 + index), index, state, frame)
    chains = list(state["unresolved_candidates"].values())
    assert len(chains) == 2
    assert all([row["frame"] for row in c["observations"]] == [1, 2] for c in chains)
    assert all(c["trajectory_direction"] == (1.0, 0.0) for c in chains)


def test_empty_frames_expire_unresolved_evidence_in_video_seconds():
    for fps in (16.0, 30.0, 49.0):
        pipeline, state = _pipeline(), _state()
        state["source_fps"] = fps
        state["unresolved_candidates"] = {1: {"last_frame": 0}}
        pipeline._link_inference_application_ids([], state, int(fps))
        assert 1 in state["unresolved_candidates"]
        pipeline._link_inference_application_ids([], state, int(fps) + 1)
        assert not state["unresolved_candidates"]


def test_terminated_identity_cannot_be_revived_by_low_support_or_same_byte():
    for score in (0.5, 0.95):
        pipeline, state = _pipeline(), _state()
        _seed_track(pipeline, state)
        state["identities"][1]["state"] = state["identities"][1]["status"] = "TERMINATED"
        _stub_geometry(pipeline, 0.0, 1.0)
        result = pipeline._link_inference_application_ids(
            [_observation(score, tracker_id=1 if score > 0.7 else None)], state, 1)
        assert state["identities"][1]["state"] == "TERMINATED"
        assert result[0] is None if score < 0.7 else result[0][:2] == (2, "NEW")


def test_same_byte_recovery_uses_registry_fps_not_previous_video_fps():
    pipeline, state = _pipeline(), _state()
    pipeline.source_fps = 16.0
    state["source_fps"] = 49.0
    _seed_track(pipeline, state)
    # This 100-frame absence is within 3 seconds only in the current video.
    result = pipeline._link_inference_application_ids([_observation(0.95)], state, 100)
    assert result[0][:2] == (1, "KEEP")


def test_lost_survivor_does_not_resolve_group_from_old_confirmation():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state, frame=20)
    _seed_track(pipeline, state, center_x=100.0, tracker_id=2, frame=20)
    state["identities"][1]["state"] = state["identities"][1]["status"] = "LOST"
    state["identities"][2]["state"] = state["identities"][2]["status"] = "TERMINATED"
    state["overlap_groups"] = {1: {"overlap_group_id": 1, "identity_ids": [1, 2],
                                  "start_frame": 10, "last_frame": 12, "state": "UNRESOLVED"}}
    pipeline._update_overlap_group_states(state, 22)
    assert state["overlap_groups"][1]["identity_ids"] == [1]
    assert state["overlap_groups"][1]["state"] == "UNRESOLVED"
    pipeline._link_inference_application_ids([_observation(.95)], state, 23)
    assert state["overlap_groups"][1]["state"] == "RESOLVED"


def test_expired_group_is_never_reopened_by_a_new_merged_observation():
    pipeline, state = _pipeline(), _state()
    pipeline._link_inference_application_ids([_observation(.95, 10, 1), _observation(.95, 90, 2)], state, 0)
    expired = {"overlap_group_id": 1, "identity_ids": [1, 2], "start_frame": -10,
               "last_frame": -5, "state": "EXPIRED", "merged_detection_indices": []}
    state["overlap_groups"] = {1: expired}
    state["next_overlap_group_id"] = 2
    pipeline._link_inference_application_ids([_observation(.95, 50, 3, width=100)], state, 1)
    assert expired["state"] == "EXPIRED"
    assert expired["last_frame"] == -5
    assert not expired["merged_detection_indices"]
    assert 2 in state["overlap_groups"]


def test_visible_but_rejected_byte_does_not_prevent_time_based_expiry():
    for fps in (16.0, 30.0, 49.0):
        pipeline, state = _pipeline(), _state()
        state["source_fps"] = fps
        _seed_track(pipeline, state)
        pipeline._update_occlusion_states([_observation(.95, 500)], state, int(3 * fps) + 1)
        assert state["identities"][1]["state"] == "TERMINATED"


def test_unresolved_evidence_is_retired_when_original_byte_recovers():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state)
    state["identities"][1]["state"] = state["identities"][1]["status"] = "LOST"
    pipeline._defer_unresolved_new_detection(_observation(.95, 12), 0, state, 1)
    result = pipeline._link_inference_application_ids([_observation(.95, 13)], state, 2)
    assert result[0][0] == 1
    assert not state["unresolved_candidates"]
    archive = state["unresolved_candidate_archive"]
    assert archive[0]["state"] == "RESOLVED"
    assert archive[0]["resolved_application_id"] == 1
    assert archive[0]["observations"][0]["confidence"] == .95


def test_clear_joint_assignment_overrides_only_local_near_tie():
    pipeline, state = _pipeline(), _state()
    # ID 1 has a local near tie between X and Y. ID 2 can explain only Y,
    # making A->X/B->Y the uniquely feasible complete assignment.
    pipeline._link_inference_application_ids([_observation(.95, 0, 1), _observation(.95, 30, 2)], state, 0)
    for identity in state["identities"].values():
        identity["state"] = identity["status"] = "UNRESOLVED"
    state["overlap_groups"] = {1: {"overlap_group_id": 1, "identity_ids": [1, 2],
                                  "start_frame": 0, "last_frame": 0, "state": "UNRESOLVED"}}
    result = pipeline._reidentify_occluded_tracks(
        [_observation(.95, -10, 11), _observation(.95, 10, 12)], state, 1, set(), set())
    assert {(row[0], row[1]) for row in result} == {(0, 1), (1, 2)}


def test_predicted_exit_does_not_create_phantom_overlap_at_stale_anchor():
    pipeline, state = _pipeline(), _state()
    state.update(source_fps=16.129, frame_size=(1280, 1024))
    _seed_track(pipeline, state, center_x=355.56, tracker_id=1, frame=194)
    old = state["identities"][1]
    old.update(state="LOST", status="LOST", trusted_center=(355.56, 936.65),
               last_center=(355.56, 936.65), high_conf_history=[], velocity=(-3.61, 27.46))
    _seed_track(pipeline, state, center_x=433, tracker_id=2, frame=214)
    recent = state["identities"][2]
    recent.update(trusted_center=(433, 914), last_center=(433, 914))
    observation = {"tracker_id": 3, "confidence": .95, "center": (410, 937),
                   "bbox": (300, 850, 520, 1024)}
    state["blocked_high_detections"] = set()
    pipeline._mark_merged_occlusion_observations([observation], state, 215)
    assert not state["blocked_high_detections"]
    assert not state["overlap_groups"]


def test_self_consistent_unresolved_chain_cannot_bridge_unrelated_old_identity():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state, center_x=608.37, width=130, frame=0)
    identity = state["identities"][1]
    identity.update(state="UNRESOLVED", status="UNRESOLVED")
    candidate = {"candidate_id": 1, "possible_identities": [1], "last_frame": 7,
                 "observations": [
                     {"frame": 4, "center": (880.11, 10), "bbox": _box(880.11, width=100), "tracker_id": 23},
                     {"frame": 7, "center": (812.01, 10), "bbox": _box(812.01, width=120), "tracker_id": 29}],
                 "byte_track_ids": [23, 29]}
    state["unresolved_candidates"] = {1: candidate}
    detection = _observation(.95, 812.30, 29, width=120)
    assert pipeline._unresolved_candidate_trajectory_evidence(detection, 1, state, 8) is None
    assert pipeline._reidentify_occluded_tracks([detection], state, 8, set(), set()) == []


def test_sustained_ambiguous_overlap_expires_at_same_video_time_for_all_fps():
    for fps in (16, 30, 49):
        pipeline, state = _pipeline(), _state()
        state["source_fps"] = fps
        pipeline._link_inference_application_ids(
            [_observation(.95, 10, 1), _observation(.95, 90, 2)], state, 0)
        for frame in range(1, 3 * fps + 1):
            result = pipeline._link_inference_application_ids(
                [_observation(.95, 50, 3, width=100)], state, frame)
            assert result == [None]
            assert state["next_id"] == 3
        pipeline._link_inference_application_ids([], state, 3 * fps + 1)
        assert all(i["state"] == "TERMINATED" for i in state["identities"].values())
        assert all(g["state"] == "EXPIRED" for g in state["overlap_groups"].values())
        assert all(len({r["frame"] for r in c["observations"]}) == len(c["observations"])
                   for c in state["unresolved_candidates"].values())


def test_existing_candidate_cannot_acquire_a_different_possible_identity():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state, center_x=10, tracker_id=1)
    state["identities"][1]["state"] = state["identities"][1]["status"] = "LOST"
    pipeline._defer_unresolved_new_detection(_observation(.95, 12, 11), 0, state, 1)
    state["identities"][2] = pipeline._new_inference_identity(2, _observation(.95, 20, 2), 0)
    state["identities"][2]["state"] = state["identities"][2]["status"] = "LOST"
    pipeline._defer_unresolved_new_detection(_observation(.95, 13, 11), 0, state, 2)
    assert state["unresolved_candidates"][1]["possible_identities"] == [1]


def test_confirmed_contact_persists_until_separation_at_every_video_fps():
    for fps in (16, 30, 49):
        pipeline, state = _pipeline(), _state()
        state["source_fps"] = fps
        for frame in range(fps + 1):
            results = pipeline._link_inference_application_ids(
                [_observation(.95, 0, 1), _observation(.95, 18, 2)], state, frame)
            assert [row[0] for row in results] == [1, 2]
        assert len(state["overlap_groups"]) == 1
        group = state["overlap_groups"][1]
        assert group["state"] == "CONTACT"
        assert group["start_frame"] == 1 and group["last_frame"] == fps
        results = pipeline._link_inference_application_ids(
            [_observation(.95, -10, 1), _observation(.95, 28, 2)], state, fps + 1)
        assert [row[0] for row in results] == [1, 2]
        assert group["state"] == "RESOLVED"
        pipeline._link_inference_application_ids(
            [_observation(.95, -10, 1), _observation(.95, 28, 2)], state, fps + 2)
        assert len(state["overlap_groups"]) == 1


def test_frame_469_wrong_candidate_cannot_steal_plausible_original_identity():
    pipeline, state = _pipeline(), _state()
    _seed_track(pipeline, state, tracker_id=916, frame=465)
    identity = state["identities"][1]
    old_box = (420.926392, 92.620331, 504.491516, 190.345856)
    old_center = (462.708954, 141.483093)
    identity.update(status="LOST", state="LOST", last_bbox=old_box,
                    trusted_bbox=old_box, last_high_conf_bbox=old_box,
                    last_center=old_center, trusted_center=old_center,
                    last_high_conf_center=old_center, high_conf_history=[])
    state["unresolved_candidates"] = {1: {
        "candidate_id": 1, "possible_identities": [1], "last_frame": 468,
        "observations": [
            {"frame": 466, "center": (389.257202, 185.548096),
             "bbox": (375.284515, 102.333344, 403.229889, 268.762848), "tracker_id": 996},
            {"frame": 468, "center": (387.371429, 310.700768),
             "bbox": (374.371887, 234.708237, 400.370972, 386.693298), "tracker_id": 1011}],
        "byte_track_ids": [996, 1011]}}
    correct = {"tracker_id": 1014, "confidence": .967,
               "center": (426.700012, 123.435358),
               "bbox": (384.956665, 50.099781, 468.443359, 196.770935)}
    wrong = {"tracker_id": 1011, "confidence": .842,
             "center": (387.394623, 373.529922),
             "bbox": (373.884064, 294.463654, 400.905182, 452.596191)}
    recovered = pipeline._reidentify_occluded_tracks([correct, wrong], state, 469, set(), set())
    assert [(r[0], r[1]) for r in recovered] == [(0, 1)]
