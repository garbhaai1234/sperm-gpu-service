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
            {"frame": 1, "center": (40.0, 10.0), "bbox": _box(40.0), "tracker_id": 31},
            {"frame": 2, "center": (50.0, 10.0), "bbox": _box(50.0), "tracker_id": 32},
        ],
        "byte_track_ids": [31, 32],
    }
    state["unresolved_candidates"] = {1: candidate}
    detection = _observation(0.95, center_x=60.0, tracker_id=33)
    recovered = pipeline._reidentify_occluded_tracks([detection], state, 3, set(), set())

    assert len(recovered) == 1
    assert recovered[0][1] == 1
    assert recovered[0][2]["unresolved_candidate_evidence"]["direction"] == (10.0, 0.0)


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
