"""CSRT identity lifecycle. OpenCV's update result is stubbed; the association is real."""

import numpy as np

from sperm_pipeline.tracking import CSRTConfig, CSRTSpermTracker


def _box(cx, cy, width=80.0, height=70.0):
    return [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0]


def _detection(cx, cy):
    bbox = _box(cx, cy)
    return {
        "bbox": bbox,
        "confidence": 0.9,
        "track_id": None,
        "center": ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
        "label": 2,
        "class_name": "sperm",
    }


def _tracker():
    tracker = CSRTSpermTracker(CSRTConfig(detector_interval=10, tracking_debug=False))

    def init(_self, frame, bbox):
        return object(), [float(value) for value in bbox[:4]]

    def update_bbox(csrt_tracker, _frame):
        for state in tracker._tracks.values():
            if state.get("tracker") is not csrt_tracker:
                continue
            x1, y1, x2, y2 = state["bbox"]
            shift = tracker._shift
            return [x1 + shift, y1, x2 + shift, y2]
        return None

    tracker._shift = 0.0
    tracker._init_csrt = init.__get__(tracker, CSRTSpermTracker)
    tracker._csrt_update_bbox = update_bbox
    return tracker


def _frame():
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _ids(detections):
    return [detection.get("track_id") for detection in detections]


def test_drifted_csrt_does_not_mint_new_ids_for_seven_sperm():
    tracker = _tracker()
    frame = _frame()
    centers = [(100.0 + index * 200.0, 120.0) for index in range(7)]
    detections = [_detection(cx, cy) for cx, cy in centers]
    first = tracker.update(detections, 0, frame=frame, reinitialize=True)
    assert _ids(first) == [1, 2, 3, 4, 5, 6, 7]

    tracker._shift = 400.0
    for frame_index in range(1, 10):
        tracker.update([], frame_index, frame=frame, reinitialize=False)

    tracker._shift = 0.0
    recovered = [_detection(cx, cy) for cx, cy in centers]
    second = tracker.update(recovered, 10, frame=frame, reinitialize=True)
    assert _ids(second) == [1, 2, 3, 4, 5, 6, 7]
    assert tracker.diagnostics()["unique_ids_created"] == 7
    assert tracker.diagnostics()["new_ids"] == 7
    assert tracker.diagnostics()["reidentifications"] == 7
    assert tracker.diagnostics()["max_simultaneous_active"] == 7
    frames = [point[0] for point in tracker.get_trajectory(1)]
    assert 0 in frames and 10 in frames
    assert frames == sorted(frames)


def test_rejected_csrt_box_does_not_clear_lost_age_or_split_the_trajectory():
    tracker = _tracker()
    frame = _frame()
    created = tracker.update([_detection(100, 100)], 0, frame=frame, reinitialize=True)
    assert created[0]["track_id"] == 1
    tracker._shift = 400.0
    tracker.update([], 1, frame=frame, reinitialize=False)
    assert tracker._tracks[1]["lost_age"] == 1
    assert tracker._tracks[1]["status"] == "lost"
    tracker._shift = 0.0
    tracker.update([], 2, frame=frame, reinitialize=False)
    assert tracker._tracks[1]["lost_age"] == 2
    assert [point[0] for point in tracker.get_trajectory(1)] == [0]

    recovered = tracker.update([_detection(110, 105)], 10, frame=frame, reinitialize=True)
    assert recovered[0]["track_id"] == 1
    assert [point[0] for point in tracker.get_trajectory(1)] == [0, 10]


def test_ambiguous_crossing_does_not_create_ids():
    tracker = _tracker()
    frame = _frame()
    tracker.update(
        [_detection(100, 100), _detection(180, 100)],
        0,
        frame=frame,
        reinitialize=True,
    )
    crossed = tracker.update(
        [_detection(140, 100), _detection(140, 108)],
        10,
        frame=frame,
        reinitialize=True,
    )
    assert _ids(crossed) == [None, None]
    assert tracker.diagnostics()["unique_ids_created"] == 2
    assert tracker.diagnostics()["id_switches"] == 2


def test_retired_id_is_not_reused_for_a_later_sperm():
    tracker = _tracker()
    tracker.config.max_reid_frames = 5
    tracker.config.strong_reid_max_frames = 5
    frame = _frame()
    tracker.update([_detection(100, 100)], 0, frame=frame, reinitialize=True)
    tracker._shift = 400.0
    for frame_index in range(1, 8):
        reinitialize = frame_index % 10 == 0
        tracker.update([], frame_index, frame=frame, reinitialize=reinitialize)
    assert 1 not in tracker._tracks
    assert tracker.get_trajectory(1)
    created = tracker.update([_detection(100, 100)], 10, frame=frame, reinitialize=True)
    assert created[0]["track_id"] == 2
    assert [point[0] for point in tracker.get_trajectory(1)] == [0]


def test_validated_csrt_motion_keeps_the_same_id():
    tracker = _tracker()
    frame = _frame()
    tracker.update([_detection(100, 100)], 0, frame=frame, reinitialize=True)
    tracker._shift = 5.0
    for frame_index in range(1, 10):
        output = tracker.update([], frame_index, frame=frame, reinitialize=False)
        assert output[0]["track_id"] == 1
    moved = tracker.update([_detection(145, 100)], 10, frame=frame, reinitialize=True)
    assert moved[0]["track_id"] == 1
    assert tracker.diagnostics()["new_ids"] == 1
    assert tracker.diagnostics()["rejected_csrt_predictions"] == 0
    assert tracker.diagnostics()["csrt_predictions"] == 10


def test_strong_overlap_ignores_a_large_box_shape_change():
    """Frame 90: 7.8px and IoU 0.34 must stay the same ID when the box side ratio is 2.45."""
    tracker = _tracker()
    frame = _frame()
    wide = [0.0, 20.0, 40.0, 80.0]
    created = tracker.update([
        {"bbox": wide, "confidence": 0.9, "track_id": None, "center": (20.0, 50.0), "label": 2}
    ], 0, frame=frame, reinitialize=True)
    assert created[0]["track_id"] == 1
    tracker._shift = 400.0
    tracker.update([], 1, frame=frame, reinitialize=False)
    # Same center, side ratio 2.45, IoU about 0.34.
    changed = [-29.0, 30.0, 69.0, 70.0]
    recovered = tracker.update([
        {"bbox": changed, "confidence": 0.9, "track_id": None, "center": (20.0, 50.0), "label": 2}
    ], 10, frame=frame, reinitialize=True)
    assert recovered[0]["track_id"] == 1
    assert tracker.diagnostics()["strong_recoveries"] >= 1
    assert tracker.diagnostics()["recovery_rejected_by_size"] == 0


def test_strong_match_can_recover_after_the_normal_window():
    """Frame 60: 26px and IoU 0.41 at a 40-frame gap is still the same ID."""
    tracker = _tracker()
    frame = _frame()
    original = [0.0, 0.0, 80.0, 70.0]
    tracker.update([
        {"bbox": original, "confidence": 0.9, "track_id": None, "center": (40.0, 35.0), "label": 2}
    ], 0, frame=frame, reinitialize=True)
    tracker._shift = 400.0
    for frame_index in range(1, 40):
        tracker.update([], frame_index, frame=frame, reinitialize=False)
    assert 1 in tracker._tracks
    shifted = [26.0, 0.0, 106.0, 70.0]
    recovered = tracker.update([
        {"bbox": shifted, "confidence": 0.9, "track_id": None, "center": (66.0, 35.0), "label": 2}
    ], 40, frame=frame, reinitialize=True)
    assert recovered[0]["track_id"] == 1


def test_drifted_trajectory_does_not_recover_a_distant_sperm():
    tracker = _tracker()
    frame = _frame()
    tracker.update([_detection(100, 100)], 0, frame=frame, reinitialize=True)
    tracker._shift = 400.0
    tracker.update([], 1, frame=frame, reinitialize=False)
    tracker.trajectories[1] = [
        (0, 100.0, 100.0),
        (7, 430.0, 430.0),
        (10, 500.0, 500.0),
    ]
    created = tracker.update([_detection(500, 500)], 10, frame=frame, reinitialize=True)
    assert created[0]["track_id"] == 2


def test_fast_sperm_keeps_its_id_from_confirmed_velocity():
    """120px from the last center, about 60px from the predicted center, same ID."""
    tracker = _tracker()
    frame = _frame()
    tracker._shift = 0.0
    tracker.update([_detection(100, 100)], 0, frame=frame, reinitialize=True)
    for frame_index, center_x in ((10, 160.0), (20, 220.0)):
        for step in range(frame_index - 9, frame_index):
            tracker.update([], step, frame=frame, reinitialize=False)
        kept = tracker.update([_detection(center_x, 100)], frame_index, frame=frame, reinitialize=True)
        assert kept[0]["track_id"] == 1
    for step in range(21, 30):
        tracker.update([], step, frame=frame, reinitialize=False)
    moved = tracker.update([_detection(340, 100)], 30, frame=frame, reinitialize=True)
    assert moved[0]["track_id"] == 1
    assert tracker.diagnostics()["motion_assisted_matches"] >= 1
    assert tracker.diagnostics()["new_ids"] == 1


def test_motion_prediction_does_not_absorb_a_different_sperm():
    tracker = _tracker()
    frame = _frame()
    tracker._shift = 0.0
    tracker.update([_detection(100, 100)], 0, frame=frame, reinitialize=True)
    tracker.update([_detection(160, 100)], 10, frame=frame, reinitialize=True)
    tracker.update([_detection(220, 100)], 20, frame=frame, reinitialize=True)
    created = tracker.update(
        [_detection(280, 100), _detection(220, 220)],
        30,
        frame=frame,
        reinitialize=True,
    )
    assert _ids(created) == [1, 2]


def test_valid_csrt_bridges_one_detector_miss_and_keeps_the_id():
    tracker = _tracker()
    tracker._shift = 2.0
    frame = _frame()
    tracker.update([_detection(100, 100)], 0, frame=frame, reinitialize=True)
    for frame_index in range(1, 10):
        tracker.update([], frame_index, frame=frame, reinitialize=False)
    missed = tracker.update([], 10, frame=frame, reinitialize=True)
    assert missed == []
    assert tracker._tracks[1]["status"] == "predicted_during_detector_miss"
    assert tracker._tracks[1]["detector_miss_count"] == 1
    assert tracker._tracks[1]["lost_age"] == 0
    assert 10 in [point[0] for point in tracker.get_trajectory(1)]
    for frame_index in range(11, 20):
        tracker.update([], frame_index, frame=frame, reinitialize=False)
    returned = tracker.update([_detection(140, 100)], 20, frame=frame, reinitialize=True)
    assert returned[0]["track_id"] == 1
    assert tracker.diagnostics()["new_ids"] == 1
    assert tracker.diagnostics()["recovered_after_detector_miss"] == 1
    assert tracker._tracks[1]["detector_miss_count"] == 0


def test_detector_miss_does_not_keep_an_invalid_csrt_track():
    tracker = _tracker()
    tracker._shift = 400.0
    frame = _frame()
    tracker.update([_detection(100, 100)], 0, frame=frame, reinitialize=True)
    tracker.update([], 1, frame=frame, reinitialize=False)
    tracker.update([], 10, frame=frame, reinitialize=True)
    assert tracker._tracks[1]["status"] == "lost"
    assert tracker.diagnostics()["detector_misses"] == 0
    assert tracker.diagnostics()["lost_because_csrt_failed"] >= 1


def test_third_detector_miss_retires_the_bridge():
    tracker = _tracker()
    tracker._shift = 0.0
    frame = _frame()
    tracker.update([_detection(100, 100)], 0, frame=frame, reinitialize=True)
    tracker.update([], 10, frame=frame, reinitialize=True)
    tracker.update([], 20, frame=frame, reinitialize=True)
    assert tracker._tracks[1]["status"] == "predicted_during_detector_miss"
    assert tracker._tracks[1]["detector_miss_count"] == 2
    tracker.update([], 30, frame=frame, reinitialize=True)
    assert tracker._tracks[1]["status"] == "lost"
    assert tracker.diagnostics()["new_ids"] == 1
    assert tracker.detector_miss_report()["two_detector_misses_survived"] == 1


def test_a_distant_detection_during_a_miss_gets_its_own_id():
    tracker = _tracker()
    tracker._shift = 0.0
    frame = _frame()
    tracker.update([_detection(100, 100)], 0, frame=frame, reinitialize=True)
    created = tracker.update([_detection(400, 400)], 10, frame=frame, reinitialize=True)
    assert created[0]["track_id"] == 2
    assert tracker._tracks[1]["status"] == "predicted_during_detector_miss"
    assert tracker.diagnostics()["new_ids"] == 2


def test_far_detection_does_not_recover():
    tracker = _tracker()
    frame = _frame()
    tracker.update([_detection(100, 100)], 0, frame=frame, reinitialize=True)
    tracker._shift = 400.0
    tracker.update([], 1, frame=frame, reinitialize=False)
    created = tracker.update([_detection(500, 500)], 10, frame=frame, reinitialize=True)
    assert created[0]["track_id"] == 2
