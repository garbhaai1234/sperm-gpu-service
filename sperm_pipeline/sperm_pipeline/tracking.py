"""
Sperm Tracking Module
Handles trajectory tracking and management for detected sperm
"""

import cv2
import math
import numpy as np
from dataclasses import dataclass, replace
from typing import List, Dict, Tuple, Optional, Callable
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class SpermTracker:
    """
    Manages analysis identities using motion-aware, one-to-one association.
    """

    # Initial association gates retain the existing 80px operating envelope.
    # They are named so video-specific calibration can be done without changing
    # the matching implementation.
    MAX_MATCH_DISTANCE = 80.0
    MAX_DISTANCE_PER_GAP = 12.0
    MAX_PREDICTION_HORIZON = 10
    MAX_LOST_FRAMES = 10
    MAX_HISTORY_POINTS = 5
    MAX_BBOX_AREA_RATIO = 4.0
    AMBIGUITY_MARGIN = 0.02
    INVALID_COST = 1.0e6

    def __init__(self, max_trajectory_length: int = 1000, max_match_distance: float = 80.0):
        """
        Args:
            max_trajectory_length: Maximum number of points to keep per trajectory
            max_match_distance: Maximum pixel distance to match a detection to an existing track
        """
        self.max_trajectory_length = max_trajectory_length
        self.max_match_distance = float(max_match_distance)
        self.trajectories: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
        self.active_tracks: set = set()
        self.next_id = 1
        self.frame_idx = 0
        self._lost_age: Dict[int, int] = {}  # track_id -> frames since last seen
        self.max_lost_frames = self.MAX_LOST_FRAMES
        self.last_bboxes: Dict[int, List[float]] = {}
        self.last_confidence: Dict[int, Optional[float]] = {}
        self.track_status: Dict[int, str] = {}
        self.association_diagnostics: List[Dict] = []
        self.diagnostics_counts = defaultdict(int)

    @staticmethod
    def _valid_center(center) -> bool:
        try:
            return center is not None and len(center) >= 2 and np.isfinite(
                [float(center[0]), float(center[1])]
            ).all()
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _bbox(detection: Dict) -> Optional[List[float]]:
        bbox = detection.get("bbox")
        try:
            values = [float(value) for value in bbox[:4]]
        except (TypeError, ValueError, IndexError):
            return None
        if len(values) != 4 or not np.isfinite(values).all() or values[2] <= values[0] or values[3] <= values[1]:
            return None
        return values

    @staticmethod
    def _distance(first, second) -> float:
        return float(np.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1])))

    @staticmethod
    def _iou(first, second) -> float:
        if first is None or second is None:
            return 0.0
        ix1, iy1 = max(first[0], second[0]), max(first[1], second[1])
        ix2, iy2 = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = area_a + area_b - intersection
        return float(intersection / union) if union > 0 else 0.0

    @staticmethod
    def _area_ratio(first, second) -> Optional[float]:
        if first is None or second is None:
            return None
        area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        if min(area_a, area_b) <= 0:
            return None
        return float(max(area_a, area_b) / min(area_a, area_b))

    def _motion(self, track_id: int, frame_idx: int):
        points = self.trajectories.get(track_id, [])[-self.MAX_HISTORY_POINTS:]
        if len(points) < 2:
            return (0.0, 0.0), False, (0.0, 0.0)
        slopes = []
        for previous, current in zip(points, points[1:]):
            elapsed = int(current[0]) - int(previous[0])
            if elapsed > 0:
                slopes.append(((current[1] - previous[1]) / elapsed,
                               (current[2] - previous[2]) / elapsed))
        if not slopes:
            return (0.0, 0.0), False, (0.0, 0.0)
        velocity = (float(np.median([pair[0] for pair in slopes])),
                    float(np.median([pair[1] for pair in slopes])))
        direction = (points[-1][1] - points[0][1], points[-1][2] - points[0][2])
        return velocity, True, direction

    def _candidate(self, track_id: int, detection: Dict, frame_idx: int) -> Dict:
        points = self.trajectories[track_id]
        last_frame, last_x, last_y = points[-1]
        last_center = (float(last_x), float(last_y))
        gap = max(1, int(frame_idx) - int(last_frame))
        velocity, has_motion, direction = self._motion(track_id, frame_idx)
        horizon = min(gap, self.MAX_PREDICTION_HORIZON)
        predicted = (last_center[0] + velocity[0] * horizon,
                     last_center[1] + velocity[1] * horizon)
        center = detection["center"]
        last_distance = self._distance(last_center, center)
        predicted_distance = self._distance(predicted, center)
        old_box = self.last_bboxes.get(track_id)
        new_box = self._bbox(detection)
        overlap = self._iou(old_box, new_box)
        size_ratio = self._area_ratio(old_box, new_box)
        distance_gate = self.max_match_distance + self.MAX_DISTANCE_PER_GAP * (gap - 1)
        # Use motion prediction as the primary gate after a track has enough
        # history; retain last-center proximity as supporting evidence.
        gate_distance = predicted_distance if has_motion else last_distance
        distance_ok = gate_distance <= distance_gate
        geometry_ok = size_ratio is None or size_ratio <= self.MAX_BBOX_AREA_RATIO or overlap >= 0.30
        direction_consistency = None
        motion_ok = True
        displacement = (center[0] - last_center[0], center[1] - last_center[1])
        if has_motion and np.hypot(*direction) > 0.5 and np.hypot(*displacement) > 0.5:
            cosine = float(np.dot(direction, displacement) /
                           (np.hypot(*direction) * np.hypot(*displacement)))
            direction_consistency = max(-1.0, min(1.0, cosine))
            motion_ok = direction_consistency >= -0.25
        valid = distance_ok and geometry_ok and motion_ok
        # Normalized motion cost plus weak geometry/confidence terms. A high
        # confidence value can never make a gated-out candidate valid.
        cost = (0.63 * gate_distance / max(distance_gate, 1.0)
                + 0.20 * last_distance / max(distance_gate * 1.5, 1.0)
                + 0.10 * (1.0 - overlap)
                + 0.05 * (min(size_ratio or 1.0, self.MAX_BBOX_AREA_RATIO) - 1.0)
                / max(self.MAX_BBOX_AREA_RATIO - 1.0, 1.0))
        confidence = detection.get("confidence")
        if confidence is not None:
            cost += 0.02 * (1.0 - max(0.0, min(1.0, float(confidence))))
        return {
            "track_id": int(track_id), "last_center": list(last_center),
            "predicted_center": list(predicted), "detection_center": list(center),
            "last_center_distance": last_distance,
            "predicted_center_distance": predicted_distance, "iou": overlap,
            "bbox_size_ratio": size_ratio, "frame_gap": gap,
            "motion_consistency": direction_consistency, "motion_ok": motion_ok,
            "distance_ok": distance_ok, "geometry_ok": geometry_ok,
            "cost": float(cost), "valid": bool(valid),
            "detection_confidence": None if confidence is None else float(confidence),
            "reason": ("ACCEPTED" if valid else "MOTION_FAILED" if not motion_ok
                       else "DISTANCE_FAILED" if not distance_ok else "GEOMETRY_FAILED"),
        }

    def update(self, detections: List[Dict], frame_idx: int) -> List[Dict]:
        """
        Match detections to active/recently lost tracks using motion-aware
        gated global assignment. Ambiguous observations are left unassigned.

        Args:
            detections: List of dicts with 'center' (x, y) and 'bbox' keys
            frame_idx: Current frame index from the pipeline

        Returns:
            The same detections list with 'track_id' assigned on each dict
        """
        self.frame_idx = frame_idx

        valid_detections = [i for i, detection in enumerate(detections)
                            if self._valid_center(detection.get("center"))]
        candidates = {}
        track_ids = [tid for tid in self.trajectories
                     if self.trajectories[tid]
                     and int(frame_idx) - int(self.trajectories[tid][-1][0]) <= self.max_lost_frames]
        for di in valid_detections:
            for tid in track_ids:
                candidates[(di, tid)] = self._candidate(tid, detections[di], frame_idx)

        # Mark near-equal alternatives ambiguous before solving. This applies
        # on either side of the bipartite graph, preventing forced crossings.
        ambiguous_dets, ambiguous_tracks = set(), set()
        for di in valid_detections:
            ranked = sorted((c for (idx, _), c in candidates.items()
                             if idx == di and c["valid"]), key=lambda c: c["cost"])
            if len(ranked) > 1 and ranked[1]["cost"] - ranked[0]["cost"] <= self.AMBIGUITY_MARGIN:
                ambiguous_dets.add(di)
        for tid in track_ids:
            ranked = sorted((c for (_, track), c in candidates.items()
                             if track == tid and c["valid"]), key=lambda c: c["cost"])
            if len(ranked) > 1 and ranked[1]["cost"] - ranked[0]["cost"] <= self.AMBIGUITY_MARGIN:
                ambiguous_tracks.add(tid)
        for di in ambiguous_dets:
            ambiguous_tracks.update(
                track_id for (index, track_id), candidate in candidates.items()
                if index == di and candidate["valid"]
            )

        usable_dets = [di for di in valid_detections if di not in ambiguous_dets]
        usable_tracks = [tid for tid in track_ids if tid not in ambiguous_tracks]
        assignments = {}
        if usable_dets and usable_tracks:
            matrix = np.full((len(usable_dets), len(usable_tracks)), self.INVALID_COST)
            for row, di in enumerate(usable_dets):
                for col, tid in enumerate(usable_tracks):
                    candidate = candidates[(di, tid)]
                    if candidate["valid"]:
                        matrix[row, col] = candidate["cost"]
            try:
                from scipy.optimize import linear_sum_assignment
                rows, cols = linear_sum_assignment(matrix)
                pairs = zip(rows.tolist(), cols.tolist())
            except Exception:
                pairs_list = sorted((matrix[row, col], row, col)
                                    for row in range(matrix.shape[0])
                                    for col in range(matrix.shape[1])
                                    if matrix[row, col] < self.INVALID_COST)
                used_rows, used_cols, selected = set(), set(), []
                for _cost, row, col in pairs_list:
                    if row not in used_rows and col not in used_cols:
                        selected.append((row, col))
                        used_rows.add(row)
                        used_cols.add(col)
                pairs = selected
            for row, col in pairs:
                if matrix[row, col] < self.INVALID_COST:
                    assignments[usable_dets[row]] = usable_tracks[col]

        current_seen = set()
        for di, detection in enumerate(detections):
            center = detection.get("center")
            detection["track_id"] = None
            if di not in valid_detections:
                continue
            if di in assignments:
                tid = assignments[di]
            else:
                plausible = [c for (idx, _), c in candidates.items()
                             if idx == di and c["valid"]]
                if plausible:
                    reason = "AMBIGUOUS"
                    self.diagnostics_counts[reason] += 1
                    self._record_diagnostic(frame_idx, di, None, detection, None,
                                            plausible, reason, False)
                    continue
                tid = self.next_id
                self.next_id += 1
                reason = "NEW_TRACK"
                self.diagnostics_counts[reason] += 1
            if tid in assignments.values():
                candidate = candidates.get((di, tid))
                last_frame = self.trajectories[tid][-1][0]
                prior_status = self.track_status.get(tid)
                if prior_status == "OCCLUDED":
                    reason = "OCCLUSION_RECOVERY"
                elif int(frame_idx) - int(last_frame) <= 1:
                    reason = "ACTIVE_CONTINUATION"
                else:
                    reason = "LOST_REASSOCIATION"
                self.diagnostics_counts[reason] += 1
            else:
                candidate = None
            detection["track_id"] = tid
            self.trajectories[tid].append((int(frame_idx), float(center[0]), float(center[1])))
            if len(self.trajectories[tid]) > self.max_trajectory_length:
                self.trajectories[tid] = self.trajectories[tid][-self.max_trajectory_length:]
            bbox = self._bbox(detection)
            if bbox is not None:
                self.last_bboxes[tid] = bbox
            self.last_confidence[tid] = detection.get("confidence")
            self.track_status[tid] = "CONFIRMED"
            self._lost_age.pop(tid, None)
            current_seen.add(tid)
            per_detection = [value for (index, _), value in candidates.items() if index == di]
            self._record_diagnostic(frame_idx, di, tid, detection, candidate,
                                    per_detection, reason, True)

        # Lost state is derived from source frame indices, so frame skipping
        # does not accidentally extend or shorten the configured recovery age.
        for tid in list(self.trajectories):
            if tid in current_seen:
                continue
            last_seen = int(self.trajectories[tid][-1][0])
            age = max(0, int(frame_idx) - last_seen)
            if age <= self.max_lost_frames:
                self._lost_age[tid] = age
                self.track_status[tid] = "OCCLUDED" if tid in ambiguous_tracks else "LOST"
            else:
                already_reported = self.track_status.get(tid) == "TERMINATED_REPORTED"
                self.track_status[tid] = "TERMINATED"
                self._lost_age.pop(tid, None)
                if not already_reported:
                    self.association_diagnostics.append({
                        "frame": int(frame_idx), "detection_index": None,
                        "track_id": int(tid), "accepted": False,
                        "ambiguous": False, "reason": "TRACK_EXPIRED",
                        "frame_gap": age,
                    })
                    self.diagnostics_counts["TRACK_EXPIRED"] += 1
                    self.track_status[tid] = "TERMINATED_REPORTED"
        self.active_tracks = {tid for tid, status in self.track_status.items()
                              if status in {"CONFIRMED", "LOST", "OCCLUDED"}}
        return detections

    def _record_diagnostic(self, frame_idx, detection_index, track_id, detection,
                           selected, candidates, reason, accepted):
        alternatives = sorted((candidate for candidate in candidates
                               if candidate.get("valid")), key=lambda item: item["cost"])
        row = {
            "frame": int(frame_idx), "detection_index": int(detection_index),
            "track_id": track_id,
            "detection_confidence": detection.get("confidence"),
            "last_center": None if selected is None else selected.get("last_center"),
            "predicted_center": None if selected is None else selected.get("predicted_center"),
            "detection_center": detection.get("center"),
            "last_center_distance": None if selected is None else selected.get("last_center_distance"),
            "predicted_distance": None if selected is None else selected.get("predicted_center_distance"),
            "iou": None if selected is None else selected.get("iou"),
            "bbox_size_ratio": None if selected is None else selected.get("bbox_size_ratio"),
            "frame_gap": None if selected is None else selected.get("frame_gap"),
            "motion_consistency": None if selected is None else selected.get("motion_consistency"),
            "assignment_cost": None if selected is None else selected.get("cost"),
            "best_candidate": alternatives[0] if alternatives else None,
            "second_best_candidate": alternatives[1] if len(alternatives) > 1 else None,
            "ambiguous": reason == "AMBIGUOUS", "accepted": bool(accepted),
            "reason": reason,
        }
        row["candidate_evaluations"] = candidates
        self.association_diagnostics.append(row)
        if len(self.association_diagnostics) > 1_000_000:
            del self.association_diagnostics[:100_000]

    # Keep backward compatibility with old API
    def update_trajectories(self, detections: List[Dict]) -> Dict[int, List[Tuple]]:
        self.update(detections, self.frame_idx)
        self.frame_idx += 1
        return dict(self.trajectories)

    def get_trajectory(self, track_id: int) -> List[Tuple[int, float, float]]:
        return self.trajectories.get(track_id, [])

    def get_all_trajectories(self) -> Dict[int, List[Tuple[int, float, float]]]:
        return dict(self.trajectories)

    def get_active_trajectories(self) -> Dict[int, List[Tuple[int, float, float]]]:
        return {tid: traj for tid, traj in self.trajectories.items()
                if tid in self.active_tracks}

    def reset(self):
        self.trajectories.clear()
        self.active_tracks.clear()
        self._lost_age.clear()
        self.last_bboxes.clear()
        self.last_confidence.clear()
        self.track_status.clear()
        self.association_diagnostics.clear()
        self.diagnostics_counts.clear()
        self.next_id = 1
        self.frame_idx = 0


@dataclass
class CSRTConfig:
    """Configurable CSRT parameters. Defaults are initial values, not tuned ones.

    Association distances are in pixels. On the 1280x1024 test video, boxes that
    stayed on one sperm moved about 2px in one frame (90th percentile about 11px,
    99th about 50px) and about 6px across 10 frames (90th percentile about 42px,
    99th about 190px). The gates below use those measurements. They are not
    clinical speeds.
    """

    padding: float = 3.0
    psr_threshold: float = 0.035
    filter_lr: float = 0.02
    template_size: float = 200.0
    number_of_scales: int = 33
    scale_step: float = 1.02
    scale_lr: float = 0.025
    use_segmentation: bool = True
    detector_interval: int = 10
    max_lost_frames: int = 10
    min_match_iou: float = 0.3
    max_trajectory_length: int = 1000
    # Live detector match. Same 80px center gate the previous CSRT matcher used.
    max_match_distance: float = 80.0
    # A CSRT box is rejected when it jumps farther than this from the last
    # accepted center, or from the position implied by the last detector fix.
    # 60px sits just above the measured one-frame 99th percentile (~50px).
    max_csrt_center_jump: float = 60.0
    max_csrt_area_change: float = 2.5
    min_csrt_iou: float = 0.05
    # How long a missed sperm can be recovered before its ID is retired.
    # 30 frames is about 1.9s at 16.13 fps, three detector cycles at interval 10.
    max_reid_frames: int = 30
    # Cap on the recovery search. The per-frame growth below reaches 188px at a
    # 10-frame gap, which covers the measured 10-frame 99th percentile (~190px).
    max_reid_distance: float = 200.0
    reid_distance_per_extra_frame: float = 12.0
    min_reid_iou: float = 0.05
    max_size_ratio: float = 2.0
    # A close, overlapping detection is the same sperm even if the Mask R-CNN
    # box changed shape. Frame 90 of the microscope video was 7.8px away with
    # IoU 0.34 and a side ratio of 2.45; the size gate alone must not split it.
    strong_reid_iou: float = 0.25
    strong_reid_distance: float = 40.0
    # Normal recovery stays at max_reid_frames (30, about 1.9s at 16.13 fps).
    # A strong match may wait longer. 60 frames is about 3.7s, which covers the
    # frame-60 case (26px, IoU 0.41, 40 frames since the last detector fix).
    strong_reid_max_frames: int = 60
    # Assignments whose costs are this close are left unmatched. A wrong ID
    # swap is worse than one missed frame.
    ambiguity_margin: float = 0.12
    # Motion-assisted live match. The 80px gate above is unchanged.
    # At 16.13 fps and detector_interval 10, confirmations are about 0.62s apart.
    # A steady 12px/frame sperm is then about 120px from its last confirmation
    # but only a few pixels from the position predicted by that velocity.
    # Velocity is pixels per frame, from actual frame indices, so it follows the
    # video clock instead of assuming 30 fps.
    use_motion_prediction: bool = True
    prediction_history_frames: int = 5
    max_prediction_distance: float = 120.0
    prediction_weight: float = 0.4
    velocity_smoothing: float = 0.5
    max_prediction_horizon: int = 15
    # Displacement must point the same way as the recent velocity. 50 degrees
    # leaves room for a curved path without accepting a reversal.
    max_direction_difference_deg: float = 50.0
    # How many consecutive detector frames a confirmed ID may miss while CSRT
    # is still valid. 2 intervals is about 1.24s at 16.13 fps with interval 10.
    # A third miss, or any rejected CSRT box, sends the track to lost.
    max_detector_miss_intervals: int = 2
    tracking_debug: bool = False

    def with_detector_interval(self, detector_interval: int) -> "CSRTConfig":
        return replace(self, detector_interval=int(detector_interval))


def _find_csrt_backend() -> Optional[Tuple[Callable, Optional[type]]]:
    """Return the first available CSRT constructor and its parameter type."""
    legacy = getattr(cv2, "legacy", None)
    candidates = []

    tracker_cls = getattr(cv2, "TrackerCSRT", None)
    if tracker_cls is not None:
        candidates.append((
            getattr(tracker_cls, "create", None),
            getattr(cv2, "TrackerCSRT_Params", None),
        ))
    if legacy is not None:
        legacy_cls = getattr(legacy, "TrackerCSRT", None)
        if legacy_cls is not None:
            candidates.append((
                getattr(legacy_cls, "create", None),
                getattr(legacy, "TrackerCSRT_Params", None) or getattr(cv2, "TrackerCSRT_Params", None),
            ))
    candidates.append((
        getattr(cv2, "TrackerCSRT_create", None),
        getattr(cv2, "TrackerCSRT_Params", None),
    ))
    if legacy is not None:
        candidates.append((
            getattr(legacy, "TrackerCSRT_create", None),
            getattr(legacy, "TrackerCSRT_Params", None) or getattr(cv2, "TrackerCSRT_Params", None),
        ))

    for create, params_type in candidates:
        if callable(create):
            return create, params_type if isinstance(params_type, type) else None
    return None


_CSRT_BACKEND: Optional[Tuple[Callable, Optional[type]]] = None
_CSRT_BACKEND_RESOLVED = False


def _resolve_csrt_backend() -> Optional[Tuple[Callable, Optional[type]]]:
    global _CSRT_BACKEND, _CSRT_BACKEND_RESOLVED
    if not _CSRT_BACKEND_RESOLVED:
        _CSRT_BACKEND = _find_csrt_backend()
        _CSRT_BACKEND_RESOLVED = True
    return _CSRT_BACKEND


def csrt_available() -> bool:
    return _resolve_csrt_backend() is not None


class CSRTSpermTracker:
    """
    One OpenCV CSRT tracker per sperm.

    Detector frames are the only confirmation of an identity. A CSRT box is a
    prediction: it is stored only when it stays near the last confirmed sperm,
    and it never clears the lost counter. A detection that misses every live
    track is offered to recently lost tracks before a new ID is created.
    """

    def __init__(self, config: Optional[CSRTConfig] = None):
        self.config = config or CSRTConfig()
        backend = _resolve_csrt_backend()
        if backend is None:
            raise RuntimeError(
                "CSRT tracking was selected, but OpenCV CSRT is not available "
                "(cv2.TrackerCSRT.create, cv2.legacy.TrackerCSRT.create, "
                "cv2.TrackerCSRT_create, cv2.legacy.TrackerCSRT_create). "
                "opencv-python does not include CSRT. Install opencv-contrib-python "
                "in the application environment before using tracking_method='csrt'. "
                "The distance-based tracker was not used as a fallback."
            )
        self._backend = backend
        self.trajectories: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
        self.active_tracks: set = set()
        self.frame_idx = 0
        self._lost_age: Dict[int, int] = {}
        self._tracks: Dict[int, Dict] = {}
        self._masks: Dict[int, np.ndarray] = {}
        self._mask_centroids: Dict[int, Tuple[float, float]] = {}
        self._next_track_id = 1
        self._diagnostics = self._empty_diagnostics()
        self.recovery_log: List[Dict] = []
        self.new_id_log: List[Dict] = []
        self._miss_survival: Dict[int, int] = {}

    def is_detector_frame(self, frame_idx: int) -> bool:
        interval = int(self.config.detector_interval)
        if interval <= 1:
            return True
        return int(frame_idx) % interval == 0

    def update(
        self,
        detections: List[Dict],
        frame_idx: int,
        frame: Optional[np.ndarray] = None,
        reinitialize: bool = False,
    ) -> List[Dict]:
        """
        Track sperm on one frame.

        reinitialize=True corrects trackers from detector observations.
        reinitialize=False ignores detections and returns CSRT boxes only.
        """
        self.frame_idx = frame_idx
        image = self._prepare_frame(frame)
        if reinitialize:
            return self._update_with_detections(detections or [], image, frame_idx)
        return self._update_tracks_only(image, frame_idx)

    def update_trajectories(
        self,
        detections: List[Dict],
        frame: Optional[np.ndarray] = None,
    ) -> Dict[int, List[Tuple]]:
        self.update(
            detections,
            self.frame_idx,
            frame=frame,
            reinitialize=self.is_detector_frame(self.frame_idx),
        )
        self.frame_idx += 1
        return dict(self.trajectories)

    def get_trajectory(self, track_id: int) -> List[Tuple[int, float, float]]:
        return self.trajectories.get(track_id, [])

    def get_all_trajectories(self) -> Dict[int, List[Tuple[int, float, float]]]:
        return dict(self.trajectories)

    def get_active_trajectories(self) -> Dict[int, List[Tuple[int, float, float]]]:
        return {tid: traj for tid, traj in self.trajectories.items()
                if tid in self.active_tracks}

    def remember_mask(
        self,
        track_id: int,
        mask: np.ndarray,
        centroid: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Store a real segmentation mask for later display. Does not reinitialize CSRT."""
        if track_id not in self._tracks or mask is None:
            return
        stored = np.asarray(mask, dtype=np.float32)
        if stored.ndim == 3:
            stored = stored[0]
        self._masks[track_id] = stored.copy()
        if centroid is not None and centroid[0] is not None and centroid[1] is not None:
            self._mask_centroids[track_id] = (float(centroid[0]), float(centroid[1]))

    def get_mask_state(
        self,
        track_id: int,
    ) -> Optional[Tuple[np.ndarray, Optional[Tuple[float, float]]]]:
        mask = self._masks.get(track_id)
        if mask is None:
            return None
        return mask, self._mask_centroids.get(track_id)

    def reset(self):
        self.trajectories.clear()
        self.active_tracks.clear()
        self._lost_age.clear()
        self._tracks.clear()
        self._masks.clear()
        self._mask_centroids.clear()
        self.frame_idx = 0
        self._next_track_id = 1
        self._diagnostics = self._empty_diagnostics()
        self.recovery_log = []
        self.new_id_log = []
        self._miss_survival = {}

    @staticmethod
    def _empty_diagnostics() -> Dict[str, int]:
        return {
            "unique_ids_created": 0,
            "max_simultaneous_active": 0,
            "detector_frames": 0,
            "new_ids": 0,
            "reidentifications": 0,
            "id_switches": 0,
            "csrt_predictions": 0,
            "rejected_csrt_predictions": 0,
            "dropped_tracks": 0,
            "strong_recoveries": 0,
            "normal_recoveries": 0,
            "recovery_rejected_by_size": 0,
            "recovery_rejected_by_age": 0,
            "recovery_rejected_by_distance": 0,
            "recovery_rejected_by_iou": 0,
            "new_true_new_detection": 0,
            "new_active_match_failed": 0,
            "new_other": 0,
            "motion_assisted_matches": 0,
            "motion_assisted_recoveries": 0,
            "detector_misses": 0,
            "recovered_after_detector_miss": 0,
            "lost_because_csrt_failed": 0,
        }

    def diagnostics(self) -> Dict[str, int]:
        """Counts for this tracker instance. Analysis and inference each have their own."""
        values = dict(self._diagnostics)
        values["unique_ids_created"] = max(0, int(self._next_track_id) - 1)
        values["max_track_id"] = values["unique_ids_created"]
        return values

    def diagnostics_summary(self) -> str:
        values = self.diagnostics()
        return "\n".join([
            f"Total unique IDs created: {values['unique_ids_created']}",
            f"Maximum simultaneous active tracks: {values['max_simultaneous_active']}",
            f"Number of detector frames: {values['detector_frames']}",
            f"Number of new IDs: {values['new_ids']}",
            f"Number of re-identifications: {values['reidentifications']}",
            f"Strong recoveries: {values['strong_recoveries']}",
            f"Normal recoveries: {values['normal_recoveries']}",
            f"Recovery rejected by size: {values['recovery_rejected_by_size']}",
            f"Recovery rejected by age: {values['recovery_rejected_by_age']}",
            f"Recovery rejected by distance: {values['recovery_rejected_by_distance']}",
            f"Recovery rejected by IoU: {values['recovery_rejected_by_iou']}",
            f"Number of ID switches: {values['id_switches']}",
            f"Number of CSRT predictions: {values['csrt_predictions']}",
            f"Number of rejected CSRT predictions: {values['rejected_csrt_predictions']}",
            f"Number of dropped tracks: {values['dropped_tracks']}",
            f"Motion-assisted matches: {values['motion_assisted_matches']}",
            f"Motion-assisted recoveries: {values['motion_assisted_recoveries']}",
            f"Detector misses bridged: {values['detector_misses']}",
            f"Tracks recovered after a detector miss: {values['recovered_after_detector_miss']}",
            f"Tracks lost because CSRT failed: {values['lost_because_csrt_failed']}",
        ])

    def detector_miss_report(self) -> Dict[str, int]:
        """How many tracks stayed alive through consecutive Mask R-CNN misses."""
        depths = list(self._miss_survival.values())
        return {
            "tracks_surviving_at_least_1": sum(1 for depth in depths if depth >= 1),
            "tracks_surviving_at_least_2": sum(1 for depth in depths if depth >= 2),
            "tracks_surviving_at_least_3": sum(1 for depth in depths if depth >= 3),
            "one_detector_miss_survived": sum(1 for depth in depths if depth == 1),
            "two_detector_misses_survived": sum(1 for depth in depths if depth == 2),
            "three_detector_misses_survived": sum(1 for depth in depths if depth == 3),
            "more_than_three_detector_misses_survived": sum(1 for depth in depths if depth > 3),
        }

    def _prepare_frame(self, frame: Optional[np.ndarray]) -> np.ndarray:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("CSRTSpermTracker.update requires a BGR image frame")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        return frame

    def _make_tracker(self):
        create, params_type = self._backend
        params = None
        if params_type is not None:
            params = params_type()
            values = {
                "padding": float(self.config.padding),
                "psr_threshold": float(self.config.psr_threshold),
                "filter_lr": float(self.config.filter_lr),
                "template_size": float(self.config.template_size),
                "number_of_scales": int(self.config.number_of_scales),
                "scale_step": float(self.config.scale_step),
                "scale_lr": float(self.config.scale_lr),
                "use_segmentation": bool(self.config.use_segmentation),
            }
            for name, value in values.items():
                if not hasattr(params, name):
                    continue
                try:
                    setattr(params, name, value)
                except Exception as exc:
                    logger.warning("Could not set CSRT parameter %s: %s", name, exc)
        else:
            logger.warning("OpenCV CSRT parameter object is unavailable; using tracker defaults")

        attempts = []
        if params is not None:
            attempts.append(lambda: create(params))
            attempts.append(lambda: create(parameters=params))
        attempts.append(lambda: create())
        errors = []
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                errors.append(exc)
        raise RuntimeError(f"OpenCV CSRT could not be created: {errors[-1] if errors else 'unknown error'}")

    def _init_csrt(self, frame: np.ndarray, bbox) -> Tuple[Optional[object], Optional[List[float]]]:
        xywh = self._xyxy_to_xywh(bbox, frame.shape)
        if xywh is None:
            return None, None
        # OpenCV 4.10 CSRT rejects float boxes. Pass a clipped integer box.
        height, width = frame.shape[:2]
        x = max(0, min(int(np.floor(xywh[0])), width - 1))
        y = max(0, min(int(np.floor(xywh[1])), height - 1))
        box_w = max(1, min(int(np.ceil(xywh[2])), width - x))
        box_h = max(1, min(int(np.ceil(xywh[3])), height - y))
        integer_box = (int(x), int(y), int(box_w), int(box_h))
        try:
            tracker = self._make_tracker()
            ok = tracker.init(frame, integer_box)
        except Exception as exc:
            logger.warning("CSRT initialization failed: %s", exc)
            return None, None
        if ok is False:
            logger.warning("CSRT initialization rejected the bounding box")
            return None, None
        return tracker, self._xywh_to_xyxy(integer_box, frame.shape)

    def _csrt_update_bbox(self, tracker, frame: np.ndarray) -> Optional[List[float]]:
        try:
            result = tracker.update(frame)
        except Exception as exc:
            logger.debug("CSRT update failed: %s", exc)
            return None
        if isinstance(result, tuple):
            if len(result) >= 2 and isinstance(result[0], (bool, np.bool_)):
                if not bool(result[0]):
                    return None
                box = result[1]
            elif len(result) >= 1 and not isinstance(result[0], (bool, np.bool_)):
                box = result[0]
            else:
                return None
        else:
            box = result
        if box is None:
            return None
        try:
            return self._xywh_to_xyxy(box, frame.shape)
        except Exception:
            return None

    def _update_with_detections(
        self,
        detections: List[Dict],
        frame: np.ndarray,
        frame_idx: int,
    ) -> List[Dict]:
        """Match detector boxes to tracks. CSRT success on this frame is not confirmation."""
        self._diagnostics["detector_frames"] += 1
        predictions = {
            tid: self._predict_csrt(state, frame, frame_idx, count_diagnostics=True)
            for tid, state in list(self._tracks.items())
        }
        assignments, ambiguous = self._assign_detections(detections, predictions, frame_idx)

        seen = set()
        for det_index, detection in enumerate(detections):
            bbox = detection.get("bbox")
            if not self._valid_xyxy(bbox):
                continue
            if det_index in ambiguous:
                rival = ambiguous[det_index]
                self._log_track_event(
                    frame_idx, det_index, rival.get("track_id"), "UNMATCHED_TRACK",
                    rival.get("iou", 0.0), rival.get("distance"), rival.get("lost_age", 0),
                    rival.get("update_success", False), "ambiguous",
                )
                continue
            if det_index not in assignments:
                created_id = self._create_track(detection, bbox, frame, frame_idx, det_index)
                if created_id is not None:
                    seen.add(created_id)
                continue

            info = assignments[det_index]
            tid = info["track_id"]
            state = self._tracks.get(tid)
            if state is None:
                continue
            tracker, clipped = self._init_csrt(frame, bbox)
            if tracker is None or clipped is None:
                logger.warning("CSRT tracker was not reinitialized for detection %s", det_index)
                self._advance_lost(tid, frame_idx, det_index, "init_failed")
                continue
            self._confirm_track(tid, state, detection, clipped, tracker, frame_idx, info, det_index)
            seen.add(tid)

        for tid in list(self._tracks.keys()):
            if tid in seen:
                continue
            state = self._tracks.get(tid)
            prediction = predictions.get(tid) or {}
            already_lost = state is None or state.get("status") == "lost" or int(state.get("lost_age", 0)) > 0
            if already_lost:
                self._advance_lost(tid, frame_idx, None, "unmatched_detection", prediction=prediction)
                continue
            if self._hold_through_detector_miss(tid, frame_idx, prediction):
                continue
            reason = "detector_miss_limit" if prediction.get("valid") else (prediction.get("reason") or "csrt_invalid")
            self._advance_lost(tid, frame_idx, None, reason, prediction=prediction)

        self._finish_frame()
        return detections

    def _update_tracks_only(self, frame: np.ndarray, frame_idx: int) -> List[Dict]:
        """Store a CSRT box only when it passes the spatial checks. Never clear lost_age here."""
        outputs: List[Dict] = []
        for tid, state in list(self._tracks.items()):
            if state.get("status") == "lost" or int(state.get("lost_age", 0)) > 0 or state.get("tracker") is None:
                self._advance_lost(tid, frame_idx, None, "already_lost")
                continue
            prediction = self._predict_csrt(state, frame, frame_idx, count_diagnostics=True)
            if not prediction["valid"]:
                self._advance_lost(tid, frame_idx, None, prediction["reason"], prediction=prediction)
                continue
            bbox = prediction["bbox"]
            center = self._center_of(bbox)
            state["bbox"] = bbox
            if int(state.get("detector_miss_count", 0)) > 0:
                state["status"] = "predicted_during_detector_miss"
            else:
                state["status"] = "tracking"
            self._append_trajectory(tid, frame_idx, center)
            outputs.append({
                "bbox": bbox,
                "confidence": state.get("confidence", 0.0),
                "track_id": tid,
                "center": center,
                "label": state.get("label"),
                "class_name": state.get("class_name"),
            })
        self._finish_frame()
        return outputs

    def _predict_csrt(self, state: Dict, frame: np.ndarray, frame_idx: int, count_diagnostics: bool) -> Dict:
        """Run CSRT and decide whether the box is a plausible continuation.

        A returned box is not enough. The lost counter is left unchanged here.
        """
        tracker = state.get("tracker")
        if tracker is None or state.get("status") == "lost" or int(state.get("lost_age", 0)) > 0:
            return {
                "update_success": False,
                "valid": False,
                "reason": "not_tracking",
                "bbox": None,
                "iou": 0.0,
                "jump": None,
            }
        bbox = self._csrt_update_bbox(tracker, frame)
        if bbox is None:
            if count_diagnostics:
                self._diagnostics["rejected_csrt_predictions"] += 1
            self._log_track_event(
                frame_idx, None, state.get("track_id"), "CSRT_REJECTED",
                0.0, None, int(state.get("lost_age", 0)), False, "no_box",
            )
            confirmed_center = state.get("confirmed_center") or self._center_of(state.get("bbox"))
            return {
                "update_success": False,
                "valid": False,
                "reason": "no_box",
                "bbox": None,
                "raw_bbox": None,
                "iou": 0.0,
                "jump": None,
                "diagnostics": {
                    "center": None,
                    "width": None,
                    "height": None,
                    "area": None,
                    "distance_from_previous_csrt": None,
                    "distance_from_last_detector_center": None,
                    "distance_from_expected_center": None,
                    "area_ratio_previous_csrt": None,
                    "area_ratio_confirmed": None,
                    "iou_previous_csrt": None,
                    "iou_confirmed": None,
                    "jump_check_passed": None,
                    "area_check_passed": None,
                    "iou_check_passed": None,
                    "distance_check_passed": None,
                    "confirmed_frame": state.get("confirmed_frame"),
                    "confirmed_center": confirmed_center,
                    "track_state": state.get("status"),
                    "internal_failure": True,
                },
            }

        last_bbox = state.get("bbox")
        center = self._center_of(bbox)
        jump = self._center_distance(center, self._center_of(last_bbox))
        iou = self._iou(bbox, last_bbox)
        max_jump = float(self.config.max_csrt_center_jump)
        area_ratio_previous = self._area_ratio(bbox, last_bbox)
        jump_pass = jump <= max_jump
        area_pass = area_ratio_previous <= float(self.config.max_csrt_area_change)
        iou_pass = not (
            iou < float(self.config.min_csrt_iou) and jump > max_jump * 0.5
        )
        expected = self._expected_center(state, frame_idx)
        off_expected = self._center_distance(center, expected)
        distance_pass = off_expected <= max_jump
        confirmed_bbox = state.get("confirmed_bbox") or last_bbox
        confirmed_center = state.get("confirmed_center") or self._center_of(last_bbox)
        reason = "ok"
        valid = True
        if jump > max_jump:
            valid, reason = False, "center_jump"
        elif area_ratio_previous > float(self.config.max_csrt_area_change):
            valid, reason = False, "area_change"
        elif iou < float(self.config.min_csrt_iou) and jump > max_jump * 0.5:
            valid, reason = False, "low_iou"
        elif not distance_pass:
            valid, reason = False, "off_confirmed"

        if count_diagnostics:
            key = "csrt_predictions" if valid else "rejected_csrt_predictions"
            self._diagnostics[key] += 1
        self._log_track_event(
            frame_idx, None, state.get("track_id"),
            "CSRT_PREDICTION" if valid else "CSRT_REJECTED",
            iou, jump, int(state.get("lost_age", 0)), True, reason,
        )
        width = max(0.0, float(bbox[2]) - float(bbox[0]))
        height = max(0.0, float(bbox[3]) - float(bbox[1]))
        return {
            "update_success": True,
            "valid": valid,
            "reason": reason,
            "bbox": bbox if valid else None,
            "raw_bbox": bbox,
            "iou": iou,
            "jump": jump,
            "diagnostics": {
                "center": center,
                "width": width,
                "height": height,
                "area": width * height,
                "distance_from_previous_csrt": jump,
                "distance_from_last_detector_center": self._center_distance(center, confirmed_center),
                "distance_from_expected_center": off_expected,
                "area_ratio_previous_csrt": area_ratio_previous,
                "area_ratio_confirmed": self._area_ratio(bbox, confirmed_bbox),
                "iou_previous_csrt": iou,
                "iou_confirmed": self._iou(bbox, confirmed_bbox),
                "jump_check_passed": jump_pass,
                "area_check_passed": area_pass,
                "iou_check_passed": iou_pass,
                "distance_check_passed": distance_pass,
                "confirmed_frame": state.get("confirmed_frame"),
                "confirmed_center": confirmed_center,
                "track_state": state.get("status"),
                "internal_failure": False,
            },
        }

    def _assign_detections(
        self,
        detections: List[Dict],
        predictions: Dict[int, Dict],
        frame_idx: int,
    ) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
        """Global one-to-one assignment. Ambiguous pairs are returned unmatched."""
        det_indices = [
            index for index, detection in enumerate(detections)
            if self._valid_xyxy(detection.get("bbox"))
        ]
        track_ids = [
            tid for tid, state in self._tracks.items()
            if int(state.get("lost_age", 0)) <= int(self.config.strong_reid_max_frames)
        ]
        track_ids.sort()
        gated: List[Tuple[int, int, float, Dict]] = []
        scored: List[Dict] = []
        for det_index in det_indices:
            for tid in track_ids:
                decision = self._score_pair(
                    detections[det_index], self._tracks[tid], predictions.get(tid), frame_idx,
                )
                if decision is None:
                    continue
                decision["new_detection_index"] = det_index
                decision["old_track_id"] = tid
                scored.append(decision)
                if decision["accept"]:
                    features = decision["features"]
                    features["track_id"] = tid
                    gated.append((det_index, tid, features["cost"], features))
        self._log_recovery_candidates(scored)

        if not det_indices or not track_ids or not gated:
            return {}, {}

        det_row = {det_index: row for row, det_index in enumerate(det_indices)}
        track_col = {tid: col for col, tid in enumerate(track_ids)}
        big = 1.0e6
        cost = np.full((len(det_indices), len(track_ids)), big, dtype=np.float64)
        for det_index, tid, pair_cost, _features in gated:
            cost[det_row[det_index], track_col[tid]] = pair_cost

        chosen = []
        for row, col in self._solve_assignment(cost):
            if cost[row, col] >= big * 0.5:
                continue
            chosen.append((det_indices[row], track_ids[col], float(cost[row, col])))

        margin = float(self.config.ambiguity_margin)
        ambiguous_dets = set()
        assigned_tracks = set()
        for det_index, tid, pair_cost in chosen:
            rivals = [
                item for item in gated
                if item[3]["cost"] <= pair_cost + margin
                and ((item[0] == det_index and item[1] != tid) or (item[1] == tid and item[0] != det_index))
            ]
            if rivals:
                ambiguous_dets.add(det_index)
                for rival_det, rival_tid, _rival_cost, _features in rivals:
                    ambiguous_dets.add(rival_det)
                    assigned_tracks.add(rival_tid)
                assigned_tracks.add(tid)

        assignments: Dict[int, Dict] = {}
        used_tracks = set()
        for det_index, tid, _pair_cost in chosen:
            if det_index in ambiguous_dets or tid in assigned_tracks or tid in used_tracks:
                continue
            features = next(item[3] for item in gated if item[0] == det_index and item[1] == tid)
            assignments[det_index] = features
            used_tracks.add(tid)

        # A detection that touched an existing track but was not assigned must not
        # become a new sperm. That includes ambiguous crossings and duplicate boxes.
        suppressed: Dict[int, Dict] = {}
        best_by_det: Dict[int, Dict] = {}
        for det_index, _tid, _pair_cost, features in gated:
            current = best_by_det.get(det_index)
            if current is None or features["cost"] < current["cost"]:
                best_by_det[det_index] = features
        for det_index, features in best_by_det.items():
            if det_index in assignments:
                continue
            suppressed[det_index] = features
            if det_index in ambiguous_dets:
                self._diagnostics["id_switches"] += 1
        return assignments, suppressed

    def _log_recovery_candidates(self, scored: List[Dict]) -> None:
        grouped: Dict[int, List[Dict]] = defaultdict(list)
        cap = float(self.config.max_reid_distance) + 20.0
        for decision in scored:
            if not decision.get("lost"):
                continue
            if min(decision["center_distance"], decision["predicted_center_distance"]) > cap:
                continue
            grouped[int(decision["new_detection_index"])].append(decision)
        for decisions in grouped.values():
            costs = sorted(
                float(item["match_cost"]) for item in decisions if item.get("match_cost") is not None
            )
            for item in decisions:
                second = None
                if item.get("match_cost") is not None and len(costs) > 1:
                    second = costs[1] if costs[0] == float(item["match_cost"]) else costs[0]
                self.recovery_log.append({
                    "old_track_id": item.get("old_track_id"),
                    "new_detection_index": item.get("new_detection_index"),
                    "frame": item.get("frame"),
                    "lost_age": item.get("lost_age"),
                    "frames_since_last_confirmation": item.get("frames_since_last_confirmation"),
                    "center_distance": item.get("center_distance"),
                    "IoU": item.get("iou"),
                    "size_ratio": item.get("size_ratio"),
                    "predicted_center_distance": item.get("predicted_center_distance"),
                    "match_cost": item.get("match_cost"),
                    "second_best_cost": second,
                    "tier": item.get("tier"),
                    "reject": item.get("reject"),
                })

    def _pair_features(self, detection: Dict, state: Dict, prediction: Optional[Dict], frame_idx: int) -> Optional[Dict]:
        decision = self._score_pair(detection, state, prediction, frame_idx)
        if decision is None or not decision["accept"]:
            return None
        return decision["features"]

    def _score_pair(
        self,
        detection: Dict,
        state: Dict,
        prediction: Optional[Dict],
        frame_idx: int,
    ) -> Optional[Dict]:
        """Tiered detector association. Size is a soft cost, not a veto of a strong match."""
        bbox = detection.get("bbox")
        confirmed_bbox = state.get("confirmed_bbox") or state.get("bbox")
        if not self._valid_xyxy(bbox) or not self._valid_xyxy(confirmed_bbox):
            return None
        if not self._labels_compatible(detection, state):
            return None

        det_center = self._center_of(bbox)
        confirmed_center = state["confirmed_center"]
        gap = max(1, int(frame_idx) - int(state["confirmed_frame"]))
        legacy_center = self._trajectory_prediction(state, frame_idx)
        motion = self._motion_estimate(state, frame_idx)
        predicted_center = motion["predicted_center"]
        dist_confirmed = self._center_distance(det_center, confirmed_center)
        legacy_dist = self._center_distance(det_center, legacy_center)
        dist_pred = self._center_distance(det_center, predicted_center)
        direction_diff = self._direction_difference(motion["velocity"], det_center, confirmed_center)
        iou_confirmed = self._iou(bbox, confirmed_bbox)
        dist_csrt = None
        iou_csrt = 0.0
        if prediction and prediction.get("valid") and self._valid_xyxy(prediction.get("bbox")):
            dist_csrt = self._center_distance(det_center, self._center_of(prediction["bbox"]))
            iou_csrt = self._iou(bbox, prediction["bbox"])

        iou = max(iou_confirmed, iou_csrt)
        side = self._side_ratio(bbox, confirmed_bbox)
        lost = state.get("status") == "lost" or int(state.get("lost_age", 0)) > 0
        within_cap = min(dist_confirmed, legacy_dist) <= self._reid_distance_limit(gap)
        confirmed_within_cap = dist_confirmed <= self._reid_distance_limit(gap)
        strong = (
            iou >= float(self.config.strong_reid_iou)
            and min(dist_confirmed, legacy_dist) <= float(self.config.strong_reid_distance)
            and confirmed_within_cap
        )
        distance = min(dist_confirmed, legacy_dist)
        if dist_csrt is not None:
            distance = min(distance, dist_csrt)
        motion_ok = self._motion_assisted(
            motion, dist_pred, dist_confirmed, side, direction_diff, gap,
        )

        reject = None
        tier = None
        event = None
        if lost:
            normal_spatial = within_cap and (
                iou >= float(self.config.min_reid_iou)
                or dist_confirmed <= float(self.config.max_match_distance)
            )
            predicted_close = (
                legacy_dist <= float(self.config.strong_reid_distance)
                and confirmed_within_cap
            )
            if strong and gap <= int(self.config.strong_reid_max_frames) and within_cap:
                tier = "strong"
                event = "REID"
            elif (
                predicted_close
                and gap <= int(self.config.strong_reid_max_frames)
                and within_cap
                and side <= float(self.config.max_size_ratio)
            ):
                tier = "normal"
                event = "REID"
            elif (
                gap <= int(self.config.max_reid_frames)
                and normal_spatial
                and side <= float(self.config.max_size_ratio)
            ):
                tier = "normal"
                event = "REID"
            elif motion_ok and confirmed_within_cap:
                tier = "motion"
                event = "REID"
            elif strong and gap > int(self.config.strong_reid_max_frames):
                reject = "RECOVERY_TOO_OLD"
            elif (not strong) and gap > int(self.config.max_reid_frames) and normal_spatial:
                reject = "RECOVERY_TOO_OLD"
            elif normal_spatial and side > float(self.config.max_size_ratio) and not strong:
                reject = "RECOVERY_SIZE_REJECTED"
            elif dist_confirmed > float(self.config.max_reid_distance):
                reject = "RECOVERY_DISTANCE_TOO_LARGE"
            elif iou < float(self.config.min_reid_iou) and dist_confirmed > float(self.config.max_match_distance):
                reject = "RECOVERY_IOU_TOO_LOW"
            else:
                reject = "RECOVERY_DISTANCE_TOO_LARGE"
        else:
            ordinary = (
                iou >= float(self.config.min_match_iou)
                or dist_confirmed <= float(self.config.max_match_distance)
                or (dist_csrt is not None and dist_csrt <= float(self.config.max_match_distance))
            )
            if not ordinary and not motion_ok:
                reject = "ACTIVE_MATCH_FAILED"
            elif (not strong) and side > float(self.config.max_size_ratio) and not motion_ok:
                reject = "RECOVERY_SIZE_REJECTED"
            elif motion_ok and not ordinary:
                event = "MATCH_MOTION"
                tier = "motion"
            elif iou >= float(self.config.min_match_iou):
                event = "MATCH_IOU"
                tier = "active"
            else:
                event = "MATCH_CENTER"
                tier = "active"

        scale = max(float(self.config.max_match_distance), 1.0)
        size_cost = min(1.0, abs(math.log(max(self._area_ratio(bbox, confirmed_bbox), 1e-6))))
        # A strong spatial match keeps a small size penalty so shape cannot outrank it.
        size_weight = 0.05 if strong else 0.15
        if tier == "motion":
            cost = self._motion_cost(dist_pred, dist_confirmed, iou, size_cost, direction_diff, gap)
        else:
            cost = (
                0.45 * min(1.0, distance / scale)
                + 0.30 * (1.0 - iou)
                + size_weight * size_cost
                + 0.15 * min(1.0, legacy_dist / scale)
            )
        accepted = event is not None
        record = {
            "accept": accepted,
            "tier": tier,
            "reject": reject,
            "track_id": state.get("track_id"),
            "lost_age": int(state.get("lost_age", 0)),
            "frames_since_last_confirmation": gap,
            "center_distance": round(float(dist_confirmed), 2),
            "predicted_center": (round(float(predicted_center[0]), 2), round(float(predicted_center[1]), 2)),
            "predicted_center_distance": round(float(dist_pred), 2),
            "prediction_velocity": (round(float(motion["velocity"][0]), 4), round(float(motion["velocity"][1]), 4)),
            "prediction_confidence": round(float(motion["confidence"]), 4),
            "trajectory_direction_difference": None if direction_diff is None else round(float(direction_diff), 2),
            "prediction_source": motion.get("source"),
            "prediction_drifted": bool(motion.get("drifted")),
            "confirmed_center": (round(float(confirmed_center[0]), 2), round(float(confirmed_center[1]), 2)),
            "detected_center": (round(float(det_center[0]), 2), round(float(det_center[1]), 2)),
            "iou": round(float(iou), 4),
            "size_ratio": round(float(side), 3),
            "match_cost": None if not accepted else round(float(cost), 4),
            "frame": int(frame_idx),
        }
        record["lost"] = lost
        if not accepted:
            return record
        record["features"] = {
            "track_id": None,
            "cost": float(cost),
            "distance": float(min(dist_confirmed, legacy_dist) if lost else distance),
            "iou": float(iou),
            "event": event,
            "tier": tier,
            "size_ratio": float(side),
            "predicted_center_x": float(predicted_center[0]),
            "predicted_center_y": float(predicted_center[1]),
            "predicted_center_distance": float(dist_pred),
            "prediction_velocity_x": float(motion["velocity"][0]),
            "prediction_velocity_y": float(motion["velocity"][1]),
            "prediction_confidence": float(motion["confidence"]),
            "trajectory_direction_difference": None if direction_diff is None else float(direction_diff),
            "frames_since_last_confirmation": gap,
            "lost_age": int(state.get("lost_age", 0)),
            "update_success": bool(prediction.get("update_success")) if prediction else False,
            "validation": (prediction or {}).get("reason", "not_tracking"),
            "reject": None,
        }
        return record

    def _motion_estimate(self, state: Dict, frame_idx: int) -> Dict:
        """Predict the next center from confirmed motion, not from a rejected CSRT box.

        Three or more detector confirmations are preferred. Validated CSRT points
        are used only while they remain inside the existing recovery cap of the
        last confirmation, so a drifted filter cannot steer the prediction.
        """
        confirmed = state.get("confirmed_center") or (0.0, 0.0)
        cx, cy = float(confirmed[0]), float(confirmed[1])
        confirmed_frame = int(state.get("confirmed_frame", frame_idx))
        gap = max(0, int(frame_idx) - confirmed_frame)
        empty = {
            "predicted_center": (cx, cy),
            "velocity": (0.0, 0.0),
            "confidence": 0.0,
            "source": "none",
            "samples": 0,
            "drifted": False,
        }
        if not bool(self.config.use_motion_prediction) or gap <= 0 or gap > int(self.config.max_prediction_horizon):
            return empty

        history = list(state.get("confirmed_points") or [])
        if not history:
            history = [(confirmed_frame, cx, cy)]
        tid = state.get("track_id")
        trail = self.trajectories.get(int(tid), []) if tid is not None else []
        cap = float(self.config.max_reid_distance)
        usable = []
        drifted = False
        for frame, x, y in trail:
            if int(frame) < confirmed_frame:
                continue
            if math.hypot(float(x) - cx, float(y) - cy) <= cap:
                usable.append((int(frame), float(x), float(y)))
            else:
                drifted = True
        window = max(3, int(self.config.prediction_history_frames))
        if len(history) >= 3:
            source_points = history[-window:]
            source = "confirmed"
        elif len(usable) >= 3 and not drifted:
            source_points = usable[-window:]
            source = "validated_csrt"
        else:
            empty["drifted"] = drifted
            return empty

        velocity = self._robust_velocity(source_points)
        if velocity is None:
            empty["drifted"] = drifted
            return empty
        vx, vy, consistency = velocity
        if math.hypot(vx, vy) > float(self.config.max_csrt_center_jump):
            empty["drifted"] = True
            return empty
        if source == "confirmed" and state.get("velocity_ready"):
            blend = float(self.config.velocity_smoothing)
            previous = state.get("velocity") or (0.0, 0.0)
            vx = blend * vx + (1.0 - blend) * float(previous[0])
            vy = blend * vy + (1.0 - blend) * float(previous[1])
        sample_score = min(1.0, max(0.0, (len(source_points) - 2) / 2.0))
        confidence = sample_score * (0.4 + 0.6 * consistency)
        return {
            "predicted_center": (cx + vx * gap, cy + vy * gap),
            "velocity": (vx, vy),
            "confidence": float(confidence),
            "source": source,
            "samples": len(source_points),
            "drifted": drifted,
        }

    def _robust_velocity(self, points: List[Tuple]) -> Optional[Tuple[float, float, float]]:
        """Median step velocity. Needs at least three positions, not a single pair."""
        steps = []
        for earlier, later in zip(points, points[1:]):
            dt = int(later[0]) - int(earlier[0])
            if dt <= 0:
                continue
            steps.append((
                (float(later[1]) - float(earlier[1])) / float(dt),
                (float(later[2]) - float(earlier[2])) / float(dt),
            ))
        if len(steps) < 2:
            return None
        vx = float(np.median([step[0] for step in steps]))
        vy = float(np.median([step[1] for step in steps]))
        angles = []
        for first, second in zip(steps, steps[1:]):
            angles.append(self._angle_between(first, second))
        usable_angles = [angle for angle in angles if angle is not None]
        mean_angle = float(np.mean(usable_angles)) if usable_angles else 0.0
        consistency = max(0.0, 1.0 - mean_angle / 90.0)
        return vx, vy, consistency

    def _motion_assisted(
        self,
        motion: Dict,
        predicted_distance: float,
        confirmed_distance: float,
        side_ratio: float,
        direction_difference: Optional[float],
        gap: int,
    ) -> bool:
        """Accept a moving sperm without widening the 80px live gate."""
        if not bool(self.config.use_motion_prediction):
            return False
        if float(motion.get("confidence") or 0.0) < 0.5:
            return False
        if gap <= 0 or gap > int(self.config.max_prediction_horizon):
            return False
        if confirmed_distance <= float(self.config.max_match_distance):
            return False
        if confirmed_distance > float(self.config.max_reid_distance):
            return False
        if predicted_distance > float(self.config.max_prediction_distance):
            return False
        if predicted_distance + 1.0 >= confirmed_distance:
            return False
        if side_ratio > float(self.config.max_size_ratio):
            return False
        speed = math.hypot(float(motion["velocity"][0]), float(motion["velocity"][1]))
        if speed * float(gap) < 0.5 * confirmed_distance:
            return False
        if direction_difference is None or direction_difference > float(self.config.max_direction_difference_deg):
            return False
        return True

    def _motion_cost(
        self,
        predicted_distance: float,
        confirmed_distance: float,
        iou: float,
        size_cost: float,
        direction_difference: Optional[float],
        gap: int,
    ) -> float:
        horizon = max(int(self.config.max_prediction_horizon), 1)
        gap_frac = min(1.0, float(gap) / float(horizon))
        weight_pred = float(self.config.prediction_weight) * (0.5 + 0.5 * gap_frac)
        weight_last = 0.20 * (1.0 - 0.5 * gap_frac)
        weight_iou = 0.25
        weight_size = 0.10
        weight_dir = 0.15
        total = weight_pred + weight_last + weight_iou + weight_size + weight_dir
        direction_term = 0.0 if direction_difference is None else min(1.0, float(direction_difference) / 180.0)
        raw = (
            weight_pred * min(1.0, predicted_distance / max(float(self.config.max_prediction_distance), 1.0))
            + weight_last * min(1.0, confirmed_distance / max(float(self.config.max_reid_distance), 1.0))
            + weight_iou * (1.0 - iou)
            + weight_size * size_cost
            + weight_dir * direction_term
        )
        return float(raw / total)

    def _direction_difference(
        self,
        velocity: Tuple[float, float],
        detected_center: Tuple[float, float],
        confirmed_center: Tuple[float, float],
    ) -> Optional[float]:
        displacement = (
            float(detected_center[0]) - float(confirmed_center[0]),
            float(detected_center[1]) - float(confirmed_center[1]),
        )
        return self._angle_between(velocity, displacement)

    @staticmethod
    def _angle_between(first: Tuple[float, float], second: Tuple[float, float]) -> Optional[float]:
        first_norm = math.hypot(float(first[0]), float(first[1]))
        second_norm = math.hypot(float(second[0]), float(second[1]))
        if first_norm < 1.0 or second_norm < 1.0:
            return None
        cosine = (
            float(first[0]) * float(second[0]) + float(first[1]) * float(second[1])
        ) / (first_norm * second_norm)
        cosine = max(-1.0, min(1.0, cosine))
        return float(math.degrees(math.acos(cosine)))

    def _trajectory_prediction(self, state: Dict, frame_idx: int) -> Tuple[float, float]:
        """Extrapolate from the last stored points. Falls back to the last detector fix."""
        tid = state.get("track_id")
        points = self.trajectories.get(int(tid), []) if tid is not None else []
        if len(points) >= 2:
            last_f, last_x, last_y = points[-1]
            anchor = points[max(0, len(points) - 6)]
            for point in reversed(points[:-1]):
                if int(last_f) - int(point[0]) >= 3:
                    anchor = point
                    break
            dt = max(1, int(last_f) - int(anchor[0]))
            vx = (float(last_x) - float(anchor[1])) / float(dt)
            vy = (float(last_y) - float(anchor[2])) / float(dt)
            ahead = max(0, int(frame_idx) - int(last_f))
            return (float(last_x) + vx * ahead, float(last_y) + vy * ahead)
        return self._expected_center(state, frame_idx)

    def _evidence_class(self, decision: Dict) -> str:
        """Diagnostic label only. It does not change the assignment."""
        predicted = decision.get("predicted_center_distance")
        last = decision.get("center_distance")
        confidence = float(decision.get("prediction_confidence") or 0.0)
        direction = decision.get("trajectory_direction_difference")
        size_ratio = decision.get("size_ratio")
        iou = float(decision.get("iou") or 0.0)
        if predicted is None or last is None:
            return "AMBIGUOUS"
        same = (
            confidence >= 0.5
            and predicted <= float(self.config.max_prediction_distance)
            and last > float(self.config.max_match_distance)
            and predicted + 5.0 < last
            and direction is not None
            and direction <= float(self.config.max_direction_difference_deg)
            and (size_ratio is None or size_ratio <= float(self.config.max_size_ratio))
        )
        if same:
            return "LIKELY_SAME_SPERM"
        if last >= float(self.config.max_reid_distance) or predicted > float(self.config.max_prediction_distance):
            return "LIKELY_DIFFERENT_SPERM"
        if iou < float(self.config.min_reid_iou) and last > float(self.config.max_match_distance):
            return "LIKELY_DIFFERENT_SPERM"
        return "AMBIGUOUS"

    def _record_new_id(self, detection: Dict, frame_idx: int, det_index: int) -> None:
        nearest = None
        nearest_distance = None
        for state in self._tracks.values():
            decision = self._score_pair(detection, state, None, frame_idx)
            if decision is None:
                continue
            distance = min(decision["center_distance"], decision["predicted_center_distance"])
            if nearest is None or distance < nearest_distance:
                nearest = decision
                nearest_distance = distance
        if nearest is None or (
            nearest["center_distance"] > float(self.config.max_reid_distance)
            and nearest["iou"] < float(self.config.min_reid_iou)
        ):
            label = "TRUE_NEW_DETECTION"
        else:
            label = nearest.get("reject") or "OTHER"
        key = {
            "TRUE_NEW_DETECTION": "new_true_new_detection",
            "RECOVERY_TOO_OLD": "recovery_rejected_by_age",
            "RECOVERY_DISTANCE_TOO_LARGE": "recovery_rejected_by_distance",
            "RECOVERY_IOU_TOO_LOW": "recovery_rejected_by_iou",
            "RECOVERY_SIZE_REJECTED": "recovery_rejected_by_size",
            "ACTIVE_MATCH_FAILED": "new_active_match_failed",
        }.get(label, "new_other")
        self._diagnostics[key] = self._diagnostics.get(key, 0) + 1
        evidence = "TRUE_NEW_DETECTION" if nearest is None else self._evidence_class(nearest)
        active_kind = None
        if label == "ACTIVE_MATCH_FAILED" and nearest is not None:
            if nearest.get("prediction_drifted"):
                active_kind = "CSRT_DRIFTED"
            elif int(nearest.get("lost_age") or 0) > 0:
                active_kind = "ALREADY_LOST"
            elif evidence == "LIKELY_SAME_SPERM":
                active_kind = "SAME_SPERM"
            elif evidence == "AMBIGUOUS":
                active_kind = "AMBIGUOUS"
            else:
                active_kind = "DIFFERENT_SPERM"
        predicted = None if nearest is None else nearest.get("predicted_center")
        confirmed = None if nearest is None else nearest.get("confirmed_center")
        detected = None if nearest is None else nearest.get("detected_center")
        velocity = None if nearest is None else nearest.get("prediction_velocity")
        self.new_id_log.append({
            "frame": int(frame_idx),
            "detection_index": int(det_index),
            "new_id_reason": label,
            "reason": label,
            "evidence_class": evidence,
            "active_failure_kind": active_kind,
            "nearest_track_id": None if nearest is None else nearest.get("track_id"),
            "old_track_id": None if nearest is None else nearest.get("track_id"),
            "lost_age": None if nearest is None else nearest.get("lost_age"),
            "frames_since_last_confirmation": None if nearest is None else nearest.get("frames_since_last_confirmation"),
            "time_since_last_confirmation_frames": None if nearest is None else nearest.get("frames_since_last_confirmation"),
            "last_confirmed_center": confirmed,
            "detected_center": detected,
            "predicted_center": predicted,
            "center_distance": None if nearest is None else nearest.get("center_distance"),
            "distance_from_last_center": None if nearest is None else nearest.get("center_distance"),
            "predicted_center_distance": None if nearest is None else nearest.get("predicted_center_distance"),
            "prediction_distance": None if nearest is None else nearest.get("predicted_center_distance"),
            "prediction_based_candidate": bool(
                nearest is not None
                and (nearest.get("prediction_confidence") or 0.0) >= 0.5
                and (nearest.get("predicted_center_distance") or 1e9) <= float(self.config.max_prediction_distance)
                and (nearest.get("center_distance") or 0.0) > float(self.config.max_match_distance)
            ),
            "prediction_velocity_x": None if velocity is None else velocity[0],
            "prediction_velocity_y": None if velocity is None else velocity[1],
            "prediction_confidence": None if nearest is None else nearest.get("prediction_confidence"),
            "trajectory_direction_difference": None if nearest is None else nearest.get("trajectory_direction_difference"),
            "iou": None if nearest is None else nearest.get("iou"),
            "size_ratio": None if nearest is None else nearest.get("size_ratio"),
            "match_cost": None if nearest is None else nearest.get("match_cost"),
        })

    def _confirm_track(
        self,
        tid: int,
        state: Dict,
        detection: Dict,
        clipped: List[float],
        tracker,
        frame_idx: int,
        info: Dict,
        det_index: Optional[int] = None,
    ) -> None:
        """Detector association. Keeps the existing trajectory and clears lost age."""
        center = self._center_of(clipped)
        was_lost = state.get("status") == "lost" or int(state.get("lost_age", 0)) > 0
        gap = int(frame_idx) - int(state.get("confirmed_frame", frame_idx))
        if gap > 0 and state.get("confirmed_center") is not None:
            previous = state["confirmed_center"]
            state["velocity"] = (
                (center[0] - previous[0]) / float(gap),
                (center[1] - previous[1]) / float(gap),
            )
        state["tracker"] = tracker
        state["bbox"] = clipped
        state["confirmed_bbox"] = list(clipped)
        state["confirmed_center"] = center
        state["confirmed_frame"] = int(frame_idx)
        history = list(state.get("confirmed_points") or [])
        history.append((int(frame_idx), float(center[0]), float(center[1])))
        state["confirmed_points"] = history[-8:]
        state["velocity_ready"] = len(state["confirmed_points"]) >= 3
        state["status"] = "confirmed"
        state["lost_age"] = 0
        state["confidence"] = float(detection.get("confidence") or 0.0)
        state["label"] = detection.get("label")
        state["class_name"] = detection.get("class_name")
        self._lost_age.pop(tid, None)
        detection["bbox"] = clipped
        detection["track_id"] = tid
        detection["center"] = center
        self._append_trajectory(tid, frame_idx, center)
        if detection.get("mask") is not None:
            self.remember_mask(tid, detection["mask"], center)
        came_from_detector_miss = int(state.get("detector_miss_count", 0)) > 0
        state["detector_miss_count"] = 0
        if was_lost or info.get("event") == "REID":
            self._diagnostics["reidentifications"] += 1
            if info.get("tier") == "strong":
                self._diagnostics["strong_recoveries"] += 1
            elif info.get("tier") == "motion":
                self._diagnostics["motion_assisted_recoveries"] += 1
            else:
                self._diagnostics["normal_recoveries"] += 1
            event = "REID"
        elif info.get("tier") == "motion":
            self._diagnostics["motion_assisted_matches"] += 1
            event = info.get("event") or "MATCH_MOTION"
        else:
            event = info.get("event") or "MATCH_GLOBAL"
        if came_from_detector_miss and not was_lost:
            self._diagnostics["recovered_after_detector_miss"] += 1
            event = "DETECTOR_CONFIRMED"
        self._log_track_event(
            frame_idx, det_index, tid, event,
            info.get("iou", 0.0), info.get("distance"), 0,
            info.get("update_success", False), info.get("validation", "detector"),
        )

    def _create_track(self, detection: Dict, bbox, frame: np.ndarray, frame_idx: int, det_index: int) -> Optional[int]:
        tracker, clipped = self._init_csrt(frame, bbox)
        if tracker is None or clipped is None:
            logger.warning("CSRT tracker was not reinitialized for detection %s", det_index)
            return None
        self._record_new_id(detection, frame_idx, det_index)
        tid = self._allocate_track_id()
        if self.new_id_log:
            self.new_id_log[-1]["new_id"] = tid
        center = self._center_of(clipped)
        self._tracks[tid] = {
            "tracker": tracker,
            "bbox": clipped,
            "confirmed_bbox": list(clipped),
            "confirmed_center": center,
            "confirmed_frame": int(frame_idx),
            "confirmed_points": [(int(frame_idx), float(center[0]), float(center[1]))],
            "velocity": (0.0, 0.0),
            "velocity_ready": False,
            "status": "confirmed",
            "lost_age": 0,
            "detector_miss_count": 0,
            "confidence": float(detection.get("confidence") or 0.0),
            "label": detection.get("label"),
            "class_name": detection.get("class_name"),
            "track_id": tid,
        }
        detection["bbox"] = clipped
        detection["track_id"] = tid
        detection["center"] = center
        self._append_trajectory(tid, frame_idx, center)
        if detection.get("mask") is not None:
            self.remember_mask(tid, detection["mask"], center)
        self._diagnostics["new_ids"] += 1
        self._diagnostics["unique_ids_created"] = self._next_track_id - 1
        self._log_track_event(
            frame_idx, det_index, tid, "NEW_ID",
            0.0, None, 0, False, "no_existing_track",
        )
        return tid

    def _hold_through_detector_miss(self, tid: int, frame_idx: int, prediction: Dict) -> bool:
        """Keep a confirmed ID through a short Mask R-CNN miss when CSRT is still valid.

        The prediction is position only. It does not confirm the ID and it does
        not store a morphology mask.
        """
        state = self._tracks.get(tid)
        if state is None or not prediction.get("valid") or not self._valid_xyxy(prediction.get("bbox")):
            return False
        misses = int(state.get("detector_miss_count", 0))
        if misses >= int(self.config.max_detector_miss_intervals):
            return False
        bbox = prediction["bbox"]
        center = self._center_of(bbox)
        state["bbox"] = bbox
        state["detector_miss_count"] = misses + 1
        state["status"] = "predicted_during_detector_miss"
        self._append_trajectory(tid, frame_idx, center)
        survived = int(state["detector_miss_count"])
        if survived > int(self._miss_survival.get(tid, 0)):
            self._miss_survival[tid] = survived
        self._diagnostics["detector_misses"] += 1
        self._log_track_event(
            frame_idx, None, tid, "DETECTOR_MISS",
            prediction.get("iou", 0.0), prediction.get("jump"), 0,
            True, "predicted_during_detector_miss",
        )
        return True

    def _advance_lost(
        self,
        tid: int,
        frame_idx: int,
        det_index: Optional[int],
        reason: str,
        prediction: Optional[Dict] = None,
    ) -> None:
        state = self._tracks.get(tid)
        if state is None:
            return
        entered = state.get("status") != "lost"
        state["status"] = "lost"
        state["detector_miss_count"] = 0
        state["tracker"] = None
        state["lost_age"] = int(state.get("lost_age", 0)) + 1
        self._lost_age[tid] = state["lost_age"]
        prediction = prediction or {}
        if entered and reason in {"center_jump", "area_change", "low_iou", "off_confirmed", "no_box", "csrt_invalid"}:
            self._diagnostics["lost_because_csrt_failed"] += 1
        if entered:
            self._log_track_event(
                frame_idx, det_index, tid, "TRACK_LOST",
                prediction.get("iou", 0.0), prediction.get("jump"), state["lost_age"],
                bool(prediction.get("update_success", False)), reason,
            )
        else:
            self._log_track_event(
                frame_idx, det_index, tid, "UNMATCHED_TRACK",
                prediction.get("iou", 0.0), prediction.get("jump"), state["lost_age"],
                bool(prediction.get("update_success", False)), reason,
            )
        if state["lost_age"] > int(self.config.strong_reid_max_frames):
            self._drop_track(tid)
            self._log_track_event(
                frame_idx, det_index, tid, "TRACK_DROPPED",
                0.0, None, state["lost_age"], False, reason,
            )

    def _finish_frame(self) -> None:
        self._refresh_active_tracks()
        active = len(self.active_tracks)
        if active > self._diagnostics["max_simultaneous_active"]:
            self._diagnostics["max_simultaneous_active"] = active

    def _log_track_event(
        self,
        frame_idx: int,
        det_index: Optional[int],
        track_id: Optional[int],
        event: str,
        iou: float,
        distance: Optional[float],
        lost_age: int,
        update_success: bool,
        validation: str,
    ) -> None:
        if not bool(self.config.tracking_debug):
            return
        distance_text = "na" if distance is None else f"{float(distance):.2f}"
        logger.info(
            "frame=%s detection_index=%s track_id=%s event=%s IoU=%.3f center_distance=%s "
            "lost_age=%s CSRT_update_success=%s CSRT_validation_result=%s "
            "active_track_count=%s unique_id_count=%s max_track_id=%s",
            int(frame_idx),
            "na" if det_index is None else int(det_index),
            "na" if track_id is None else int(track_id),
            event,
            float(iou or 0.0),
            distance_text,
            int(lost_age),
            bool(update_success),
            validation,
            len(self.active_tracks),
            max(0, self._next_track_id - 1),
            max(0, self._next_track_id - 1),
        )

    def _reid_distance_limit(self, gap: int) -> float:
        grown = float(self.config.max_match_distance) + float(self.config.reid_distance_per_extra_frame) * max(0, int(gap) - 1)
        return min(float(self.config.max_reid_distance), grown)

    def _expected_center(self, state: Dict, frame_idx: int) -> Tuple[float, float]:
        gap = max(0, int(frame_idx) - int(state.get("confirmed_frame", frame_idx)))
        velocity = state.get("velocity") or (0.0, 0.0)
        center = state.get("confirmed_center") or self._center_of(state["bbox"])
        return (float(center[0]) + float(velocity[0]) * gap, float(center[1]) + float(velocity[1]) * gap)

    def _solve_assignment(self, cost: np.ndarray) -> List[Tuple[int, int]]:
        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(cost)
            return list(zip(rows.tolist(), cols.tolist()))
        except Exception:
            logger.warning("Hungarian assignment was unavailable; using gated one-to-one fallback")
            pairs = [
                (float(cost[row, col]), int(row), int(col))
                for row in range(cost.shape[0])
                for col in range(cost.shape[1])
                if cost[row, col] < 1.0e5
            ]
            pairs.sort(key=lambda item: item[0])
            used_rows, used_cols = set(), set()
            chosen = []
            for _pair_cost, row, col in pairs:
                if row in used_rows or col in used_cols:
                    continue
                used_rows.add(row)
                used_cols.add(col)
                chosen.append((row, col))
            return chosen

    def _append_trajectory(self, track_id: int, frame_idx: int, center: Tuple[float, float]) -> None:
        if center is None or center[0] is None or center[1] is None:
            return
        self.trajectories[track_id].append((int(frame_idx), float(center[0]), float(center[1])))
        max_length = int(self.config.max_trajectory_length)
        if len(self.trajectories[track_id]) > max_length:
            self.trajectories[track_id] = self.trajectories[track_id][-max_length:]

    def _refresh_active_tracks(self) -> None:
        max_lost = int(self.config.max_lost_frames)
        self.active_tracks = {
            tid for tid, state in self._tracks.items()
            if state.get("status") != "lost" or int(state.get("lost_age", 0)) <= max_lost
        }

    def _allocate_track_id(self) -> int:
        """Next ID for this video. Retired IDs are not given to a different sperm."""
        track_id = int(self._next_track_id)
        self._next_track_id = track_id + 1
        return track_id

    def _drop_track(self, track_id: int) -> None:
        """Remove the live tracker. The trajectory list for this ID stays."""
        self._tracks.pop(track_id, None)
        self._lost_age.pop(track_id, None)
        self._masks.pop(track_id, None)
        self._mask_centroids.pop(track_id, None)
        self.active_tracks.discard(track_id)
        self._diagnostics["dropped_tracks"] += 1

    @staticmethod
    def _center_distance(first: Tuple[float, float], second: Tuple[float, float]) -> float:
        return float(np.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1])))

    @staticmethod
    def _box_area(bbox) -> float:
        return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))

    def _area_ratio(self, box_a, box_b) -> float:
        area_a = self._box_area(box_a)
        area_b = self._box_area(box_b)
        if area_a <= 0.0 or area_b <= 0.0:
            return float("inf")
        return max(area_a, area_b) / min(area_a, area_b)

    @staticmethod
    def _side_ratio(box_a, box_b) -> float:
        aw = max(1.0, float(box_a[2]) - float(box_a[0]))
        ah = max(1.0, float(box_a[3]) - float(box_a[1]))
        bw = max(1.0, float(box_b[2]) - float(box_b[0]))
        bh = max(1.0, float(box_b[3]) - float(box_b[1]))
        return max(aw / bw, bw / aw, ah / bh, bh / ah)

    @staticmethod
    def _labels_compatible(detection: Dict, state: Dict) -> bool:
        detected = detection.get("label")
        stored = state.get("label")
        if detected is None or stored is None:
            return True
        try:
            return int(detected) == int(stored)
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _center_of(bbox) -> Tuple[float, float]:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))

    @staticmethod
    def _valid_xyxy(bbox) -> bool:
        if bbox is None or len(bbox) < 4:
            return False
        try:
            values = [float(v) for v in bbox[:4]]
        except (TypeError, ValueError):
            return False
        if not np.isfinite(values).all():
            return False
        return (values[2] - values[0]) >= 1.0 and (values[3] - values[1]) >= 1.0

    @staticmethod
    def _xyxy_to_xywh(bbox, shape) -> Optional[Tuple[float, float, float, float]]:
        height, width = shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        x1 = min(max(x1, 0.0), max(float(width) - 1.0, 0.0))
        y1 = min(max(y1, 0.0), max(float(height) - 1.0, 0.0))
        x2 = min(max(x2, 0.0), max(float(width) - 1.0, 0.0))
        y2 = min(max(y2, 0.0), max(float(height) - 1.0, 0.0))
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < 1.0 or box_h < 1.0:
            return None
        return (x1, y1, box_w, box_h)

    @staticmethod
    def _xywh_to_xyxy(box, shape) -> Optional[List[float]]:
        height, width = shape[:2]
        x, y, box_w, box_h = [float(v) for v in list(box)[:4]]
        if not np.isfinite([x, y, box_w, box_h]).all() or box_w < 1.0 or box_h < 1.0:
            return None
        x1 = min(max(x, 0.0), max(float(width) - 1.0, 0.0))
        y1 = min(max(y, 0.0), max(float(height) - 1.0, 0.0))
        x2 = min(max(x + box_w, 0.0), max(float(width) - 1.0, 0.0))
        y2 = min(max(y + box_h, 0.0), max(float(height) - 1.0, 0.0))
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _iou(box_a, box_b) -> float:
        try:
            ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
            bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
        except (TypeError, ValueError):
            return 0.0
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if intersection <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0.0 else 0.0
