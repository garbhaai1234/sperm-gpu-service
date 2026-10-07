"""
Live Camera Stream Module
Handles real-time sperm detection and tracking from:
  1. USB-connected cameras (mobile phone via USB to a microscope)
  2. WebSocket frames (mobile phone camera via QR code / browser)
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Dict, List, Tuple, Callable
from collections import defaultdict

from .detection import SpermDetector
from .tracking import SpermTracker
from .motility import MotilityAnalyzer, UNCALIBRATED_VELOCITY_WARNING

logger = logging.getLogger(__name__)


class LiveStreamProcessor:
    """
    Real-time sperm detection and motility tracking from a USB camera source.
    Designed for mobile phones connected via USB cable to a microscope.
    """

    def __init__(
        self,
        detector: SpermDetector,
        device_index: int = 0,
        resolution: Tuple[int, int] = (1280, 720),
        target_fps: int = 30,
    ):
        """
        Args:
            detector: Pre-loaded SpermDetector instance (shared with pipeline)
            device_index: Camera device index (0 = default, 1 = second camera, etc.)
            resolution: Desired capture resolution (width, height)
            target_fps: Target frames per second for processing
        """
        self.detector = detector
        self.device_index = device_index
        self.resolution = resolution
        self.target_fps = target_fps

        # Per-session tracker and motility analyzer
        self.tracker = SpermTracker(max_trajectory_length=500, max_match_distance=80.0)
        self.motility_analyzer = MotilityAnalyzer()

        # State
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._lock = threading.Lock()
        self._frame_idx = 0
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_annotated: Optional[np.ndarray] = None
        self._stats: Dict = {}
        self._thread: Optional[threading.Thread] = None

    # ── Camera management ──────────────────────────────────────────────

    def list_cameras(self, max_check: int = 10) -> List[Dict]:
        """Probe available camera devices and return info for each."""
        cameras = []
        for idx in range(max_check):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cameras.append(
                    {
                        "index": idx,
                        "resolution": f"{w}x{h}",
                        "fps": fps,
                        "backend": cap.getBackendName(),
                    }
                )
                cap.release()
            else:
                cap.release()
        return cameras

    def open_camera(self, device_index: Optional[int] = None) -> bool:
        """
        Open the USB camera device.

        Args:
            device_index: Override default device index

        Returns:
            True if camera opened successfully
        """
        idx = device_index if device_index is not None else self.device_index
        with self._lock:
            if self._cap is not None:
                self._cap.release()

            # Try DirectShow first (Windows), then default backend
            self._cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(idx)

            if not self._cap.isOpened():
                logger.error(f"Failed to open camera at index {idx}")
                self._cap = None
                return False

            # Set resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
            logger.info(
                f"Camera opened: index={idx}, "
                f"resolution={actual_w}x{actual_h}, fps={actual_fps}"
            )
            self.device_index = idx
            return True

    def close_camera(self):
        """Release the camera device."""
        self.stop()
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        logger.info("Camera closed")

    # ── Processing loop ────────────────────────────────────────────────

    def start(self):
        """Start the background processing loop."""
        if self._running:
            return
        if self._cap is None or not self._cap.isOpened():
            if not self.open_camera():
                raise RuntimeError("Cannot start: camera not available")

        self._running = True
        self._frame_idx = 0
        self.tracker.reset()
        self._thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._thread.start()
        logger.info("Live stream processing started")

    def stop(self):
        """Stop the background processing loop."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Live stream processing stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def _processing_loop(self):
        """Main frame capture + detection + tracking loop."""
        frame_interval = 1.0 / self.target_fps

        while self._running:
            t0 = time.time()

            with self._lock:
                if self._cap is None or not self._cap.isOpened():
                    self._running = False
                    break
                ret, frame = self._cap.read()

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            self._latest_frame = frame

            # Detect
            detections = self.detector.detect_and_track(frame)

            # Track
            detections = self.tracker.update(detections, self._frame_idx)

            # Annotate
            annotated = self._draw_annotations(frame, detections)
            self._latest_annotated = annotated

            # Compute stats
            all_traj = self.tracker.get_all_trajectories()
            active_traj = self.tracker.get_active_trajectories()

            # Quick motility classification for active tracks
            fps_est = self.target_fps
            motility_counts = {"progressive": 0, "non_progressive": 0, "immotile": 0}
            for tid, traj in active_traj.items():
                if len(traj) >= 5:
                    mot = self.motility_analyzer.analyze_motility(traj, fps_est)
                    label = mot.get("label", "immotile")
                    if label in motility_counts:
                        motility_counts[label] += 1

            self._stats = {
                "frame_idx": self._frame_idx,
                "total_tracked": len(all_traj),
                "active_count": len(active_traj),
                "detections_this_frame": len(detections),
                "motility": motility_counts,
                "velocity_unit": self._velocity_unit(),
                "calibration_warning": self._calibration_warning(),
                "fps_actual": round(1.0 / max(time.time() - t0, 0.001), 1),
            }

            self._frame_idx += 1

            # Throttle to target FPS
            elapsed = time.time() - t0
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    # ── Annotation drawing ─────────────────────────────────────────────

    def _draw_annotations(
        self, frame: np.ndarray, detections: List[Dict]
    ) -> np.ndarray:
        """Draw detection boxes, trajectory trails, and ID labels."""
        overlay = frame.copy()
        all_traj = self.tracker.get_all_trajectories()
        active_ids = self.tracker.active_tracks

        TAIL_LENGTH = 60
        LINE_THICKNESS = 2
        HEAD_RADIUS = 5
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.5
        FONT_THICKNESS = 1

        for tid in active_ids:
            traj = all_traj.get(tid, [])
            if not traj:
                continue

            # Keep only points up to current frame
            pts = [(fi, x, y) for fi, x, y in traj if fi <= self._frame_idx]
            if not pts:
                continue
            pts = pts[-TAIL_LENGTH:]

            color = self._track_color(tid)
            coords = [(int(x), int(y)) for _, x, y in pts]

            # Draw fading trajectory line
            n = len(coords)
            for i in range(1, n):
                alpha = 0.3 + 0.7 * (i / n)
                faded = tuple(int(c * alpha) for c in color)
                cv2.line(
                    overlay,
                    coords[i - 1],
                    coords[i],
                    faded,
                    LINE_THICKNESS,
                    cv2.LINE_AA,
                )

            # Head marker
            head = coords[-1]
            cv2.circle(overlay, head, HEAD_RADIUS, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, head, HEAD_RADIUS + 1, (255, 255, 255), 1, cv2.LINE_AA)

            # ID label
            label = f"ID:{tid}"
            (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
            lx = head[0] - tw // 2
            ly = head[1] - 12
            cv2.rectangle(
                overlay, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), (0, 0, 0), -1
            )
            cv2.putText(
                overlay,
                label,
                (lx, ly),
                FONT,
                FONT_SCALE,
                color,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        # Top-left summary
        total = len(all_traj)
        active = len(active_ids)
        summary = f"Tracked sperm: {total}"
        cv2.putText(overlay, summary, (10, 30), FONT, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, summary, (10, 30), FONT, 1.0, (0, 0, 0), 1, cv2.LINE_AA)

        # FPS indicator
        fps_text = f"FPS: {self._stats.get('fps_actual', 0)}"
        cv2.putText(
            overlay, fps_text, (10, 60), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )

        return overlay

    @staticmethod
    def _track_color(tid: int) -> Tuple[int, int, int]:
        """Generate a unique bright color for each track ID."""
        hue = int((tid * 37) % 180)
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    # ── Frame access ───────────────────────────────────────────────────

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest raw frame."""
        return self._latest_frame

    def get_latest_annotated(self) -> Optional[np.ndarray]:
        """Get the latest annotated frame with detections and trajectories."""
        return self._latest_annotated

    def get_jpeg_frame(self, quality: int = 80) -> Optional[bytes]:
        """Get the latest annotated frame as JPEG bytes for streaming."""
        frame = self._latest_annotated
        if frame is None:
            return None
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ret:
            return None
        return buf.tobytes()

    def get_stats(self) -> Dict:
        """Get current tracking statistics."""
        return self._stats.copy()

    def get_motility_summary(self) -> Dict:
        """Get motility summary for all tracked sperm so far."""
        all_traj = self.tracker.get_all_trajectories()
        fps_est = self.target_fps
        results = {}
        for tid, traj in all_traj.items():
            if len(traj) >= 3:
                results[tid] = self.motility_analyzer.analyze_motility(traj, fps_est)
        counts = {"progressive": 0, "non_progressive": 0, "immotile": 0}
        for r in results.values():
            label = r.get("label", "immotile")
            if label in counts:
                counts[label] += 1
        return {
            "total_tracked": len(all_traj),
            "motility_counts": counts,
            "velocity_unit": self._velocity_unit(),
            "calibration_warning": self._calibration_warning(),
            "per_sperm": {str(k): v for k, v in results.items()},
        }

    def _velocity_unit(self) -> str:
        if self.motility_analyzer.micrometers_per_pixel is None:
            return "px/s"
        return "µm/s"

    def _calibration_warning(self) -> Optional[str]:
        if self.motility_analyzer.micrometers_per_pixel is None:
            return UNCALIBRATED_VELOCITY_WARNING
        return None

    def reset_tracking(self):
        """Reset all tracking state (start fresh analysis)."""
        self.tracker.reset()
        self._frame_idx = 0
        self._stats = {}
        logger.info("Tracking state reset")

    # ── WebSocket frame input (QR code / browser camera) ───────────────

    def process_frame(self, jpeg_bytes: bytes) -> Optional[bytes]:
        """
        Process a single JPEG frame received from a WebSocket client
        (mobile phone browser camera).

        Args:
            jpeg_bytes: Raw JPEG image bytes from the mobile browser

        Returns:
            Annotated JPEG bytes to send back, or None on decode failure
        """
        t0 = time.time()

        # Decode JPEG to numpy array
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        self._latest_frame = frame

        # Detect
        detections = self.detector.detect_and_track(frame)

        # Track
        detections = self.tracker.update(detections, self._frame_idx)

        # Annotate
        annotated = self._draw_annotations(frame, detections)
        self._latest_annotated = annotated

        # Compute stats
        all_traj = self.tracker.get_all_trajectories()
        active_traj = self.tracker.get_active_trajectories()

        motility_counts = {"progressive": 0, "non_progressive": 0, "immotile": 0}
        for tid, traj in active_traj.items():
            if len(traj) >= 5:
                mot = self.motility_analyzer.analyze_motility(traj, self.target_fps)
                label = mot.get("label", "immotile")
                if label in motility_counts:
                    motility_counts[label] += 1

        self._stats = {
            "type": "stats",
            "frame_idx": self._frame_idx,
            "total_tracked": len(all_traj),
            "active_count": len(active_traj),
            "detections_this_frame": len(detections),
            "motility": motility_counts,
            "velocity_unit": self._velocity_unit(),
            "calibration_warning": self._calibration_warning(),
            "fps_actual": round(1.0 / max(time.time() - t0, 0.001), 1),
        }

        self._frame_idx += 1

        # Encode annotated frame back to JPEG
        ret, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            return None
        return buf.tobytes()

    def start_websocket_mode(self):
        """
        Initialize for WebSocket frame input mode (no USB camera needed).
        Call this before process_frame().
        """
        self._running = True
        self._frame_idx = 0
        self.tracker.reset()
        self._stats = {}
        logger.info("Live stream started in WebSocket mode (QR code camera)")

    def stop_websocket_mode(self):
        """Stop WebSocket mode and return final summary."""
        self._running = False
        logger.info("WebSocket mode stopped")
