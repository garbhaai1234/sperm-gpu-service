"""
Main Sperm Analysis Pipeline
Integrates all components for end-to-end sperm quality analysis
"""

import cv2
import numpy as np
import os
import json
import time
import csv
import torch
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Callable
import logging
from collections import defaultdict
from itertools import permutations

from .grid_tracking import GridConfig, GridDiagnostics
from .detection import SpermDetector
from .tracking import CSRTConfig, CSRTSpermTracker, SpermTracker
from .segmentation import SpermSegmentation
from .morphology import MorphologyAnalyzer
from .motility import MotilityAnalyzer, UNCALIBRATED_VELOCITY_WARNING

logger = logging.getLogger(__name__)


class SpermIdentityRegistry(dict):
    """Video-local monotonic application-ID registry with optional future capacity."""

    def allocate_id(self) -> Optional[int]:
        capacity = self.get("identity_capacity")
        candidate = int(self.get("next_id", 1))
        if capacity is not None and candidate > int(capacity):
            return None
        self["next_id"] = candidate + 1
        return candidate


class SpermAnalysisPipeline:
    """
    Main pipeline for comprehensive sperm quality analysis
    """

    CONFIDENCE_THRESHOLD = 0.45
    LOW_SCORE_MIN = 0.30
    MORPHOLOGY_THRESHOLD = 0.70
    LOW_SCORE_MAX_DISTANCE = 40.0
    LOW_SCORE_MIN_IOU = 0.30
    MAX_IDENTITY_SUPPORT_FRAMES = 3
    LOW_SCORE_AMBIGUITY_MAX_COST_MARGIN = 0.10
    LOW_SCORE_MOTION_MIN_SPEED = 1.0
    DEFAULT_TARGET_LABELS = {1}
    MASK_THRESHOLD = 0.50
    HIGH_COUNT_MASK_CURRENT_WEIGHT = 0.65
    LOW_COUNT_MASK_CURRENT_WEIGHT = 0.75
    MIN_MASK_AREA = 6
    MORPHOLOGY_KERNEL_SIZE = 3
    MORPHOLOGY_CLOSE_ITERATIONS = 1
    MASK_STATE_MAX_AGE = 45
    MASK_OPACITY = 0.50
    SPERM_MASK_COLOR = (0, 255, 0)
    DRAW_TRACK_IDS = True
    DRAW_CENTRE_DOT = False
    DRAW_TRAILS = False
    TRAIL_LENGTH = 40
    
    def __init__(self, 
                 detection_model_path: str,
                 maskrcnn_model_path: str,
                 hnk_model_path: str,
                 device: str = "cpu",
                 frame_skip: int = 1,
                 morph_every_n: int = 5,
                 maskrcnn_downscale: float = 1.0,
                 maskrcnn_threshold: float = 0.7,
                 maskrcnn_sperm_class_ids: Optional[str] = None,
                 sliced_inference: bool = False,
                 slice_size: int = 320,
                 slice_overlap: float = 0.25,
                 tracking_method: str = "distance",
                 detector_interval: int = 10,
                 csrt_config: Optional[CSRTConfig] = None,
                 micrometers_per_pixel: Optional[float] = None,
                 show_tracking_grid: bool = False,
                 grid_rows: int = 8, grid_columns: int = 8,
                 grid_history_length: int = 40):
        """
        Initialize the complete sperm analysis pipeline
        
        Args:
            detection_model_path: Path to YOLOv8 detection model
            maskrcnn_model_path: Path to Mask R-CNN segmentation model
            hnk_model_path: Path to Head-Neck-Tail segmentation model
            device: Device to run inference on
            frame_skip: Process every N frames (1 = all frames)
            morph_every_n: Compute morphology every N frames
            maskrcnn_downscale: Downscale factor for Mask R-CNN (0.5-1.0)
            maskrcnn_threshold: Confidence threshold for Mask R-CNN visualization
            maskrcnn_sperm_class_ids: Comma-separated Mask R-CNN label ids to keep
            sliced_inference: Run tiled detection and segmentation inference
            slice_size: Tile width/height in pixels for sliced inference
            slice_overlap: Fractional tile overlap for sliced inference
            tracking_method: "distance" keeps SpermTracker; "csrt" selects CSRT
            detector_interval: Frames between CSRT detector corrections
            csrt_config: Optional CSRT parameter set
            micrometers_per_pixel: Measured µm per pixel. None until calibrated.
        """
        self.grid_config = GridConfig(show_tracking_grid, grid_rows, grid_columns, grid_history_length)
        self.grid_paths = {}
        self.device = device
        self.frame_skip = frame_skip
        self.morph_every_n = morph_every_n
        self.maskrcnn_downscale = maskrcnn_downscale
        self.sliced_inference = sliced_inference
        self.slice_size = slice_size
        self.slice_overlap = slice_overlap
        self.tracking_method = (tracking_method or "distance").strip().lower()
        self.detector_interval = int(detector_interval)
        self.csrt_config = None
        
        # Initialize components
        self.detector = SpermDetector(
            detection_model_path,
            device,
            sliced_inference=sliced_inference,
            slice_size=slice_size,
            slice_overlap=slice_overlap,
        )
        if self.tracking_method == "distance":
            self.tracker = SpermTracker()
        elif self.tracking_method == "csrt":
            self.csrt_config = (
                csrt_config.with_detector_interval(self.detector_interval)
                if csrt_config is not None
                else CSRTConfig(
                    detector_interval=self.detector_interval,
                    tracking_debug=os.environ.get("SPERM_PIPELINE_TRACKING_DEBUG", "").strip().lower()
                    in {"1", "true", "yes", "on"},
                )
            )
            self.tracker = CSRTSpermTracker(self.csrt_config)
        else:
            raise ValueError(
                f"Unknown tracking_method '{tracking_method}'. Expected 'distance' or 'csrt'."
            )
        self.segmenter = SpermSegmentation(
            maskrcnn_model_path,
            hnk_model_path,
            device,
            threshold=maskrcnn_threshold,
            sperm_class_ids=maskrcnn_sperm_class_ids,
            sliced_inference=sliced_inference,
            slice_size=slice_size,
            slice_overlap=slice_overlap,
        )
        self.morphology_analyzer = MorphologyAnalyzer()
        self.micrometers_per_pixel = micrometers_per_pixel
        self.motility_analyzer = MotilityAnalyzer(
            micrometers_per_pixel=micrometers_per_pixel
        )
        
        # Results storage
        self.trajectories = {}
        self.morphology_results = {}
        self.motility_results = {}
        self.application_trajectories = {}
        self.inference_tracking_rows = []
        self._canonical_identity_available = False
        self.frame_count = 0
        self.source_fps = None
        self.identity_registry_path = None
        self.identity_quality_events_path = None
        self.tracking_quality_summary_path = None
        self.overlap_review_path = None
        
        logger.info(
            "Sperm analysis pipeline initialized successfully (tracking_method=%s)",
            self.tracking_method,
        )
    def _maskrcnn_target_labels(self) -> set:
        return set(getattr(self.segmenter, "sperm_class_ids", None) or self.DEFAULT_TARGET_LABELS)

    def _maskrcnn_class_name(self, label_id: int) -> str:
        classes = getattr(self.segmenter, "maskrcnn_classes", None) or []
        if 0 <= label_id < len(classes):
            return str(classes[label_id])
        return f"Class {label_id}"


    def _reset_run_state(self):
        """Reset mutable state before processing a new video."""
        self.grid_paths = {}
        self.tracker.reset()
        self.trajectories = {}
        self.morphology_results = {}
        self.motility_results = {}
        self.application_trajectories = {}
        self.inference_tracking_rows = []
        self._canonical_identity_available = False
        self.frame_count = 0
        self.identity_registry_path = None
        self.identity_quality_events_path = None
        self.tracking_quality_summary_path = None
        self.overlap_review_path = None
    
    def process_video(self, video_path: str, output_dir: str, 
                     progress_callback: Optional[Callable] = None) -> Dict:
        """
        Process a complete video for sperm analysis
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save outputs
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing analysis results and output paths
        """
        self._reset_run_state()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.source_fps = float(fps) if fps is not None else None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Processing video: {total_frames} frames at {fps} FPS")
        
        # Initialize processed video writer. The final inference video owns
        # its writer inside _generate_inference_video.
        processed_writer = self._init_processed_video_writer(
            output_dir, fps, width, height
        )
        
        try:
            # Process frames
            frame_idx = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if requested
                if frame_idx % self.frame_skip != 0:
                    frame_idx += 1
                    continue
                
                # Process single frame for analysis metadata. Inference video
                # generation happens later from Mask R-CNN detections.
                self._process_frame(frame, frame_idx, fps)
                
                # Update progress
                processed_frames += 1
                if progress_callback and processed_frames % 10 == 0:
                    progress = min(80, 10 + (processed_frames / max(1, total_frames // self.frame_skip)) * 70)
                    progress_callback(f"Processed {processed_frames} frames...", progress)
                
                frame_idx += 1
            
            # The video inference identity is canonical for uploaded analysis.
            # Discard morphology keyed by the separate diagnostic tracker; the
            # inference pass will attach morphology directly to application IDs.
            self.morphology_results = defaultdict(list)

            # Generate processed video with the existing processed-output logic.
            self._generate_processed_video(video_path, processed_writer, fps)
            if processed_writer:
                processed_writer.release()
                processed_writer = None

            if progress_callback:
                progress_callback("Generating inference video...", 82)
            self._generate_inference_video(video_path, output_dir, fps, width, height)
            
            # Analyze results
            if progress_callback:
                progress_callback("Analyzing results...", 85)
            
            # Compute final analysis
            analysis_results = self._compute_final_analysis(fps)
            
            # Save results
            output_paths = self._save_results(output_dir, analysis_results)
            
            if progress_callback:
                progress_callback("Analysis complete!", 100)
            
            return {
                'analysis_results': analysis_results,
                'output_paths': output_paths,
                'video_properties': {
                    'total_frames': total_frames,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'velocity_unit': (
                        'µm/s' if self.micrometers_per_pixel else 'px/s'
                    ),
                    'calibration_warning': (
                        None
                        if self.micrometers_per_pixel
                        else UNCALIBRATED_VELOCITY_WARNING
                    ),
                }
            }
            
        finally:
            cap.release()
            if processed_writer:
                processed_writer.release()
    
    def _process_frame(self, frame: np.ndarray, frame_idx: int, fps: float):
        """
        Process a single frame for analysis metadata.

        Args:
            frame: Input frame
            frame_idx: Frame index
            fps: Frames per second
        """
        if self.tracking_method == "csrt":
            self._process_frame_csrt(frame, frame_idx, fps)
            return

        # Detect sperm heads
        detections = self.detector.detect_and_track(frame)

        # Assign stable track IDs via tracker
        detections = self.tracker.update(detections, frame_idx)

        # Morphology for uploaded analysis is computed in the inference pass,
        # where the canonical application ID and the accepted mask coexist.
    
    def _analyze_morphology_for_frame(self, frame: np.ndarray, 
                                    detections: List[Dict], 
                                    segments: List[Dict]):
        """
        Analyze morphology for detections in current frame
        
        Args:
            frame: Current frame
            detections: List of detections
            segments: List of full sperm segments
        """
        for detection in detections:
            track_id = detection.get('application_id', detection.get('track_id'))
            if track_id is None:
                continue
            
            # Find matching segment
            segment = self.segmenter.match_detection_to_segment(detection, segments)
            if segment is None:
                continue
            
            # Crop region for subpart analysis
            bbox = segment['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            x1, y1 = max(0, x1-3), max(0, y1-3)
            x2, y2 = min(frame.shape[1], x2+3), min(frame.shape[0], y2+3)
            
            crop_img = frame[y1:y2, x1:x2]
            crop_mask = segment['mask'][y1:y2, x1:x2]
            
            # Segment subparts
            subparts = self.segmenter.segment_subparts(crop_img, crop_mask)
            
            # Analyze morphology
            morph_result = self.morphology_analyzer.analyze_morphology(
                subparts['head'], subparts['neck'], subparts['tail']
            )
            
            # Store result
            if track_id not in self.morphology_results:
                self.morphology_results[track_id] = []
            self.morphology_results[track_id].append(morph_result)
    
    def _generate_processed_video(self, video_path: str,
                                 writer: cv2.VideoWriter, fps: float):
        """
        Generate the processed video with direct Mask R-CNN sperm masks.
        This mirrors the notebook-style output: green mask overlay, green
        bounding box, and class/score text from the active Mask R-CNN checkpoint.
        """
        if writer is None:
            logger.error("Processed video writer is None, cannot generate video")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video for processed output: {video_path}")
            return

        color = (0, 255, 0)
        frames_written = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output = frame.copy()
            height, width = output.shape[:2]
            segments = self.segmenter.segment_full_sperm(frame)

            for segment in segments:
                mask = np.asarray(segment['mask'], dtype=np.float32)
                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
                mask_bin = mask > 0.5

                colored_mask = np.zeros_like(output, dtype=np.uint8)
                colored_mask[mask_bin] = color
                output = cv2.addWeighted(output, 1.0, colored_mask, 0.5, 0)

                bbox = segment['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width - 1, x2), min(height - 1, y2)
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

                label_id = int(segment.get('label', next(iter(self._maskrcnn_target_labels()))))
                class_name = segment.get('class_name') or self._maskrcnn_class_name(label_id)
                score = float(segment.get('confidence', 0.0))
                text = f"{class_name}: {score:.2f}"
                cv2.putText(
                    output, text, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

            writer.write(output)
            frames_written += 1

            if frames_written % 100 == 0:
                logger.info(f"Processed Mask R-CNN video: {frames_written} frames written")

        cap.release()
        logger.info(f"Processed Mask R-CNN video generation complete: {frames_written} frames written")

    class _ByteTrackDetections:
        """Small adapter exposing the fields Ultralytics BYTETracker expects."""

        def __init__(self, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray):
            self.xyxy = boxes.astype(np.float32, copy=False)
            if len(boxes) == 0:
                self.xywh = np.empty((0, 4), dtype=np.float32)
            else:
                xywh = self.xyxy.copy()
                xywh[:, 0] = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
                xywh[:, 1] = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0
                xywh[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
                xywh[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
                self.xywh = xywh
            self.conf = scores.astype(np.float32, copy=False)
            self.cls = labels.astype(np.float32, copy=False)

    def _create_bytetrack(self, video_fps: float):
        from ultralytics.trackers.byte_tracker import BYTETracker

        args = SimpleNamespace(
            track_high_thresh=0.25,
            track_low_thresh=0.10,
            new_track_thresh=0.25,
            track_buffer=40,
            match_thresh=0.75,
            fuse_score=True,
        )
        return BYTETracker(args, frame_rate=max(1, int(round(video_fps or 30.0))))

    def _run_maskrcnn_inference(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Return normal high-score detections and isolated identity-only candidates."""
        if self.segmenter.maskrcnn_model is None:
            return {
                'boxes': np.empty((0, 4), dtype=np.float32),
                'scores': np.empty((0,), dtype=np.float32),
                'labels': np.empty((0,), dtype=np.int64),
                'masks': np.empty((0, frame.shape[0], frame.shape[1]), dtype=np.float32),
                'identity_boxes': np.empty((0, 4), dtype=np.float32),
                'identity_scores': np.empty((0,), dtype=np.float32),
                'identity_labels': np.empty((0,), dtype=np.int64),
                'identity_masks': np.empty((0, frame.shape[0], frame.shape[1]), dtype=np.float32),
            }

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(self.device) / 255.0

        with torch.inference_mode():
            prediction = self.segmenter.maskrcnn_model([tensor])[0]

        boxes = prediction.get('boxes', torch.empty((0, 4), device=tensor.device)).detach().cpu().numpy()
        scores = prediction.get('scores', torch.empty((0,), device=tensor.device)).detach().cpu().numpy()
        labels_tensor = prediction.get('labels')
        labels = (
            labels_tensor.detach().cpu().numpy()
            if labels_tensor is not None
            else np.ones((len(scores),), dtype=np.int64)
        )
        target_labels = self._maskrcnn_target_labels()
        masks_tensor = prediction.get('masks')
        masks = (
            masks_tensor[:, 0].detach().cpu().numpy()
            if masks_tensor is not None and len(masks_tensor) > 0
            else np.empty((0, height, width), dtype=np.float32)
        )

        valid_label = np.asarray([int(label) in target_labels for label in labels], dtype=bool)
        high_keep = (
            valid_label & (scores >= self.CONFIDENCE_THRESHOLD)
            & (scores > self.MORPHOLOGY_THRESHOLD)
        )
        low_keep = valid_label & (scores >= self.LOW_SCORE_MIN) & (scores <= self.MORPHOLOGY_THRESHOLD)

        boxes = boxes.astype(np.float32, copy=False)
        scores = scores.astype(np.float32, copy=False)
        labels = labels.astype(np.int64, copy=False)
        masks = masks.astype(np.float32, copy=False)
        high_boxes, high_scores, high_labels, high_masks = (
            boxes[high_keep], scores[high_keep], labels[high_keep], masks[high_keep]
        )
        identity_boxes, identity_scores, identity_labels, identity_masks = (
            boxes[low_keep], scores[low_keep], labels[low_keep], masks[low_keep]
        )

        if masks.ndim == 3 and masks.shape[1:3] != (height, width):
            masks = np.stack([
                cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
                for mask in masks
            ]).astype(np.float32, copy=False)
            high_masks = masks[high_keep]
            identity_masks = masks[low_keep]

        for selected_boxes in (high_boxes, identity_boxes):
            if len(selected_boxes):
                selected_boxes[:, [0, 2]] = np.clip(selected_boxes[:, [0, 2]], 0, width - 1)
                selected_boxes[:, [1, 3]] = np.clip(selected_boxes[:, [1, 3]], 0, height - 1)

        return {
            'boxes': high_boxes, 'scores': high_scores, 'labels': high_labels, 'masks': high_masks,
            'identity_boxes': identity_boxes, 'identity_scores': identity_scores,
            'identity_labels': identity_labels, 'identity_masks': identity_masks,
        }

    @staticmethod
    def _box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
        area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
        denom = area_a + area_b - inter
        return inter / denom if denom > 0.0 else 0.0

    def _match_tracks_to_detection_indices(self, tracks: np.ndarray, boxes: np.ndarray) -> Dict[int, int]:
        if len(tracks) == 0 or len(boxes) == 0:
            return {}

        candidates = []
        for track_idx, track in enumerate(tracks):
            for det_idx, box in enumerate(boxes):
                iou = self._box_iou_xyxy(track[:4], box)
                if iou > 0.0:
                    candidates.append((iou, track_idx, det_idx))

        matches = {}
        used_tracks = set()
        used_detections = set()
        for _, track_idx, det_idx in sorted(candidates, reverse=True):
            if track_idx in used_tracks or det_idx in used_detections:
                continue
            matches[track_idx] = det_idx
            used_tracks.add(track_idx)
            used_detections.add(det_idx)
        return matches

    @staticmethod
    def _mask_centroid(probability_mask: np.ndarray, fallback_box: np.ndarray) -> Tuple[float, float]:
        binary = probability_mask >= 0.50
        moments = cv2.moments(binary.astype(np.uint8))
        if moments['m00'] > 0:
            return (float(moments['m10'] / moments['m00']), float(moments['m01'] / moments['m00']))
        return (float((fallback_box[0] + fallback_box[2]) / 2.0), float((fallback_box[1] + fallback_box[3]) / 2.0))

    def _smooth_track_mask(
        self,
        tracker_id: int,
        current_probability: np.ndarray,
        current_centroid: Tuple[float, float],
        frame_index: int,
        mask_states: Dict[int, Dict],
        current_weight: float,
    ) -> np.ndarray:
        previous = mask_states.get(tracker_id)
        if previous is None or previous['probability'].shape != current_probability.shape:
            smoothed = current_probability
        else:
            prev_centroid = previous['centroid']
            dx = current_centroid[0] - prev_centroid[0]
            dy = current_centroid[1] - prev_centroid[1]
            height, width = current_probability.shape[:2]
            transform = np.float32([[1, 0, dx], [0, 1, dy]])
            translated_previous = cv2.warpAffine(
                previous['probability'],
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            smoothed = (
                current_weight * current_probability
                + (1.0 - current_weight) * translated_previous
            )

        mask_states[tracker_id] = {
            'probability': smoothed.astype(np.float32, copy=False),
            'centroid': current_centroid,
            'last_seen': frame_index,
        }
        return smoothed

    def _process_frame_csrt(self, frame: np.ndarray, frame_idx: int, fps: float):
        """CSRT analysis path. Morphology frequency stays on morph_every_n."""
        is_detector = self.tracker.is_detector_frame(frame_idx)
        is_morphology = frame_idx % self.morph_every_n == 0

        if is_detector:
            segments = self.segmenter.segment_full_sperm(frame)
            detections = [self._segment_to_detection(segment) for segment in segments]
            detections = self.tracker.update(
                detections, frame_idx, frame=frame, reinitialize=True
            )
        else:
            segments = []
            detections = self.tracker.update(
                [], frame_idx, frame=frame, reinitialize=False
            )

        if is_morphology and not is_detector:
            segments = self.segmenter.segment_full_sperm(frame)
        if is_morphology and segments:
            self._analyze_morphology_for_frame(frame, detections, segments)
        if segments:
            self._associate_csrt_masks(self.tracker, detections, segments)

    @staticmethod
    def _segment_to_detection(segment: Dict) -> Dict:
        bbox = segment.get("bbox")
        if hasattr(bbox, "tolist"):
            bbox = bbox.tolist()
        if bbox is not None and len(bbox) >= 4:
            bbox = [float(v) for v in bbox[:4]]
            center = (
                float((bbox[0] + bbox[2]) / 2.0),
                float((bbox[1] + bbox[3]) / 2.0),
            )
        else:
            bbox = None
            center = (None, None)
        return {
            "bbox": bbox,
            "confidence": float(segment.get("confidence", 0.0)),
            "track_id": None,
            "center": center,
            "mask": segment.get("mask"),
            "label": segment.get("label"),
            "class_name": segment.get("class_name"),
        }

    def _associate_csrt_masks(self, tracker: CSRTSpermTracker, detections: List[Dict], segments: List[Dict]) -> None:
        """Attach a fresh Mask R-CNN mask to an existing CSRT track. Does not reinitialize CSRT."""
        for detection in detections:
            track_id = detection.get("track_id")
            if track_id is None:
                continue
            segment = self.segmenter.match_detection_to_segment(detection, segments)
            if segment is None or segment.get("mask") is None:
                continue
            detection["mask"] = segment["mask"]
            tracker.remember_mask(track_id, segment["mask"], detection.get("center"))

    def _attach_shifted_csrt_mask(self, tracker: CSRTSpermTracker, detection: Dict) -> None:
        track_id = detection.get("track_id")
        if track_id is None:
            return
        state = tracker.get_mask_state(track_id)
        if state is None:
            return
        mask, centroid = state
        center = detection.get("center")
        if centroid is None or center is None or center[0] is None or center[1] is None:
            return
        dx = float(center[0]) - float(centroid[0])
        dy = float(center[1]) - float(centroid[1])
        shifted = self._translate_mask(mask, dx, dy)
        detection["mask"] = shifted
        tracker.remember_mask(track_id, shifted, center)

    @staticmethod
    def _translate_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
        mask = np.asarray(mask, dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[0]
        height, width = mask.shape[:2]
        transform = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(
            mask,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def _draw_csrt_annotation(self, output: np.ndarray, detection: Dict) -> None:
        """Green mask, green box, and green track id for one CSRT detection."""
        color = (0, 255, 0)
        opacity = 0.5
        height, width = output.shape[:2]
        mask = detection.get("mask")
        if mask is not None:
            mask = np.asarray(mask, dtype=np.float32)
            if mask.ndim == 3:
                mask = mask[0]
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
            mask_bin = mask > 0.5
            if np.any(mask_bin):
                colored_mask = np.zeros_like(output, dtype=np.uint8)
                colored_mask[mask_bin] = color
                blended = cv2.addWeighted(output, 1.0, colored_mask, opacity, 0)
                output[:, :, :] = blended

        bbox = detection.get("bbox")
        if not bbox or len(bbox) < 4:
            return
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        application_id = detection.get("application_id", detection.get("track_id"))
        if application_id is not None:
            cv2.putText(
                output,
                f"ID: {application_id}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    def _generate_csrt_inference_video(
        self,
        video_path: str,
        output_dir: str,
        fps: float,
        width: int,
        height: int,
    ):
        """CSRT inference video. The ByteTrack inference path is left unchanged."""
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            logger.error(f"Could not open video for CSRT inference output: {video_path}")
            return

        if not fps or fps <= 0:
            fps = 30.0

        temp_path = os.path.join(output_dir, "output_inference_video.avi")
        writer = cv2.VideoWriter(
            temp_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not initialize inference video writer: {temp_path}")

        tracker = CSRTSpermTracker(self.csrt_config or CSRTConfig(detector_interval=self.detector_interval))
        csrt_identity_state = self._initialize_application_identity_state(
            width, height, fps, identity_capacity=None
        )
        csrt_identity_state["application_by_tracker"] = {}
        grid = GridDiagnostics(os.path.basename(os.path.normpath(output_dir)), width, height, fps,
                               getattr(self, "grid_config", GridConfig()))
        metadata_rows: List[Dict] = []
        frame_index = 0
        frames_written = 0

        try:
            while True:
                success, frame = capture.read()
                if not success or frame is None:
                    break

                is_detector = tracker.is_detector_frame(frame_index)
                is_morphology = frame_index % self.morph_every_n == 0
                if is_detector:
                    segments = self.segmenter.segment_full_sperm(frame)
                    detections = [self._segment_to_detection(segment) for segment in segments]
                    detections = tracker.update(
                        detections, frame_index, frame=frame, reinitialize=True
                    )
                    if segments:
                        self._associate_csrt_masks(tracker, detections, segments)
                elif is_morphology:
                    detections = tracker.update([], frame_index, frame=frame, reinitialize=False)
                    segments = self.segmenter.segment_full_sperm(frame)
                    if segments:
                        self._associate_csrt_masks(tracker, detections, segments)
                else:
                    detections = tracker.update([], frame_index, frame=frame, reinitialize=False)
                    for detection in detections:
                        self._attach_shifted_csrt_mask(tracker, detection)

                for detection in detections:
                    tracker_id = detection.get("track_id")
                    if tracker_id is not None:
                        detection["application_id"] = self._csrt_application_id(
                            int(tracker_id), csrt_identity_state
                        )
                        item = dict(detection)
                        item["tracker_id"] = int(tracker_id)
                        app_id = int(detection["application_id"])
                        if app_id not in csrt_identity_state["identities"]:
                            csrt_identity_state["identities"][app_id] = self._new_inference_identity(
                                app_id, item, frame_index
                            )
                        self._touch_inference_identity(
                            csrt_identity_state, app_id, item, frame_index,
                            int(tracker_id), update_velocity=True,
                        )

                if is_morphology and segments:
                    self._analyze_morphology_for_frame(frame, detections, segments)

                grid_items, grid_state = self._csrt_grid_snapshot(
                    detections, tracker, csrt_identity_state, is_detector
                )
                grid.capture(frame_index, grid_items,
                             [(d["application_id"], "CSRT_UPDATE", "existing CSRT canonical assignment") for d in grid_items],
                             grid_state, tracker_kind="csrt")
                output = frame.copy()
                for detection in detections:
                    if detection.get("track_id") is None:
                        continue
                    self._draw_csrt_annotation(output, detection)
                    bbox = detection.get("bbox") or [0, 0, 0, 0]
                    center = detection.get("center") or (0.0, 0.0)
                    mask = detection.get("mask")
                    mask_area = 0
                    if mask is not None:
                        mask_arr = np.asarray(mask)
                        if mask_arr.ndim == 3:
                            mask_arr = mask_arr[0]
                        mask_area = int((mask_arr > 0.5).sum())
                    label = detection.get("label")
                    confidence = detection.get("confidence")
                    metadata_rows.append({
                        "frame_index": int(frame_index),
                        "timestamp_seconds": float(frame_index / fps),
                        "tracker_id": int(detection["track_id"]),
                        "application_id": int(detection["application_id"]),
                        "class_id": int(label) if label is not None else 0,
                        "confidence": float(confidence) if confidence is not None else 0.0,
                        "centroid_x": float(center[0]) if center[0] is not None else 0.0,
                        "centroid_y": float(center[1]) if center[1] is not None else 0.0,
                        "x1": float(bbox[0]),
                        "y1": float(bbox[1]),
                        "x2": float(bbox[2]),
                        "y2": float(bbox[3]),
                        "mask_area_pixels": mask_area,
                    })

                writer.write(output)
                frames_written += 1
                frame_index += 1
                if frames_written % 100 == 0:
                    logger.info(f"CSRT inference video: {frames_written} frames written")
        finally:
            capture.release()
            writer.release()

        meta_dir = os.path.join(output_dir, "meta")
        os.makedirs(meta_dir, exist_ok=True)
        metadata_path = os.path.join(meta_dir, "inference_tracking.csv")
        fieldnames = [
            "frame_index", "timestamp_seconds", "tracker_id", "application_id", "class_id",
            "confidence", "centroid_x", "centroid_y", "x1", "y1", "x2", "y2",
            "mask_area_pixels",
        ]
        with open(metadata_path, "w", newline="") as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(metadata_rows)
        self.inference_tracking_metadata_path = metadata_path
        self.inference_tracking_rows = metadata_rows
        self.application_trajectories = self._trajectories_from_inference_rows(
            metadata_rows, require_morphology_valid=False
        )
        self._canonical_identity_available = True
        self._write_application_identity_artifacts(
            video_path, meta_dir, fps, width, height, frames_written,
            metadata_rows, csrt_identity_state,
        )
        self._finish_grid_diagnostics(grid, video_path, output_dir)
        self.inference_video_path = self._convert_to_mp4(temp_path)
        logger.info(
            f"CSRT inference video generation complete: {frames_written} frames written; "
            f"metadata rows={len(metadata_rows)}"
        )
        self._write_csrt_diagnostics(tracker, meta_dir, "csrt_inference_diagnostics.json", "inference")

    # Video-local labels only. These do not change ByteTrack's track_buffer or IDs.
    # Stage 1 keeps the ByteTrack ID already stored on an application track.
    # A 1-frame step within 45px is always kept. That covers the 4.7px
    # frame-24 case and the measured p99 step of a continuing track (43px).
    # A larger step is kept only when the previous box still overlaps.
    # Each extra missed frame, up to 10, adds 8px to the close band and 12px
    # to the overlap band.
    # Stage 2 restores a different ByteTrack ID only from the last center:
    # 18px on the next frame, plus 5px per extra missed frame, for 10 frames.
    # A reserved ID cannot be taken by another detection.
    INFERENCE_APP_ID_MAX_LOST_FRAMES = 10
    INFERENCE_APP_ID_STRONG_MAX_GAP = 10
    INFERENCE_APP_ID_STRONG_BASE_DIST = 80.0
    INFERENCE_APP_ID_STRONG_DIST_PER_EXTRA_FRAME = 12.0
    INFERENCE_APP_ID_STRONG_CLOSE_DIST = 45.0
    INFERENCE_APP_ID_STRONG_CLOSE_PER_EXTRA_FRAME = 8.0
    INFERENCE_APP_ID_STRONG_MIN_IOU = 0.20
    INFERENCE_APP_ID_REASSOC_BASE_DIST = 18.0
    INFERENCE_APP_ID_REASSOC_DIST_PER_EXTRA_FRAME = 5.0
    INFERENCE_APP_ID_REASSOC_MAX_SIDE_RATIO = 2.0
    INFERENCE_APP_ID_OCCLUDED_MAX_SECONDS = 3.0
    INFERENCE_APP_ID_REID_BASE_DISTANCE = 20.0
    INFERENCE_APP_ID_REID_DISTANCE_PER_FRAME = 7.0
    INFERENCE_APP_ID_REID_MAX_DISTANCE = 100.0
    INFERENCE_APP_ID_REID_AMBIGUITY_MARGIN = 0.10
    INFERENCE_APP_ID_ASSOCIATION_AMBIGUITY_MARGIN = 0.10
    INFERENCE_APP_ID_SAME_BYTE_STRONG_IOU = 0.85
    INFERENCE_APP_ID_UNRESOLVED_CANDIDATE_SECONDS = 1.0
    INFERENCE_APP_ID_FRAGMENT_GAP_SECONDS = 1.0
    POSSIBLE_SWITCH_MIN_IOU = 0.75
    POSSIBLE_SWITCH_MAX_CENTER_FRACTION = 0.10
    POSSIBLE_SWITCH_MAX_PREDICTION_FRACTION = 0.25

    @staticmethod
    def _record_lifecycle_event(state, frame, event, application_id, **details):
        row = {"frame": int(frame), "event": event, "application_id": int(application_id)}
        row.update(details)
        state.setdefault("lifecycle_events", []).append(row)

    @staticmethod
    def _new_inference_identity(application_id, item, frame_index):
        """Create the persistent video-local record for one canonical identity."""
        center = tuple(float(v) for v in item["center"])
        bbox = tuple(float(v) for v in item["bbox"])
        byte_id = item.get("tracker_id")
        return {
            "application_id": int(application_id),
            "state": "CONFIRMED",
            "status": "CONFIRMED",  # compatibility with the existing linker
            "first_seen_frame": int(frame_index),
            "last_seen_frame": int(frame_index),
            "last_high_conf_frame": int(frame_index),
            "last_center": center,
            "last_bbox": bbox,
            "trusted_center": center,
            "trusted_bbox": bbox,
            "last_high_conf_center": center,
            "last_high_conf_bbox": bbox,
            "last_frame": int(frame_index),
            "last_observation_center": center,
            "last_observation_bbox": bbox,
            "last_observation_frame": int(frame_index),
            "previous_center": None,
            "velocity": (0.0, 0.0),
            "trajectory_history": [(int(frame_index), center, bbox)],
            "high_conf_history": [(int(frame_index), center, bbox)],
            "current_byte_id": int(byte_id) if byte_id is not None else None,
            "byte_id_history": ([{"frame": int(frame_index), "tracker_id": int(byte_id)}]
                                if byte_id is not None else []),
            "byte_track_ids": ({int(byte_id)} if byte_id is not None else set()),
            "overlap_partners": set(),
            "occlusion_partners": set(),
            "last_occlusion_partners": set(),
            "occlusion_start_frame": None,
            "lost_since_frame": None,
            "identity_support_count": 0,
            "low_score_supported": False,
            "created_frame": int(frame_index),
            "created_reason": "new_high_confidence_detection",
        }

    @staticmethod
    def _initialize_application_identity_state(width, height, fps, identity_capacity=None):
        """Start a fresh, uncapped identity session for each uploaded video."""
        return SpermIdentityRegistry({
            "next_id": 1,
            "identity_capacity": identity_capacity,
            "identities": {},
            "byte_owner": {},
            "frame_size": (int(width), int(height)),
            "source_fps": float(fps),
            "lifecycle_events": [],
            "overlap_groups": {},
            "next_overlap_group_id": 1,
            "unresolved_candidates": {},
            "next_unresolved_candidate_id": 1,
            "identity_assignment_diagnostics": [],
            "low_score_diagnostics": [],
            "quality_events": [],
        })

    def _identity_recovery_window_frames(self, fps=None):
        """Keep lost identities for a video-time window, not a fixed frame count."""
        source_fps = fps if fps and np.isfinite(float(fps)) and float(fps) > 0 else 30.0
        return max(1, int(np.ceil(float(source_fps) * self.INFERENCE_APP_ID_OCCLUDED_MAX_SECONDS)))

    def _mark_merged_occlusion_observations(self, observations, state, frame_index):
        """Block merged boxes and preserve the pre-overlap identity group state."""
        identities = state.setdefault("identities", {})
        state.setdefault("overlap_groups", {})
        for index, item in enumerate(observations):
            if float(item.get("confidence", 0.0)) <= self.MORPHOLOGY_THRESHOLD:
                continue
            x1, y1, x2, y2 = item["bbox"]
            participants = []
            for app_id, identity in identities.items():
                if identity.get("status") == "TERMINATED":
                    continue
                anchor = identity.get("trusted_center", identity.get("last_high_conf_center", identity.get("last_center")))
                bbox = identity.get("trusted_bbox", identity.get("last_high_conf_bbox", identity.get("last_bbox")))
                if anchor is None or bbox is None:
                    continue
                anchor_frame = identity.get("last_high_conf_frame", identity.get("last_frame", frame_index))
                gap = int(frame_index) - int(anchor_frame)
                if gap > self._identity_recovery_window_frames(state.get("source_fps")):
                    continue
                # An absent member's old box is not current overlap evidence.
                # In particular, an extrapolated exit from the field must not
                # turn a different sperm at that old location into a merge.
                if gap > 1:
                    velocity = self._recent_inference_velocity(identity)
                    predicted = (anchor[0] + velocity[0] * gap, anchor[1] + velocity[1] * gap)
                    width, height = state.get("frame_size", (None, None))
                    if (width is not None and height is not None
                            and not (0 <= predicted[0] < width and 0 <= predicted[1] < height)):
                        continue
                center_inside = x1 <= anchor[0] <= x2 and y1 <= anchor[1] <= y2
                if center_inside:
                    participants.append(int(app_id))
            if len(participants) < 2:
                continue
            state["blocked_high_detections"].add(index)
            item["occlusion_merged"] = True
            group = next((group for group in state["overlap_groups"].values()
                          if set(group["identity_ids"]) == set(participants)
                          and group.get("state") not in {"RESOLVED", "EXPIRED"}), None)
            if group is None:
                group_id = int(state.setdefault("next_overlap_group_id", 1))
                state["next_overlap_group_id"] = group_id + 1
                pre_state = {}
                for app_id in participants:
                    identity = identities[app_id]
                    pre_state[str(app_id)] = {
                        "frame": int(identity.get("last_high_conf_frame", identity.get("last_frame", frame_index))),
                        "center": identity.get("trusted_center", identity.get("last_high_conf_center", identity.get("last_center"))),
                        "bbox": identity.get("trusted_bbox", identity.get("last_high_conf_bbox", identity.get("last_bbox"))),
                        "velocity": tuple(identity.get("velocity") or (0.0, 0.0)),
                        "direction": tuple(identity.get("velocity") or (0.0, 0.0)),
                        "byte_ids": sorted(identity.get("byte_track_ids", set())),
                    }
                group = {
                    "overlap_group_id": group_id,
                    "identity_ids": sorted(participants),
                    "start_frame": int(frame_index),
                    "last_frame": int(frame_index),
                    "state": "OCCLUDED",
                    "merged_detection_indices": [],
                    "pre_overlap_state": pre_state,
                    "unresolved_frames": [],
                }
                state["overlap_groups"][group_id] = group
                state.setdefault("quality_events", []).append({
                    "frame": int(frame_index), "event": "OVERLAP_GROUP_STARTED",
                    "overlap_group_id": group_id, "application_ids": sorted(participants),
                })
            group["last_frame"] = int(frame_index)
            group["state"] = "OCCLUDED"
            group["merged_detection_indices"].append({"frame": int(frame_index), "detection_index": int(index)})
            for app_id in participants:
                identity = identities[app_id]
                partners = set(participants) - {app_id}
                previous_status = identity.get("status")
                identity["status"] = "OCCLUDED"
                identity["state"] = "OCCLUDED"
                identity.setdefault("occlusion_start_frame", int(frame_index))
                identity.setdefault("lost_since_frame", int(frame_index))
                identity.setdefault("occlusion_partners", set()).update(partners)
                identity.setdefault("last_occlusion_partners", set()).update(partners)
                identity.setdefault("overlap_partners", set()).update(partners)
                if previous_status != "OCCLUDED":
                    self._record_lifecycle_event(
                        state, frame_index, "ID_OCCLUDED", app_id,
                        occlusion_partners=sorted(partners), merged_detection_index=index,
                        byte_track_id=item.get("tracker_id"),
                    )
        # Record likely close-contact pairs even before a merged box is emitted.
        # This retains pre-contact motion for the later joint recovery stage.
        active = [(int(app_id), identity) for app_id, identity in identities.items()
                  if identity.get("state", identity.get("status")) in {"CONFIRMED", "LOW_SUPPORT"}
                  and identity.get("trusted_center", identity.get("last_high_conf_center")) is not None]
        visible_count = sum(1 for item in observations
                            if float(item.get("confidence", 0.0)) > self.MORPHOLOGY_THRESHOLD)
        if visible_count <= len(active):
            for pos, (left_id, left) in enumerate(active):
                for right_id, right in active[pos + 1:]:
                    left_center = left.get("trusted_center", left.get("last_high_conf_center"))
                    right_center = right.get("trusted_center", right.get("last_high_conf_center"))
                    left_box = left.get("trusted_bbox", left.get("last_high_conf_bbox"))
                    right_box = right.get("trusted_bbox", right.get("last_high_conf_bbox"))
                    if left_box is None or right_box is None:
                        continue
                    scale = max(1.0, *(abs(float(v)) for v in (
                        left_box[2] - left_box[0], left_box[3] - left_box[1],
                        right_box[2] - right_box[0], right_box[3] - right_box[1])))
                    gap = float(np.hypot(left_center[0] - right_center[0],
                                         left_center[1] - right_center[1]))
                    if gap > scale * 0.75 and self._bbox_iou(left_box, right_box) <= 0.05:
                        continue
                    group = next((g for g in state["overlap_groups"].values()
                                  if set(g["identity_ids"]) == {left_id, right_id}
                                  and g.get("state") not in {"RESOLVED", "EXPIRED"}), None)
                    if group is None:
                        group_id = int(state.setdefault("next_overlap_group_id", 1))
                        state["next_overlap_group_id"] = group_id + 1
                        group = {
                            "overlap_group_id": group_id,
                            "identity_ids": [left_id, right_id],
                            "start_frame": int(frame_index),
                            "last_frame": int(frame_index),
                            "state": "CONTACT",
                            "merged_detection_indices": [],
                            "pre_overlap_state": {
                                str(app_id): {
                                    "frame": int(identity.get("last_high_conf_frame", frame_index)),
                                    "center": identity.get("trusted_center", identity.get("last_high_conf_center")),
                                    "bbox": identity.get("trusted_bbox", identity.get("last_high_conf_bbox")),
                                    "velocity": tuple(identity.get("velocity") or (0.0, 0.0)),
                                    "direction": tuple(identity.get("velocity") or (0.0, 0.0)),
                                    "byte_ids": sorted(identity.get("byte_track_ids", set())),
                                }
                                for app_id, identity in ((left_id, left), (right_id, right))
                            },
                            "unresolved_frames": [],
                        }
                        state["overlap_groups"][group_id] = group
                        state.setdefault("quality_events", []).append({
                            "frame": int(frame_index), "event": "OVERLAP_GROUP_STARTED",
                            "overlap_group_id": group_id,
                            "application_ids": [left_id, right_id],
                            "trigger": "close_contact_with_detection_count_drop",
                        })
                    group["last_frame"] = int(frame_index)
                    for app_id, partner in ((left_id, right_id), (right_id, left_id)):
                        identities[app_id].setdefault("overlap_partners", set()).add(partner)

    def _update_occlusion_states(self, observations, state, frame_index):
        """Keep briefly absent identities available for conservative recovery."""
        identities = state.setdefault("identities", {})
        blocked = state.get("blocked_high_detections", set())
        visible_bytes = {int(item["tracker_id"]) for i, item in enumerate(observations)
                         if i not in blocked and item.get("tracker_id") is not None}
        visible_boxes = [item for i, item in enumerate(observations)
                         if i not in blocked and float(item.get("confidence", 0.0)) > self.MORPHOLOGY_THRESHOLD]
        for app_id, identity in identities.items():
            if identity.get("status") == "TERMINATED":
                continue
            last_frame = identity.get("last_high_conf_frame", identity.get("last_frame"))
            if last_frame is None:
                continue
            gap = int(frame_index) - int(last_frame)
            if gap <= 0:
                continue
            if gap > self._identity_recovery_window_frames(state.get("source_fps")):
                if identity.get("status") != "TERMINATED":
                    identity["status"] = "TERMINATED"
                    identity["state"] = "TERMINATED"
                    self._record_lifecycle_event(state, frame_index, "ID_TERMINATED", app_id,
                                                 last_reliable_frame=last_frame, frame_gap=gap)
                continue
            byte_id = identity.get("current_byte_id")
            if byte_id is not None and int(byte_id) in visible_bytes:
                continue
            center = identity.get("last_high_conf_center", identity.get("last_center"))
            bbox = identity.get("last_high_conf_bbox", identity.get("last_bbox"))
            if center is None or bbox is None:
                continue
            close_partners = set()
            for item in visible_boxes:
                width = max(1.0, abs(bbox[2] - bbox[0]))
                height = max(1.0, abs(bbox[3] - bbox[1]))
                distance = float(np.hypot(item["center"][0] - center[0], item["center"][1] - center[1]))
                near = distance <= max(width, height) * 1.5 or self._bbox_iou(item["bbox"], bbox) > 0.0
                if near:
                    partner = state.get("byte_owner", {}).get(int(item["tracker_id"]))
                    if partner is not None and partner != app_id:
                        close_partners.add(int(partner))
            old_status = identity.get("status")
            new_status = "OCCLUDED" if close_partners or old_status == "OCCLUDED" else "LOST"
            identity["status"] = new_status
            identity["state"] = new_status
            identity.setdefault("lost_since_frame", int(frame_index))
            if new_status == "OCCLUDED":
                identity.setdefault("occlusion_start_frame", int(frame_index))
                identity.setdefault("occlusion_partners", set()).update(close_partners)
                identity.setdefault("last_occlusion_partners", set()).update(close_partners)
                if old_status != "OCCLUDED":
                    self._record_lifecycle_event(state, frame_index, "ID_OCCLUDED", app_id,
                                                 occlusion_partners=sorted(close_partners),
                                                 last_reliable_frame=last_frame)
            elif old_status not in {"LOST", "OCCLUDED"}:
                    self._record_lifecycle_event(state, frame_index, "ID_LOST", app_id,
                                             last_reliable_frame=last_frame)

    def _update_overlap_group_states(self, state, frame_index):
        """Keep overlap groups until every member is safely confirmed or expired."""
        for group in state.get("overlap_groups", {}).values():
            if group.get("state") in {"RESOLVED", "EXPIRED"}:
                continue
            # Keep a permanent audit trail, but remove unavailable members from
            # the active group so they cannot block recovery or leave a stale
            # UNRESOLVED group influencing future associations.
            original_ids = [int(value) for value in group.get(
                "all_identity_ids", group.get("identity_ids", []))]
            group.setdefault("all_identity_ids", list(original_ids))
            retired = set(int(value) for value in group.get("retired_identity_ids", []))
            active_ids = []
            for app_id in original_ids:
                identity = state.get("identities", {}).get(app_id)
                if identity is None or identity.get("state", identity.get("status")) == "TERMINATED":
                    retired.add(app_id)
                else:
                    active_ids.append(app_id)
            group["retired_identity_ids"] = sorted(retired)
            group["identity_ids"] = active_ids
            if not active_ids:
                group["state"] = "EXPIRED"
                group["resolved_frame"] = int(frame_index)
                continue
            members = [state["identities"][app_id] for app_id in active_ids]
            # Once a surviving member has a reliable observation after contact,
            # a terminated partner no longer makes the group ambiguous.
            lone_member_recovered = (
                len(active_ids) == 1
                and members[0].get("state", members[0].get("status")) == "CONFIRMED"
                and int(members[0].get("last_high_conf_frame", -1))
                > int(group.get("last_frame", group.get("start_frame", -1)))
            )
            all_confirmed_now = all(
                member.get("state", member.get("status")) == "CONFIRMED"
                and member.get("last_seen_frame") == int(frame_index)
                for member in members
            )
            still_in_contact = False
            if all_confirmed_now and len(members) > 1:
                for pos, left in enumerate(members):
                    for right in members[pos + 1:]:
                        left_box = left.get("trusted_bbox", left.get("last_high_conf_bbox"))
                        right_box = right.get("trusted_bbox", right.get("last_high_conf_bbox"))
                        left_center = left.get("trusted_center", left.get("last_high_conf_center"))
                        right_center = right.get("trusted_center", right.get("last_high_conf_center"))
                        if any(value is None for value in (left_box, right_box, left_center, right_center)):
                            continue
                        scale = max(1.0, left_box[2] - left_box[0], left_box[3] - left_box[1],
                                    right_box[2] - right_box[0], right_box[3] - right_box[1])
                        distance = float(np.hypot(left_center[0] - right_center[0],
                                                  left_center[1] - right_center[1]))
                        if distance <= scale * 0.75 or self._bbox_iou(left_box, right_box) > 0.05:
                            still_in_contact = True
                if still_in_contact:
                    # Confirmation is not separation. Keep one physical contact
                    # episode instead of resolving/recreating it every frame.
                    group["state"] = "CONTACT"
                    group["last_frame"] = int(frame_index)
                    continue
            if all_confirmed_now or lone_member_recovered:
                group["state"] = "RESOLVED"
                group["resolved_frame"] = int(frame_index)
                group["resolution"] = (
                    "REMAINING_MEMBER_CONFIRMED_AFTER_PARTNER_TERMINATED"
                    if lone_member_recovered else "ALL_MEMBERS_CONFIRMED"
                )
                state.setdefault("quality_events", []).append({
                    "frame": int(frame_index), "event": "OVERLAP_GROUP_RESOLVED",
                    "overlap_group_id": group["overlap_group_id"],
                    "application_ids": list(group.get("all_identity_ids", active_ids)),
                    "retired_identity_ids": sorted(retired),
                    "resolution": group["resolution"],
                })
            elif int(frame_index) > int(group.get("last_frame", frame_index)):
                frames = group.setdefault("unresolved_frames", [])
                if not frames or frames[-1] != int(frame_index):
                    frames.append(int(frame_index))

    def _reidentify_occluded_tracks(self, observations, state, frame_index, reserved_det, reserved_app):
        """Match recent lost IDs to high detections; reject close competing explanations."""
        eligible = [
            (app_id, identity) for app_id, identity in state.get("identities", {}).items()
            if identity.get("status") in {"OCCLUDED", "LOST", "UNRESOLVED"} and app_id not in reserved_app
        ]
        high_indices = [i for i, item in enumerate(observations)
                        if i not in reserved_det and i not in state.get("blocked_high_detections", set())
                        and float(item.get("confidence", 0.0)) > self.MORPHOLOGY_THRESHOLD]
        candidates_by_detection = defaultdict(list)
        for index in high_indices:
            item = observations[index]
            for app_id, identity in eligible:
                anchor_frame = identity.get("last_high_conf_frame", identity.get("last_frame"))
                anchor = identity.get("last_high_conf_center", identity.get("last_center"))
                anchor_bbox = identity.get("last_high_conf_bbox", identity.get("last_bbox"))
                gap = int(frame_index) - int(anchor_frame) if anchor_frame is not None else 999999
                reason = None
                metrics = None
                if gap < 1 or gap > self._identity_recovery_window_frames(state.get("source_fps")):
                    reason = "GAP_TOO_LARGE"
                elif anchor is None or anchor_bbox is None:
                    reason = "NO_PREVIOUS_CANDIDATE"
                else:
                    velocity = self._recent_inference_velocity(identity)
                    predicted = (
                        anchor[0] + velocity[0] * gap,
                        anchor[1] + velocity[1] * gap,
                    )
                    actual = float(np.hypot(item["center"][0] - anchor[0], item["center"][1] - anchor[1]))
                    pred_dist = float(np.hypot(item["center"][0] - predicted[0], item["center"][1] - predicted[1]))
                    side_ratio = self._bbox_long_side_ratio(item["bbox"], anchor_bbox)
                    limit = min(
                        self.INFERENCE_APP_ID_REID_MAX_DISTANCE,
                        self.INFERENCE_APP_ID_REID_BASE_DISTANCE
                        + self.INFERENCE_APP_ID_REID_DISTANCE_PER_FRAME * gap,
                    )
                    speed = float(np.hypot(*velocity))
                    dot = ((item["center"][0] - anchor[0]) * velocity[0]
                           + (item["center"][1] - anchor[1]) * velocity[1])
                    motion_opposes = speed > self.LOW_SCORE_MOTION_MIN_SPEED and dot < 0
                    metrics = {
                        "gap": gap, "dist_last": actual, "dist_anchor": actual,
                        "dist_pred": pred_dist, "iou": self._bbox_iou(item["bbox"], anchor_bbox),
                        "iou_anchor": self._bbox_iou(item["bbox"], anchor_bbox),
                        "side_ratio": side_ratio, "speed": speed,
                        "motion_opposes": motion_opposes, "previous_center": anchor,
                        "previous_bbox": anchor_bbox, "anchor_center": anchor,
                        "anchor_bbox": anchor_bbox, "predicted_center": predicted,
                    }
                    trajectory = self._unresolved_candidate_trajectory_evidence(
                        item, app_id, state, frame_index
                    )
                    if trajectory is not None:
                        metrics["unresolved_candidate_evidence"] = trajectory
                    if side_ratio > self.INFERENCE_APP_ID_REASSOC_MAX_SIDE_RATIO:
                        reason = "SIZE_INCONSISTENT"
                    elif ((motion_opposes or
                           (speed > self.LOW_SCORE_MOTION_MIN_SPEED and pred_dist > limit))
                          and not (trajectory is not None
                                   and trajectory["predicted_distance"] <= limit)):
                        reason = "MOTION_INCONSISTENT"
                    elif actual > limit or pred_dist > limit:
                        if trajectory is None or trajectory["predicted_distance"] > limit:
                            reason = "OLD_ID_TOO_FAR"
                    else:
                        # Lower is better; all terms reward prediction and size continuity.
                        cost = (pred_dist / max(1.0, limit) + actual / max(1.0, limit)
                                + abs(np.log(max(1e-6, side_ratio))))
                        if trajectory is not None:
                            cost += trajectory["predicted_distance"] / max(1.0, limit)
                        candidates_by_detection[index].append((float(cost), app_id, metrics))
                        continue
                    if reason is None:
                        cost = (trajectory["predicted_distance"] / max(1.0, limit)
                                + actual / max(1.0, limit)
                                + abs(np.log(max(1e-6, side_ratio))))
                        candidates_by_detection[index].append((float(cost), app_id, metrics))
                        continue
                self._record_identity_assignment_diagnostic(
                    state, frame_index, index, item, app_id, identity.get("current_byte_id"),
                    metrics, "OCCLUDED_REIDENTIFICATION", False, reason or "NO_PREVIOUS_CANDIDATE")
                rejection_reason = reason or "NO_PREVIOUS_CANDIDATE"
                if metrics is None:
                    rejection_rank = float("inf")
                else:
                    normalizer = max(
                        1.0,
                        self.INFERENCE_APP_ID_REID_BASE_DISTANCE
                        + self.INFERENCE_APP_ID_REID_DISTANCE_PER_FRAME * max(0, metrics["gap"] - 1),
                    )
                    rejection_rank = (
                        metrics["dist_pred"] / normalizer
                        + metrics["dist_last"] / normalizer
                        + abs(np.log(max(1e-6, metrics["side_ratio"])))
                    )
                previous_rank = state.setdefault("pending_new_id_rejection_rank", {}).get(index, float("inf"))
                if rejection_rank < previous_rank or index not in state["pending_new_id_reasons"]:
                    state["pending_new_id_reasons"][index] = rejection_reason
                    state["pending_new_id_rejection_rank"][index] = rejection_rank
        # Reject near-tied per-detection and per-identity claims before solving
        # the complete one-to-one assignment.
        ambiguous_detections = set()
        ambiguous_ids = set()
        for index in high_indices:
            options = sorted(candidates_by_detection.get(index, []), key=lambda row: (row[0], row[1]))
            if len(options) > 1 and options[1][0] - options[0][0] <= self.INFERENCE_APP_ID_REID_AMBIGUITY_MARGIN:
                ambiguous_detections.add(index)
                ambiguous_ids.update(row[1] for row in options[:2])
        by_identity = defaultdict(list)
        for index in high_indices:
            for cost, app_id, metrics in candidates_by_detection.get(index, []):
                by_identity[app_id].append((cost, index, metrics))
        for app_id, claims in by_identity.items():
            claims.sort(key=lambda row: (row[0], row[1]))
            if len(claims) > 1 and claims[1][0] - claims[0][0] <= self.INFERENCE_APP_ID_REID_AMBIGUITY_MARGIN:
                ambiguous_ids.add(app_id)
                ambiguous_detections.update(row[1] for row in claims[:2])

        # Recover an overlap group as a unit. A partial set of reappearances
        # cannot safely consume only one member of an unresolved group.
        for group in state.get("overlap_groups", {}).values():
            if group.get("state") in {"RESOLVED", "EXPIRED"}:
                continue
            group_ids = [int(v) for v in group.get("identity_ids", [])
                         if int(v) in {app for app, _ in eligible}]
            if len(group_ids) < 2:
                continue
            relevant_dets = [index for index in high_indices
                             if any(app_id in group_ids for _, app_id, _ in candidates_by_detection.get(index, []))]
            if not relevant_dets:
                continue
            lookup = {(index, app_id): (cost, metrics)
                      for index in relevant_dets
                      for cost, app_id, metrics in candidates_by_detection.get(index, [])
                      if app_id in group_ids}
            if len(relevant_dets) != len(group_ids):
                ambiguous_ids.update(group_ids)
                ambiguous_detections.update(relevant_dets)
                group["state"] = "UNRESOLVED"
                group.setdefault("unresolved_frames", []).append(int(frame_index))
                for app_id in group_ids:
                    identity = state["identities"][app_id]
                    identity["state"] = identity["status"] = "UNRESOLVED"
                    self._record_lifecycle_event(
                        state, frame_index, "ID_UNRESOLVED", app_id,
                        reason="AMBIGUOUS_OVERLAP_GROUP",
                        overlap_group_id=group["overlap_group_id"],
                    )
                state.setdefault("quality_events", []).append({
                    "frame": int(frame_index), "event": "OVERLAP_GROUP_AMBIGUOUS",
                    "overlap_group_id": group["overlap_group_id"],
                    "application_ids": group_ids, "detection_indices": relevant_dets,
                    "reason": "INCOMPLETE_GROUP_OBSERVATIONS",
                })
                continue

            matrix = np.full((len(group_ids), len(relevant_dets)), 1.0e6, dtype=np.float64)
            for app_row, app_id in enumerate(group_ids):
                for det_col, det_index in enumerate(relevant_dets):
                    if (det_index, app_id) in lookup:
                        matrix[app_row, det_col] = lookup[(det_index, app_id)][0]

            def solve_group(costs):
                try:
                    from scipy.optimize import linear_sum_assignment
                    rows, cols = linear_sum_assignment(costs)
                    if len(rows) != len(group_ids) or any(costs[r, c] >= 1.0e6 for r, c in zip(rows, cols)):
                        return None
                    return float(sum(costs[r, c] for r, c in zip(rows, cols))), list(zip(rows.tolist(), cols.tolist()))
                except Exception:
                    choices = []
                    for chosen_dets in permutations(range(len(relevant_dets)), len(group_ids)):
                        if len(set(chosen_dets)) != len(group_ids):
                            continue
                        value = sum(costs[row, col] for row, col in enumerate(chosen_dets))
                        if value < 1.0e6 * len(group_ids):
                            choices.append((float(value), list(enumerate(chosen_dets))))
                    return min(choices, key=lambda item: item[0]) if choices else None

            best = solve_group(matrix)
            alternatives = []
            if best is not None:
                for app_row, det_col in best[1]:
                    alternative_matrix = matrix.copy()
                    alternative_matrix[app_row, det_col] = 1.0e6
                    alternative = solve_group(alternative_matrix)
                    if alternative is not None:
                        alternatives.append(alternative[0])
            if best is None or (alternatives and min(alternatives) - best[0]
                                <= self.INFERENCE_APP_ID_REID_AMBIGUITY_MARGIN):
                ambiguous_ids.update(group_ids)
                ambiguous_detections.update(relevant_dets)
                group["state"] = "UNRESOLVED"
                group.setdefault("unresolved_frames", []).append(int(frame_index))
                for app_id in group_ids:
                    identity = state["identities"][app_id]
                    identity["state"] = identity["status"] = "UNRESOLVED"
                    self._record_lifecycle_event(state, frame_index, "ID_UNRESOLVED", app_id,
                                                 reason="AMBIGUOUS_OVERLAP_GROUP",
                                                 overlap_group_id=group["overlap_group_id"])
                state.setdefault("quality_events", []).append({
                    "frame": int(frame_index), "event": "OVERLAP_GROUP_AMBIGUOUS",
                    "overlap_group_id": group["overlap_group_id"],
                    "application_ids": group_ids, "detection_indices": relevant_dets,
                    "best_cost": best[0] if best else None,
                    "second_best_cost": min(alternatives) if alternatives else None,
                })
            else:
                # Local near-ties are not group ambiguity when the complete
                # assignment has a clear winner. Only override local rejection
                # for an isolated group: competing outside identities/groups
                # must retain the conservative global ambiguity protection.
                isolated = (
                    all(app in group_ids for index in relevant_dets
                        for _, app, _ in candidates_by_detection.get(index, []))
                    and not any(
                        other is not group
                        and other.get("state") not in {"RESOLVED", "EXPIRED"}
                        and set(other.get("identity_ids", [])).intersection(group_ids)
                        for other in state.get("overlap_groups", {}).values()
                    )
                )
                if isolated:
                    ambiguous_ids.difference_update(group_ids)
                    ambiguous_detections.difference_update(relevant_dets)
                    chosen = {relevant_dets[col]: group_ids[row] for row, col in best[1]}
                    for index in relevant_dets:
                        candidates_by_detection[index] = [
                            edge for edge in candidates_by_detection[index]
                            if edge[1] == chosen[index]
                        ]

        for index in sorted(ambiguous_detections):
            options = sorted(candidates_by_detection.get(index, []), key=lambda row: row[0])
            state["blocked_high_detections"].add(index)
            state["pending_new_id_reasons"][index] = "AMBIGUOUS_REIDENTIFICATION"
            state.setdefault("high_priority_app_candidates", set()).update(
                app_id for _, app_id, _ in options
            )
            best_cost = options[0][0] if options else None
            second_cost = options[1][0] if len(options) > 1 else None
            for cost, app_id, metrics in options:
                if app_id not in ambiguous_ids and index not in ambiguous_detections:
                    continue
                self._record_identity_assignment_diagnostic(
                    state, frame_index, index, observations[index], app_id,
                    state["identities"][app_id].get("current_byte_id"), metrics,
                    "AMBIGUOUS_REJECT", False, "AMBIGUOUS_REIDENTIFICATION", True,
                    best_cost, second_cost)

        valid_edges = [(cost, index, app_id, metrics)
                       for index in high_indices
                       if index not in ambiguous_detections
                       for cost, app_id, metrics in candidates_by_detection.get(index, [])
                       if app_id not in ambiguous_ids]
        det_ids = sorted({row[1] for row in valid_edges})
        app_ids = sorted({row[2] for row in valid_edges})
        if not det_ids or not app_ids:
            return []
        matrix = np.full((len(app_ids), len(det_ids)), 1.0e6, dtype=np.float64)
        lookup = {}
        app_row = {value: i for i, value in enumerate(app_ids)}
        det_col = {value: i for i, value in enumerate(det_ids)}
        for cost, index, app_id, metrics in valid_edges:
            matrix[app_row[app_id], det_col[index]] = cost
            lookup[(app_id, index)] = (cost, metrics)
        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(matrix)
            pairs = zip(rows.tolist(), cols.tolist())
        except Exception:
            ranked = sorted((matrix[r, c], r, c) for r in range(len(app_ids)) for c in range(len(det_ids)))
            used_r, used_c, picked = set(), set(), []
            for cost, row, col in ranked:
                if cost >= 1.0e6 or row in used_r or col in used_c:
                    continue
                used_r.add(row); used_c.add(col); picked.append((row, col))
            pairs = picked
        resolved = []
        for row, col in pairs:
            app_id, index = app_ids[row], det_ids[col]
            if matrix[row, col] >= 1.0e6 or (app_id, index) not in lookup:
                continue
            cost, metrics = lookup[(app_id, index)]
            alternatives = sorted(value[0] for value in candidates_by_detection[index])
            second = alternatives[1] if len(alternatives) > 1 else None
            resolved.append((index, app_id, metrics, cost, second))
        return resolved

    def _unresolved_candidate_trajectory_evidence(self, item, application_id, state, frame_index):
        """Return trajectory support from a still-live unresolved observation chain."""
        fps = max(1.0, float(state.get("source_fps") or 30.0))
        lifetime = max(1, int(round(self.INFERENCE_APP_ID_UNRESOLVED_CANDIDATE_SECONDS * fps)))
        best = None
        for candidate in state.get("unresolved_candidates", {}).values():
            if int(application_id) not in set(candidate.get("possible_identities", [])):
                continue
            history = candidate.get("observations", [])
            # One ambiguous point has no directional information and must not
            # bias a group assignment toward either member.
            if len(history) < 2:
                continue
            identity = state.get("identities", {}).get(application_id, {})
            anchor = identity.get("trusted_center", identity.get("last_high_conf_center"))
            anchor_box = identity.get("trusted_bbox", identity.get("last_high_conf_bbox"))
            anchor_frame = identity.get("last_high_conf_frame")
            if anchor is None or anchor_box is None or anchor_frame is None:
                continue
            # Candidate self-consistency is not evidence that it belongs to
            # this old identity. Its first point must be spatially anchored to
            # that identity, inside the existing maximum recovery envelope.
            first = history[0]
            first_gap = int(first["frame"]) - int(anchor_frame)
            if first_gap < 1:
                continue
            first_limit = min(self.INFERENCE_APP_ID_REID_MAX_DISTANCE,
                self.INFERENCE_APP_ID_REID_BASE_DISTANCE
                + self.INFERENCE_APP_ID_REID_DISTANCE_PER_FRAME * first_gap)
            first_distance = float(np.hypot(first["center"][0] - anchor[0],
                                             first["center"][1] - anchor[1]))
            if (first_distance > first_limit
                    or self._bbox_long_side_ratio(first["bbox"], anchor_box)
                    > self.INFERENCE_APP_ID_REASSOC_MAX_SIDE_RATIO):
                continue
            previous = history[-1]
            gap = int(frame_index) - int(previous["frame"])
            if gap < 1 or gap > lifetime:
                continue
            velocity = (0.0, 0.0)
            if len(history) >= 2:
                before = history[-2]
                elapsed = int(previous["frame"]) - int(before["frame"])
                if elapsed > 0:
                    velocity = tuple(
                        (float(previous["center"][axis]) - float(before["center"][axis])) / elapsed
                        for axis in range(2)
                    )
            predicted = tuple(float(previous["center"][axis]) + velocity[axis] * gap
                              for axis in range(2))
            distance = float(np.hypot(item["center"][0] - predicted[0],
                                      item["center"][1] - predicted[1]))
            box = previous["bbox"]
            scale = max(1.0, abs(float(box[2]) - float(box[0])),
                        abs(float(box[3]) - float(box[1])))
            # Reuse the existing candidate continuity envelope; this adds
            # trajectory ranking evidence without widening the spatial gate.
            if distance > max(20.0, 2.0 * scale):
                continue
            evidence = {
                "candidate_id": int(candidate["candidate_id"]),
                "predicted_center": predicted,
                "predicted_distance": distance,
                "direction": velocity,
                "elapsed_frames": gap,
                "byte_track_ids": list(candidate.get("byte_track_ids", [])),
            }
            if best is None or evidence["predicted_distance"] < best["predicted_distance"]:
                best = evidence
        return best

    @staticmethod
    def _new_id_reason(item, state):
        width, height = state.get("frame_size", (None, None))
        if width is not None and height is not None:
            x1, y1, x2, y2 = item["bbox"]
            margin = max(2.0, min(float(width), float(height)) * 0.02)
            if x1 <= margin or y1 <= margin or x2 >= float(width) - margin or y2 >= float(height) - margin:
                return "TRUE_NEW_ENTRANT"
        return "NO_PREVIOUS_CANDIDATE"

    def _expire_unresolved_candidates(self, state, frame_index):
        """Age pending evidence on every frame, including frames without detections."""
        fps = max(1.0, float(state.get("source_fps") or 30.0))
        pending = state.setdefault("unresolved_candidates", {})
        lifetime = max(1, int(round(self.INFERENCE_APP_ID_UNRESOLVED_CANDIDATE_SECONDS * fps)))
        for key in list(pending):
            if int(frame_index) - int(pending[key]["last_frame"]) > lifetime:
                candidate = pending[key]
                candidate["state"] = "EXPIRED"
                candidate["end_frame"] = int(frame_index)
                state.setdefault("unresolved_candidate_archive", []).append(candidate)
                state.setdefault("quality_events", []).append({
                    "frame": int(frame_index), "event": "UNRESOLVED_CANDIDATE_EXPIRED",
                    "candidate_id": int(key),
                })
                del pending[key]

    def _resolve_unresolved_candidate(self, state, item, application_id, frame_index, metrics):
        """Retire evidence only when this accepted observation identifies its chain."""
        pending = state.get("unresolved_candidates", {})
        evidence = (metrics or {}).get("unresolved_candidate_evidence")
        candidates = []
        for key, candidate in pending.items():
            if application_id not in candidate.get("possible_identities", []):
                continue
            if int(candidate["last_frame"]) >= int(frame_index):
                continue
            history = candidate.get("observations", [])
            same_byte = (history and item.get("tracker_id") is not None
                         and history[-1].get("tracker_id") == item["tracker_id"])
            if same_byte or (evidence and key == evidence["candidate_id"]):
                candidates.append(key)
        # Multiple plausible chains are still ambiguous evidence, even if the
        # current detection itself could be assigned safely.
        if len(candidates) != 1:
            return
        key = candidates[0]
        candidate = pending.pop(key)
        candidate.update(state="RESOLVED", end_frame=int(frame_index),
                         resolved_application_id=int(application_id))
        state.setdefault("unresolved_candidate_archive", []).append(candidate)
        state.setdefault("quality_events", []).append({
            "frame": int(frame_index), "event": "UNRESOLVED_CANDIDATE_RESOLVED",
            "candidate_id": int(key), "application_id": int(application_id),
            "observed_frames": len(candidate.get("observations", [])),
        })

    def _defer_unresolved_new_detection(self, item, detection_index, state, frame_index):
        """Delay ID creation when a recent lost/occluded identity remains plausible."""
        self._expire_unresolved_candidates(state, frame_index)
        pending = state.setdefault("unresolved_candidates", {})

        plausible_ids = []
        for app_id, identity in state.get("identities", {}).items():
            lifecycle_state = identity.get("state", identity.get("status"))
            if (lifecycle_state not in {"LOST", "OCCLUDED", "UNRESOLVED"}
                    and int(app_id) not in state.get("high_priority_app_candidates", set())):
                continue
            anchor = identity.get("trusted_center", identity.get("last_high_conf_center"))
            bbox = identity.get("trusted_bbox", identity.get("last_high_conf_bbox"))
            anchor_frame = identity.get("last_high_conf_frame")
            if anchor is None or bbox is None or anchor_frame is None:
                continue
            gap = int(frame_index) - int(anchor_frame)
            if gap < 1 or gap > self._identity_recovery_window_frames(state.get("source_fps")):
                continue
            velocity = self._recent_inference_velocity(identity)
            predicted = (anchor[0] + velocity[0] * gap, anchor[1] + velocity[1] * gap)
            predicted_distance = float(np.hypot(item["center"][0] - predicted[0],
                                                item["center"][1] - predicted[1]))
            last_distance = float(np.hypot(item["center"][0] - anchor[0],
                                           item["center"][1] - anchor[1]))
            scale = max(1.0, abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1]))
            ratio = self._bbox_long_side_ratio(item["bbox"], bbox)
            if ratio <= 3.0 and min(predicted_distance, last_distance) <= max(60.0, 3.0 * scale):
                plausible_ids.append({
                    "application_id": int(app_id), "predicted_distance": predicted_distance,
                    "last_center_distance": last_distance, "bbox_size_ratio": ratio,
                    "frame_gap": gap,
                })
        if not plausible_ids:
            return False

        # Keep a bounded temporal record for subsequent recovery attempts.
        candidate_key = None
        for key, candidate in pending.items():
            # A trajectory may contain at most one detection per frame. Mixing
            # simultaneous boxes invents motion and destroys directional evidence.
            if int(candidate["last_frame"]) >= int(frame_index):
                continue
            if not set(candidate.get("possible_identities", [])).intersection(
                    entry["application_id"] for entry in plausible_ids):
                continue
            old_center = candidate["last_center"]
            old_bbox = candidate["last_bbox"]
            scale = max(1.0, abs(old_bbox[2] - old_bbox[0]), abs(old_bbox[3] - old_bbox[1]))
            if (np.hypot(item["center"][0] - old_center[0], item["center"][1] - old_center[1])
                    <= max(20.0, 2.0 * scale)):
                candidate_key = key
                break
        if candidate_key is None:
            candidate_key = int(state.setdefault("next_unresolved_candidate_id", 1))
            state["next_unresolved_candidate_id"] = candidate_key + 1
            pending[candidate_key] = {"candidate_id": candidate_key, "first_frame": int(frame_index),
                                      "observations": [], "possible_identities": []}
        candidate = pending[candidate_key]
        history = candidate.setdefault("observations", [])
        direction = (0.0, 0.0)
        if history:
            previous = history[-1]
            elapsed = int(frame_index) - int(previous["frame"])
            if elapsed > 0:
                direction = tuple(
                    (float(item["center"][axis]) - float(previous["center"][axis])) / elapsed
                    for axis in range(2)
                )
        candidate["last_frame"] = int(frame_index)
        candidate["last_center"] = tuple(float(v) for v in item["center"])
        candidate["last_bbox"] = tuple(float(v) for v in item["bbox"])
        candidate["trajectory_direction"] = direction
        candidate["observations"].append({
            "frame": int(frame_index), "detection_index": int(detection_index),
            "center": tuple(float(v) for v in item["center"]),
            "bbox": tuple(float(v) for v in item["bbox"]),
            "confidence": float(item.get("confidence", 0.0)),
            "tracker_id": item.get("tracker_id"),
            "direction": direction,
        })
        byte_id = item.get("tracker_id")
        if byte_id is not None:
            history_ids = candidate.setdefault("byte_id_history", [])
            if not history_ids or history_ids[-1].get("tracker_id") != int(byte_id):
                history_ids.append({"frame": int(frame_index), "tracker_id": int(byte_id)})
            candidate["byte_track_ids"] = sorted({row["tracker_id"] for row in history_ids})
        possible = {entry["application_id"] for entry in plausible_ids}
        previous_possible = set(candidate.get("possible_identities", []))
        candidate["possible_identities"] = sorted(
            possible.intersection(previous_possible) if previous_possible else possible)
        item["identity_state"] = "UNRESOLVED"
        item["unresolved_candidate_id"] = int(candidate_key)
        state.setdefault("quality_events", []).append({
            "frame": int(frame_index), "event": "UNRESOLVED_CANDIDATE",
            "candidate_id": int(candidate_key), "detection_index": int(detection_index),
            "possible_identities": candidate["possible_identities"],
            "candidates": plausible_ids,
        })
        state["identity_assignment_diagnostics"].append({
            "frame": int(frame_index), "detection_index": int(detection_index),
            "detector_score": float(item.get("confidence", 0.0)),
            "byte_track_id": item.get("tracker_id"), "candidate_application_id": None,
            "current_center": item.get("center"), "current_bbox": item.get("bbox"),
            "assignment_type": "UNRESOLVED_CANDIDATE", "accepted": False,
            "ambiguous": True, "rejection_reason": "PLAUSIBLE_LOST_OR_OCCLUDED_IDENTITY",
            "possible_identities": candidate["possible_identities"],
        })
        return True

    def _possible_fragmentation_candidates(self, item, state, frame_index):
        """Return diagnostic-only prior identities that could explain a new ID."""
        fps = max(1.0, float(state.get("source_fps") or 30.0))
        max_gap = max(1, int(round(self.INFERENCE_APP_ID_FRAGMENT_GAP_SECONDS * fps)))
        candidates = []
        for app_id, identity in state.get("identities", {}).items():
            anchor = identity.get("trusted_center", identity.get("last_high_conf_center"))
            bbox = identity.get("trusted_bbox", identity.get("last_high_conf_bbox"))
            anchor_frame = identity.get("last_high_conf_frame")
            if anchor is None or bbox is None or anchor_frame is None:
                continue
            gap = int(frame_index) - int(anchor_frame)
            if gap < 1 or gap > max_gap:
                continue
            velocity = self._recent_inference_velocity(identity)
            predicted = (anchor[0] + velocity[0] * gap, anchor[1] + velocity[1] * gap)
            predicted_distance = float(np.hypot(item["center"][0] - predicted[0],
                                                item["center"][1] - predicted[1]))
            ratio = self._bbox_long_side_ratio(item["bbox"], bbox)
            scale = max(1.0, abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1]))
            if ratio <= 2.5 and predicted_distance <= max(80.0, 4.0 * scale):
                old_velocity = np.asarray(velocity, dtype=float)
                new_vector = np.asarray(item["center"], dtype=float) - np.asarray(anchor, dtype=float)
                norm = float(np.linalg.norm(old_velocity) * np.linalg.norm(new_vector))
                direction_cosine = float(np.dot(old_velocity, new_vector) / norm) if norm > 0 else None
                candidates.append({
                    "old_application_id": int(app_id), "frame_gap": gap,
                    "predicted_distance": predicted_distance,
                    "bbox_size_ratio": ratio, "direction_cosine": direction_cosine,
                    "old_state": identity.get("state", identity.get("status")),
                })
        return sorted(candidates, key=lambda row: (row["predicted_distance"], row["old_application_id"]))

    def _build_tracking_quality_summary(self, frame_count, fps, width, height, rows, identity_state):
        by_frame = defaultdict(set)
        track_lengths = defaultdict(int)
        byte_to_app_frame = defaultdict(dict)
        for row in rows:
            app_id = row.get("application_id")
            if app_id is None:
                continue
            app_id = int(app_id)
            by_frame[int(row["frame_index"])].add(app_id)
            track_lengths[app_id] += 1
            byte_id = row.get("tracker_id")
            if byte_id not in (None, ""):
                byte_to_app_frame[int(byte_id)][int(row["frame_index"])] = app_id
        same_byte_splits = 0
        for mapping in byte_to_app_frame.values():
            frames = sorted(mapping)
            same_byte_splits += sum(mapping[b] != mapping[a] for a, b in zip(frames, frames[1:]) if b - a <= 2)
        lifecycle = identity_state.get("lifecycle_events", [])
        quality_events = identity_state.get("quality_events", [])
        diagnostics = identity_state.get("identity_assignment_diagnostics", [])
        event_counts = defaultdict(int)
        for event in lifecycle:
            event_counts[event.get("event")] += 1
        quality_counts = defaultdict(int)
        for event in quality_events:
            quality_counts[event.get("event")] += 1
        ambiguous_keys = {
            (int(row.get("frame", -1)), int(row.get("detection_index", -1)))
            for row in diagnostics if row.get("ambiguous")
            and row.get("assignment_type") != "UNRESOLVED_CANDIDATE"
        }
        unresolved_candidate_keys = {
            int(event["candidate_id"]) for event in quality_events
            if event.get("event") == "UNRESOLVED_CANDIDATE" and event.get("candidate_id") is not None
        }
        ambiguous_group_keys = {
            (int(event.get("overlap_group_id", -1)), int(event.get("frame", -1)))
            for event in quality_events if event.get("event") == "OVERLAP_GROUP_AMBIGUOUS"
        }
        fragmentation_keys = {
            (int(event.get("new_application_id", -1)), int(event.get("frame", -1)))
            for event in quality_events if event.get("event") == "POSSIBLE_ID_FRAGMENTATION"
        }
        switch_keys = {
            (int(event.get("previous_application_id", -1)),
             int(event.get("new_application_id", -1)), int(event.get("frame", -1)))
            for event in quality_events if event.get("event") == "POSSIBLE_ID_SWITCH"
        }
        lengths = list(track_lengths.values())
        return {
            "frames": int(frame_count), "fps": float(fps),
            "resolution": [int(width), int(height)],
            "application_ids_created": int(event_counts["ID_CREATED"]),
            "maximum_simultaneous_ids": max((len(ids) for ids in by_frame.values()), default=0),
            "new_id_events": int(event_counts["ID_CREATED"]),
            "same_bytetrack_app_splits": int(same_byte_splits),
            "lost_events": int(event_counts["ID_LOST"]),
            "occlusion_events": int(event_counts["ID_OCCLUDED"]),
            "lost_recoveries": sum(1 for row in lifecycle if row.get("event") == "ID_REIDENTIFIED" and not row.get("overlap_partners")),
            "occlusion_recoveries": sum(1 for row in lifecycle if row.get("event") == "ID_REIDENTIFIED" and row.get("overlap_partners")),
            "ambiguous_events": len(ambiguous_keys),
            "unresolved_events": int(len(unresolved_candidate_keys) + len(ambiguous_group_keys)),
            "possible_fragmentations": len(fragmentation_keys),
            "possible_id_switches": len(switch_keys),
            "median_track_length": float(np.median(lengths)) if lengths else 0.0,
            "short_track_count": sum(length <= 5 for length in lengths),
            "unresolved_candidate_observations": sum(len(row.get("observations", [])) for row in identity_state.get("unresolved_candidates", {}).values()),
        }

    def _write_overlap_identity_review(self, video_path, meta_dir, fps, rows, lifecycle, frame_count):
        """Select visible contact/approach evidence, not occlusion labels alone."""
        review_dir = os.path.join(meta_dir, "overlap_identity_review")
        os.makedirs(review_dir, exist_ok=True)
        events_by_frame = defaultdict(list)
        for row in lifecycle:
            if row.get("event") in {"ID_OCCLUDED", "ID_REIDENTIFIED"}:
                events_by_frame[int(row["frame"])].append(row)
        visible = defaultdict(dict)
        for row in rows:
            visible[int(row["frame_index"])][int(row["application_id"])] = row
        contact_events = []
        previous_contact = {}
        offset = max(1, int(round(max(1.0, float(fps)) * 0.5)))
        for frame, detections in sorted(visible.items()):
            ids = sorted(detections)
            for pos, left in enumerate(ids):
                for right in ids[pos + 1:]:
                    boxes = [tuple(float(detections[i][k]) for k in ("x1", "y1", "x2", "y2"))
                             for i in (left, right)]
                    iou = self._bbox_iou(*boxes)
                    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
                    distance = float(np.hypot(centers[0][0] - centers[1][0], centers[0][1] - centers[1][1]))
                    scale = max(1.0, *(max(b[2] - b[0], b[3] - b[1]) for b in boxes))
                    if iou <= 0.05 and distance > 1.5 * scale:
                        continue
                    pair = (left, right)
                    if frame - previous_contact.get(pair, -100000) < offset * 2:
                        continue
                    before = [f for f in range(max(0, frame - offset * 2), frame)
                              if left in visible[f] and right in visible[f]]
                    after = [f for f in range(frame + 1, min(frame_count, frame + offset * 3))
                             if left in visible[f] or right in visible[f]]
                    if not before or not after:
                        continue
                    before_frame = min(before, key=lambda f: abs(f - (frame - offset)))
                    after_frame = min(after, key=lambda f: abs(f - (frame + offset)))
                    before_boxes = [tuple(float(visible[before_frame][i][k]) for k in ("x1", "y1", "x2", "y2"))
                                    for i in (left, right)]
                    before_centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in before_boxes]
                    before_distance = float(np.hypot(before_centers[0][0] - before_centers[1][0],
                                                       before_centers[0][1] - before_centers[1][1]))
                    disappearance = any(left not in visible[f] or right not in visible[f]
                                        for f in range(frame + 1, min(frame_count, frame + offset + 1)))
                    if iou <= 0.05 and not (before_distance > distance + .1 * scale and disappearance):
                        continue
                    contact_events.append({"frame": frame, "ids": pair, "iou": iou,
                                           "before": before_frame, "after": after_frame,
                                           "evidence": ("two_visible_identity_boxes_in_contact" if iou > .05
                                                        else "approach_followed_by_missing_detection")})
                    previous_contact[pair] = frame
        # A review pool can exceed 20: reviewers must reject samples where the
        # physical objects are not visible instead of counting lifecycle labels.
        if len(contact_events) > 40:
            indices = np.linspace(0, len(contact_events) - 1, 40, dtype=int)
            contact_events = [contact_events[index] for index in indices]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"events": 0, "directory": review_dir, "error": "video_open_failed"}
        offset = max(1, int(round(max(1.0, float(fps)) * 0.5)))
        frame_rows = defaultdict(list)
        by_identity = defaultdict(list)
        for row in rows:
            frame_rows[int(row["frame_index"])].append(row)
            by_identity[int(row["application_id"])].append(row)
        review_index = []
        try:
            for event_number, event in enumerate(contact_events, 1):
                event_frame = event["frame"]
                event_ids = set(event["ids"])
                sample_frames = [event["before"], event_frame, event["after"]]
                nearby_boxes = []
                for sample_frame in sample_frames:
                    for row in frame_rows.get(sample_frame, []):
                        if int(row.get("application_id", -1)) in event_ids:
                            nearby_boxes.append([float(row[k]) for k in ("x1", "y1", "x2", "y2")])
                if nearby_boxes:
                    max_side = max(max(box[2] - box[0], box[3] - box[1]) for box in nearby_boxes)
                    pad = max(48, int(round(max_side)))
                    roi = (max(0, int(min(box[0] for box in nearby_boxes)) - pad),
                           max(0, int(min(box[1] for box in nearby_boxes)) - pad),
                           int(max(box[2] for box in nearby_boxes)) + pad,
                           int(max(box[3] for box in nearby_boxes)) + pad)
                else:
                    roi = (0, 0, 0, 0)
                cards = []
                for label, frame_no in zip(("BEFORE", "DURING", "AFTER"), sample_frames):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    for row in frame_rows.get(frame_no, []):
                        x1, y1, x2, y2 = [int(round(float(row[k]))) for k in ("x1", "y1", "x2", "y2")]
                        app_id = int(row["application_id"])
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        history = [r for r in by_identity[app_id] if int(r["frame_index"]) < frame_no][-1:]
                        if history:
                            prev = history[0]
                            start = (int(round(float(prev["centroid_x"]))), int(round(float(prev["centroid_y"]))))
                            end = (int(round(float(row["centroid_x"]))), int(round(float(row["centroid_y"]))))
                            cv2.arrowedLine(frame, start, end, (255, 255, 0), 2, tipLength=0.35)
                        cv2.putText(frame, f"ID:{app_id} BT:{row.get('tracker_id')} {label}",
                                    (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    if roi[2] > roi[0] and roi[3] > roi[1]:
                        x1, y1 = roi[0], roi[1]
                        x2, y2 = min(frame.shape[1], roi[2]), min(frame.shape[0], roi[3])
                        frame = frame[y1:y2, x1:x2].copy()
                        for row in frame_rows.get(frame_no, []):
                            if int(row.get("application_id", -1)) in event_ids:
                                cv2.rectangle(frame,
                                    (int(float(row["x1"]) - x1), int(float(row["y1"]) - y1)),
                                    (int(float(row["x2"]) - x1), int(float(row["y2"]) - y1)),
                                    (0, 255, 0), 2)
                    cv2.putText(frame, f"frame={frame_no} {label} IDs={sorted(event_ids)}", (12, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    cards.append(frame)
                if cards:
                    target_h = min(card.shape[0] for card in cards)
                    resized = [cv2.resize(card, (int(card.shape[1] * target_h / card.shape[0]), target_h))
                               for card in cards]
                    contact = np.concatenate(resized, axis=1)
                    path = os.path.join(review_dir, f"overlap_{event_number:03d}_frame_{event_frame:06d}.jpg")
                    cv2.imwrite(path, contact)
                    review_index.append({"event_frame": event_frame, "samples": sample_frames,
                                         "application_ids": sorted(event_ids), "contact_iou": event["iou"],
                                         "selection_evidence": event["evidence"],
                                         "classification": "UNCERTAIN",
                                         "contact_sheet": os.path.basename(path)})
        finally:
            cap.release()
        with open(os.path.join(review_dir, "index.json"), "w") as index_file:
            json.dump({"events": len(review_index), "review_items": review_index,
                       "note": "Diagnostic contacts are not ground-truth identity validation."}, index_file, indent=2)
        return {"events": len(review_index), "directory": review_dir}

    def _write_application_identity_artifacts(self, video_path, meta_dir, fps, width, height,
                                              frame_count, rows, identity_state):
        """Persist registry snapshots, per-video quality metrics, and visual review."""
        registry_path = os.path.join(meta_dir, "application_identity_registry.json")
        registry = []
        for app_id, identity in sorted(identity_state.get("identities", {}).items()):
            record = dict(identity)
            record["application_id"] = int(app_id)
            for key in ("byte_track_ids", "overlap_partners", "occlusion_partners", "last_occlusion_partners"):
                if key in record:
                    record[key] = sorted(record[key])
            registry.append(record)
        with open(registry_path, "w") as registry_file:
            json.dump({"identity_capacity": identity_state.get("identity_capacity"),
                       "allocated_identity_count": len(registry), "identities": registry,
                       "overlap_groups": list(identity_state.get("overlap_groups", {}).values()),
                       "unresolved_candidate_archive": identity_state.get("unresolved_candidate_archive", []),
                       "unresolved_candidates": list(identity_state.get("unresolved_candidates", {}).values())},
                      registry_file, indent=2)
        quality_events_path = os.path.join(meta_dir, "application_identity_quality_events.json")
        with open(quality_events_path, "w") as event_file:
            json.dump(identity_state.get("quality_events", []), event_file, indent=2)
        quality = self._build_tracking_quality_summary(frame_count, fps, width, height, rows, identity_state)
        quality_path = os.path.join(meta_dir, "tracking_quality_summary.json")
        with open(quality_path, "w") as quality_file:
            json.dump(quality, quality_file, indent=2)
        review = self._write_overlap_identity_review(
            video_path, meta_dir, fps, rows, identity_state.get("lifecycle_events", []), frame_count
        )
        self.identity_registry_path = registry_path
        self.identity_quality_events_path = quality_events_path
        self.tracking_quality_summary_path = quality_path
        self.overlap_review_path = review["directory"]
        identity_state["tracking_quality_summary"] = quality
        identity_state["overlap_review"] = review
        return registry_path, quality_path, review["directory"]

    def _link_inference_application_ids(
        self,
        observations: List[Dict],
        state: Dict,
        frame_index: int,
    ) -> List[tuple]:
        """Assign IDs in continuation, reassociation, re-ID, low support, new-ID order.

        Stage 1 reserves an application ID when the same ByteTrack ID continues
        near its own last box. Stage 2 reconnects an unmatched detection only to
        a recently lost track that passes the existing distance, overlap, and
        box-size gates. Stage 3 recovers an eligible occluded ID using trusted
        trajectory state. Stage 4 supports low-confidence identity only. Stage 5
        creates IDs only for unmatched high-confidence detections.
        One detection receives one ID, and one ID is given to one detection.
        """
        state.setdefault("byte_owner", {})
        state.setdefault("low_score_diagnostics", [])
        state.setdefault("identity_assignment_diagnostics", [])
        state.setdefault("lifecycle_events", [])
        state.setdefault("overlap_groups", {})
        state.setdefault("unresolved_candidates", {})
        state.setdefault("quality_events", [])
        state["pending_new_id_reasons"] = {}
        state["pending_new_id_rejection_rank"] = {}
        state["blocked_high_detections"] = set()
        state["high_priority_app_candidates"] = set()
        self._expire_unresolved_candidates(state, frame_index)
        self._mark_merged_occlusion_observations(observations, state, frame_index)
        self._update_occlusion_states(observations, state, frame_index)
        self._update_overlap_group_states(state, frame_index)
        events: List[Optional[tuple]] = [None] * len(observations)
        reserved_det = set()
        reserved_app = set()
        for index, item in enumerate(observations):
            score = float(item.get("confidence", 0.0))
            item["identity_only"] = self.LOW_SCORE_MIN <= score <= self.MORPHOLOGY_THRESHOLD
            item["morphology_valid"] = score > self.MORPHOLOGY_THRESHOLD

        strong_pairs = []
        for index, item in enumerate(observations):
            if index in state["blocked_high_detections"] or float(item.get("confidence", 0.0)) <= self.MORPHOLOGY_THRESHOLD:
                continue
            byte_id = int(item["tracker_id"])
            application_id = state["byte_owner"].get(byte_id)
            if application_id is None:
                continue
            identity = state["identities"].get(application_id)
            if identity is None or identity.get("status") == "TERMINATED":
                continue
            metrics = self._strong_continuation_metrics(
                item, identity, frame_index, fps=state.get("source_fps"))
            if metrics is None:
                rejected_metrics = self._identity_geometry(item, identity, frame_index)
                if rejected_metrics is not None:
                    box = item.get("bbox")
                    characteristic = max(1.0, abs(box[2] - box[0]), abs(box[3] - box[1]))
                    # ByteTrack continuity plus near-identical geometry wins over
                    # noisy one-step direction estimates at an otherwise stable box.
                    if (
                        self._bbox_iou(item["bbox"], identity.get("last_bbox", item["bbox"]))
                        >= self.INFERENCE_APP_ID_SAME_BYTE_STRONG_IOU
                        and rejected_metrics["dist_last"] <= max(3.0, characteristic * 0.05)
                        and rejected_metrics["dist_pred"] <= max(6.0, characteristic * 0.20)
                    ):
                        metrics = rejected_metrics
                    elif (
                        rejected_metrics.get("motion_opposes")
                        and rejected_metrics.get("gap", self.INFERENCE_APP_ID_STRONG_MAX_GAP + 1)
                        <= self.INFERENCE_APP_ID_STRONG_MAX_GAP
                        and rejected_metrics.get("side_ratio", float("inf"))
                        <= self.INFERENCE_APP_ID_REASSOC_MAX_SIDE_RATIO
                        and rejected_metrics.get("iou", 0.0)
                        >= self.INFERENCE_APP_ID_STRONG_MIN_IOU
                        and rejected_metrics["dist_last"]
                        <= self.INFERENCE_APP_ID_STRONG_BASE_DIST
                        + self.INFERENCE_APP_ID_STRONG_DIST_PER_EXTRA_FRAME
                        * (rejected_metrics["gap"] - 1)
                        and rejected_metrics["dist_pred"]
                        <= self.INFERENCE_APP_ID_STRONG_BASE_DIST
                        + self.INFERENCE_APP_ID_STRONG_DIST_PER_EXTRA_FRAME
                        * (rejected_metrics["gap"] - 1)
                    ):
                        # A disagreeing one-step direction estimate is not
                        # enough to split a continuing ByteTrack detection when
                        # overlap, size, prediction, and displacement all pass
                        # the existing strong-continuation bounds.
                        metrics = rejected_metrics
                if metrics is not None:
                    pass
                elif rejected_metrics is not None:
                    reason = "MOTION_INCONSISTENT" if rejected_metrics.get("motion_opposes") else "continuation_gate_failed"
                    self._record_identity_assignment_diagnostic(
                        state, frame_index, index, item, application_id,
                        identity.get("current_byte_id"), rejected_metrics,
                        "HIGH_CONTINUATION", False, reason,
                    )
            if metrics is None:
                continue
            strong_pairs.append((metrics["dist_last"], index, application_id, metrics))
        strong_pairs.sort(key=lambda row: (row[0], row[1], row[2]))
        for _distance, index, application_id, metrics in strong_pairs:
            if index in reserved_det or application_id in reserved_app:
                self._record_identity_assignment_diagnostic(
                    state, frame_index, index, observations[index], application_id,
                    state["identities"][application_id].get("current_byte_id"), metrics,
                    "HIGH_CONTINUATION", False, "ALREADY_RESERVED",
                )
                continue
            self._record_identity_assignment_diagnostic(
                state, frame_index, index, observations[index], application_id,
                state["identities"][application_id].get("current_byte_id"),
                metrics, "HIGH_CONTINUATION", True, "accepted",
                best_candidate_score=metrics["dist_last"],
            )
            self._assign_existing_application_id(
                events, state, observations[index], index, application_id,
                frame_index, "KEEP", "strong_continuation", metrics, update_velocity=True,
            )
            reserved_det.add(index)
            reserved_app.add(application_id)
            self._reset_inference_identity_support(state, application_id)

        reassoc_pairs = []
        for index, item in enumerate(observations):
            if index in reserved_det or index in state["blocked_high_detections"] or float(item.get("confidence", 0.0)) <= self.MORPHOLOGY_THRESHOLD:
                continue
            for application_id, identity in state["identities"].items():
                if identity.get("status") in {"OCCLUDED", "TERMINATED"}:
                    continue
                metrics = self._reassociation_metrics(item, identity, frame_index)
                if metrics is None:
                    geometry = self._identity_geometry(item, identity, frame_index)
                    if geometry is not None:
                        reason = "MOTION_INCONSISTENT" if geometry.get("motion_opposes") else "reassociation_gate_failed"
                        self._record_identity_assignment_diagnostic(
                            state, frame_index, index, item, application_id,
                            identity.get("current_byte_id"), geometry,
                            "HIGH_REASSOCIATION", False, reason,
                        )
                    continue
                if application_id in reserved_app:
                    self._record_identity_assignment_diagnostic(
                        state, frame_index, index, item, application_id,
                        identity.get("current_byte_id"), metrics,
                        "HIGH_REASSOCIATION", False, "ALREADY_RESERVED",
                        best_candidate_score=metrics["dist_last"],
                    )
                    continue
                reassoc_pairs.append((metrics["dist_last"], index, application_id, metrics))
        proposal_rows = []
        by_detection = defaultdict(list)
        by_application = defaultdict(list)
        for _distance, index, application_id, metrics in reassoc_pairs:
            limit = self.INFERENCE_APP_ID_REASSOC_BASE_DIST + self.INFERENCE_APP_ID_REASSOC_DIST_PER_EXTRA_FRAME * (metrics["gap"] - 1)
            cost = (metrics["dist_pred"] / max(1.0, limit)
                    + metrics["dist_last"] / max(1.0, limit)
                    + (1.0 - metrics["iou"])
                    + abs(np.log(max(1e-6, metrics["side_ratio"]))))
            proposal = (float(cost), index, application_id, metrics)
            proposal_rows.append(proposal)
            by_detection[index].append(proposal)
            by_application[application_id].append(proposal)

        ambiguous_detections = set()
        ambiguous_applications = set()
        for ranked in by_detection.values():
            ranked.sort(key=lambda row: (row[0], row[2]))
            if len(ranked) > 1 and ranked[1][0] - ranked[0][0] <= self.INFERENCE_APP_ID_ASSOCIATION_AMBIGUITY_MARGIN:
                ambiguous_detections.add(ranked[0][1])
                ambiguous_applications.update(row[2] for row in ranked[:2])
        for ranked in by_application.values():
            ranked.sort(key=lambda row: (row[0], row[1]))
            if len(ranked) > 1 and ranked[1][0] - ranked[0][0] <= self.INFERENCE_APP_ID_ASSOCIATION_AMBIGUITY_MARGIN:
                ambiguous_applications.add(ranked[0][2])
                ambiguous_detections.update(row[1] for row in ranked[:2])

        for cost, index, application_id, metrics in proposal_rows:
            if index in ambiguous_detections or application_id in ambiguous_applications:
                state["blocked_high_detections"].add(index)
                state["high_priority_app_candidates"].add(application_id)
                alternatives = sorted(by_detection[index], key=lambda row: row[0])
                best = alternatives[0][0]
                second = alternatives[1][0] if len(alternatives) > 1 else None
                self._record_identity_assignment_diagnostic(
                    state, frame_index, index, observations[index], application_id,
                    state["identities"][application_id].get("current_byte_id"),
                    metrics, "AMBIGUOUS_REJECT", False, "AMBIGUOUS_HIGH_REASSOCIATION",
                    True, best, second,
                )

        usable = [row for row in proposal_rows
                  if row[1] not in ambiguous_detections
                  and row[2] not in ambiguous_applications
                  and row[1] not in reserved_det and row[2] not in reserved_app]
        detection_indices = sorted({row[1] for row in usable})
        application_ids = sorted({row[2] for row in usable})
        assignments = []
        if detection_indices and application_ids:
            cost_matrix = np.full((len(detection_indices), len(application_ids)), 1.0e6)
            det_col = {value: index for index, value in enumerate(detection_indices)}
            app_col = {value: index for index, value in enumerate(application_ids)}
            proposal_lookup = {}
            for cost, index, application_id, metrics in usable:
                cost_matrix[det_col[index], app_col[application_id]] = cost
                proposal_lookup[(index, application_id)] = (cost, metrics)
            try:
                from scipy.optimize import linear_sum_assignment
                rows, cols = linear_sum_assignment(cost_matrix)
                pairs = zip(rows.tolist(), cols.tolist())
            except Exception:
                pairs_list = sorted((cost_matrix[row, col], row, col)
                                    for row in range(cost_matrix.shape[0])
                                    for col in range(cost_matrix.shape[1]))
                used_rows, used_cols, selected = set(), set(), []
                for cost, row, col in pairs_list:
                    if cost >= 1.0e6 or row in used_rows or col in used_cols:
                        continue
                    used_rows.add(row); used_cols.add(col); selected.append((row, col))
                pairs = selected
            for row, col in pairs:
                index, application_id = detection_indices[row], application_ids[col]
                if (index, application_id) in proposal_lookup:
                    cost, metrics = proposal_lookup[(index, application_id)]
                    assignments.append((index, application_id, metrics, cost))

        for index, application_id, metrics, cost in assignments:
            self._record_identity_assignment_diagnostic(
                state, frame_index, index, observations[index], application_id,
                state["identities"][application_id].get("current_byte_id"),
                metrics, "HIGH_REASSOCIATION", True, "accepted",
                best_candidate_score=cost,
                second_best_candidate_score=(
                    sorted(row[0] for row in by_detection[index])[1]
                    if len(by_detection[index]) > 1 else None
                ),
            )
            self._assign_existing_application_id(
                events, state, observations[index], index, application_id,
                frame_index, "REASSOCIATED", "lost_track_reassociation", metrics,
                update_velocity=False,
            )
            reserved_det.add(index)
            reserved_app.add(application_id)
            self._reset_inference_identity_support(state, application_id)

        # Occluded identities get a motion- and size-aware recovery attempt
        # before low-score support and before any new application ID is created.
        occluded_matches = self._reidentify_occluded_tracks(
            observations, state, frame_index, reserved_det, reserved_app
        )
        for index, application_id, metrics, score, second_score in occluded_matches:
            identity = state["identities"][application_id]
            previous_byte_id = identity.get("current_byte_id")
            self._record_identity_assignment_diagnostic(
                state, frame_index, index, observations[index], application_id,
                previous_byte_id, metrics, "OCCLUDED_REIDENTIFICATION", True,
                "accepted", best_candidate_score=score,
                second_best_candidate_score=second_score,
            )
            self._record_lifecycle_event(
                state, frame_index, "ID_REIDENTIFIED", application_id,
                old_application_id=application_id,
                old_byte_track_id=previous_byte_id,
                new_byte_track_id=observations[index].get("tracker_id"),
                last_reliable_frame=identity.get("last_high_conf_frame"),
                reappearance_frame=frame_index,
                frame_gap=metrics["gap"],
                predicted_distance=metrics["dist_pred"],
                actual_distance=metrics["dist_last"],
                size_ratio=metrics["side_ratio"],
                motion_consistent=not metrics["motion_opposes"],
                overlap_partners=sorted(identity.get("last_occlusion_partners", set())),
                reason="motion_size_prediction_match",
            )
            self._record_lifecycle_event(
                state, frame_index, "ID_CONFIRMED", application_id,
                byte_track_id=observations[index].get("tracker_id"),
                score=float(observations[index].get("confidence", 0.0)),
                after_occlusion=True,
            )
            self._assign_existing_application_id(
                events, state, observations[index], index, application_id,
                frame_index, "REIDENTIFIED", "occluded_track_reidentification",
                metrics, update_velocity=True,
            )
            reserved_det.add(index)
            reserved_app.add(application_id)
            self._reset_inference_identity_support(state, application_id)

        # Low-score candidates are considered only after high-confidence
        # continuation, reassociation, and occlusion recovery have reserved IDs.
        low_matches = self._resolve_low_score_identity_support(
            observations, state, frame_index, reserved_det, reserved_app
        )
        for index, application_id, metrics, cost, second_cost in low_matches:
            identity = state["identities"][application_id]
            identity["identity_support_count"] = int(identity.get("identity_support_count", 0)) + 1
            self._record_identity_assignment_diagnostic(
                state, frame_index, index, observations[index], application_id,
                identity.get("current_byte_id"), metrics, "LOW_IDENTITY_SUPPORT",
                True, "accepted", best_candidate_score=cost,
                second_best_candidate_score=second_cost,
            )
            self._record_low_score_diagnostic(
                state, frame_index, application_id,
                float(observations[index].get("confidence", 0.0)), metrics,
                int(identity.get("identity_support_count", 0)),
                True, "accepted", True, False,
            )
            identity["low_score_supported"] = True
            weak_center = tuple(float(v) for v in observations[index]["center"])
            weak_bbox = tuple(float(v) for v in observations[index]["bbox"])
            identity["last_observation_center"] = weak_center
            identity["last_observation_bbox"] = weak_bbox
            identity["last_observation_frame"] = frame_index
            identity["last_seen_frame"] = int(frame_index)
            identity.setdefault("observation_history", []).append({
                "frame": int(frame_index), "center": weak_center, "bbox": weak_bbox,
                "confidence": float(observations[index].get("confidence", 0.0)),
                "byte_track_id": observations[index].get("tracker_id"),
                "identity_only": True,
            })
            if len(identity["observation_history"]) > 512:
                del identity["observation_history"][:-512]
            identity["status"] = "LOW_SUPPORT"
            identity["state"] = "LOW_SUPPORT"
            self._record_lifecycle_event(
                state, frame_index, "ID_LOW_SUPPORT", application_id,
                detection_index=index, byte_track_id=observations[index].get("tracker_id"),
                support_count=identity["identity_support_count"],
            )
            events[index] = (application_id, "IDENTITY_ONLY", "low_score_identity_support")
            reserved_det.add(index)
            reserved_app.add(application_id)

        for index in sorted(state["blocked_high_detections"]):
            if (index < len(observations)
                    and float(observations[index].get("confidence", 0.0)) > self.MORPHOLOGY_THRESHOLD):
                self._defer_unresolved_new_detection(observations[index], index, state, frame_index)

        unmatched = [
            index for index, item in enumerate(observations)
            if index not in reserved_det
            and index not in state["blocked_high_detections"]
            and float(item.get("confidence", 0.0)) > self.MORPHOLOGY_THRESHOLD
        ]
        unmatched.sort(key=lambda index: (
            observations[index]["center"][1],
            observations[index]["center"][0],
            int(observations[index]["tracker_id"]),
        ))
        for index in unmatched:
            item = observations[index]
            score = float(item.get("confidence", 0.0))
            if score <= self.MORPHOLOGY_THRESHOLD:
                continue
            if self._defer_unresolved_new_detection(item, index, state, frame_index):
                continue
            if hasattr(state, "allocate_id"):
                application_id = state.allocate_id()
            else:
                capacity = state.get("identity_capacity")
                application_id = None
                if capacity is None or int(state["next_id"]) <= int(capacity):
                    application_id = int(state["next_id"])
                    state["next_id"] = application_id + 1
            if application_id is None:
                capacity = state.get("identity_capacity")
                item["identity_state"] = "UNRESOLVED"
                state.setdefault("quality_events", []).append({
                    "frame": int(frame_index), "event": "NEW_ID_BLOCKED_CAPACITY",
                    "detection_index": int(index), "identity_capacity": int(capacity),
                })
                state["identity_assignment_diagnostics"].append({
                    "frame": int(frame_index), "detection_index": int(index),
                    "detector_score": score, "byte_track_id": item.get("tracker_id"),
                    "candidate_application_id": None, "current_center": item.get("center"),
                    "current_bbox": item.get("bbox"), "assignment_type": "NEW_ID_BLOCKED_CAPACITY",
                    "accepted": False, "ambiguous": False, "rejection_reason": "IDENTITY_CAPACITY_REACHED",
                })
                continue
            byte_id = int(item["tracker_id"])
            new_reason = state["pending_new_id_reasons"].get(index)
            if new_reason in {None, "NO_PREVIOUS_CANDIDATE"}:
                new_reason = self._new_id_reason(item, state)
            possible_fragments = self._possible_fragmentation_candidates(item, state, frame_index)
            if possible_fragments:
                state.setdefault("quality_events", []).append({
                    "frame": int(frame_index), "event": "POSSIBLE_ID_FRAGMENTATION",
                    "new_application_id": int(application_id),
                    "detection_index": int(index), "candidates": possible_fragments,
                })
            rejected_same_byte = [row for row in state["identity_assignment_diagnostics"]
                                  if row.get("frame") == int(frame_index)
                                  and row.get("detection_index") == int(index)
                                  and row.get("byte_track_id") == byte_id
                                  and not row.get("accepted")
                                  and row.get("assignment_type") == "HIGH_CONTINUATION"
                                  and row.get("candidate_application_id") is not None]
            strong_switch_evidence = []
            for rejected in rejected_same_byte:
                old_bbox = rejected.get("previous_bbox")
                bbox = item.get("bbox")
                if not old_bbox or not bbox:
                    continue
                scale = max(1.0, abs(float(bbox[2]) - float(bbox[0])),
                            abs(float(bbox[3]) - float(bbox[1])))
                last_distance = rejected.get("distance_to_last_observation")
                predicted_distance = rejected.get("predicted_center_distance")
                iou = rejected.get("iou_to_last_observation")
                if (iou is not None and last_distance is not None and predicted_distance is not None
                        and float(iou) >= self.POSSIBLE_SWITCH_MIN_IOU
                        and float(last_distance) <= max(3.0, scale * self.POSSIBLE_SWITCH_MAX_CENTER_FRACTION)
                        and float(predicted_distance) <= max(6.0, scale * self.POSSIBLE_SWITCH_MAX_PREDICTION_FRACTION)):
                    strong_switch_evidence.append(rejected)
            for rejected in sorted(strong_switch_evidence, key=lambda row: (
                float(row.get("distance_to_last_observation") or 0.0),
                float(row.get("predicted_center_distance") or 0.0),
                int(row.get("candidate_application_id")),
            ))[:1]:
                state.setdefault("quality_events", []).append({
                    "frame": int(frame_index), "event": "POSSIBLE_ID_SWITCH",
                    "previous_application_id": int(rejected["candidate_application_id"]),
                    "new_application_id": int(application_id), "detection_index": int(index),
                    "byte_track_id": byte_id, "reason": rejected.get("rejection_reason"),
                    "metrics": {key: rejected.get(key) for key in (
                        "distance_to_last_observation", "iou_to_last_observation",
                        "predicted_center_distance", "frame_gap", "bbox_size_ratio")},
                })
            identity = self._new_inference_identity(application_id, item, frame_index)
            identity.update({
                "created_reason": "no reliable existing identity",
                "returned_to_high_confidence": False,
                "last_lifecycle_state": "CONFIRMED",
                "last_new_id_reason": new_reason,
            })
            state["identities"][application_id] = identity
            self._record_lifecycle_event(
                state, frame_index, "ID_CREATED", application_id,
                new_id_reason=new_reason, byte_track_id=byte_id,
                bbox=item["bbox"], center=item["center"], score=score,
            )
            self._record_lifecycle_event(
                state, frame_index, "ID_ALLOCATED", application_id,
                byte_track_id=byte_id, identity_state="CONFIRMED",
            )
            self._record_lifecycle_event(
                state, frame_index, "ID_CONFIRMED", application_id,
                byte_track_id=byte_id, score=score,
            )
            self._record_identity_assignment_diagnostic(
                state, frame_index, index, item, application_id, None, None,
                "NEW_HIGH_ID", True, new_reason,
            )
            events[index] = (application_id, "NEW", "no reliable existing identity")
            reserved_app.add(application_id)
            self._touch_inference_identity(
                state, application_id, item, frame_index, byte_id, update_velocity=False
            )
            logger.debug(
                "frame=%s bytetrack_id=%s application_id=%s decision=NEW "
                "reason=no_reliable_existing_identity",
                frame_index,
                byte_id,
                application_id,
            )

        for application_id, identity in state["identities"].items():
            if application_id in reserved_app:
                if any(events[i] and events[i][1] == "IDENTITY_ONLY" for i in range(len(events)) if events[i] and events[i][0] == application_id):
                    identity["status"] = identity["state"] = "LOW_SUPPORT"
                else:
                    identity["status"] = identity["state"] = "CONFIRMED"
            elif identity.get("status") in {"CONFIRMED", "LOW_SUPPORT", "IDENTITY_SUPPORTED"}:
                identity["status"] = "LOST"
                identity["state"] = "LOST"
                identity.setdefault("lost_since_frame", int(frame_index))
                self._record_lifecycle_event(
                    state, frame_index, "ID_LOST", application_id,
                    last_reliable_frame=identity.get("last_high_conf_frame"),
                )
        self._update_overlap_group_states(state, frame_index)
        return events

    def _assign_existing_application_id(
        self, events, state, item, index, application_id, frame_index,
        decision, reason, metrics, update_velocity,
    ):
        self._resolve_unresolved_candidate(state, item, application_id, frame_index, metrics)
        events[index] = (application_id, decision, reason)
        self._touch_inference_identity(
            state,
            application_id,
            item,
            frame_index,
            int(item["tracker_id"]),
            update_velocity=update_velocity,
        )
        if decision == "KEEP":
            logger.debug(
                "frame=%s bytetrack_id=%s previous_application_id=%s decision=KEEP "
                "reason=strong_continuation distance=%.1f iou=%.2f",
                frame_index,
                int(item["tracker_id"]),
                application_id,
                metrics["dist_last"],
                metrics["iou"],
            )
        else:
            logger.debug(
                "frame=%s bytetrack_id=%s application_id=%s decision=REASSOCIATED "
                "distance=%.1f frame_gap=%s",
                frame_index,
                int(item["tracker_id"]),
                application_id,
                metrics["dist_last"],
                metrics["gap"],
            )

    def _resolve_low_score_identity_support(
        self, observations, state, frame_index, reserved_det, reserved_app
    ):
        proposals = []
        for index, item in enumerate(observations):
            if index in reserved_det or not (
                self.LOW_SCORE_MIN <= float(item.get("confidence", 0.0)) <= self.MORPHOLOGY_THRESHOLD
            ):
                continue
            candidates = []
            had_geometry = False
            for application_id, identity in state["identities"].items():
                if identity.get("status") == "TERMINATED":
                    continue
                if application_id in state.get("high_priority_app_candidates", set()):
                    metrics = self._low_identity_geometry(item, identity, frame_index)
                    if metrics is not None:
                        had_geometry = True
                        self._record_identity_assignment_diagnostic(
                            state, frame_index, index, item, application_id,
                            identity.get("current_byte_id"), metrics,
                            "LOW_REJECT", False, "HIGH_PRIORITY_CONFLICT",
                        )
                    continue
                metrics = self._low_identity_geometry(item, identity, frame_index)
                if metrics is None:
                    continue
                had_geometry = True
                count = int(identity.get("identity_support_count", 0))
                if metrics["gap"] > 1:
                    count = 0
                    identity["identity_support_count"] = 0
                reason = None
                distance_failed = (
                    metrics["dist_last"] > self.LOW_SCORE_MAX_DISTANCE
                    or metrics["dist_anchor"] > self.LOW_SCORE_MAX_DISTANCE
                )
                iou_failed = (
                    metrics["iou"] < self.LOW_SCORE_MIN_IOU
                    or metrics["iou_anchor"] < self.LOW_SCORE_MIN_IOU
                )
                if distance_failed and iou_failed:
                    reason = "BOTH_FAILED"
                elif distance_failed:
                    reason = "DISTANCE_FAILED"
                elif iou_failed:
                    reason = "IOU_FAILED"
                elif count >= self.MAX_IDENTITY_SUPPORT_FRAMES:
                    reason = "SUPPORT_CAP_REACHED"
                elif metrics["motion_opposes"]:
                    reason = "MOTION_INCONSISTENT"
                elif metrics["speed"] > self.LOW_SCORE_MOTION_MIN_SPEED and metrics["dist_pred"] > self.LOW_SCORE_MAX_DISTANCE:
                    reason = "HIGH_ANCHOR_INCONSISTENT"

                cost = self._low_identity_candidate_cost(metrics)
                if reason is not None:
                    self._record_identity_assignment_diagnostic(
                        state, frame_index, index, item, application_id,
                        identity.get("current_byte_id"), metrics,
                        "LOW_REJECT", False, reason,
                        best_candidate_score=cost,
                    )
                    self._record_low_score_diagnostic(
                        state, frame_index, application_id,
                        float(item.get("confidence", 0.0)), metrics, count,
                        False, reason.lower(), True, False,
                    )
                    continue
                candidates.append((cost, application_id, metrics))

            if not had_geometry:
                if reserved_app:
                    reserved_id = min(reserved_app)
                    self._record_identity_assignment_diagnostic(
                        state, frame_index, index, item, reserved_id,
                        state["identities"].get(reserved_id, {}).get("current_byte_id"),
                        None, "LOW_REJECT", False, "HIGH_PRIORITY_CONFLICT",
                    )
                    self._record_low_score_diagnostic(
                        state, frame_index, reserved_id,
                        float(item.get("confidence", 0.0)), None, 0,
                        False, "high_priority_conflict", True, False,
                    )
                    continue
                self._record_identity_assignment_diagnostic(
                    state, frame_index, index, item, None, None, None,
                    "LOW_REJECT", False, "NO_EXISTING_ID",
                )
                self._record_low_score_diagnostic(
                    state, frame_index, None, float(item.get("confidence", 0.0)),
                    None, 0, False, "no_existing_track", True, False,
                )
                continue
            if not candidates:
                continue
            candidates.sort(key=lambda row: (row[0], row[1]))
            best = candidates[0]
            second = candidates[1] if len(candidates) > 1 else None
            second_cost = None if second is None else float(second[0])
            if second is not None and second[0] - best[0] <= self.LOW_SCORE_AMBIGUITY_MAX_COST_MARGIN:
                for cost, application_id, metrics in candidates[:2]:
                    identity = state["identities"][application_id]
                    self._record_identity_assignment_diagnostic(
                        state, frame_index, index, item, application_id,
                        identity.get("current_byte_id"), metrics,
                        "AMBIGUOUS_REJECT", False, "AMBIGUOUS_MATCH",
                        ambiguous=True, best_candidate_score=float(best[0]),
                        second_best_candidate_score=second_cost,
                    )
                self._record_low_score_diagnostic(
                    state, frame_index, best[1], float(item.get("confidence", 0.0)),
                    best[2], int(state["identities"][best[1]].get("identity_support_count", 0)),
                    False, "ambiguous_match", True, False,
                )
                continue
            proposals.append((index, best[1], best[2], float(best[0]), second_cost))

        by_application_id = defaultdict(list)
        for proposal in proposals:
            by_application_id[proposal[1]].append(proposal)
        accepted = []
        for application_id, claims in by_application_id.items():
            if application_id in reserved_app or len(claims) > 1:
                for index, _app, metrics, cost, second_cost in claims:
                    item = observations[index]
                    self._record_identity_assignment_diagnostic(
                        state, frame_index, index, item, application_id,
                        state["identities"][application_id].get("current_byte_id"),
                        metrics, "LOW_REJECT", False,
                        "HIGH_PRIORITY_CONFLICT" if application_id in reserved_app else "ALREADY_RESERVED",
                        best_candidate_score=cost, second_best_candidate_score=second_cost,
                    )
                    self._record_low_score_diagnostic(
                        state, frame_index, application_id, float(item.get("confidence", 0.0)),
                        metrics, int(state["identities"][application_id].get("identity_support_count", 0)),
                        False, "assignment_conflict", True, False,
                    )
                continue
            accepted.extend(claims)
        return accepted

    def _low_identity_geometry(self, item, identity, frame_index):
        anchor_center = identity.get("last_high_conf_center", identity.get("last_center"))
        anchor_bbox = identity.get("last_high_conf_bbox", identity.get("last_bbox"))
        anchor_frame = identity.get("last_high_conf_frame", identity.get("last_frame"))
        previous_center = identity.get("last_observation_center", anchor_center)
        previous_bbox = identity.get("last_observation_bbox", anchor_bbox)
        previous_frame = identity.get("last_observation_frame", anchor_frame)
        if anchor_center is None or anchor_bbox is None or previous_center is None or previous_bbox is None:
            return None
        gap = int(frame_index) - int(previous_frame)
        anchor_gap = int(frame_index) - int(anchor_frame)
        if gap < 1 or anchor_gap < 1:
            return None
        center = item["center"]
        velocity = identity.get("velocity") or (0.0, 0.0)
        speed = float(np.hypot(velocity[0], velocity[1]))
        displacement = (center[0] - anchor_center[0], center[1] - anchor_center[1])
        motion_opposes = (
            speed > self.LOW_SCORE_MOTION_MIN_SPEED
            and displacement[0] * velocity[0] + displacement[1] * velocity[1] < 0.0
        )
        predicted = self._predicted_inference_center(identity, frame_index)
        return {
            "gap": gap,
            "anchor_gap": anchor_gap,
            "dist_last": float(np.hypot(center[0] - previous_center[0], center[1] - previous_center[1])),
            "iou": self._bbox_iou(item["bbox"], previous_bbox),
            "dist_anchor": float(np.hypot(center[0] - anchor_center[0], center[1] - anchor_center[1])),
            "iou_anchor": self._bbox_iou(item["bbox"], anchor_bbox),
            "dist_pred": float(np.hypot(center[0] - predicted[0], center[1] - predicted[1])),
            "speed": speed,
            "motion_opposes": bool(motion_opposes),
            "side_ratio": self._bbox_long_side_ratio(item["bbox"], previous_bbox),
            "previous_center": previous_center,
            "previous_bbox": previous_bbox,
            "anchor_center": anchor_center,
            "anchor_bbox": anchor_bbox,
            "predicted_center": predicted,
        }

    @staticmethod
    def _low_identity_candidate_cost(metrics):
        return float((
            metrics["dist_last"] / SpermAnalysisPipeline.LOW_SCORE_MAX_DISTANCE
            + metrics["dist_anchor"] / SpermAnalysisPipeline.LOW_SCORE_MAX_DISTANCE
            + metrics["dist_pred"] / SpermAnalysisPipeline.LOW_SCORE_MAX_DISTANCE
            + (1.0 - metrics["iou"])
            + (1.0 - metrics["iou_anchor"])
        ) / 5.0)

    @staticmethod
    def _record_identity_assignment_diagnostic(
        state, frame, detection_index, item, application_id, previous_byte_id,
        metrics, assignment_type, accepted, reason, ambiguous=False,
        best_candidate_score=None, second_best_candidate_score=None,
    ):
        row = {
            "frame": int(frame),
            "detection_index": int(detection_index),
            "detector_score": float(item.get("confidence", 0.0)),
            "byte_track_id": item.get("tracker_id"),
            "candidate_application_id": application_id,
            "previous_byte_track_id": previous_byte_id,
            "distance_to_last_observation": None if metrics is None else metrics.get("dist_observation", metrics.get("dist_last")),
            "iou_to_last_observation": None if metrics is None else metrics.get("iou_observation", metrics.get("iou")),
            "distance_to_high_anchor": None if metrics is None else metrics.get("dist_anchor", metrics.get("dist_last")),
            "iou_to_high_anchor": None if metrics is None else metrics.get("iou_anchor", metrics.get("iou")),
            "predicted_center_distance": None if metrics is None else metrics.get("dist_pred"),
            "frame_gap": None if metrics is None else metrics.get("gap"),
            "bbox_size_ratio": None if metrics is None else metrics.get("side_ratio"),
            "motion_opposes": None if metrics is None else metrics.get("motion_opposes"),
            "motion_consistent": None if metrics is None else not bool(metrics.get("motion_opposes")),
            "low_support_count": None if application_id is None else int(state["identities"].get(application_id, {}).get("identity_support_count", 0)),
            "previous_center": None if metrics is None else metrics.get("previous_center", metrics.get("anchor_center")),
            "current_center": item.get("center"),
            "previous_bbox": None if metrics is None else metrics.get("previous_bbox", metrics.get("anchor_bbox")),
            "current_bbox": item.get("bbox"),
            "predicted_center": None if metrics is None else metrics.get("predicted_center"),
            "unresolved_candidate_evidence": None if metrics is None else metrics.get("unresolved_candidate_evidence"),
            "best_candidate_score": best_candidate_score,
            "second_best_candidate_score": second_best_candidate_score,
            "assignment_cost": best_candidate_score,
            "assignment_type": assignment_type,
            "accepted": bool(accepted),
            "ambiguous": bool(ambiguous),
            "rejection_reason": reason,
        }
        state["identity_assignment_diagnostics"].append(row)

    @staticmethod
    def _reset_inference_identity_support(state, application_id):
        identity = state["identities"][application_id]
        if identity.get("low_score_supported"):
            identity["returned_to_high_confidence"] = True
        identity["identity_support_count"] = 0

    @staticmethod
    def _record_low_score_diagnostic(state, frame, track_id, score, metrics, support_count,
                                     accepted, reason, identity_only, morphology_valid):
        row = {
            "frame_number": int(frame),
            "track_id": track_id,
            "detection_score": float(score),
            "center_distance": None if metrics is None else float(metrics["dist_last"]),
            "iou": None if metrics is None else float(metrics["iou"]),
            "identity_support_count": int(support_count),
            "accepted": bool(accepted),
            "rejection_reason": reason,
            "identity_only": bool(identity_only),
            "morphology_valid": bool(morphology_valid),
        }
        state["low_score_diagnostics"].append(row)
        logger.info("low_score_identity %s", row)

    def _strong_continuation_metrics(self, item: Dict, identity: Dict, frame_index: int, fps=None):
        if identity.get("current_byte_id") != int(item["tracker_id"]):
            return None
        metrics = self._identity_geometry(item, identity, frame_index)
        if metrics is None:
            return None
        gap = metrics["gap"]
        if gap < 1:
            return None
        distance = metrics["dist_last"]
        # The same ByteTrack ID can be absent from the drawn frames because the
        # mask filter drops it. A later box that is still near the last center
        # is the same track, including gaps longer than the 10-frame reassociation
        # window. The distance cap stops a large jump from keeping the ID.
        if gap > self.INFERENCE_APP_ID_STRONG_MAX_GAP:
            distance_limit = min(100.0, 20.0 + 7.0 * gap)
            if (
                gap <= self._identity_recovery_window_frames(fps)
                and distance <= distance_limit
                and (metrics["speed"] <= self.LOW_SCORE_MOTION_MIN_SPEED
                     or metrics["dist_pred"] <= distance_limit)
            ):
                return metrics
            return None
        close_limit = (
            self.INFERENCE_APP_ID_STRONG_CLOSE_DIST
            + self.INFERENCE_APP_ID_STRONG_CLOSE_PER_EXTRA_FRAME * (gap - 1)
        )
        overlap_limit = (
            self.INFERENCE_APP_ID_STRONG_BASE_DIST
            + self.INFERENCE_APP_ID_STRONG_DIST_PER_EXTRA_FRAME * (gap - 1)
        )
        if (
            metrics["speed"] > self.LOW_SCORE_MOTION_MIN_SPEED
            and metrics["dist_pred"] > close_limit
        ):
            return None
        if metrics.get("motion_opposes") and not (
            distance <= close_limit
            and metrics["iou"] >= self.INFERENCE_APP_ID_STRONG_MIN_IOU
            and metrics["dist_pred"] <= close_limit
        ):
            return None
        if distance <= close_limit:
            return metrics
        if distance <= overlap_limit and metrics["iou"] >= self.INFERENCE_APP_ID_STRONG_MIN_IOU:
            return metrics
        return None

    def _reassociation_metrics(self, item: Dict, identity: Dict, frame_index: int):
        metrics = self._identity_geometry(item, identity, frame_index)
        if metrics is None:
            return None
        gap = metrics["gap"]
        if gap < 1 or gap > self.INFERENCE_APP_ID_MAX_LOST_FRAMES:
            return None
        if metrics.get("motion_opposes"):
            return None
        limit = (
            self.INFERENCE_APP_ID_REASSOC_BASE_DIST
            + self.INFERENCE_APP_ID_REASSOC_DIST_PER_EXTRA_FRAME * (gap - 1)
        )
        if metrics["dist_last"] > limit:
            return None
        if metrics["side_ratio"] > self.INFERENCE_APP_ID_REASSOC_MAX_SIDE_RATIO:
            return None
        if gap == 1 and metrics["iou"] < 0.10 and metrics["dist_last"] > 12.0:
            return None
        # A lost track with a validated velocity must also agree with that motion.
        # Zero velocity means there is no same-sperm motion estimate yet.
        velocity = identity.get("velocity") or (0.0, 0.0)
        speed = float(np.hypot(velocity[0], velocity[1]))
        if speed > 1.0 and metrics["dist_pred"] > limit:
            return None
        return metrics

    def _identity_geometry(self, item: Dict, identity: Dict, frame_index: int):
        if identity.get("last_center") is None or identity.get("last_bbox") is None:
            return None
        gap = frame_index - int(identity["last_frame"])
        if gap < 1:
            return None
        predicted = self._predicted_inference_center(identity, frame_index)
        center = item["center"]
        velocity = identity.get("velocity") or (0.0, 0.0)
        speed = float(np.hypot(velocity[0], velocity[1]))
        motion_opposes = (
            speed > self.LOW_SCORE_MOTION_MIN_SPEED
            and (center[0] - identity["last_center"][0]) * velocity[0]
            + (center[1] - identity["last_center"][1]) * velocity[1] < 0.0
        )
        observation_center = identity.get("last_observation_center", identity["last_center"])
        observation_bbox = identity.get("last_observation_bbox", identity["last_bbox"])
        return {
            "gap": gap,
            "dist_last": float(np.hypot(
                center[0] - identity["last_center"][0],
                center[1] - identity["last_center"][1],
            )),
            "dist_observation": float(np.hypot(
                center[0] - observation_center[0], center[1] - observation_center[1],
            )),
            "iou_observation": self._bbox_iou(item["bbox"], observation_bbox),
            "dist_pred": float(np.hypot(center[0] - predicted[0], center[1] - predicted[1])),
            "iou": self._bbox_iou(item["bbox"], identity["last_bbox"]),
            "side_ratio": self._bbox_long_side_ratio(item["bbox"], identity["last_bbox"]),
            "dist_anchor": float(np.hypot(
                center[0] - identity["last_center"][0], center[1] - identity["last_center"][1],
            )),
            "iou_anchor": self._bbox_iou(item["bbox"], identity["last_bbox"]),
            "speed": speed,
            "motion_opposes": bool(motion_opposes),
            "previous_center": observation_center,
            "previous_bbox": observation_bbox,
            "anchor_center": identity["last_center"],
            "anchor_bbox": identity["last_bbox"],
            "predicted_center": predicted,
        }

    @staticmethod
    def _bbox_iou(a, b) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - intersection
        if union <= 0:
            return 0.0
        return float(intersection / union)

    @staticmethod
    def _bbox_long_side_ratio(a, b) -> float:
        a_long = max(1.0, abs(a[2] - a[0]), abs(a[3] - a[1]))
        b_long = max(1.0, abs(b[2] - b[0]), abs(b[3] - b[1]))
        return float(max(a_long / b_long, b_long / a_long))

    @staticmethod
    def _predicted_inference_center(identity: Dict, frame_index: int) -> tuple:
        last = identity["last_center"]
        velocity = identity.get("velocity") or (0.0, 0.0)
        gap = frame_index - int(identity["last_frame"])
        return (last[0] + velocity[0] * gap, last[1] + velocity[1] * gap)

    @staticmethod
    def _recent_inference_velocity(identity: Dict) -> tuple:
        """Use recent trusted high-confidence history to smooth re-ID direction."""
        history = identity.get("high_conf_history", [])
        unique = {}
        for frame, center, _bbox in history:
            unique[int(frame)] = center
        samples = sorted(unique.items())[-4:]
        if len(samples) >= 2:
            first_frame, first_center = samples[0]
            last_frame, last_center = samples[-1]
            gap = last_frame - first_frame
            if gap > 0:
                return (
                    (last_center[0] - first_center[0]) / gap,
                    (last_center[1] - first_center[1]) / gap,
                )
        return identity.get("velocity") or (0.0, 0.0)

    @staticmethod
    def _touch_inference_identity(
        state, application_id, item, frame_index, byte_id, update_velocity=False
    ):
        identity = state["identities"][application_id]
        center = item["center"]
        if (
            update_velocity
            and identity.get("last_center") is not None
            and int(identity["last_frame"]) != frame_index
        ):
            gap = frame_index - int(identity["last_frame"])
            if gap > 0:
                identity["previous_center"] = identity["last_center"]
                identity["velocity"] = (
                    (center[0] - identity["last_center"][0]) / gap,
                    (center[1] - identity["last_center"][1]) / gap,
                )
        elif not update_velocity:
            # A new ByteTrack ID must not turn the jump from another sperm
            # into this identity's velocity. The next validated continuation
            # of this same detection supplies the motion.
            identity["velocity"] = (0.0, 0.0)
            identity["previous_center"] = None
        previous_owner = state.setdefault("byte_owner", {}).get(int(byte_id))
        if previous_owner is not None and previous_owner != application_id:
            previous = state["identities"].get(previous_owner)
            if previous is not None and previous.get("current_byte_id") == int(byte_id):
                previous["current_byte_id"] = None
        center = tuple(float(v) for v in center)
        bbox = tuple(float(v) for v in item["bbox"])
        identity["last_center"] = center
        identity["last_bbox"] = bbox
        identity["trusted_center"] = center
        identity["trusted_bbox"] = bbox
        identity["last_frame"] = int(frame_index)
        identity["last_seen_frame"] = int(frame_index)
        identity["last_high_conf_frame"] = int(frame_index)
        identity["last_observation_center"] = center
        identity["last_observation_bbox"] = bbox
        identity["last_observation_frame"] = int(frame_index)
        identity["last_high_conf_center"] = center
        identity["last_high_conf_bbox"] = bbox
        history = identity.setdefault("high_conf_history", [])
        if not history or int(history[-1][0]) != int(frame_index):
            history.append((int(frame_index), center, bbox))
        if len(history) > 8:
            del history[:-8]
        trajectory = identity.setdefault("trajectory_history", [])
        if not trajectory or int(trajectory[-1][0]) != int(frame_index):
            trajectory.append((int(frame_index), center, bbox))
        if len(trajectory) > 256:
            del trajectory[:-256]
        identity.setdefault("observation_history", []).append({
            "frame": int(frame_index), "center": center, "bbox": bbox,
            "confidence": float(item.get("confidence", 0.0)),
            "byte_track_id": int(byte_id) if byte_id is not None else None,
        })
        if len(identity["observation_history"]) > 512:
            del identity["observation_history"][:-512]
        identity["current_byte_id"] = int(byte_id)
        identity.setdefault("byte_track_ids", set()).add(int(byte_id))
        identity.setdefault("byte_id_history", [])
        if (not identity["byte_id_history"] or
                identity["byte_id_history"][-1].get("tracker_id") != int(byte_id)):
            identity["byte_id_history"].append({"frame": int(frame_index), "tracker_id": int(byte_id)})
        identity["status"] = identity["state"] = "CONFIRMED"
        identity["lost_since_frame"] = None
        identity["occlusion_start_frame"] = None
        identity["occlusion_partners"] = set()
        state["byte_owner"][int(byte_id)] = application_id

    def _generate_inference_video(
        self,
        video_path: str,
        output_dir: str,
        fps: float,
        width: int,
        height: int,
    ):
        """Generate output_inference_video with Mask R-CNN, ByteTrack, and mask smoothing."""
        if self.tracking_method == "csrt":
            self._generate_csrt_inference_video(video_path, output_dir, fps, width, height)
            return

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            logger.error(f"Could not open video for inference output: {video_path}")
            return

        if not fps or fps <= 0:
            fps = 30.0

        temp_path = os.path.join(output_dir, 'output_inference_video.avi')
        writer = cv2.VideoWriter(
            temp_path,
            cv2.VideoWriter_fourcc(*'MJPG'),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not initialize inference video writer: {temp_path}")

        tracker = self._create_bytetrack(fps)
        # Video-local labels. ByteTrack IDs stay in tracker_id and mask smoothing.
        identity_state = self._initialize_application_identity_state(
            width, height, fps, identity_capacity=None
        )
        mask_states: Dict[int, Dict] = {}
        grid = GridDiagnostics(os.path.basename(os.path.normpath(output_dir)), width, height, fps,
                               getattr(self, "grid_config", GridConfig()))
        metadata_rows: List[Dict] = []
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.MORPHOLOGY_KERNEL_SIZE, self.MORPHOLOGY_KERNEL_SIZE),
        )
        frame_index = 0
        frames_written = 0

        try:
            while True:
                success, frame = capture.read()
                if not success or frame is None:
                    break

                output = frame.copy()
                detections = self._run_maskrcnn_inference(frame)
                boxes = detections['boxes']
                scores = detections['scores']
                labels = detections['labels']
                masks = detections['masks']
                identity_boxes = detections['identity_boxes']
                identity_scores = detections['identity_scores']
                identity_labels = detections['identity_labels']
                identity_masks = detections['identity_masks']

                byte_detections = self._ByteTrackDetections(boxes, scores, labels)
                tracks = tracker.update(byte_detections, img=frame)
                track_matches = self._match_tracks_to_detection_indices(tracks, boxes)
                current_weight = (
                    self.HIGH_COUNT_MASK_CURRENT_WEIGHT
                    if len(boxes) > 50
                    else self.LOW_COUNT_MASK_CURRENT_WEIGHT
                )

                pending_labels = []
                grid_matched_indices = set()
                for track_idx, det_idx in track_matches.items():
                    track = tracks[track_idx]
                    x1, y1, x2, y2 = track[:4].astype(np.float32)
                    tracker_id = int(track[4])
                    confidence = float(scores[det_idx])
                    class_id = int(labels[det_idx])
                    probability = masks[det_idx]
                    if probability.shape[:2] != (height, width):
                        probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)

                    centroid = self._mask_centroid(probability, np.asarray([x1, y1, x2, y2], dtype=np.float32))
                    smoothed_probability = self._smooth_track_mask(
                        tracker_id,
                        probability,
                        centroid,
                        frame_index,
                        mask_states,
                        current_weight,
                    )
                    smoothed_binary = (smoothed_probability >= self.MASK_THRESHOLD)
                    smoothed_binary = cv2.morphologyEx(
                        smoothed_binary.astype(np.uint8),
                        cv2.MORPH_CLOSE,
                        kernel,
                        iterations=self.MORPHOLOGY_CLOSE_ITERATIONS,
                    )
                    mask_area = int(smoothed_binary.sum())
                    if mask_area < self.MIN_MASK_AREA:
                        continue

                    colored_mask = np.zeros_like(output, dtype=np.uint8)
                    colored_mask[smoothed_binary.astype(bool)] = self.SPERM_MASK_COLOR
                    output = cv2.addWeighted(output, 1.0, colored_mask, self.MASK_OPACITY, 0)

                    bx1 = max(0, min(width - 1, int(round(x1))))
                    by1 = max(0, min(height - 1, int(round(y1))))
                    bx2 = max(0, min(width - 1, int(round(x2))))
                    by2 = max(0, min(height - 1, int(round(y2))))
                    cv2.rectangle(output, (bx1, by1), (bx2, by2), self.SPERM_MASK_COLOR, 2)
                    if self.DRAW_CENTRE_DOT:
                        cv2.circle(output, (int(round(centroid[0])), int(round(centroid[1]))), 2, self.SPERM_MASK_COLOR, -1)

                    grid_matched_indices.add(det_idx)
                    pending_labels.append({
                        'tracker_id': tracker_id,
                        'bbox': (float(x1), float(y1), float(x2), float(y2)),
                        'center': ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0),
                        'text_origin': (bx1, max(0, by1 - 8)),
                        'class_id': class_id,
                        'confidence': confidence,
                        'centroid': centroid,
                        'mask_area': mask_area,
                        '_mask_binary': smoothed_binary,
                    })

                # Low-score boxes never enter ByteTrack, its mask smoother, or
                # the ordinary detection overlay. They reach only the identity
                # support linker and are drawn only if that linker accepts them.
                for identity_index, box in enumerate(identity_boxes):
                    x1, y1, x2, y2 = [float(value) for value in box]
                    probability = identity_masks[identity_index]
                    if probability.shape[:2] != (height, width):
                        probability = cv2.resize(
                            probability, (width, height), interpolation=cv2.INTER_LINEAR
                        )
                    identity_binary = probability >= self.MASK_THRESHOLD
                    identity_area = int(identity_binary.sum())
                    bx1 = max(0, min(width - 1, int(round(x1))))
                    by1 = max(0, min(height - 1, int(round(y1))))
                    bx2 = max(0, min(width - 1, int(round(x2))))
                    by2 = max(0, min(height - 1, int(round(y2))))
                    pending_labels.append({
                        'tracker_id': None,
                        'bbox': (x1, y1, x2, y2),
                        'center': ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                        'text_origin': (bx1, max(0, by1 - 8)),
                        'class_id': int(identity_labels[identity_index]),
                        'confidence': float(identity_scores[identity_index]),
                        'centroid': ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                        'mask_area': identity_area,
                        '_identity_mask_binary': identity_binary,
                    })

                links = self._link_inference_application_ids(
                    pending_labels, identity_state, frame_index
                )
                # Include detector observations that never reached the linker, without IDs.
                grid_unlinked = [dict(bbox=tuple(map(float, box)), tracker_id=None,
                                      confidence=float(scores[index]))
                                 for index, box in enumerate(boxes)
                                 if index not in grid_matched_indices]
                grid.capture(frame_index, pending_labels + grid_unlinked,
                             links + [None] * len(grid_unlinked), identity_state,
                             predict=self._predicted_inference_center)
                for item, link in zip(pending_labels, links):
                    if link is None:
                        continue
                    application_id, _event, _reason = link
                    if item.get("identity_only"):
                        low_mask = np.zeros_like(output, dtype=np.uint8)
                        low_mask[item["_identity_mask_binary"]] = self.SPERM_MASK_COLOR
                        output = cv2.addWeighted(output, 1.0, low_mask, self.MASK_OPACITY, 0)
                        bbox = item["bbox"]
                        cv2.rectangle(
                            output,
                            (int(round(bbox[0])), int(round(bbox[1]))),
                            (int(round(bbox[2])), int(round(bbox[3]))),
                            self.SPERM_MASK_COLOR, 2,
                        )
                    if self.DRAW_TRACK_IDS:
                        cv2.putText(
                            output,
                            f"ID:{application_id}",
                            item['text_origin'],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            self.SPERM_MASK_COLOR,
                            2,
                            cv2.LINE_AA,
                        )
                    if (
                        item.get("morphology_valid")
                        and frame_index % self.morph_every_n == 0
                        and item.get("_mask_binary") is not None
                    ):
                        self._analyze_application_morphology(
                            frame, item, int(application_id)
                        )
                    metadata_rows.append({
                        'frame_index': int(frame_index),
                        'timestamp_seconds': float(frame_index / fps),
                        'tracker_id': item['tracker_id'],
                        'application_id': application_id,
                        'class_id': item['class_id'],
                        'confidence': item['confidence'],
                        'centroid_x': float(item['centroid'][0]),
                        'centroid_y': float(item['centroid'][1]),
                        'x1': item['bbox'][0],
                        'y1': item['bbox'][1],
                        'x2': item['bbox'][2],
                        'y2': item['bbox'][3],
                        'mask_area_pixels': item['mask_area'],
                        'identity_only': bool(item.get('identity_only', False)),
                        'morphology_valid': bool(item.get('morphology_valid', False)),
                    })

                stale_ids = [
                    tracker_id
                    for tracker_id, state in mask_states.items()
                    if frame_index - int(state['last_seen']) > self.MASK_STATE_MAX_AGE
                ]
                for tracker_id in stale_ids:
                    del mask_states[tracker_id]

                writer.write(output)
                frames_written += 1
                frame_index += 1

                if frames_written % 100 == 0:
                    logger.info(f"Inference Mask R-CNN video: {frames_written} frames written")
        finally:
            capture.release()
            writer.release()

        meta_dir = os.path.join(output_dir, 'meta')
        os.makedirs(meta_dir, exist_ok=True)
        metadata_path = os.path.join(meta_dir, 'inference_tracking.csv')
        fieldnames = [
            'frame_index', 'timestamp_seconds', 'tracker_id', 'application_id',
            'class_id', 'confidence', 'centroid_x', 'centroid_y', 'x1', 'y1',
            'x2', 'y2', 'mask_area_pixels', 'identity_only', 'morphology_valid'
        ]
        with open(metadata_path, 'w', newline='') as f:
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(metadata_rows)
        self.inference_tracking_metadata_path = metadata_path
        self.inference_tracking_rows = metadata_rows
        self.application_trajectories = self._trajectories_from_inference_rows(
            metadata_rows, require_morphology_valid=True
        )
        self._canonical_identity_available = True
        diagnostics_path = os.path.join(meta_dir, "low_score_identity_diagnostics.json")
        for diagnostic in identity_state["low_score_diagnostics"]:
            track_id = diagnostic.get("track_id")
            identity = identity_state["identities"].get(track_id)
            diagnostic["track_returned_to_high_confidence"] = bool(
                identity and identity.get("returned_to_high_confidence", False)
            )
        with open(diagnostics_path, "w") as diagnostics_file:
            json.dump(identity_state["low_score_diagnostics"], diagnostics_file, indent=2)
        with open(os.path.join(meta_dir, "identity_assignment_diagnostics.json"), "w") as diagnostics_file:
            json.dump(identity_state["identity_assignment_diagnostics"], diagnostics_file, indent=2)
        with open(os.path.join(meta_dir, "application_id_lifecycle.json"), "w") as lifecycle_file:
            json.dump(identity_state["lifecycle_events"], lifecycle_file, indent=2)
        self._write_application_identity_artifacts(
            video_path, meta_dir, fps, width, height, frames_written,
            metadata_rows, identity_state,
        )
        self._finish_grid_diagnostics(grid, video_path, output_dir)
        self.inference_video_path = self._convert_to_mp4(temp_path)
        logger.info(
            f"Inference Mask R-CNN video generation complete: {frames_written} frames written; "
            f"metadata rows={len(metadata_rows)}"
        )

    @staticmethod
    def _csrt_grid_snapshot(detections, tracker, identity_state, is_detector):
        """Copy CSRT provenance; image-tracker boxes are predictions, not detections."""
        items, identities = [], {}
        for detection in detections:
            app = detection.get("application_id")
            if app is None:
                continue
            tid = detection.get("track_id")
            current = tracker._tracks.get(tid, {})
            items.append(dict(detection, tracker_id=tid,
                              coordinate_kind="observed" if is_detector else "predicted",
                              confidence=detection.get("confidence") if is_detector else None))
            identities[app] = {"state": str(current.get("status", "UNKNOWN")).upper()}
        return items, dict(identity_state, identities=identities)

    def _finish_grid_diagnostics(self, grid, video_path, output_dir):
        """Export separately; no diagnostic values enter tracking, morphology or CASA."""
        self.grid_paths = grid.write(output_dir)
        if grid.config.show_tracking_grid:
            path = os.path.join(output_dir, "output_grid_tracking_video.avi")
            grid.render(video_path, path)
            self.grid_paths["grid_tracking_video_path"] = self._convert_to_mp4(path)

    @staticmethod
    def _trajectories_from_inference_rows(rows, require_morphology_valid=True):
        """Build uploaded trajectories from the identity labels drawn on video."""
        trajectories = defaultdict(list)
        for row in rows:
            application_id = row.get("application_id")
            if application_id is None:
                continue
            if require_morphology_valid and not row.get("morphology_valid", False):
                continue
            frame = int(row["frame_index"])
            center_x = (float(row["x1"]) + float(row["x2"])) / 2.0
            center_y = (float(row["y1"]) + float(row["y2"])) / 2.0
            trajectories[int(application_id)].append((frame, center_x, center_y))
        return {track_id: sorted(points) for track_id, points in trajectories.items()}

    @staticmethod
    def _csrt_application_id(tracker_id, identity_state):
        """Map CSRT implementation IDs into the uploaded identity namespace."""
        application_by_tracker = identity_state["application_by_tracker"]
        tracker_id = int(tracker_id)
        if tracker_id not in application_by_tracker:
            if hasattr(identity_state, "allocate_id"):
                app_id = identity_state.allocate_id()
                if app_id is None:
                    return None
                application_by_tracker[tracker_id] = int(app_id)
            else:
                application_by_tracker[tracker_id] = int(identity_state["next_id"])
                identity_state["next_id"] += 1
        return application_by_tracker[tracker_id]

    def _analyze_application_morphology(self, frame, observation, application_id):
        """Compute morphology against the same accepted mask carrying the video ID."""
        x1, y1, x2, y2 = [int(round(float(value))) for value in observation["bbox"]]
        x1, y1 = max(0, x1 - 3), max(0, y1 - 3)
        x2, y2 = min(frame.shape[1], x2 + 3), min(frame.shape[0], y2 + 3)
        if x2 <= x1 or y2 <= y1:
            return
        mask = np.asarray(observation["_mask_binary"], dtype=np.uint8)
        subparts = self.segmenter.segment_subparts(
            frame[y1:y2, x1:x2], mask[y1:y2, x1:x2]
        )
        result = self.morphology_analyzer.analyze_morphology(
            subparts["head"], subparts["neck"], subparts["tail"]
        )
        self.morphology_results.setdefault(int(application_id), []).append(result)

    def _uploaded_trajectories(self):
        """Return the canonical video-local identity trajectories for uploaded jobs."""
        if getattr(self, "_canonical_identity_available", False):
            return self.application_trajectories
        return self.tracker.get_all_trajectories()

    def _identify_good_quality_sperm(self) -> set:
        """
        Identify sperm with good quality based on morphology and motility
        
        Returns:
            Set of track IDs for good quality sperm
        """
        good_sperm = set()
        
        # Analyze motility for all trajectories
        all_trajectories = self._uploaded_trajectories()
        self.motility_results = self.motility_analyzer.analyze_batch(all_trajectories, 30.0)  # Assume 30 FPS
        
        for track_id in all_trajectories.keys():
            # Check morphology
            morph_ok = False
            if track_id in self.morphology_results and self.morphology_results[track_id]:
                # Use median morphology result
                morph_scores = [r['score'] for r in self.morphology_results[track_id]]
                morph_ok = np.median(morph_scores) > 0.5
            
            # Check motility
            mot_ok = False
            if track_id in self.motility_results:
                mot_label = self.motility_results[track_id]['label']
                mot_ok = mot_label == 'progressive'
            
            if morph_ok and mot_ok:
                good_sperm.add(track_id)
        
        return good_sperm
    
    def _compute_final_analysis(self, fps: float) -> List[Dict]:
        """
        Compute final analysis results
        
        Args:
            fps: Frames per second
            
        Returns:
            List of analysis results for each sperm
        """
        results = []
        all_trajectories = self._uploaded_trajectories()
        if fps is None or not np.isfinite(float(fps)) or float(fps) <= 0:
            raise ValueError(
                f"Cannot compute CASA velocities: video FPS is invalid ({fps!r}). "
                "No 30 FPS fallback is applied."
            )
        if self.micrometers_per_pixel is not None and (
            not np.isfinite(float(self.micrometers_per_pixel))
            or float(self.micrometers_per_pixel) <= 0
        ):
            raise ValueError(
                "micrometers_per_pixel must be a positive measured scale when it is set. "
                "Leave it blank to report velocities in px/s. No default scale is applied."
            )
        self.motility_results = self.motility_analyzer.analyze_batch(
            all_trajectories,
            fps
        )

        for track_id in all_trajectories.keys():
            # Get morphology summary
            morph_summary = self._get_morphology_summary(track_id)
            
            # Get motility summary
            mot_summary = self.motility_results.get(track_id, {})
            
            # Determine overall status
            morph_ok = morph_summary.get('status') == 'normal'
            mot_ok = mot_summary.get('label') == 'progressive'
            
            status = 'good' if (morph_ok and mot_ok) else 'defective'
            
            feats = morph_summary.get('features', {})
            result = {
                'sperm_id': track_id,
                'status': status,
                'morphology_score': morph_summary.get('score', 0.0),
                'motility_score': 1.0 if mot_ok else (0.5 if mot_summary.get('label') == 'non_progressive' else 0.0),
                'morphology_features': feats,
                'motility_features': mot_summary,
                'velocity_unit': mot_summary.get('velocity_unit'),
                'calibration_warning': mot_summary.get('calibration_warning'),
                # Flattened fields expected by templates/index.html Detailed Report table
                'head_length': float(feats.get('head_length', 0.0)),
                'head_width': float(feats.get('head_width', 0.0)),
                'head_circularity': float(feats.get('head_circularity', 0.0)),
                'tail_length': float(feats.get('tail_length', 0.0)),
                'neck_angle_deg': float(feats.get('neck_angle_deg', 0.0)),
                'explainability': {
                    'morphology': morph_summary.get('reasons', []),
                    'motility': self._motility_notes(mot_summary)
                }
            }
            
            results.append(result)
        
        return results

    @staticmethod
    def _motility_notes(mot_summary: Dict) -> List[str]:
        notes = [mot_summary.get('label', 'unknown')]
        unit = mot_summary.get('velocity_unit')
        if unit:
            notes.append(str(unit))
        warning = mot_summary.get('calibration_warning')
        if warning:
            notes.append(str(warning))
        return notes
    
    def _get_morphology_summary(self, track_id: int) -> Dict:
        """
        Get morphology summary for a track ID
        
        Args:
            track_id: Track ID
            
        Returns:
            Morphology summary dictionary
        """
        if track_id not in self.morphology_results or not self.morphology_results[track_id]:
            return {
                'status': 'defective',
                'score': 0.0,
                'reasons': ['no morphology data'],
                'features': {}
            }
        
        # Use median values across all frames
        morph_data = self.morphology_results[track_id]
        
        # Aggregate features
        features = {}
        for key in ['head_length', 'head_width', 'head_circularity', 'tail_length', 'neck_angle_deg']:
            values = [m['features'].get(key, 0) for m in morph_data if key in m['features']]
            features[key] = float(np.median(values)) if values else 0.0
        
        # Aggregate status and score
        scores = [m['score'] for m in morph_data]
        reasons = []
        for m in morph_data:
            reasons.extend(m['reasons'])
        
        # Remove duplicates
        reasons = list(set(reasons))
        
        status = 'normal' if np.median(scores) > 0.5 else 'defective'
        
        return {
            'status': status,
            'score': float(np.median(scores)),
            'reasons': reasons,
            'features': features
        }
    
    def _write_csrt_diagnostics(self, tracker, meta_dir: str, filename: str, pass_name: str) -> None:
        """Write the CSRT identity summary. Distance tracking has no diagnostics object."""
        if not hasattr(tracker, "diagnostics_summary"):
            return
        logger.info("CSRT %s tracking diagnostics:\n%s", pass_name, tracker.diagnostics_summary())
        os.makedirs(meta_dir, exist_ok=True)
        path = os.path.join(meta_dir, filename)
        with open(path, "w") as handle:
            json.dump(tracker.diagnostics(), handle, indent=2)

    def _save_results(self, output_dir: str, analysis_results: List[Dict]) -> Dict:
        """
        Save analysis results to files
        
        Args:
            output_dir: Output directory
            analysis_results: Analysis results
            
        Returns:
            Dictionary of output file paths
        """
        # Create meta directory
        meta_dir = os.path.join(output_dir, 'meta')
        os.makedirs(meta_dir, exist_ok=True)
        
        # Save JSON results
        json_path = os.path.join(meta_dir, 'summary.json')
        with open(json_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Save trajectories
        trajectories_path = os.path.join(meta_dir, 'trajectories.json')
        trajectories_data = {
            'fps': self.source_fps,
            'trajectories': {str(k): [(int(fi), float(x), float(y)) for fi, x, y in v]
                           for k, v in self._uploaded_trajectories().items()}
        }
        with open(trajectories_path, 'w') as f:
            json.dump(trajectories_data, f, indent=2)

        if hasattr(self.tracker, "diagnostics_summary"):
            self._write_csrt_diagnostics(
                self.tracker, meta_dir, "csrt_tracking_diagnostics.json", "analysis",
            )
        analysis_id_diagnostics_path = None
        if hasattr(self.tracker, "association_diagnostics"):
            analysis_id_diagnostics_path = os.path.join(
                meta_dir, "analysis_id_diagnostics.json"
            )
            with open(analysis_id_diagnostics_path, "w") as diagnostics_file:
                json.dump({
                    "counts": dict(self.tracker.diagnostics_counts),
                    "decisions": self.tracker.association_diagnostics,
                }, diagnostics_file, indent=2)
        
        # Save CSV results
        try:
            import pandas as pd
            
            # Summary CSV
            csv_data = []
            for result in analysis_results:
                csv_data.append({
                    'sperm_id': result['sperm_id'],
                    'status': result['status'],
                    'morphology_score': result['morphology_score'],
                    'motility_score': result['motility_score'],
                    'morphology_issues': ';'.join(result['explainability']['morphology']),
                    'motility_issues': ';'.join(result['explainability']['motility'])
                })
            
            csv_path = os.path.join(meta_dir, 'summary.csv')
            pd.DataFrame(csv_data).to_csv(csv_path, index=False)
            
        except ImportError:
            logger.warning("pandas not available, skipping CSV export")
            csv_path = None
        
        # Use actual written video paths if available
        processed_path = getattr(self, 'processed_video_path', os.path.join(output_dir, 'output_processed_video.mp4'))
        inference_path = getattr(self, 'inference_video_path', os.path.join(output_dir, 'output_inference_video.mp4'))

        # Convert AVI to browser-compatible MP4 (H.264) using ffmpeg
        processed_path = self._convert_to_mp4(processed_path)
        inference_path = self._convert_to_mp4(inference_path)

        inference_tracking_path = getattr(
            self, 'inference_tracking_metadata_path',
            os.path.join(meta_dir, 'inference_tracking.csv')
        )

        return {
            **getattr(self, 'grid_paths', {}),
            'json_path': json_path,
            'csv_path': csv_path,
            'trajectories_path': trajectories_path,
            'inference_tracking_path': inference_tracking_path,
            'analysis_id_diagnostics_path': analysis_id_diagnostics_path,
            'identity_registry_path': getattr(self, 'identity_registry_path', None),
            'identity_quality_events_path': getattr(self, 'identity_quality_events_path', None),
            'tracking_quality_summary_path': getattr(self, 'tracking_quality_summary_path', None),
            'overlap_review_path': getattr(self, 'overlap_review_path', None),
            'processed_video': processed_path,
            'inference_video': inference_path
        }
    
    def _convert_to_mp4(self, video_path: str) -> str:
        """
        Convert a video file to browser-compatible MP4 (H.264) using ffmpeg.
        Returns the MP4 path if successful, or the original path if ffmpeg is unavailable.
        """
        import subprocess
        import shutil

        if not video_path or not os.path.exists(video_path):
            return video_path

        # Already MP4 — check if it's H.264, if not re-encode
        if video_path.lower().endswith('.mp4'):
            return video_path

        mp4_path = os.path.splitext(video_path)[0] + '.mp4'

        # Resolve ffmpeg even when running under systemd, where PATH is
        # typically just /usr/bin:/bin and the user's shell PATH isn't loaded.
        # Order: env override → PATH lookup → common absolute paths.
        ffmpeg_bin = (
            os.environ.get("FFMPEG_BIN")
            or shutil.which("ffmpeg")
            or next(
                (p for p in (
                    "/usr/bin/ffmpeg",
                    "/usr/local/bin/ffmpeg",
                    "/snap/bin/ffmpeg",
                    "/opt/homebrew/bin/ffmpeg",
                ) if os.path.isfile(p) and os.access(p, os.X_OK)),
                None,
            )
        )
        if not ffmpeg_bin:
            logger.warning("ffmpeg not found, skipping AVI to MP4 conversion. Install ffmpeg for browser video playback.")
            return video_path

        try:
            cmd = [
                ffmpeg_bin, '-y', '-i', video_path,
                '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '23', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                mp4_path
            ]
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            logger.info(f"Converted {video_path} -> {mp4_path} (ffmpeg={ffmpeg_bin})")
            # Remove original AVI to save space
            os.remove(video_path)
            return mp4_path
        except Exception as e:
            logger.error(f"ffmpeg conversion failed: {e}")
            return video_path

    def _init_processed_video_writer(self, output_dir: str, fps: float,
                                     width: int, height: int) -> Optional[cv2.VideoWriter]:
        """Initialize the processed-video writer only."""
        if not fps or fps <= 0:
            fps = 30.0
            logger.warning(f"Invalid FPS, using default: {fps}")

        codecs = [
            ('MJPG', 'output_processed_video.avi'),
            ('mp4v', 'output_processed_video.mp4'),
            ('XVID', 'output_processed_video.avi'),
            ('DIVX', 'output_processed_video.avi')
        ]

        for codec, filename in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            filepath = os.path.join(output_dir, filename)
            writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            if writer.isOpened():
                self.processed_video_path = filepath
                logger.info(f"Processed video writer initialized: {filepath} (codec: {codec})")
                return writer
            writer.release()

        logger.error("Failed to initialize processed video writer")
        return None
