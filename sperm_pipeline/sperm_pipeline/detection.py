"""
Sperm Detection Module
Handles YOLOv8-based sperm head detection with ByteTrack integration
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import logging

from .slicing import iter_slices, nms_xyxy

logger = logging.getLogger(__name__)

class SpermDetector:
    """
    YOLOv8-based sperm head detector with ByteTrack integration
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        sliced_inference: bool = True,
        slice_size: int = 320,
        slice_overlap: float = 0.25,
        slice_nms_iou: float = 0.45,
    ):
        """
        Initialize the sperm detector
        
        Args:
            model_path: Path to YOLOv8 model weights
            device: Device to run inference on ('cpu' or 'cuda')
            conf_threshold: Confidence threshold for detections
            sliced_inference: Run tiled inference and merge overlap detections
            slice_size: Tile width/height in pixels for sliced inference
            slice_overlap: Fractional tile overlap (0.0-0.9)
            slice_nms_iou: IoU threshold for duplicate removal across tiles
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.sliced_inference = sliced_inference
        self.slice_size = slice_size
        self.slice_overlap = slice_overlap
        self.slice_nms_iou = slice_nms_iou
        self.model = None
        self.tracker_name = "bytetrack.yaml"
        
        self._load_model()
    
    def _load_model(self):
        """Load the YOLOv8 model"""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Successfully loaded YOLOv8 model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
    
    def detect_and_track(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect sperm heads in a single frame.
        Tracking IDs are not provided here; they will be assigned by the external SpermTracker.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detection dictionaries without tracking IDs
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            if self.sliced_inference:
                return self._detect_sliced(frame)
            return self._detect_single(frame)
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def _detect_single(self, frame: np.ndarray) -> List[Dict]:
        """Run standard full-frame YOLO inference."""
        # Use per-frame detection. Using .track() per frame resets tracker state; instead
        # assign stable IDs via our SpermTracker at the pipeline level.
        results = self.model.predict(
            frame,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False
        )
        detections = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            for i, box in enumerate(boxes):
                detections.append(self._make_detection(box, confidences[i]))
        return detections

    def _detect_sliced(self, frame: np.ndarray) -> List[Dict]:
        """Run YOLO over overlapping tiles and merge detections back to frame space."""
        height, width = frame.shape[:2]
        boxes = []
        confidences = []

        for x1, y1, x2, y2 in iter_slices(
            height, width, self.slice_size, self.slice_overlap
        ):
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            results = self.model.predict(
                crop,
                device=self.device,
                conf=self.conf_threshold,
                verbose=False
            )
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                continue

            crop_boxes = results[0].boxes.xyxy.cpu().numpy()
            crop_scores = results[0].boxes.conf.cpu().numpy()
            crop_boxes[:, [0, 2]] += x1
            crop_boxes[:, [1, 3]] += y1
            crop_boxes[:, [0, 2]] = np.clip(crop_boxes[:, [0, 2]], 0, width)
            crop_boxes[:, [1, 3]] = np.clip(crop_boxes[:, [1, 3]], 0, height)

            boxes.extend(crop_boxes.tolist())
            confidences.extend(float(score) for score in crop_scores)

        if not boxes:
            return []

        keep = nms_xyxy(boxes, confidences, self.slice_nms_iou)
        return [self._make_detection(np.asarray(boxes[i]), confidences[i]) for i in keep]

    def _make_detection(self, box: np.ndarray, confidence: float) -> Dict:
        return {
            'bbox': box.tolist(),  # [x1, y1, x2, y2]
            'confidence': float(confidence),
            'track_id': None,
            'center': self._get_center(box)
        }
    
    def _get_center(self, bbox: np.ndarray) -> Tuple[float, float]:
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return (float((x1 + x2) / 2), float((y1 + y2) / 2))
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect sperm in a batch of frames
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection lists for each frame
        """
        all_detections = []
        for frame in frames:
            detections = self.detect_and_track(frame)
            all_detections.append(detections)
        return all_detections
