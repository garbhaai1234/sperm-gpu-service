"""
Sperm Segmentation Module
Handles full sperm segmentation using Mask R-CNN and subpart segmentation
"""

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
from torchvision.transforms import functional as F
from typing import List, Dict, Tuple, Optional
import logging

from .slicing import iter_slices, nms_xyxy

logger = logging.getLogger(__name__)

class SpermSegmentation:
    """
    Handles full sperm segmentation and subpart analysis
    """
    
    def __init__(self, maskrcnn_path: str, hnk_path: str, device: str = "cpu", 
                 num_classes: int = 2, threshold: float = 0.7,
                 sperm_class_ids: Optional[str] = None,
                 sliced_inference: bool = False, slice_size: int = 320,
                 slice_overlap: float = 0.25, slice_nms_iou: float = 0.35):
        """
        Initialize segmentation models
        
        Args:
            maskrcnn_path: Path to Mask R-CNN model weights
            hnk_path: Path to Head-Neck-Tail segmentation model
            device: Device to run inference on
            num_classes: Number of classes for Mask R-CNN
            threshold: Confidence threshold for segmentation
            sperm_class_ids: Comma-separated Mask R-CNN label ids to display
            sliced_inference: Run tiled Mask R-CNN inference and merge overlap segments
            slice_size: Tile width/height in pixels for sliced inference
            slice_overlap: Fractional tile overlap (0.0-0.9)
            slice_nms_iou: IoU threshold for duplicate removal across tiles
        """
        self.maskrcnn_path = maskrcnn_path
        self.hnk_path = hnk_path
        self.device = device
        self.num_classes = num_classes
        self.threshold = threshold
        self.configured_sperm_class_ids = self._parse_class_ids(sperm_class_ids)
        self.sliced_inference = sliced_inference
        self.slice_size = slice_size
        self.slice_overlap = slice_overlap
        self.slice_nms_iou = slice_nms_iou
        
        self.maskrcnn_model = None
        self.hnk_model = None
        self.maskrcnn_classes = []
        self.sperm_class_ids = self.configured_sperm_class_ids or {1}
        
        self._load_models()
    
    def _extract_maskrcnn_state(self):
        """Load a Mask R-CNN checkpoint and return its state dict plus metadata."""
        checkpoint = torch.load(self.maskrcnn_path, map_location=self.device)
        classes = []
        num_classes = None

        if isinstance(checkpoint, dict):
            classes = checkpoint.get("classes") or checkpoint.get("class_names") or []
            num_classes = checkpoint.get("num_classes")
            if "model_state_dict" in checkpoint:
                state = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state = checkpoint["model"]
            else:
                state = checkpoint
        else:
            state = checkpoint

        cleaned_state = {}
        for key, value in state.items():
            if key.startswith("module."):
                key = key.replace("module.", "", 1)
            cleaned_state[key] = value

        cls_weight = cleaned_state.get("roi_heads.box_predictor.cls_score.weight")
        if hasattr(cls_weight, "shape"):
            num_classes = int(cls_weight.shape[0])

        if isinstance(num_classes, int) and num_classes > 0:
            self.num_classes = num_classes

        self.maskrcnn_classes = list(classes)
        sperm_class_ids = (
            self.configured_sperm_class_ids
            or self._resolve_sperm_class_ids(self.maskrcnn_classes)
        )
        if isinstance(num_classes, int) and num_classes > 0:
            valid_ids = {label for label in sperm_class_ids if 0 <= label < num_classes}
            invalid_ids = sperm_class_ids - valid_ids
            if invalid_ids:
                logger.warning(
                    "Ignoring invalid Mask R-CNN sperm_class_ids=%s for num_classes=%s",
                    sorted(invalid_ids),
                    num_classes,
                )
            sperm_class_ids = valid_ids or self._resolve_sperm_class_ids(self.maskrcnn_classes)
        self.sperm_class_ids = sperm_class_ids or {1}
        return cleaned_state

    @staticmethod
    def _parse_class_ids(class_ids: Optional[str]) -> Optional[set]:
        if class_ids is None:
            return None
        ids = set()
        for value in str(class_ids).replace(";", ",").split(","):
            value = value.strip()
            if not value:
                continue
            ids.add(int(value))
        return ids or None

    @staticmethod
    def _resolve_sperm_class_ids(classes: List[str]) -> set:
        """Return the Mask R-CNN label ids that should be treated as sperm."""
        ids = {
            idx
            for idx, name in enumerate(classes)
            if str(name).strip().lower() == "sperm"
        }
        return ids or {1}

    def _load_models(self):
        """Load segmentation models"""
        try:
            state = self._extract_maskrcnn_state()

            # Load Mask R-CNN
            self.maskrcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
            
            # Update predictor heads
            in_features = self.maskrcnn_model.roi_heads.box_predictor.cls_score.in_features
            self.maskrcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, self.num_classes
            )
            
            mask_in_features = self.maskrcnn_model.roi_heads.mask_predictor.conv5_mask.in_channels
            self.maskrcnn_model.roi_heads.mask_predictor = MaskRCNNPredictor(
                mask_in_features, 256, self.num_classes
            )
            
            self.maskrcnn_model.load_state_dict(state, strict=True)
            self.maskrcnn_model.to(self.device)
            self.maskrcnn_model.eval()
            
            logger.info(
                f"Successfully loaded Mask R-CNN model from {self.maskrcnn_path} "
                f"(num_classes={self.num_classes}, sperm_class_ids={sorted(self.sperm_class_ids)})"
            )
            
        except Exception as e:
            logger.error(f"Failed to load Mask R-CNN model: {e}")
            self.maskrcnn_model = None
        
        try:
            # Load HNK model
            self.hnk_model = torch.jit.load(self.hnk_path, map_location=self.device)
            logger.info(f"Successfully loaded HNK model from {self.hnk_path}")
        except Exception:
            try:
                self.hnk_model = torch.load(self.hnk_path, map_location=self.device)
                # If a plain state dict or non-callable object is loaded, disable and fallback to heuristics
                if isinstance(self.hnk_model, dict) or not callable(getattr(self.hnk_model, "__call__", None)):
                    logger.warning(
                        "HNK model file appears to be a state dict or non-callable object; "
                        "falling back to heuristic subpart segmentation."
                    )
                    self.hnk_model = None
                else:
                    logger.info(f"Successfully loaded HNK model from {self.hnk_path}")
            except Exception as e:
                logger.error(f"Failed to load HNK model: {e}")
                self.hnk_model = None
    
    def segment_full_sperm(self, frame: np.ndarray) -> List[Dict]:
        """
        Segment full sperm instances using Mask R-CNN.

        Args:
            frame: Input frame as numpy array

        Returns:
            List of segmentation dictionaries
        """
        if self.maskrcnn_model is None:
            return []

        if self.sliced_inference:
            return self._segment_full_sperm_sliced(frame)
        return self._segment_full_sperm_single(frame)

    def _segment_full_sperm_single(self, frame: np.ndarray) -> List[Dict]:
        """Run standard full-frame Mask R-CNN inference."""
        try:
            # Convert to tensor (HWC BGR->RGB)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(self.device)

            with torch.inference_mode():
                predictions = self.maskrcnn_model(img_tensor)[0]

            segments = []
            if 'masks' in predictions and 'boxes' in predictions and 'scores' in predictions:
                pm = predictions['masks']
                pb = predictions['boxes']
                ps = predictions['scores']
                pl = predictions.get('labels')
                masks = pm.detach().cpu().numpy()
                boxes = pb.detach().cpu().numpy()
                scores = ps.detach().cpu().numpy()
                labels = (
                    pl.detach().cpu().numpy()
                    if pl is not None
                    else np.ones(len(scores), dtype=np.int64)
                )

                for i in range(len(masks)):
                    label = int(labels[i])
                    if label not in self.sperm_class_ids:
                        continue
                    if scores[i] > self.threshold:
                        segment = {
                            'mask': masks[i][0],
                            'bbox': boxes[i].tolist(),
                            'confidence': float(scores[i]),
                            'label': label,
                            'class_name': (
                                self.maskrcnn_classes[label]
                                if 0 <= label < len(self.maskrcnn_classes)
                                else 'sperm'
                            )
                        }
                        segments.append(segment)

            return segments

        except Exception as e:
            logger.error(f"Full sperm segmentation failed: {e}")
            return []

    def _segment_full_sperm_sliced(self, frame: np.ndarray) -> List[Dict]:
        """Run Mask R-CNN over overlapping tiles and return full-frame masks."""
        height, width = frame.shape[:2]
        segments = []

        for x1, y1, x2, y2 in iter_slices(
            height, width, self.slice_size, self.slice_overlap
        ):
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_segments = self._segment_full_sperm_single(crop)
            crop_h, crop_w = crop.shape[:2]
            for segment in crop_segments:
                box = np.asarray(segment['bbox'], dtype=np.float32)
                box[[0, 2]] += x1
                box[[1, 3]] += y1
                box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
                box[[1, 3]] = np.clip(box[[1, 3]], 0, height)

                crop_mask = np.asarray(segment['mask'], dtype=np.float32)
                if crop_mask.shape[:2] != (crop_h, crop_w):
                    crop_mask = cv2.resize(crop_mask, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

                full_mask = np.zeros((height, width), dtype=np.float32)
                full_mask[y1:y2, x1:x2] = crop_mask[:crop_h, :crop_w]

                segments.append({
                    'mask': full_mask,
                    'bbox': box.tolist(),
                    'confidence': float(segment['confidence']),
                    'label': segment.get('label'),
                    'class_name': segment.get('class_name', 'sperm')
                })

        if not segments:
            return []

        boxes = [segment['bbox'] for segment in segments]
        scores = [segment['confidence'] for segment in segments]
        keep = nms_xyxy(boxes, scores, self.slice_nms_iou)
        return [segments[i] for i in keep]

    def segment_subparts(self, crop_img: np.ndarray, crop_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Segment head, neck, and tail subparts
        
        Args:
            crop_img: Cropped image region
            crop_mask: Binary mask of the region
            
        Returns:
            Dictionary with 'head', 'neck', 'tail' masks
        """
        # If no valid callable model, use heuristic fallback
        if self.hnk_model is None or not callable(getattr(self.hnk_model, "__call__", None)):
            return self._heuristic_subpart_segmentation(crop_mask)
        
        try:
            with torch.no_grad():
                inp = torch.from_numpy(crop_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                inp = inp.to(self.device)
                
                logits = self.hnk_model(inp)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                cls_map = np.argmax(probs, axis=0).astype(np.uint8)
                
                head = (cls_map == 1).astype(np.uint8)
                neck = (cls_map == 2).astype(np.uint8)
                tail = (cls_map == 3).astype(np.uint8)
                
                return {
                    'head': head,
                    'neck': neck,
                    'tail': tail
                }
                
        except Exception as e:
            logger.error(f"Subpart segmentation failed: {e}")
            return self._heuristic_subpart_segmentation(crop_mask)
    
    def _heuristic_subpart_segmentation(self, binary_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Fallback heuristic segmentation when model is not available
        
        Args:
            binary_mask: Binary mask of sperm
            
        Returns:
            Dictionary with heuristic head, neck, tail masks
        """
        from skimage import measure, morphology
        from skimage.morphology import skeletonize
        
        # Find the most circular region as head
        labeled = measure.label(binary_mask.astype(np.uint8), connectivity=2)
        regions = measure.regionprops(labeled)
        
        if not regions:
            return {'head': np.zeros_like(binary_mask), 
                   'neck': np.zeros_like(binary_mask), 
                   'tail': np.zeros_like(binary_mask)}
        
        # Find most circular region
        best_region = None
        best_circularity = -1.0
        
        for region in regions:
            if region.perimeter > 0:
                circularity = 4.0 * np.pi * region.area / (region.perimeter ** 2)
                if circularity > best_circularity:
                    best_circularity = circularity
                    best_region = region
        
        if best_region is None:
            return {'head': np.zeros_like(binary_mask), 
                   'neck': np.zeros_like(binary_mask), 
                   'tail': np.zeros_like(binary_mask)}
        
        # Create head mask
        head_mask = (labeled == best_region.label).astype(np.uint8)
        
        # Create skeleton for tail
        skeleton = skeletonize(binary_mask.astype(bool)).astype(np.uint8)
        tail_mask = (np.logical_and(skeleton > 0, head_mask == 0)).astype(np.uint8)
        
        # Create neck as boundary region
        head_boundary = morphology.binary_dilation(head_mask, morphology.disk(1)) ^ head_mask.astype(bool)
        neck_mask = np.logical_and(binary_mask > 0, head_boundary).astype(np.uint8)
        neck_mask = morphology.binary_dilation(neck_mask, morphology.disk(1)).astype(np.uint8)
        
        return {
            'head': head_mask,
            'neck': neck_mask,
            'tail': tail_mask
        }
    
    def match_detection_to_segment(self, detection: Dict, segments: List[Dict]) -> Optional[Dict]:
        """
        Match a detection to the best corresponding segmentation
        
        Args:
            detection: Detection dictionary with bbox
            segments: List of segmentation dictionaries
            
        Returns:
            Best matching segment or None
        """
        if not segments:
            return None
        
        best_iou = 0.0
        best_segment = None
        
        for segment in segments:
            iou = self._compute_iou(detection['bbox'], segment['bbox'])
            if iou > best_iou and iou > 0.1:  # Minimum IoU threshold
                best_iou = iou
                best_segment = segment
        
        return best_segment
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Compute IoU between two bounding boxes
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
