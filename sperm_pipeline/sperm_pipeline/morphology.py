"""
Morphology Analysis Module
Handles sperm morphology assessment based on head, neck, and tail characteristics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from skimage import measure, morphology
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

class MorphologyAnalyzer:
    """
    Analyzes sperm morphology based on head, neck, and tail characteristics
    """
    
    def __init__(self, 
                 head_circularity_min: float = 0.70,
                 head_length_range: Tuple[float, float] = (5, 60),
                 head_width_range: Tuple[float, float] = (3, 40),
                 neck_angle_max: float = 45.0,
                 tail_length_min: float = 15.0):
        """
        Initialize morphology analyzer with clinical criteria
        
        Args:
            head_circularity_min: Minimum circularity for normal head shape
            head_length_range: Acceptable head length range in pixels
            head_width_range: Acceptable head width range in pixels
            neck_angle_max: Maximum acceptable neck angle in degrees
            tail_length_min: Minimum tail length in pixels
        """
        self.head_circularity_min = head_circularity_min
        self.head_length_range = head_length_range
        self.head_width_range = head_width_range
        self.neck_angle_max = neck_angle_max
        self.tail_length_min = tail_length_min
    
    def analyze_morphology(self, head_mask: np.ndarray, neck_mask: np.ndarray, 
                          tail_mask: np.ndarray) -> Dict:
        """
        Analyze morphology of a sperm based on subpart masks
        
        Args:
            head_mask: Binary mask of head region
            neck_mask: Binary mask of neck region
            tail_mask: Binary mask of tail region
            
        Returns:
            Dictionary containing morphology analysis results
        """
        # Analyze head properties
        head_props = self._analyze_head(head_mask)
        
        # Analyze tail properties
        tail_props = self._analyze_tail(tail_mask)
        
        # Analyze neck angle
        neck_angle = self._compute_neck_angle(head_props['centroid'], tail_mask)
        
        # Combine all features
        features = {
            'head_length': head_props['length'],
            'head_width': head_props['width'],
            'head_circularity': head_props['circularity'],
            'head_area': head_props['area'],
            'tail_length': tail_props['length'],
            'neck_angle_deg': neck_angle,
            'centroid': head_props['centroid']
        }
        
        # Classify morphology
        status, score, reasons = self._classify_morphology(features)
        
        return {
            'features': features,
            'status': status,
            'score': score,
            'reasons': reasons
        }
    
    def _analyze_head(self, head_mask: np.ndarray) -> Dict:
        """
        Analyze head morphology properties
        
        Args:
            head_mask: Binary mask of head region
            
        Returns:
            Dictionary of head properties
        """
        if np.sum(head_mask) == 0:
            return {
                'area': 0.0,
                'length': 0.0,
                'width': 0.0,
                'circularity': 0.0,
                'centroid': (0.0, 0.0)
            }
        
        labeled = measure.label(head_mask.astype(np.uint8), connectivity=2)
        regions = measure.regionprops(labeled)
        
        if not regions:
            return {
                'area': 0.0,
                'length': 0.0,
                'width': 0.0,
                'circularity': 0.0,
                'centroid': (0.0, 0.0)
            }
        
        # Get largest region
        region = max(regions, key=lambda r: r.area)
        
        area = float(region.area)
        perimeter = float(region.perimeter) if region.perimeter > 0 else 1.0
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        
        # Get major and minor axis lengths
        if region.inertia_tensor_eigvals is not None and len(region.inertia_tensor_eigvals) == 2:
            major_axis = float(region.major_axis_length)
            minor_axis = float(region.minor_axis_length)
        else:
            # Fallback to bounding box dimensions
            minr, minc, maxr, maxc = region.bbox
            major_axis = max(maxr - minr, maxc - minc)
            minor_axis = min(maxr - minr, maxc - minc)
        
        centroid = (float(region.centroid[1]), float(region.centroid[0]))
        
        return {
            'area': area,
            'length': major_axis,
            'width': minor_axis,
            'circularity': circularity,
            'centroid': centroid
        }
    
    def _analyze_tail(self, tail_mask: np.ndarray) -> Dict:
        """
        Analyze tail morphology properties
        
        Args:
            tail_mask: Binary mask of tail region
            
        Returns:
            Dictionary of tail properties
        """
        if np.sum(tail_mask) == 0:
            return {'length': 0.0}
        
        # Skeletonize tail to get length
        skeleton = skeletonize(tail_mask.astype(bool)).astype(np.uint8)
        tail_length = float(np.sum(skeleton))
        
        return {'length': tail_length}
    
    def _compute_neck_angle(self, head_centroid: Tuple[float, float], 
                           tail_mask: np.ndarray) -> float:
        """
        Compute neck angle based on head centroid and tail direction
        
        Args:
            head_centroid: (x, y) coordinates of head centroid
            tail_mask: Binary mask of tail region
            
        Returns:
            Neck angle in degrees
        """
        if np.sum(tail_mask) == 0:
            return 0.0
        
        ys, xs = np.where(tail_mask > 0)
        if len(xs) < 2:
            return 0.0
        
        coords = np.stack([xs, ys], axis=1)
        
        # Find furthest point from head
        dists = cdist([head_centroid], coords)[0]
        furthest_idx = np.argmax(dists)
        furthest_point = coords[furthest_idx]
        
        # Calculate direction vector
        direction_vec = furthest_point - np.array([head_centroid[0], head_centroid[1]])
        
        # Calculate principal axis of tail
        centered_coords = coords - np.mean(coords, axis=0)
        if len(centered_coords) < 2:
            return 0.0
        
        try:
            _, _, Vt = np.linalg.svd(centered_coords, full_matrices=False)
            principal_axis = Vt[0]
        except np.linalg.LinAlgError:
            return 0.0
        
        # Calculate angle between direction vector and principal axis
        direction_angle = np.arctan2(direction_vec[1], direction_vec[0])
        principal_angle = np.arctan2(principal_axis[1], principal_axis[0])
        
        angle_diff = np.abs((direction_angle - principal_angle + np.pi) % (2 * np.pi) - np.pi)
        
        return float(np.degrees(angle_diff))
    
    def _classify_morphology(self, features: Dict) -> Tuple[str, float, List[str]]:
        """
        Classify morphology as normal or defective based on features
        
        Args:
            features: Dictionary of morphology features
            
        Returns:
            Tuple of (status, score, reasons)
        """
        reasons = []
        score = 1.0
        
        # Check head circularity
        circularity = features.get('head_circularity', 0.0)
        if circularity < self.head_circularity_min:
            reasons.append("low head circularity")
            score -= 0.2
        
        # Check head length
        head_length = features.get('head_length', 0.0)
        if not (self.head_length_range[0] <= head_length <= self.head_length_range[1]):
            reasons.append("abnormal head length")
            score -= 0.2
        
        # Check head width
        head_width = features.get('head_width', 0.0)
        if not (self.head_width_range[0] <= head_width <= self.head_width_range[1]):
            reasons.append("abnormal head width")
            score -= 0.2
        
        # Check neck angle
        neck_angle = features.get('neck_angle_deg', 0.0)
        if neck_angle > self.neck_angle_max:
            reasons.append("excess neck angle")
            score -= 0.2
        
        # Check tail length
        tail_length = features.get('tail_length', 0.0)
        if tail_length < self.tail_length_min:
            reasons.append("short tail length")
            score -= 0.2
        
        # Determine final status
        status = "normal" if len(reasons) == 0 else "defective"
        score = max(0.0, min(1.0, score))
        
        return status, score, reasons
