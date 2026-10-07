"""
Motility Analysis Module
Handles sperm motility classification based on trajectory analysis.

A configured positive micrometers_per_pixel converts coordinates before the
velocity formulas:

    x_um = x_pixel * micrometers_per_pixel
    y_um = y_pixel * micrometers_per_pixel

Those velocities are labeled µm/s. When the scale is missing, the same formulas
run on the pixel coordinates and the velocities are labeled px/s. No default
scale is substituted.

The pixel positions are the existing tracker coordinates. In this project those
are full-frame bounding-box centers, not verified sperm-head centroids.

VAP uses an application-defined average path, not a universal CASA standard:
a time-local mean of observed points, with the first and last points left at
their measured positions. The smoother never zero-pads toward the origin.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

UNCALIBRATED_VELOCITY_WARNING = (
    "Microscope calibration is not configured. Velocities are reported in "
    "pixels/second and are not calibrated CASA µm/s."
)

class MotilityAnalyzer:
    """
    Analyzes sperm motility based on trajectory characteristics
    """
    
    def __init__(self,
                 window_seconds: float = 1.0,
                 min_frames_for_class: int = 3,
                 immobile_disp_threshold: float = 2.0,
                 speed_threshold: float = 5.0,
                 linearity_threshold: float = 0.7,
                 micrometers_per_pixel: Optional[float] = None,
                 vap_smooth_seconds: float = 0.2):
        """
        Initialize motility analyzer.

        micrometers_per_pixel must be a measured microscope/camera scale.
        This constructor does not supply a default scale.

        The numeric classification thresholds are the historical pixel-based
        cutoffs (displacement in pixels, speed in pixels/second). They are not
        µm/s clinical thresholds.
        """
        self.window_seconds = window_seconds
        self.min_frames_for_class = min_frames_for_class
        self.immobile_disp_threshold = immobile_disp_threshold
        self.speed_threshold = speed_threshold
        self.linearity_threshold = linearity_threshold
        self.vap_smooth_seconds = float(vap_smooth_seconds)
        if micrometers_per_pixel is None:
            self.micrometers_per_pixel = None
        else:
            scale = float(micrometers_per_pixel)
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(
                    "micrometers_per_pixel must be a positive measured scale, "
                    f"not {micrometers_per_pixel!r}."
                )
            self.micrometers_per_pixel = scale
    
    def analyze_motility(self, trajectory: List[Tuple[int, float, float]], 
                        fps: float) -> Dict:
        """
        Analyze motility characteristics of a sperm trajectory
        
        Args:
            trajectory: List of (frame_idx, x, y) tuples
            fps: Frames per second of the video
            
        Returns:
            Dictionary containing motility analysis results
        """
        fps = self._validated_fps(fps)
        if len(trajectory) < self.min_frames_for_class:
            return self._result(label='immotile', reason='too_few_frames')

        # Keep points inside the last window_seconds of frame time.
        # Point count is not used as a substitute for elapsed time.
        trajectory = self._select_time_window(trajectory, fps)

        coords_px = np.array([(p[1], p[2]) for p in trajectory], dtype=float)
        if len(coords_px) < 2:
            return self._result(label='immotile', reason='insufficient_data')

        if self.micrometers_per_pixel is None:
            coords = coords_px
        else:
            coords = coords_px * self.micrometers_per_pixel
        metrics = self._calculate_motility_metrics(coords, trajectory, fps)
        
        # Classify motility
        label = self._classify_motility(metrics)
        
        return self._result(
            vcl=metrics['VCL'],
            vsl=metrics['VSL'],
            vap=metrics['VAP'],
            lin=metrics['LIN'],
            path_deviation=metrics['path_deviation'],
            elapsed_time_s=metrics['elapsed_time_s'],
            label=label,
        )

    def _validated_fps(self, fps: float) -> float:
        try:
            value = float(fps)
        except (TypeError, ValueError):
            raise ValueError(
                f"Cannot compute CASA velocities: video FPS is invalid ({fps!r}). "
                "No 30 FPS fallback is applied."
            )
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"Cannot compute CASA velocities: video FPS is invalid ({fps!r}). "
                "No 30 FPS fallback is applied."
            )
        return value

    def _select_time_window(self, trajectory: List[Tuple[int, float, float]],
                            fps: float) -> List[Tuple[int, float, float]]:
        """Keep observations whose frame index falls in the last window_seconds."""
        ordered = sorted(trajectory, key=lambda point: point[0])
        last_frame = ordered[-1][0]
        earliest = last_frame - (self.window_seconds * fps)
        selected = [point for point in ordered if point[0] >= earliest]
        return selected or ordered

    def _smooth_average_path(self, coords_um: np.ndarray,
                             frames: np.ndarray,
                             fps: float) -> np.ndarray:
        """
        Application-defined average path.

        Each interior point is the mean of observed points whose frame index
        is within vap_smooth_seconds/2. The mean uses only real samples, so
        endpoints are not padded with zeros. The first and last smoothed
        points are the measured endpoints.
        """
        count = len(coords_um)
        smoothed = np.array(coords_um, dtype=float, copy=True)
        if count < 3:
            return smoothed
        half_window_frames = (self.vap_smooth_seconds * fps) / 2.0
        for index in range(1, count - 1):
            neighbors = np.abs(frames - frames[index]) <= half_window_frames
            smoothed[index] = np.mean(coords_um[neighbors], axis=0)
        smoothed[0] = coords_um[0]
        smoothed[-1] = coords_um[-1]
        return smoothed

    def _result(self,
                label: str,
                reason: Optional[str] = None,
                vcl: float = 0.0,
                vsl: float = 0.0,
                vap: float = 0.0,
                lin: float = 0.0,
                path_deviation: float = 0.0,
                elapsed_time_s: Optional[float] = None) -> Dict:
        if self.micrometers_per_pixel is None:
            velocity_unit = 'px/s'
            calibration_warning = UNCALIBRATED_VELOCITY_WARNING
        else:
            velocity_unit = 'µm/s'
            calibration_warning = None
        result = {
            'VCL': None if vcl is None else float(vcl),
            'VSL': None if vsl is None else float(vsl),
            'VAP': None if vap is None else float(vap),
            'LIN': None if lin is None else float(lin),
            'LIN_percent': None if lin is None else float(lin) * 100.0,
            'path_deviation': float(path_deviation),
            'label': label,
            'velocity_unit': velocity_unit,
            'calibration_warning': calibration_warning,
            'linearity_unit': 'ratio',
            'coordinate_source': 'bounding_box_center_pixels',
            'micrometers_per_pixel': self.micrometers_per_pixel,
            'elapsed_time_s': elapsed_time_s,
            'vap_method': 'time_local_mean_fixed_endpoints',
            # These cutoffs are still the historical pixel values. They are
            # not validated µm/s clinical thresholds.
            'classification_threshold_unit': 'legacy_px_and_px_per_s',
        }
        if reason is not None:
            result['reason'] = reason
        return result

    def _calculate_motility_metrics(self, coords: np.ndarray, 
                                   trajectory: List[Tuple[int, float, float]], 
                                   fps: float) -> Dict:
        """
        Calculate detailed motility metrics
        
        Args:
            coords: Array of (x, y) coordinates
            trajectory: Original trajectory with frame indices
            fps: Frames per second
            
        Returns:
            Dictionary of motility metrics
        """
        # coords are micrometers. Time uses the frame-index span, so a gap
        # from frame 1 to frame 5 contributes 4/fps seconds, not one frame.
        diffs = np.diff(coords, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        path_length = float(np.sum(segment_lengths)) if len(segment_lengths) > 0 else 0.0
        net_displacement = float(np.linalg.norm(coords[-1] - coords[0])) if len(coords) > 1 else 0.0

        frame_span = trajectory[-1][0] - trajectory[0][0]
        total_time = frame_span / fps if frame_span != 0 else 1.0 / fps

        VCL = path_length / total_time if total_time > 0 else 0.0
        VSL = net_displacement / total_time if total_time > 0 else 0.0

        frames = np.array([point[0] for point in trajectory], dtype=float)
        smooth_coords = self._smooth_average_path(coords, frames, fps)
        smooth_path_length = float(np.sum(np.linalg.norm(np.diff(smooth_coords, axis=0), axis=1)))
        VAP = smooth_path_length / total_time if total_time > 0 else 0.0
        path_deviation = float(np.mean(np.linalg.norm(coords - smooth_coords, axis=1)))

        LIN = VSL / VCL if VCL > 1e-6 else 0.0

        return {
            'VCL': VCL,
            'VAP': VAP,
            'VSL': VSL,
            'LIN': LIN,
            'path_deviation': path_deviation,
            'net_displacement': net_displacement,
            'path_length': path_length,
            'elapsed_time_s': total_time,
        }
    
    def _classify_motility(self, metrics: Dict) -> str:
        """
        Classify motility based on calculated metrics
        
        Args:
            metrics: Dictionary of motility metrics
            
        Returns:
            Motility classification label
        """
        VCL = metrics['VCL']
        VSL = metrics['VSL']
        LIN = metrics['LIN']
        net_displacement = metrics['net_displacement']
        
        # Numeric cutoffs below are unchanged historical pixel thresholds.
        # They are not µm/s reference-CASA or WHO cutoffs.
        if (net_displacement < self.immobile_disp_threshold and 
            VCL < self.speed_threshold):
            return 'immotile'
        
        # Check if progressive
        if VCL >= self.speed_threshold and LIN >= self.linearity_threshold:
            return 'progressive'
        
        # Otherwise non-progressive
        return 'non_progressive'
    
    def analyze_batch(self, trajectories: Dict[int, List[Tuple[int, float, float]]], 
                     fps: float) -> Dict[int, Dict]:
        """
        Analyze motility for multiple trajectories
        
        Args:
            trajectories: Dictionary of track_id -> trajectory
            fps: Frames per second
            
        Returns:
            Dictionary of track_id -> motility analysis
        """
        results = {}
        for track_id, trajectory in trajectories.items():
            results[track_id] = self.analyze_motility(trajectory, fps)
        return results
