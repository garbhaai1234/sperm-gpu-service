"""Independent checks for the current CASA velocity math. Not a clinical validation."""

import math
import unittest

import numpy as np

from sperm_pipeline.motility import MotilityAnalyzer, UNCALIBRATED_VELOCITY_WARNING
from sperm_pipeline.pipeline import SpermAnalysisPipeline


def independent(points, fps, scale, window_seconds=1.0, smooth_seconds=0.2):
    """Recompute VCL, VSL, VAP, and LIN without calling MotilityAnalyzer."""
    ordered = sorted(points, key=lambda point: point[0])
    earliest = ordered[-1][0] - window_seconds * fps
    selected = [point for point in ordered if point[0] >= earliest] or ordered
    coords = np.array([(point[1] * scale, point[2] * scale) for point in selected], dtype=float)
    frames = np.array([point[0] for point in selected], dtype=float)
    path = float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
    net = float(np.linalg.norm(coords[-1] - coords[0]))
    span = selected[-1][0] - selected[0][0]
    elapsed = span / fps if span != 0 else 1.0 / fps
    smoothed = coords.copy()
    half = (smooth_seconds * fps) / 2.0
    for index in range(1, len(coords) - 1):
        neighbors = np.abs(frames - frames[index]) <= half
        smoothed[index] = np.mean(coords[neighbors], axis=0)
    smooth_path = float(np.sum(np.linalg.norm(np.diff(smoothed, axis=0), axis=1)))
    vcl = path / elapsed
    vsl = net / elapsed
    vap = smooth_path / elapsed
    lin = vsl / vcl if vcl > 1e-6 else 0.0
    return {
        "VCL": vcl,
        "VSL": vsl,
        "VAP": vap,
        "LIN": lin,
        "elapsed": elapsed,
        "path": path,
        "net": net,
        "smooth_path": smooth_path,
        "smoothed": smoothed,
    }


class CasaMetricTests(unittest.TestCase):
    def test_known_micrometer_trajectory(self):
        points = [(0, 0.0, 0.0), (1, 3.0, 4.0), (2, 6.0, 8.0), (3, 6.0, 12.0)]
        result = MotilityAnalyzer(micrometers_per_pixel=1.0).analyze_motility(points, 30.0)
        self.assertAlmostEqual(result["VCL"], 140.0, places=6)
        self.assertAlmostEqual(result["VSL"], math.sqrt(180.0) / 0.1, places=6)
        self.assertAlmostEqual(result["LIN"], result["VSL"] / result["VCL"], places=9)
        self.assertAlmostEqual(result["LIN_percent"], result["LIN"] * 100.0, places=6)
        self.assertEqual(result["velocity_unit"], "µm/s")
        self.assertIsNone(result["calibration_warning"])
        self.assertEqual(result["linearity_unit"], "ratio")
        self.assertAlmostEqual(result["elapsed_time_s"], 0.1, places=9)

        expected = independent(points, 30.0, 1.0)
        # Interior points both average all four samples: mean = (3.75, 6).
        # Path = sqrt(3.75^2+6^2) + 0 + sqrt(2.25^2+6^2).
        self.assertAlmostEqual(expected["smooth_path"], math.sqrt(50.0625) + math.sqrt(41.0625), places=6)
        self.assertAlmostEqual(expected["VAP"], (math.sqrt(50.0625) + math.sqrt(41.0625)) / 0.1, places=6)
        for key in ("VCL", "VSL", "VAP", "LIN"):
            self.assertAlmostEqual(result[key], expected[key], places=6)

    def test_calibration_scales_pixel_velocity(self):
        pixels = [(0, 0.0, 0.0), (1, 15.0, 20.0), (2, 30.0, 40.0), (3, 30.0, 60.0)]
        scale = 0.20
        result = MotilityAnalyzer(micrometers_per_pixel=scale).analyze_motility(pixels, 30.0)
        pixel_result = MotilityAnalyzer(micrometers_per_pixel=1.0).analyze_motility(pixels, 30.0)
        self.assertAlmostEqual(result["VCL"], pixel_result["VCL"] * scale, places=6)
        self.assertAlmostEqual(result["VSL"], pixel_result["VSL"] * scale, places=6)
        self.assertAlmostEqual(result["VAP"], pixel_result["VAP"] * scale, places=6)
        self.assertAlmostEqual(result["VCL"], 140.0, places=6)
        self.assertAlmostEqual(result["LIN"], pixel_result["LIN"], places=9)

    def test_vap_does_not_explode_far_from_origin(self):
        points = [
            (0, 5000.0, 4000.0),
            (1, 5003.0, 4004.0),
            (2, 5006.0, 4008.0),
            (3, 5006.0, 4012.0),
        ]
        result = MotilityAnalyzer(micrometers_per_pixel=1.0).analyze_motility(points, 30.0)
        self.assertLess(result["VAP"], 3.0 * result["VCL"])
        self.assertAlmostEqual(result["VCL"], 140.0, places=6)
        expected = independent(points, 30.0, 1.0)
        self.assertAlmostEqual(result["VAP"], expected["VAP"], places=6)

    def test_frame_gap_uses_frame_index_time(self):
        points = [(0, 0.0, 0.0), (1, 3.0, 4.0), (5, 6.0, 8.0), (6, 6.0, 12.0)]
        result = MotilityAnalyzer(micrometers_per_pixel=1.0).analyze_motility(points, 30.0)
        self.assertAlmostEqual(result["elapsed_time_s"], 6.0 / 30.0, places=9)
        self.assertAlmostEqual(result["VCL"], 70.0, places=6)
        self.assertAlmostEqual(result["VSL"], math.sqrt(180.0) / 0.2, places=6)
        expected = independent(points, 30.0, 1.0)
        for key in ("VCL", "VSL", "VAP", "LIN"):
            self.assertAlmostEqual(result[key], expected[key], places=6)

    def test_missing_calibration_reports_pixels_per_second(self):
        points = [(0, 0.0, 0.0), (1, 3.0, 4.0), (2, 6.0, 8.0), (3, 6.0, 12.0)]
        result = MotilityAnalyzer().analyze_motility(points, 30.0)
        expected = independent(points, 30.0, 1.0)
        self.assertIsNone(result["micrometers_per_pixel"])
        self.assertEqual(result["velocity_unit"], "px/s")
        self.assertEqual(result["calibration_warning"], UNCALIBRATED_VELOCITY_WARNING)
        self.assertNotIn("reason", result)
        for key in ("VCL", "VSL", "VAP", "LIN"):
            self.assertAlmostEqual(result[key], expected[key], places=6)
        self.assertAlmostEqual(result["LIN"], result["VSL"] / result["VCL"], places=9)
        self.assertEqual(result["coordinate_source"], "bounding_box_center_pixels")

    def test_video_analysis_continues_without_calibration(self):
        points = [(0, 0.0, 0.0), (1, 3.0, 4.0), (2, 6.0, 8.0), (3, 6.0, 12.0)]
        pipeline = SpermAnalysisPipeline.__new__(SpermAnalysisPipeline)
        pipeline.micrometers_per_pixel = None
        pipeline.motility_analyzer = MotilityAnalyzer()
        pipeline.morphology_results = {}
        pipeline.tracker = type("Tracker", (), {"get_all_trajectories": lambda self: {1: points}})()
        results = pipeline._compute_final_analysis(30.0)
        self.assertEqual(results[0]["velocity_unit"], "px/s")
        self.assertEqual(results[0]["calibration_warning"], UNCALIBRATED_VELOCITY_WARNING)
        self.assertAlmostEqual(results[0]["motility_features"]["VCL"], 140.0, places=6)
        self.assertIn("px/s", results[0]["explainability"]["motility"])
        self.assertIn(UNCALIBRATED_VELOCITY_WARNING, results[0]["explainability"]["motility"])

        pipeline.micrometers_per_pixel = 0.2
        pipeline.motility_analyzer = MotilityAnalyzer(micrometers_per_pixel=0.2)
        calibrated = pipeline._compute_final_analysis(30.0)
        self.assertEqual(calibrated[0]["velocity_unit"], "µm/s")
        self.assertIsNone(calibrated[0]["calibration_warning"])
        self.assertAlmostEqual(calibrated[0]["motility_features"]["VCL"], 28.0, places=6)

    def test_invalid_fps_fails_clearly(self):
        analyzer = MotilityAnalyzer(micrometers_per_pixel=1.0)
        points = [(0, 0.0, 0.0), (1, 1.0, 1.0), (2, 2.0, 2.0)]
        with self.assertRaises(ValueError):
            analyzer.analyze_motility(points, 0.0)
        with self.assertRaises(ValueError):
            analyzer.analyze_motility(points, None)


if __name__ == "__main__":
    unittest.main()
