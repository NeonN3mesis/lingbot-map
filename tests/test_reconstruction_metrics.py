import unittest

import cv2
import numpy as np

from lingbot_map.evaluation.metrics import (
    adjacent_geometric_agreement,
    layer_thickness,
    numerical_validity,
    surface_roughness,
)


class ReconstructionMetricsTests(unittest.TestCase):
    def setUp(self):
        self.h, self.w = 48, 64
        self.intrinsic = np.repeat(
            np.array([[[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]]]),
            2, axis=0,
        )
        self.extrinsic = np.repeat(np.eye(4, dtype=np.float32)[None, :3], 2, axis=0)
        self.confidence = np.full((2, self.h, self.w), 2.0, dtype=np.float32)

    def test_validity_reports_corruption(self):
        depth = np.ones((2, self.h, self.w, 1), dtype=np.float32)
        depth[0, 0, 0, 0] = np.nan
        result = numerical_validity(depth, self.confidence, self.extrinsic)
        self.assertLess(result["depth_finite_pct"], 100.0)
        self.assertEqual(result["pose_finite_pct"], 100.0)

    def test_roughness_ranks_noisy_plane_worse(self):
        yy, xx = np.mgrid[:self.h, :self.w]
        plane = (2.0 + 0.001 * xx + 0.002 * yy).astype(np.float32)
        clean = np.repeat(plane[None, ..., None], 2, axis=0)
        rng = np.random.default_rng(7)
        noisy = clean + rng.normal(0, 0.03, clean.shape).astype(np.float32)
        clean_metric = surface_roughness(clean, self.confidence, self.intrinsic)
        noisy_metric = surface_roughness(noisy, self.confidence, self.intrinsic)
        self.assertLess(clean_metric["roughness_median"], noisy_metric["roughness_median"])

    def test_identical_frames_have_low_adjacent_disagreement(self):
        rng = np.random.default_rng(4)
        gray = rng.integers(0, 256, (self.h, self.w), dtype=np.uint8)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        rgb = np.repeat(gray[None], 3, axis=0)
        images = np.repeat(rgb[None], 2, axis=0)
        depth = np.full((2, self.h, self.w, 1), 2.0, dtype=np.float32)
        result = adjacent_geometric_agreement(
            depth, self.confidence, self.intrinsic, self.extrinsic, images
        )
        self.assertGreater(result["agreement_matches"], 20)
        self.assertLess(result["agreement_median"], 1e-6)

    def test_layer_thickness_detects_separated_planes(self):
        single = np.full((2, self.h, self.w, 1), 2.0, dtype=np.float32)
        layered = single.copy()
        layered[1] = 2.2
        roi = (8, 8, 56, 40)
        single_metric = layer_thickness(
            single, self.confidence, self.intrinsic, self.extrinsic, roi
        )
        layered_metric = layer_thickness(
            layered, self.confidence, self.intrinsic, self.extrinsic, roi
        )
        self.assertLess(single_metric["layer_thickness"], layered_metric["layer_thickness"])


if __name__ == "__main__":
    unittest.main()
