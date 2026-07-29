import unittest

import numpy as np

from lingbot_map.vis.point_cloud_viewer import PointCloudViewer


class DepthContinuityMaskTests(unittest.TestCase):
    def test_keeps_smooth_plane(self):
        depth = np.ones((2, 4, 5), dtype=np.float32)
        mask = PointCloudViewer.depth_continuity_mask(depth, 0.15)
        self.assertTrue(mask.all())

    def test_rejects_both_sides_of_depth_step(self):
        depth = np.ones((1, 3, 6), dtype=np.float32)
        depth[:, :, 3:] = 2.0
        mask = PointCloudViewer.depth_continuity_mask(depth, 0.15)
        self.assertFalse(mask[:, :, 2].any())
        self.assertFalse(mask[:, :, 3].any())
        self.assertTrue(mask[:, :, 0].all())
        self.assertTrue(mask[:, :, 5].all())

    def test_rejects_invalid_depth(self):
        depth = np.ones((1, 3, 3), dtype=np.float32)
        depth[0, 1, 1] = np.nan
        mask = PointCloudViewer.depth_continuity_mask(depth, 0.15)
        self.assertFalse(mask[0, 1, 1])


if __name__ == "__main__":
    unittest.main()
