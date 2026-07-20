import unittest

import numpy as np

from lingbot_map.vis.point_cloud_viewer import PointCloudViewer


class VoxelFusionTests(unittest.TestCase):
    def test_keeps_multiframe_voxel_and_rejects_single_frame_voxel(self):
        points = np.array([
            [[[0.01, 0.01, 0.01], [2.0, 2.0, 2.0]]],
            [[[0.02, 0.01, 0.01], [3.0, 3.0, 3.0]]],
        ], dtype=np.float32)
        colors = np.array([
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            [[[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]],
        ], dtype=np.float32)
        conf = np.full((2, 1, 2), 2.0, dtype=np.float32)

        fused_points, fused_colors, fused_conf = PointCloudViewer.fuse_multiview_voxels(
            points, colors, conf, conf_threshold=1.5,
            voxel_size=0.1, min_view_support=2, pixel_stride=1,
        )

        self.assertEqual(sum(len(x) for x in fused_points), 1)
        point = np.concatenate(fused_points)[0]
        color = np.concatenate(fused_colors)[0]
        self.assertTrue(np.allclose(point, [0.015, 0.01, 0.01]))
        self.assertTrue(np.allclose(color, [0.5, 0.0, 0.5]))
        self.assertGreater(np.concatenate(fused_conf)[0], 1.5)

    def test_requires_positive_voxel_size(self):
        data = np.zeros((1, 1, 1, 3), dtype=np.float32)
        conf = np.ones((1, 1, 1), dtype=np.float32)
        with self.assertRaises(ValueError):
            PointCloudViewer.fuse_multiview_voxels(data, data, conf, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
