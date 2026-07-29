import unittest

import numpy as np

from demo import rotation_between_vectors


class RotationBetweenVectorsTests(unittest.TestCase):
    def test_maps_source_to_target_without_scaling(self):
        source = np.array([0.0, -0.8, -0.6])
        target = np.array([0.0, -1.0, 0.0])
        rotation = rotation_between_vectors(source, target)
        self.assertTrue(np.allclose(rotation @ source, target, atol=1e-7))
        self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7))
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=7)

    def test_identity_for_matching_vectors(self):
        rotation = rotation_between_vectors([0, -1, 0], [0, -1, 0])
        self.assertTrue(np.allclose(rotation, np.eye(3)))


if __name__ == "__main__":
    unittest.main()
