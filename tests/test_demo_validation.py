import unittest

import torch

from demo import validate_predictions


def valid_predictions(frames=4):
    return {
        "pose_enc": torch.zeros(1, frames, 9),
        "depth": torch.ones(1, frames, 2, 3, 1),
        "depth_conf": torch.ones(1, frames, 2, 3),
    }


class ValidatePredictionsTest(unittest.TestCase):
    def test_accepts_finite_positive_predictions(self):
        validate_predictions(valid_predictions())

    def test_reports_first_nonfinite_frame(self):
        predictions = valid_predictions()
        predictions["depth"][0, 2, 0, 0, 0] = torch.nan

        with self.assertRaisesRegex(RuntimeError, r"depth.*first bad frame=2"):
            validate_predictions(predictions)

    def test_rejects_nonpositive_depth(self):
        predictions = valid_predictions()
        predictions["depth"].zero_()

        with self.assertRaisesRegex(RuntimeError, "depth is only 0.00% positive"):
            validate_predictions(predictions)


if __name__ == "__main__":
    unittest.main()
