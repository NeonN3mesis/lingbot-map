import unittest

from demo import parse_frame_spec


class ParseFrameSpecTests(unittest.TestCase):
    def test_parses_numbers_and_ranges(self):
        self.assertEqual(parse_frame_spec("3,8-10, 12"), {3, 8, 9, 10, 12})

    def test_empty_spec(self):
        self.assertEqual(parse_frame_spec(""), set())

    def test_rejects_reversed_range(self):
        with self.assertRaises(ValueError):
            parse_frame_spec("9-4")


if __name__ == "__main__":
    unittest.main()
