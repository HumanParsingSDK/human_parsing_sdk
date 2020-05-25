import unittest

from human_parsing.segmentation import Segmentation


class SegmentationTest(unittest.TestCase):
    def test_weight_download(self):
        try:
            Segmentation()
        except:
            self.fail("Weights are not downloads")
