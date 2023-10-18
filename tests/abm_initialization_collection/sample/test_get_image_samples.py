import unittest
from unittest import mock

import numpy as np
import pandas as pd
from aicsimageio import AICSImage

from abm_initialization_collection.sample.get_image_samples import get_image_samples


class TestGetImageSamples(unittest.TestCase):
    def test_get_image_samples(self) -> None:
        channel = 1
        array = np.array(
            [
                [
                    [0, 1, 2],
                    [3, 4, 5],
                ],
                [
                    [6, 7, 8],
                    [9, 10, 11],
                ],
            ]
        )
        sample_indices = [
            (0, 1, 0),
            (1, 1, 2),
            (1, 0, 1),
        ]

        image_mock = mock.Mock(spec=AICSImage)
        image_mock.get_image_data.return_value = array

        expected_samples = [
            (3, 0, 1, 0),
            (11, 1, 1, 2),
            (7, 1, 0, 1),
        ]

        expected = pd.DataFrame(expected_samples, columns=["id", "x", "y", "z"])

        samples = get_image_samples(image_mock, sample_indices, channel)

        image_mock.get_image_data.assert_called_with("XYZ", T=0, C=channel)
        self.assertTrue(expected.equals(samples))


if __name__ == "__main__":
    unittest.main()
