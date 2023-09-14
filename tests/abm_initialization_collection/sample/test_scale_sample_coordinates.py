import random
import unittest

import numpy as np
import pandas as pd

from abm_initialization_collection.sample.scale_sample_coordinates import scale_sample_coordinates


class TestScaleSampleCoordinates(unittest.TestCase):
    def test_scale_sample_coordinates_by_absolute(self) -> None:
        scale_xy = random.random()
        scale_z = random.random()

        sample_data = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
            ]
        )

        samples = pd.DataFrame(sample_data, columns=["id", "x", "y", "z"])
        x_scaled = sample_data[:, 1] * scale_xy
        y_scaled = sample_data[:, 2] * scale_xy
        z_scaled = sample_data[:, 3] * scale_z

        scaled_samples = scale_sample_coordinates(samples, "absolute", 0, scale_xy, scale_z)

        for expected, actual in zip(x_scaled, scaled_samples["x"]):
            self.assertAlmostEqual(expected, actual, places=5)

        for expected, actual in zip(y_scaled, scaled_samples["y"]):
            self.assertAlmostEqual(expected, actual, places=5)

        for expected, actual in zip(z_scaled, scaled_samples["z"]):
            self.assertAlmostEqual(expected, actual, places=5)

    def test_scale_sample_coordinates_by_step(self) -> None:
        scale_xy = 5
        scale_z = 10
        resolution = 20

        sample_data = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
            ]
        )

        samples = pd.DataFrame(sample_data, columns=["id", "x", "y", "z"])
        x_scaled = sample_data[:, 1] * scale_xy / resolution
        y_scaled = sample_data[:, 2] * scale_xy / resolution
        z_scaled = sample_data[:, 3] * scale_z / resolution

        scaled_samples = scale_sample_coordinates(samples, "step", resolution, scale_xy, scale_z)

        for expected, actual in zip(x_scaled, scaled_samples["x"]):
            self.assertEqual(int(expected), actual)

        for expected, actual in zip(y_scaled, scaled_samples["y"]):
            self.assertEqual(int(expected), actual)

        for expected, actual in zip(z_scaled, scaled_samples["z"]):
            self.assertEqual(int(expected), actual)


if __name__ == "__main__":
    unittest.main()
