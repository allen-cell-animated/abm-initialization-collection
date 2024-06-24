import unittest

import pandas as pd

from abm_initialization_collection.coordinate.filter_coordinate_bounds import (
    filter_coordinate_bounds,
)


class TestFilterCoordinateBounds(unittest.TestCase):
    def test_filter_coordinate_bounds_not_centered(self) -> None:
        radius = 2
        coordinates = [(0, 2, 0), (0, 4, 1), (1, 6, 2), (1, 2, 3), (2, 4, 4), (2, 6, 5)]

        expected = [(0, 4, 1), (1, 6, 2), (1, 2, 3), (2, 4, 4)]
        expected_df = pd.DataFrame(expected, columns=["x", "y", "z"])

        filtered_df = filter_coordinate_bounds(coordinates, radius, center=False)
        self.assertTrue(expected_df.astype("float").equals(filtered_df.astype("float")))

    def test_filter_coordinate_bounds_centered(self) -> None:
        radius = 2
        coordinates = [(0, 2, 0), (0, 4, 1), (1, 6, 2), (1, 2, 3), (2, 4, 4), (2, 6, 5)]

        expected = [(-1, 0, 1), (0, 2, 2), (0, -2, 3), (1, 0, 4)]
        expected_df = pd.DataFrame(expected, columns=["x", "y", "z"])

        filtered_df = filter_coordinate_bounds(coordinates, radius, center=True)
        self.assertTrue(expected_df.astype("float").equals(filtered_df.astype("float")))


if __name__ == "__main__":
    unittest.main()
