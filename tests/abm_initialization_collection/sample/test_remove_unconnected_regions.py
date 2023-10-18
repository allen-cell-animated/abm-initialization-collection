import unittest

import numpy as np
import pandas as pd

from abm_initialization_collection.sample.remove_unconnected_regions import (
    convert_to_dataframe,
    convert_to_integer_array,
    get_minimum_distance,
    get_sample_maximums,
    get_sample_minimums,
    remove_unconnected_regions,
)


class TestRemoveUnconnectedRegions(unittest.TestCase):
    def test_remove_unconnected_regions_by_connectivity(self) -> None:
        connected_threshold = 0
        samples = pd.DataFrame(
            [
                [2, 2, 0, 0],
                [2, 2, 1, 0],
                [2, 2, 0, 1],
                [2, 1, 2, 1],
                [2, 0, 2, 1],
                [1, 0, 0, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 2, 2, 0],
            ],
            columns=["id", "x", "y", "z"],
        )

        expected = pd.DataFrame(
            [
                [1, 0, 0, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [2, 2, 0, 0],
                [2, 2, 0, 1],
                [2, 2, 1, 0],
            ],
            columns=["id", "x", "y", "z"],
        )

        filtered_samples = remove_unconnected_regions(samples, connected_threshold, "connectivity")
        self.assertTrue(expected.equals(filtered_samples))

    def test_remove_unconnected_regions_by_distance(self) -> None:
        connected_threshold = 1.5
        samples = pd.DataFrame(
            [
                [2, 0, 0, 0],
                [2, 1, 1, 1],
                [2, 1, 2, 2],
                [1, 4, 4, 4],
                [1, 7, 5, 4],
                [1, 4, 3, 4],
            ],
            columns=["id", "x", "y", "z"],
        )

        expected = pd.DataFrame(
            [
                [1, 4, 3, 4],
                [1, 4, 4, 4],
                [2, 1, 1, 1],
                [2, 1, 2, 2],
            ],
            columns=["id", "x", "y", "z"],
        )

        filtered_samples = remove_unconnected_regions(samples, connected_threshold, "distance")
        self.assertTrue(expected.equals(filtered_samples))

    def test_remove_unconnected_regions_invalid_filter_throws_exception(self) -> None:
        with self.assertRaises(ValueError):
            samples = pd.DataFrame()
            remove_unconnected_regions(samples, 0, "invalid_filter")

    def test_get_sample_minimums(self) -> None:
        samples = pd.DataFrame(
            [
                [1, 0, 3, 10],
                [1, 2, 15, 40],
                [1, 6, 6, 20],
                [1, 4, 12, 50],
                [1, 8, 9, 30],
                [1, 10, 18, 60],
            ],
            columns=["id", "x", "y", "z"],
        )
        expected_minimums = (0, 3, 10)
        minimums = get_sample_minimums(samples)
        self.assertTupleEqual(expected_minimums, minimums)

    def test_get_sample_maximums(self) -> None:
        samples = pd.DataFrame(
            [
                [1, 0, 3, 10],
                [1, 2, 15, 40],
                [1, 6, 6, 20],
                [1, 4, 12, 50],
                [1, 8, 9, 30],
                [1, 10, 18, 60],
            ],
            columns=["id", "x", "y", "z"],
        )
        expected_maximums = (10, 18, 60)
        maximums = get_sample_maximums(samples)
        self.assertTupleEqual(expected_maximums, maximums)

    def test_convert_to_integer_array(self) -> None:
        samples = pd.DataFrame(
            [
                [1, 0, 0, 0],
                [2, 1, 1, 0],
                [3, 2, 1, 0],
                [4, 3, 2, 0],
                [5, 4, 2, 0],
                [6, 5, 2, 1],
            ],
            columns=["id", "x", "y", "z"],
        )
        minimums = (0, 0, 0)
        maximums = (5, 2, 1)

        expected_array = np.array(
            [
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 2, 3, 0, 0, 0],
                    [0, 0, 0, 4, 5, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 6],
                ],
            ]
        )

        array = convert_to_integer_array(samples, minimums, maximums)
        self.assertTrue(np.array_equal(expected_array, array))

    def test_convert_to_dataframe(self) -> None:
        array = np.array(
            [
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 2, 3, 0, 0, 0],
                    [0, 0, 0, 4, 5, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 6],
                ],
            ]
        )
        minimums = (0, 0, 0)

        expected_dataframe = pd.DataFrame(
            [
                [1, 0, 0, 0],
                [2, 1, 1, 0],
                [3, 2, 1, 0],
                [4, 3, 2, 0],
                [5, 4, 2, 0],
                [6, 5, 2, 1],
            ],
            columns=["id", "x", "y", "z"],
        )

        dataframe = convert_to_dataframe(array, minimums)
        self.assertTrue(expected_dataframe.equals(dataframe))

    def test_get_minimum_distance(self) -> None:
        source = np.array([[0, 0, 0]])
        targets = np.array(
            [
                [3, 2, 1],
                [1, 2, 3],
                [2, 1, 2],
                [3, 1, 2],
            ]
        )

        minimum_distance = get_minimum_distance(source, targets)
        self.assertAlmostEqual(3, minimum_distance)


if __name__ == "__main__":
    unittest.main()
