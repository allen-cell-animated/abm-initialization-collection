import unittest

import pandas as pd

from abm_initialization_collection.sample.remove_edge_regions import (
    find_edge_ids,
    remove_edge_regions,
)


class TestRemoveEdgeRegions(unittest.TestCase):
    def test_remove_edge_regions(self) -> None:
        edge_threshold = 1
        edge_padding = 0

        cell_1_data = [[1, 0, 2, 1], [1, 2, 0, 2], [1, 2, 4, 3], [1, 4, 2, 4]]
        cell_2_data = [[2, 1, 1, 1], [2, 2, 1, 2]]
        cell_3_data = [[3, 0, 3, 1], [3, 1, 3, 2]]
        cell_4_data = [[4, 3, 3, 1], [4, 3, 4, 2]]

        sample_data = cell_1_data + cell_2_data + cell_3_data + cell_4_data
        samples = pd.DataFrame(sample_data, columns=["id", "x", "y", "z"])

        expected_data = cell_2_data + cell_3_data + cell_4_data
        expected_samples = pd.DataFrame(expected_data, columns=["id", "x", "y", "z"])

        filtered_samples = remove_edge_regions(samples, edge_threshold, edge_padding)
        self.assertTrue(expected_samples.equals(filtered_samples))

    def test_find_edge_ids_no_padding_low_threshold(self) -> None:
        padding = 0
        threshold = 0
        samples = pd.DataFrame(
            [
                [1, 0, 1, 1],
                [1, 1, 1, 2],
                [1, 2, 1, 3],
                [1, 2, 0, 4],
                [1, 3, 0, 5],
                [1, 4, 0, 5],
                [2, 0, 2, 1],
                [2, 1, 2, 2],
                [3, 1, 3, 1],
                [3, 2, 3, 2],
                [3, 2, 4, 3],
            ],
            columns=["id", "x", "y", "z"],
        )
        expected_edge_ids: list[int] = [1, 2]

        edge_ids = find_edge_ids("x", samples, threshold, padding)
        self.assertListEqual(expected_edge_ids, edge_ids)

    def test_find_edge_ids_no_padding_high_threshold(self) -> None:
        padding = 0
        threshold = 3
        samples = pd.DataFrame(
            [
                [1, 0, 1, 1],
                [1, 1, 1, 2],
                [1, 2, 1, 3],
                [1, 2, 0, 4],
                [1, 3, 0, 5],
                [1, 4, 0, 5],
                [2, 0, 2, 1],
                [2, 1, 2, 2],
                [3, 1, 3, 1],
                [3, 2, 3, 2],
                [3, 2, 4, 3],
            ],
            columns=["id", "x", "y", "z"],
        )
        expected_edge_ids: list[int] = []

        edge_ids = find_edge_ids("x", samples, threshold, padding)
        self.assertListEqual(expected_edge_ids, edge_ids)

    def test_find_edge_ids_with_padding_low_threshold(self) -> None:
        padding = 1
        threshold = 0
        samples = pd.DataFrame(
            [
                [1, 0, 1, 1],
                [1, 1, 1, 2],
                [1, 2, 1, 3],
                [1, 2, 0, 4],
                [1, 3, 0, 5],
                [1, 4, 0, 5],
                [2, 0, 2, 1],
                [2, 1, 2, 2],
                [3, 1, 3, 1],
                [3, 2, 3, 2],
                [3, 2, 4, 3],
            ],
            columns=["id", "x", "y", "z"],
        )
        expected_edge_ids: list[int] = [1, 2, 3]

        edge_ids = find_edge_ids("x", samples, threshold, padding)
        self.assertListEqual(expected_edge_ids, edge_ids)

    def test_find_edge_ids_with_padding_high_threshold(self) -> None:
        padding = 1
        threshold = 3
        samples = pd.DataFrame(
            [
                [1, 0, 1, 1],
                [1, 1, 1, 2],
                [1, 2, 1, 3],
                [1, 2, 0, 4],
                [1, 3, 0, 5],
                [1, 4, 0, 5],
                [2, 0, 2, 1],
                [2, 1, 2, 2],
                [3, 1, 3, 1],
                [3, 2, 3, 2],
                [3, 2, 4, 3],
            ],
            columns=["id", "x", "y", "z"],
        )
        expected_edge_ids: list[int] = [1]

        edge_ids = find_edge_ids("x", samples, threshold, padding)
        self.assertListEqual(expected_edge_ids, edge_ids)


if __name__ == "__main__":
    unittest.main()
