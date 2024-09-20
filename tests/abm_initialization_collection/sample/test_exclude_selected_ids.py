import unittest

import pandas as pd

from abm_initialization_collection.sample.exclude_selected_ids import exclude_selected_ids


class TestExcludeSelectedIDs(unittest.TestCase):
    def test_exclude_selected_ids(self):
        cell_1_data = [[1, 1, 2, 3]]
        cell_2_data = [[2, 1, 2, 1]]
        cell_3_data = [[3, 1, 2, 2]]
        sample_data = cell_1_data + cell_2_data + cell_3_data
        samples = pd.DataFrame(sample_data, columns=["id", "x", "y", "z"])

        exclude = [2]
        expected_data = cell_1_data + cell_3_data
        expected_samples = pd.DataFrame(expected_data, columns=["id", "x", "y", "z"])

        selected_samples = exclude_selected_ids(samples, exclude)

        self.assertTrue(expected_samples.equals(selected_samples))


if __name__ == "__main__":
    unittest.main()
