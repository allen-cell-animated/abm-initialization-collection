import unittest

import pandas as pd

from abm_initialization_collection.image.select_fov_images import select_fov_images


class TestSelectFOVImages(unittest.TestCase):
    @staticmethod
    def make_metadata_entry(index, path, outlier, stage, volume):
        return {
            "this_cell_index": index,
            "fov_seg_path": f"path/to/{path}_fov_seg_path",
            "outlier": "Yes" if outlier else "No",
            "cell_stage": stage,
            "MEM_shape_volume": volume,
        }

    def test_select_fov_images_exclude_outliers(self) -> None:
        metadata = pd.DataFrame(
            [
                self.make_metadata_entry(1, "aaa", True, "M0", 150),
                self.make_metadata_entry(2, "aaa", True, "M0", 150),
                self.make_metadata_entry(3, "bbb", False, "M0", 150),
                self.make_metadata_entry(4, "bbb", False, "M0", 150),
            ]
        )
        cells_per_fov = 2
        bins = [0, 100, 200, 300]
        counts = [0, 1, 0]

        expected = [{"key": "bbb", "item": "path/to/bbb_fov_seg_path", "cell_ids": [3, 4]}]

        selected = select_fov_images(metadata, cells_per_fov, bins, counts)
        self.assertListEqual(expected, selected)

    def test_select_fov_images_exclude_cell_stage(self) -> None:
        metadata = pd.DataFrame(
            [
                self.make_metadata_entry(1, "aaa", False, "MX", 150),
                self.make_metadata_entry(2, "aaa", False, "MX", 150),
                self.make_metadata_entry(3, "bbb", False, "M0", 150),
                self.make_metadata_entry(4, "bbb", False, "M0", 150),
            ]
        )
        cells_per_fov = 2
        bins = [0, 100, 200, 300]
        counts = [0, 1, 0]

        expected = [{"key": "bbb", "item": "path/to/bbb_fov_seg_path", "cell_ids": [3, 4]}]

        selected = select_fov_images(metadata, cells_per_fov, bins, counts)
        self.assertListEqual(expected, selected)

    def test_select_fov_images_filter_by_volume(self) -> None:
        metadata = pd.DataFrame(
            [
                self.make_metadata_entry(1, "aaa", False, "M0", 50),
                self.make_metadata_entry(2, "aaa", False, "M0", 50),
                self.make_metadata_entry(3, "bbb", False, "M0", 150),
                self.make_metadata_entry(4, "bbb", False, "M0", 150),
                self.make_metadata_entry(5, "ccc", False, "M0", 250),
                self.make_metadata_entry(6, "ccc", False, "M0", 250),
            ]
        )
        cells_per_fov = 2
        bins = [0, 100, 200, 300]
        counts = [0, 1, 1]

        expected = [
            {"key": "bbb", "item": "path/to/bbb_fov_seg_path", "cell_ids": [3, 4]},
            {"key": "ccc", "item": "path/to/ccc_fov_seg_path", "cell_ids": [5, 6]},
        ]

        selected = select_fov_images(metadata, cells_per_fov, bins, counts)
        self.assertListEqual(expected, selected)

    def test_select_fov_images_filter_by_count(self) -> None:
        metadata = pd.DataFrame(
            [
                self.make_metadata_entry(1, "aaa", False, "M0", 150),
                self.make_metadata_entry(2, "aaa", False, "M0", 150),
                self.make_metadata_entry(3, "aaa", False, "M0", 150),
                self.make_metadata_entry(4, "bbb", False, "M0", 150),
                self.make_metadata_entry(5, "bbb", False, "M0", 150),
            ]
        )
        cells_per_fov = 2
        bins = [0, 100, 200, 300]
        counts = [0, 1, 0]

        expected = [{"key": "bbb", "item": "path/to/bbb_fov_seg_path", "cell_ids": [4, 5]}]

        selected = select_fov_images(metadata, cells_per_fov, bins, counts)
        self.assertListEqual(expected, selected)


if __name__ == "__main__":
    unittest.main()
