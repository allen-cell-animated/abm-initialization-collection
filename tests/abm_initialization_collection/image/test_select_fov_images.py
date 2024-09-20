import unittest

import pandas as pd

from abm_initialization_collection.image.select_fov_images import select_fov_images


class TestSelectFOVImages(unittest.TestCase):
    @staticmethod
    def make_metadata_entry(
        index: int, path: str, stage: str, volume: int, *, outlier: bool
    ) -> dict:
        return {
            "this_cell_index": index,
            "fov_seg_path": f"path/to/{path}_fov_seg_path",
            "outlier": "Yes" if outlier else "No",
            "cell_stage": stage,
            "MEM_shape_volume": volume,
        }

    def test_select_fov_images_exclude_outliers(self):
        metadata = pd.DataFrame(
            [
                self.make_metadata_entry(1, "aaa", "M0", 150, outlier=True),
                self.make_metadata_entry(2, "aaa", "M0", 150, outlier=True),
                self.make_metadata_entry(3, "bbb", "M0", 150, outlier=False),
                self.make_metadata_entry(4, "bbb", "M0", 150, outlier=False),
            ]
        )
        cells_per_fov = 2
        bins = [0, 100, 200, 300]
        counts = [0, 1, 0]

        expected = [{"key": "bbb", "item": "path/to/bbb_fov_seg_path", "cell_ids": [3, 4]}]

        selected = select_fov_images(metadata, cells_per_fov, bins, counts)
        self.assertListEqual(expected, selected)

    def test_select_fov_images_exclude_cell_stage(self):
        metadata = pd.DataFrame(
            [
                self.make_metadata_entry(1, "aaa", "MX", 150, outlier=False),
                self.make_metadata_entry(2, "aaa", "MX", 150, outlier=False),
                self.make_metadata_entry(3, "bbb", "M0", 150, outlier=False),
                self.make_metadata_entry(4, "bbb", "M0", 150, outlier=False),
            ]
        )
        cells_per_fov = 2
        bins = [0, 100, 200, 300]
        counts = [0, 1, 0]

        expected = [{"key": "bbb", "item": "path/to/bbb_fov_seg_path", "cell_ids": [3, 4]}]

        selected = select_fov_images(metadata, cells_per_fov, bins, counts)
        self.assertListEqual(expected, selected)

    def test_select_fov_images_filter_by_volume(self):
        metadata = pd.DataFrame(
            [
                self.make_metadata_entry(1, "aaa", "M0", 50, outlier=False),
                self.make_metadata_entry(2, "aaa", "M0", 50, outlier=False),
                self.make_metadata_entry(3, "bbb", "M0", 150, outlier=False),
                self.make_metadata_entry(4, "bbb", "M0", 150, outlier=False),
                self.make_metadata_entry(5, "ccc", "M0", 250, outlier=False),
                self.make_metadata_entry(6, "ccc", "M0", 250, outlier=False),
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

    def test_select_fov_images_filter_by_count(self):
        metadata = pd.DataFrame(
            [
                self.make_metadata_entry(1, "aaa", "M0", 150, outlier=False),
                self.make_metadata_entry(2, "aaa", "M0", 150, outlier=False),
                self.make_metadata_entry(3, "aaa", "M0", 150, outlier=False),
                self.make_metadata_entry(4, "bbb", "M0", 150, outlier=False),
                self.make_metadata_entry(5, "bbb", "M0", 150, outlier=False),
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
