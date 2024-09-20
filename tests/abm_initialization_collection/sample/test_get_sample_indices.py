import unittest

from abm_initialization_collection.sample.get_sample_indices import (
    get_hex_sample_indices,
    get_rect_sample_indices,
    get_sample_indices,
)


class TestMakeGridCoordinates(unittest.TestCase):
    def test_get_sample_indices_rect_grid(self):
        bounds = (2, 6, 12)
        resolution = 2
        scale_xy = 1
        scale_z = 0.5

        expected = get_rect_sample_indices(bounds, resolution, scale_xy, scale_z)

        indices = get_sample_indices("rect", bounds, resolution, scale_xy, scale_z)
        self.assertCountEqual(expected, indices)

    def test_get_sample_indices_hex_grid(self):
        bounds = (2, 6, 12)
        resolution = 2
        scale_xy = 1
        scale_z = 0.5

        expected = get_hex_sample_indices(bounds, resolution, scale_xy, scale_z)

        indices = get_sample_indices("hex", bounds, resolution, scale_xy, scale_z)
        self.assertCountEqual(expected, indices)

    def test_get_sample_indices_invalid_grid_throws_exception(self):
        with self.assertRaises(ValueError):
            bounds = (0, 0, 0)
            grid = "invalid_grid"
            get_sample_indices(grid, bounds, 0, 0, 0)

    def test_get_rect_sample_indices(self):
        bounds = (2, 6, 12)
        resolution = 2
        scale_xy = 1
        scale_z = 0.5

        expected = [
            (0, 0, 0),
            (0, 0, 4),
            (0, 0, 8),
            (0, 2, 0),
            (0, 2, 4),
            (0, 2, 8),
            (0, 4, 0),
            (0, 4, 4),
            (0, 4, 8),
        ]

        indices = get_rect_sample_indices(bounds, resolution, scale_xy, scale_z)
        self.assertCountEqual(expected, indices)

    def test_get_hex_sample_indices(self):
        bounds = (2, 6, 12)
        resolution = 2
        scale_xy = 1
        scale_z = 0.5

        expected = [
            (0, 0, 0),
            (1, 2, 0),
            (0, 3, 0),
            (1, 5, 0),
            (1, 1, 4),
            (1, 4, 4),
            (0, 1, 8),
            (1, 3, 8),
            (0, 5, 8),
        ]

        indices = get_hex_sample_indices(bounds, resolution, scale_xy, scale_z)
        self.assertCountEqual(expected, indices)


if __name__ == "__main__":
    unittest.main()
