import unittest
from unittest import mock

from bioio import BioImage

from abm_initialization_collection.image.get_image_bounds import get_image_bounds


class TestGetImageBounds(unittest.TestCase):
    def test_get_image_bounds(self) -> None:
        # Shape dimensions given in TCZYX order
        shape = (100, 200, 300, 400, 500)

        image_mock = mock.Mock(spec=BioImage)
        image_mock.shape = shape
        expected_bounds = (500, 400, 300)

        bounds = get_image_bounds(image_mock)
        self.assertEquals(expected_bounds, bounds)


if __name__ == "__main__":
    unittest.main()
