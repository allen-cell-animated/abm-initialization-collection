from typing import Tuple

from aicsimageio import AICSImage
from prefect import task


@task
def get_image_bounds(image: AICSImage) -> Tuple[int, int, int]:
    """
    Extracts image bounds in the x, y, and z directions.

    Parameters
    ----------
    image
        Image object.

    Returns
    -------
    :
        Tuple of image bounds.
    """
    _, _, z_shape, y_shape, x_shape = image.shape
    bounds = (x_shape, y_shape, z_shape)
    return bounds
