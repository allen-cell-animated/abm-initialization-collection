from bioio import BioImage


def get_image_bounds(image: BioImage) -> tuple[int, int, int]:
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
