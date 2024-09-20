from bioio import BioImage


def get_image_bounds(image: BioImage) -> tuple[int, int, int]:
    """
    Extract image bounds in the x, y, and z directions.

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
    return (x_shape, y_shape, z_shape)
