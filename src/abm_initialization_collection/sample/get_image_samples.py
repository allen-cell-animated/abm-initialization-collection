import pandas as pd
from bioio import BioImage


def get_image_samples(image: BioImage, sample_indices: list, channel: int) -> pd.DataFrame:
    """
    Sample image at given indices into list of (id, x, y, z) samples.

    Parameters
    ----------
    image
        Image object to sample.
    sample_indices
        List of sampling indices.
    channel
        Image channel to sample.

    Returns
    -------
    :
        Dataframe of image samples.
    """

    array = image.get_image_data("XYZ", T=0, C=channel)
    samples = [(array[x, y, z], x, y, z) for x, y, z in sample_indices if array[x, y, z] > 0]
    return pd.DataFrame(samples, columns=["id", "x", "y", "z"])
