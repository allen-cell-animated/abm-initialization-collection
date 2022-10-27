from typing import Optional

from prefect import task
import pandas as pd


@task
def scale_sample_coordinates(
    samples: pd.DataFrame,
    coordinate_type: Optional[str],
    resolution: float,
    scale_xy: float,
    scale_z: float,
) -> pd.DataFrame:
    """
    Scales sampled coordinates using to given coordinate type.

    The "absolute" coordinate type scales sample index coordinate into absolute
    positions (in um).
    The "step" coordinate type scales sample index coordinates by step size.
    Otherwise, samples index coordinates are not modified.

    Parameters
    ----------
    samples
        Sample cell ids and coordinates.
    coordinate_type : {'absolute', 'step', None}
        The coordinate scaling type.
    resolution
        Distance between samples (um).
    scale_xy
        Resolution scaling in x/y (um/pixel).
    scale_z
        Resolution scaling in z (um/pixel).

    Returns
    -------
    :
        Sample cell ids and scaled coordinates.
    """

    if coordinate_type == "absolute":
        samples["x"] = samples["x"] * scale_xy
        samples["y"] = samples["y"] * scale_xy
        samples["z"] = samples["z"] * scale_z
    elif coordinate_type == "step":
        samples["x"] = (samples["x"] / round(resolution / scale_xy)).astype("int32")
        samples["y"] = (samples["y"] / round(resolution / scale_xy)).astype("int32")
        samples["z"] = (samples["z"] / round(resolution / scale_z)).astype("int32")

    return samples
