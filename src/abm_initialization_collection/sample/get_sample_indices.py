from typing import Tuple, List

from prefect import task

from abm_initialization_collection.coordinate.make_grid_coordinates import (
    make_rect_grid_coordinates,
    make_hex_grid_coordinates,
)


@task
def get_sample_indices(
    grid: str,
    bounds: Tuple[int, int, int],
    resolution: float,
    scale_xy: float,
    scale_z: float,
) -> List:
    """
    Get sample indices with given bounds for selected grid type.

    Parameters
    ----------
    grid : {'rect', 'hex'}
        Type of grid.
    bounds
        Sampling bounds in the x, y, and z directions.
    resolution
        Distance between samples (um).
    scale_xy
        Resolution scaling in x/y (um/pixel).
    scale_z
        Resolution scaling in z (um/pixel).

    Returns
    -------
    :
        List of sample indices.
    """
    if grid == "rect":
        return get_rect_sample_indices(bounds, resolution, scale_xy, scale_z)

    if grid == "hex":
        return get_hex_sample_indices(bounds, resolution, scale_xy, scale_z)

    raise ValueError(f"invalid grid type {grid}")


def get_rect_sample_indices(
    bounds: Tuple[int, int, int],
    resolution: float,
    scale_xy: float,
    scale_z: float,
) -> List:
    """
    Get list of (x, y, z) sample indices for rect grid.

    Parameters
    ----------
    bounds
        Sampling bounds in the x, y, and z directions.
    resolution
        Distance between samples (um).
    scale_xy
        Resolution scaling in x/y.
    scale_z
        Resolution scaling in z.

    Returns
    -------
    :
        List of sample indices.
    """
    increment_z = round(resolution / scale_z)
    increment_xy = round(resolution / scale_xy)

    sample_coordinates = make_rect_grid_coordinates(bounds, increment_xy, increment_z)
    sample_indices = [(round(x), round(y), z) for x, y, z in sample_coordinates]

    return sample_indices


def get_hex_sample_indices(
    bounds: Tuple[int, int, int],
    resolution: float,
    scale_xy: float,
    scale_z: float,
) -> List:
    """
    Get list of (x, y, z) sample indices for hex grid.

    Sample indices are offset in sets of three z slices to form a
    face-centered cubic (FCC) packing.

    Parameters
    ----------
    bounds
        Sampling bounds in the x, y, and z directions.
    resolution
        Distance between samples (um).
    scale_xy
        Resolution scaling in x/y.
    scale_z
        Resolution scaling in z.

    Returns
    -------
    :
        List of sample indices.
    """
    increment_z = round(resolution / scale_z)
    increment_xy = round(resolution / scale_xy)

    sample_coordinates = make_hex_grid_coordinates(bounds, increment_xy, increment_z)
    sample_indices = [(round(x), round(y), z) for x, y, z in sample_coordinates]

    return sample_indices
