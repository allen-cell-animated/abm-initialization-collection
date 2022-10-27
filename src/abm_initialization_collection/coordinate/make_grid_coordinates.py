from math import floor, sqrt
from typing import Tuple, List

from prefect import task
import numpy as np
from hexalattice import hexalattice


@task
def make_grid_coordinates(
    grid: str,
    bounds: Tuple[int, int, int],
    increment_xy: float,
    increment_z: float,
) -> List:
    """
    Get all coordinates within given bounding box for selected grid type.

    Parameters
    ----------
    grid : {'rect', 'hex'}
        Type of grid.
    bounds
        Bounds in the x, y, and z directions.
    increment_xy
        Increment size in x/y directions.
    increment_z
        Increment size in z direction.

    Returns
    -------
    :
        List of grid coordinates.
    """
    if grid == "rect":
        return make_rect_grid_coordinates(bounds, increment_xy, increment_z)

    if grid == "hex":
        return make_hex_grid_coordinates(bounds, increment_xy, increment_z)

    raise ValueError(f"invalid grid type {grid}")


def make_rect_grid_coordinates(
    bounds: Tuple[int, int, int],
    increment_xy: float,
    increment_z: float,
) -> List:
    """
    Get list of bounded (x, y, z) coordinates for rect grid.

    Parameters
    ----------
    bounds
        Bounds in the x, y, and z directions.
    increment_xy
        Increment size in x/y directions.
    increment_z
        Increment size in z direction.

    Returns
    -------
    :
        List of grid coordinates.
    """
    x_bound, y_bound, z_bound = bounds

    z_indices = np.arange(0, z_bound, increment_z)
    x_indices = np.arange(0, x_bound, increment_xy)
    y_indices = np.arange(0, y_bound, increment_xy)

    coordinates = [(x, y, z) for z in z_indices for x in x_indices for y in y_indices]

    return coordinates


def make_hex_grid_coordinates(
    bounds: Tuple[int, int, int],
    increment_xy: float,
    increment_z: float,
) -> List:
    """
    Get list of bounded (x, y, z) coordinates for hex grid.

    Coordinates are offset in sets of three z slices to form a face-centered
    cubic (FCC) packing.

    Parameters
    ----------
    bounds
        Bounds in the x, y, and z directions.
    increment_xy
        Increment size in x/y directions.
    increment_z
        Increment size in z direction.

    Returns
    -------
    :
        List of grid coordinates.
    """
    x_bound, y_bound, z_bound = bounds

    z_indices = np.arange(0, z_bound, increment_z)
    z_offsets = [(i % 3) for i in range(len(z_indices))]

    xy_indices, _ = hexalattice.create_hex_grid(
        nx=floor(x_bound / increment_xy),
        ny=floor(y_bound / increment_xy * sqrt(3)),
        min_diam=increment_xy,
        align_to_origin=False,
        do_plot=False,
    )

    x_offsets = [(increment_xy / 2) if z_offset == 1 else 0 for z_offset in z_offsets]
    y_offsets = [(increment_xy / 2) * sqrt(3) / 3 * z_offset for z_offset in z_offsets]

    coordinates = [
        (x + x_offset, y + y_offset, z)
        for z, x_offset, y_offset in zip(z_indices, x_offsets, y_offsets)
        for x, y in xy_indices
        if round(x + x_offset) < x_bound and round(y + y_offset) < y_bound
    ]

    return coordinates
