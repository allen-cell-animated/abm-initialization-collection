import numpy as np
import pandas as pd


def filter_coordinate_bounds(coordinates: list, radius: float, center: bool = True) -> pd.DataFrame:
    """
    Filters list for coordinates with given radius.

    Parameters
    ----------
    coordinates
        List of (x, y, z) coordinates.
    radius
        Maximum valid radius of coordinate.

    Returns
    -------
    :
        Filtered list of coordinates.
    """
    filtered_coordinates = []
    x_center, y_center, _ = np.array(coordinates).mean(axis=0)

    for x, y, z in coordinates:
        coordinate_radius = (x - x_center) ** 2 + (y - y_center) ** 2
        if coordinate_radius <= radius**2:
            if center:
                filtered_coordinates.append((x - x_center, y - y_center, z))
            else:
                filtered_coordinates.append((x, y, z))

    return pd.DataFrame(filtered_coordinates, columns=["x", "y", "z"])
