from typing import Tuple

from prefect import task
import numpy as np
import pandas as pd
from skimage import measure
from scipy.spatial import distance


@task
def remove_unconnected_regions(
    samples: pd.DataFrame, unconnected_threshold: float, unconnected_filter: str
) -> pd.DataFrame:
    """
    Removes unconnected regions.

    Parameters
    ----------
    samples
        Sample cell ids and coordinates.
    threshold
        Distance for removing unconnected regions.
    filter
        Filter type for assigning unconnected coordinates.

    Returns
    -------
    :
        Samples with unconnected regions removed.
    """

    if unconnected_filter == "connectivity":
        return remove_unconnected_by_connectivity(samples)

    if unconnected_filter == "distance":
        return remove_unconnected_by_distance(samples, unconnected_threshold)

    raise ValueError(f"invalid filter type {unconnected_filter}")


def remove_unconnected_by_connectivity(samples: pd.DataFrame) -> pd.DataFrame:
    """
    Removes unconnected regions based on simple connectivity.

    Parameters
    ----------
    samples
        Sample cell ids and coordinates.

    Returns
    -------
    :
        Samples with unconnected regions removed.
    """
    minimums = get_sample_minimums(samples)
    maximums = get_sample_maximums(samples)

    array = convert_to_integer_array(samples, minimums, maximums)

    array_connected = np.zeros(array.shape, dtype="int")
    labels = measure.label(array, connectivity=1)

    # Sort labeled regions by size.
    regions = np.bincount(labels.flatten())[1:]
    regions_sorted = sorted(
        [(i + 1, n) for i, n in enumerate(regions)],
        key=lambda tup: tup[1],
        reverse=True,
    )

    # Iterate through all regions and copy the largest connected region to array.
    ids_added = set()
    for index, _ in regions_sorted:
        cell_id = list(set(array[labels == index]))[0]

        if cell_id not in ids_added:
            array_connected[labels == index] = cell_id
            ids_added.add(cell_id)
        else:
            print(f"Skipping unconnected region for cell id {cell_id}")

    # Convert back to dataframe.
    samples_connected = convert_to_dataframe(array_connected, minimums)
    return samples_connected.sort_values(by=["id", "x", "y", "z"]).reset_index(drop=True)


def remove_unconnected_by_distance(samples: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Removes unconnected regions based on distance.

    Parameters
    ----------
    samples
        Sample cell ids and coordinates.
    threshold
        Distance for removing unconnected regions.

    Returns
    -------
    :
        Samples with unconnected regions removed.
    """
    all_connected: list = []

    # Iterate through each id and filter out samples above the distance threshold.
    for cell_id, group in samples.groupby("id"):
        coordinates = group[["x", "y", "z"]].to_numpy()
        distances = [
            get_minimum_distance(np.array([coordinate]), coordinates) for coordinate in coordinates
        ]
        connected = [
            (cell_id, x, y, z)
            for distance, (x, y, z) in zip(distances, coordinates)
            if distance < threshold
        ]
        all_connected = all_connected + connected

    # Convert back to dataframe.
    samples_connected = pd.DataFrame(all_connected, columns=["id", "x", "y", "z"])
    return samples_connected.sort_values(by=["id", "x", "y", "z"]).reset_index(drop=True)


def get_sample_minimums(samples: pd.DataFrame) -> Tuple[int, int, int]:
    """
    Gets minimums in x, y, and z directions for samples.

    Parameters
    ----------
    samples
        Sample cell ids and coordinates.

    Returns
    -------
        Tuple of minimums.
    """
    min_x = min(samples.x)
    min_y = min(samples.y)
    min_z = min(samples.z)
    minimums = (min_x, min_y, min_z)
    return minimums


def get_sample_maximums(samples: pd.DataFrame) -> Tuple[int, int, int]:
    """
    Gets maximums in x, y, and z directions for samples.

    Parameters
    ----------
    samples
        Sample cell ids and coordinates.

    Returns
    -------
        Tuple of maximums.
    """
    max_x = max(samples.x)
    max_y = max(samples.y)
    max_z = max(samples.z)
    maximums = (max_x, max_y, max_z)
    return maximums


def convert_to_integer_array(
    samples: pd.DataFrame,
    minimums: Tuple[int, int, int],
    maximums: Tuple[int, int, int],
) -> np.ndarray:
    """
    Converts ids and coordinate samples to integer array.

    Parameters
    ----------
    samples
        Sample cell ids and coordinates.
    minimums
        Minimums in x, y, and z directions.
    maximums
        Maximums in x, y, and z directions.

    Returns
    -------
    :
        Array of ids.
    """
    length, width, height = np.subtract(maximums, minimums).astype("int32")
    array = np.zeros((height + 1, width + 1, length + 1), dtype="int32")

    coordinates = samples[["x", "y", "z"]].values - minimums
    array[tuple(np.transpose(np.flip(coordinates, axis=1)))] = samples.id

    return array


def convert_to_dataframe(array: np.ndarray, minimums: Tuple[int, int, int]) -> pd.DataFrame:
    """
    Converts integer array to ids and coordinate samples.

    Parameters
    ----------
    array
        Integer array of ids.
    minimums
        Minimums in x, y, and z directions.

    Returns
    -------
    :
        Dataframe of ids and coordinates.
    """
    min_x, min_y, min_z = minimums

    samples = [
        (array[z, y, x], x + min_x, y + min_y, z + min_z) for z, y, x in zip(*np.where(array != 0))
    ]

    return pd.DataFrame(samples, columns=["id", "x", "y", "z"])


def get_minimum_distance(source: np.ndarray, targets: np.ndarray) -> float:
    """
    Get the minimum distance from point to array of points.

    Parameters
    ----------
    source
        Coordinates of source point with shape (1, 3)
    targets
        Coordinates for N target points with shape (3, N)

    Returns
    -------
    :
        Minimum distance between source and targets.
    """
    distances = distance.cdist(source, targets)
    return np.min(distances[distances != 0])
