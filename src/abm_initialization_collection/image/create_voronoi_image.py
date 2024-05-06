from math import floor

import numpy as np
from bioio import BioImage
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt


def create_voronoi_image(image: BioImage, channel: int, iterations: int, height: int) -> np.ndarray:
    array = image.get_image_data("ZYX", T=0, C=channel)

    # Create artificial boundary for voronoi.
    mask = create_boundary_mask(array, iterations)
    lower_bound, upper_bound = get_mask_bounds(array, height)
    mask_id = np.iinfo(array.dtype).max
    array[mask == 0] = mask_id
    mask[:lower_bound, :, :] = 0
    mask[upper_bound:, :, :] = 0

    # Calculate voronoi on bounded array.
    zslice, yslice, xslice = get_array_slices(mask)
    voronoi = calculate_voronoi_array(array[zslice, yslice, xslice])

    # Remove masking ids.
    array[zslice, yslice, xslice] = voronoi
    array[mask == 0] = 0
    array[array == mask_id] = 0

    return array


def create_boundary_mask(array: np.ndarray, iterations: int) -> np.ndarray:
    """
    Creates filled boundary mask around regions in array.

    Parameters
    ----------
    array
        Image array.
    iterations
        Number of boundary estimation steps.

    Returns
    -------
    :
        Boundary mask array.
    """

    mask = np.zeros(array.shape, dtype="uint8")
    mask[array != 0] = 1

    # Expand using binary dilation to create a border.
    binary_dilation(mask, output=mask, iterations=iterations)

    # Fill holes in the mask in each z slice.
    for z in range(array.shape[0]):
        binary_fill_holes(mask[z, :, :], output=mask[z, :, :])

    return mask


def get_mask_bounds(array: np.ndarray, target_range: int) -> tuple[int, int]:
    """
    Calculates the indices of z axis bounds with given target range.

    If the current range between z axis bounds (the minimum and maximum
    indices in the z axis where there exist non-zero entries) is wider than
    the target range, the current bound indices are returned.

    Parameters
    ----------
    array
        Image array.
    target_range
        Target distance between bounds.

    Returns
    -------
    :
        Lower and upper bound indices.
    """

    lower_bound, upper_bound = np.where(np.any(array, axis=(1, 2)))[0][[0, -1]]
    current_range = upper_bound - lower_bound + 1

    if current_range < target_range:
        height_delta = target_range - current_range
        lower_offset = floor(height_delta / 2)
        upper_offset = height_delta - lower_offset
        lower_bound = lower_bound - lower_offset
        upper_bound = upper_bound + upper_offset + 1
    else:
        upper_bound = upper_bound + 1

    return (lower_bound, upper_bound)


def get_array_slices(array: np.ndarray) -> tuple[slice, slice, slice]:
    """
    Calculate bounding box slices around binary array.

    Parameters
    ----------
    array
        Binary array.

    Returns
    -------
    :
        Slices in the z, y, and x directions.
    """

    zsize, ysize, xsize = array.shape

    zmin, zmax = np.where(np.any(array, axis=(1, 2)))[0][[0, -1]]
    ymin, ymax = np.where(np.any(array, axis=(0, 2)))[0][[0, -1]]
    xmin, xmax = np.where(np.any(array, axis=(0, 1)))[0][[0, -1]]

    zslice = slice(max(zmin - 1, 0), min(zmax + 2, zsize))
    yslice = slice(max(ymin - 1, 0), min(ymax + 2, ysize))
    xslice = slice(max(xmin - 1, 0), min(xmax + 2, xsize))

    slices = (zslice, yslice, xslice)
    return slices


def calculate_voronoi_array(array: np.ndarray) -> np.ndarray:
    """
    Calculates voronoi on image array using distance transform.

    Parameters
    ----------
    array
        Image array.

    Returns
    -------
    :
        Voronoi array.
    """

    distances = distance_transform_edt(array == 0, return_distances=False, return_indices=True)
    distances = distances.astype("uint16", copy=False)

    coordinates_z = distances[0].flatten()
    coordinates_y = distances[1].flatten()
    coordinates_x = distances[2].flatten()
    voronoi = array[coordinates_z, coordinates_y, coordinates_x].reshape(array.shape)

    return voronoi
