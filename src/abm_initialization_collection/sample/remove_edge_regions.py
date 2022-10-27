from prefect import task
import pandas as pd


@task
def remove_edge_regions(
    samples: pd.DataFrame, edge_threshold: int, edge_padding: float
) -> pd.DataFrame:
    """
    Removes regions at edges.

    Parameters
    ----------
    samples
        Sample cell ids and coordinates.
    threshold
        Number of edge positions per axis needed to assign edge region.
    padding
        Distance from axis limits to assign edge positions.

    Returns
    -------
    :
        Samples with edge cells removed.
    """

    # Get ids of cell at edge.
    x_edge_ids = find_edge_ids("x", samples, edge_threshold, edge_padding)
    y_edge_ids = find_edge_ids("y", samples, edge_threshold, edge_padding)

    # Filter samples for cells not at edge.
    all_edge_ids = set(x_edge_ids + y_edge_ids)
    samples_filtered = samples[~samples["id"].isin(all_edge_ids)]

    return samples_filtered.reset_index(drop=True)


def find_edge_ids(axis: str, samples: pd.DataFrame, threshold: float, padding: float) -> list[int]:
    """
    Finds ids of cells with voxels touching edges of given axis.

    Parameters
    ----------
    axis : {'x', 'y', 'z'}
        The name of axis to check.
    samples
        Sample cell ids and coordinates.
    threshold
        Number of edge positions per axis needed to assign edge region.
    padding
        Distance from axis limits to assign edge positions.

    Returns
    -------
    :
        List of edge cell ids.
    """

    # Get min and max coordinate for given axis.
    axis_min = samples[axis].min() + padding
    axis_max = samples[axis].max() - padding

    # Check for cell ids located at edges.
    edges = samples.groupby("id").apply(
        lambda g: len(g[(g[axis] <= axis_min) | (g[axis] >= axis_max)])
    )
    edge_ids = edges[edges > threshold]

    return list(edge_ids.index)
