from math import ceil, sqrt
from typing import Optional, Tuple, List, Union

from prefect import task
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.use("Agg")
mpl.rc("figure", dpi=200)
mpl.rc("font", size=8)
mpl.rc("axes", titlesize=10, titleweight="bold")


@task
def plot_contact_sheet(
    data: pd.DataFrame, reference: Optional[pd.DataFrame] = None
) -> mpl.figure.Figure:
    z_layers = sorted(data.z.unique() if reference is None else reference.z.unique())

    n_rows, n_cols, indices = separate_rows_cols(z_layers)
    fig, axs = make_subplots(n_rows, n_cols)

    max_id = int(data.id.max())
    min_id = int(data.id.min())
    min_x = data.x.min() if reference is None else reference.x.min()
    max_x = data.x.max() if reference is None else reference.x.max()
    min_y = data.y.min() if reference is None else reference.y.min()
    max_y = data.y.max() if reference is None else reference.y.max()

    for i, j, k in indices:
        ax = select_axes(axs, i, j, n_rows, n_cols)
        ax.set_xlim([min_x - 1, max_x + 1])
        ax.set_ylim([max_y + 1, min_y - 1])

        if k is None:
            ax.axis("off")
            continue

        z_slice = data[data.z == z_layers[k]]

        patches = [plt.Circle((x, y), radius=0.5) for x, y in zip(z_slice.x, z_slice.y)]
        collection = mpl.collections.PatchCollection(patches, cmap="jet")
        collection.set_array(z_slice.id)
        collection.set_clim([min_id, max_id])
        ax.add_collection(collection)

        if reference is not None:
            z_slice_reference = reference[reference.z == z_layers[k]]
            filtered = pd.merge(z_slice, z_slice_reference, how="outer", indicator=True)
            removed = filtered[filtered["_merge"] == "right_only"]

            patches = [plt.Circle((x, y), radius=0.3) for x, y in zip(removed.x, removed.y)]
            collection = mpl.collections.PatchCollection(patches, facecolor="#ccc")
            ax.add_collection(collection)

        if np.issubdtype(z_layers[k], np.integer):
            ax.set_title(f"z = {z_layers[k]}")
        else:
            ax.set_title(f"z = {z_layers[k]:.2f}")

        ax.set_aspect("equal", adjustable="box")
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    fig.tight_layout()

    return fig


def separate_rows_cols(items: list[str]) -> Tuple[int, int, List[Tuple[int, int, Optional[int]]]]:
    n_items = len(items)
    n_cols = ceil(sqrt(len(items)))
    n_rows = ceil(len(items) / n_cols)

    all_indices = [(i, j, i * n_cols + j) for i in range(n_rows) for j in range(n_cols)]
    indices = [(i, j, k if k < n_items else None) for i, j, k in all_indices]

    return n_rows, n_cols, indices


def make_subplots(
    n_rows: int, n_cols: int
) -> Tuple[mpl.figure.Figure, Union[mpl.axes.Axes, np.ndarray]]:
    plt.close("all")
    fig, axs = plt.subplots(n_rows, n_cols, sharex="all", sharey="all")
    return fig, axs


def select_axes(
    axs: Union[mpl.axes.Axes, np.ndarray], i: int, j: int, n_rows: int, n_cols: int
) -> mpl.axes.Axes:
    if n_rows == 1 and n_cols == 1:
        return axs
    if n_rows == 1:
        return axs[j]

    return axs[i, j]
