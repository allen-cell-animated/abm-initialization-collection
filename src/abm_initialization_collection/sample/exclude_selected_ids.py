import pandas as pd
from prefect import task


@task
def exclude_selected_ids(samples: pd.DataFrame, exclude: list[int]) -> pd.DataFrame:
    """
    Filters samples to exclude given ids.

    Parameters
    ----------
    samples
        Sample cell ids and coordinates.
    exclude
        List of ids to exclude.

    Returns
    -------
    :
        Samples without excluded ids.
    """
    samples = samples[~samples.id.isin(exclude)]
    return samples.reset_index(drop=True)
