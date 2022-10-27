from prefect import task
import pandas as pd


@task
def include_selected_ids(samples: pd.DataFrame, include: list[int]) -> pd.DataFrame:
    """
    Filters samples to include given ids.

    Parameters
    ----------
    samples
        Sample cell ids and coordinates.
    include
        List of ids to include.

    Returns
    -------
    :
        Samples with included ids.
    """
    samples = samples[samples.id.isin(include)]
    return samples.reset_index(drop=True)
