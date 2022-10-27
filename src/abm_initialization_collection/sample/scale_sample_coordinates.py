from prefect import task
import pandas as pd


@task
def scale_sample_coordinates(
    samples: pd.DataFrame, coordinate_type: str, resolution: float, scale_xy: float, scale_z: float
) -> pd.DataFrame:
    if coordinate_type == "absolute":
        samples["x"] = samples["x"] * scale_xy
        samples["y"] = samples["y"] * scale_xy
        samples["z"] = samples["z"] * scale_z
    elif coordinate_type == "step":
        samples["x"] = (samples["x"] / round(resolution / scale_xy)).astype("int32")
        samples["y"] = (samples["y"] / round(resolution / scale_xy)).astype("int32")
        samples["z"] = (samples["z"] / round(resolution / scale_z)).astype("int32")

    return samples
