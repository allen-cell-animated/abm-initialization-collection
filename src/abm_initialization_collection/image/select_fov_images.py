import pandas as pd

from prefect import task


@task
def select_fov_images(metadata: pd.DataFrame, cells_per_fov: int, num_fovs: int) -> list[dict]:
    fov_count = 0
    fov_seg_paths = []

    for fov_seg_path, seg_group in metadata.groupby("fov_seg_path"):
        no_outliers = (seg_group.outlier == "No").all()
        only_m0 = (seg_group.cell_stage == "M0").all()

        if no_outliers and only_m0:
            if len(seg_group) == cells_per_fov:
                fov_count = fov_count + 1
                key = fov_seg_path.split("/")[-1].split("_")[0]

                fov_seg_paths.append(
                    {
                        "key": key,
                        "item": fov_seg_path,
                        "cell_ids": list(seg_group["this_cell_index"]),
                    }
                )

        if len(fov_seg_paths) == num_fovs:
            break

    return fov_seg_paths
