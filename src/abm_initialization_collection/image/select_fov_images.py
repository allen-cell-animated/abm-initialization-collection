import numpy as np
import pandas as pd


def select_fov_images(
    metadata: pd.DataFrame, cells_per_fov: int, bins: list[int], counts: list[int]
) -> list[dict]:
    total_count = sum(counts)
    bin_counts = [0] * len(counts)
    fovs = []

    for fov_seg_path, seg_group in metadata.groupby("fov_seg_path"):
        no_outliers = (seg_group.outlier != "Yes").all()
        only_m0 = (seg_group.cell_stage == "M0").all()

        if no_outliers and only_m0 and len(seg_group) == cells_per_fov:
            mean = seg_group["MEM_shape_volume"].mean()
            bin_index = np.digitize(mean, bins)

            if bin_counts[bin_index] < counts[bin_index - 1]:
                bin_counts[bin_index] = bin_counts[bin_index] + 1
                key = fov_seg_path.split("/")[-1].split("_")[0]
                fovs.append(
                    {
                        "key": key,
                        "item": fov_seg_path,
                        "cell_ids": list(seg_group["this_cell_index"]),
                    }
                )

            if sum(bin_counts) >= total_count:
                break

    return fovs
