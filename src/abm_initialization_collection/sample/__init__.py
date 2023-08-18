"""Tasks for initialization from samples."""

from prefect import task

from .exclude_selected_ids import exclude_selected_ids
from .get_image_samples import get_image_samples
from .get_sample_indices import get_sample_indices
from .include_selected_ids import include_selected_ids
from .remove_edge_regions import remove_edge_regions
from .remove_unconnected_regions import remove_unconnected_regions
from .scale_sample_coordinates import scale_sample_coordinates

exclude_selected_ids = task(exclude_selected_ids)
get_image_samples = task(get_image_samples)
get_sample_indices = task(get_sample_indices)
include_selected_ids = task(include_selected_ids)
remove_edge_regions = task(remove_edge_regions)
remove_unconnected_regions = task(remove_unconnected_regions)
scale_sample_coordinates = task(scale_sample_coordinates)
