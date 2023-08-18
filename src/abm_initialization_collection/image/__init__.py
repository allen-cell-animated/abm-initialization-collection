"""Tasks for initialization from images."""

from prefect import task

from .create_voronoi_image import create_voronoi_image
from .get_image_bounds import get_image_bounds
from .plot_contact_sheet import plot_contact_sheet
from .select_fov_images import select_fov_images

create_voronoi_image = task(create_voronoi_image)
get_image_bounds = task(get_image_bounds)
plot_contact_sheet = task(plot_contact_sheet)
select_fov_images = task(select_fov_images)
