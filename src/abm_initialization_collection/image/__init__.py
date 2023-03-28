import importlib
import sys

from prefect import task

from .create_voronoi_image import create_voronoi_image
from .get_image_bounds import get_image_bounds
from .plot_contact_sheet import plot_contact_sheet
from .select_fov_images import select_fov_images

TASK_MODULES = [
    create_voronoi_image,
    get_image_bounds,
    plot_contact_sheet,
    select_fov_images,
]

for task_module in TASK_MODULES:
    MODULE_NAME = task_module.__name__
    module = importlib.import_module(f".{MODULE_NAME}", package=__name__)
    setattr(sys.modules[__name__], MODULE_NAME, task(getattr(module, MODULE_NAME)))
