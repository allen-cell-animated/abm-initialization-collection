import importlib
import sys

from prefect import task

from .filter_coordinate_bounds import filter_coordinate_bounds
from .make_grid_coordinates import make_grid_coordinates

TASK_MODULES = [
    filter_coordinate_bounds,
    make_grid_coordinates,
]

for task_module in TASK_MODULES:
    MODULE_NAME = task_module.__name__
    module = importlib.import_module(f".{MODULE_NAME}", package=__name__)
    setattr(sys.modules[__name__], MODULE_NAME, task(getattr(module, MODULE_NAME)))
