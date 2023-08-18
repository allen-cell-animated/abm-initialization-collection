"""Tasks for initialization from coordinates."""

from prefect import task

from .filter_coordinate_bounds import filter_coordinate_bounds
from .make_grid_coordinates import make_grid_coordinates

filter_coordinate_bounds = task(filter_coordinate_bounds)
make_grid_coordinates = task(make_grid_coordinates)
