import math
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds
from shapely.geometry import box


@dataclass
class AverageElevationConfig:
    target_grid_path: str | Path = "output/fishnet/lishui_target_grid.gpkg"
    dem_path: str | Path = "data/static/dem/cndemalb30.tif"
    output_path: str | Path = "output/elevation/area_elev.txt"
    cell_size: float = 0.081

    @classmethod
    def from_run_context(cls, context):
        return cls(
            target_grid_path=context.output_path("fishnet", "lishui_target_grid.gpkg"),
            dem_path=context.base_data_path / "static/dem/cndemalb30.tif",
            output_path=context.output_path("elevation", "area_elev.txt"),
        )


def expanded_bounds(bounds, pad):
    xmin, ymin, xmax, ymax = bounds
    return xmin - pad, ymin - pad, xmax + pad, ymax + pad


def read_dem_window(dem_path, grid_4326, cell_size):
    with rasterio.open(dem_path) as src:
        pad = cell_size / 2
        bbox_4326 = box(*expanded_bounds(grid_4326.total_bounds, pad))
        bbox_dem = gpd.GeoSeries([bbox_4326], crs=grid_4326.crs).to_crs(src.crs).iloc[0]
        window = from_bounds(*bbox_dem.bounds, transform=src.transform)
        window = Window(
            math.floor(window.col_off),
            math.floor(window.row_off),
            math.ceil(window.width),
            math.ceil(window.height),
        )
        data = src.read(1, window=window, masked=False)
        transform = src.window_transform(window)
        nodata = src.nodata
        dem_crs = src.crs

    return data, transform, nodata, dem_crs


def grid_window(bounds, transform, shape):
    left, bottom, right, top = bounds
    inv = ~transform
    col_start, row_start = inv * (left, top)
    col_stop, row_stop = inv * (right, bottom)

    row_start = max(0, math.floor(row_start))
    col_start = max(0, math.floor(col_start))
    row_stop = min(shape[0], math.ceil(row_stop))
    col_stop = min(shape[1], math.ceil(col_stop))

    return row_start, row_stop, col_start, col_stop


def calculate_average_elevation(target_grid, dem_data, transform, nodata):
    results = []
    for row in target_grid.itertuples(index=False):
        row_start, row_stop, col_start, col_stop = grid_window(
            row.geometry.bounds, transform, dem_data.shape
        )
        values = dem_data[row_start:row_stop, col_start:col_stop]
        if values.size == 0:
            results.append((int(row.vic_id), np.nan))
            continue

        window_transform = transform * Affine.translation(col_start, row_start)
        mask = geometry_mask(
            [row.geometry],
            out_shape=values.shape,
            transform=window_transform,
            invert=True,
        )
        grid_values = values[mask]
        if nodata is not None:
            grid_values = grid_values[grid_values != nodata]
        grid_values = grid_values[np.isfinite(grid_values)]

        mean_elevation = float(grid_values.mean()) if grid_values.size else np.nan
        results.append((int(row.vic_id), mean_elevation))

    return results


def write_area_elev(output_path, results):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("vic_id,elevation\n")
        for vic_id, elevation in results:
            value = "nan" if np.isnan(elevation) else f"{elevation:.6f}"
            f.write(f"{vic_id},{value}\n")


def create_average_elevation(target_grid_path, dem_path, output_path, cell_size=0.081):
    grid_4326 = gpd.read_file(target_grid_path).to_crs("EPSG:4326")
    grid_4326 = grid_4326.sort_values("vic_id").reset_index(drop=True)

    dem_data, transform, nodata, dem_crs = read_dem_window(dem_path, grid_4326, cell_size)
    grid_dem = grid_4326.to_crs(dem_crs)
    results = calculate_average_elevation(grid_dem, dem_data, transform, nodata)
    write_area_elev(output_path, results)
    return results


def create_average_elevation_from_config(config):
    return create_average_elevation(
        target_grid_path=config.target_grid_path,
        dem_path=config.dem_path,
        output_path=config.output_path,
        cell_size=config.cell_size,
    )
