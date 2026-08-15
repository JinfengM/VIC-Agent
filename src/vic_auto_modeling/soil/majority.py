import math
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine, from_origin
from rasterio.windows import from_bounds


SOIL_CLASS_MAX = 13


@dataclass
class SoilMajorityConfig:
    target_grid_path: str | Path = "output/fishnet/lishui_target_grid.gpkg"
    top_soil_path: str | Path = "data/soil/usda_top.img"
    sub_soil_path: str | Path = "data/soil/usda_sub.img"
    top_output_path: str | Path = "output/soil/top_soil.txt"
    sub_output_path: str | Path = "output/soil/sub_soil.txt"
    cell_size: float = 0.081
    resample_ratio: float = 0.01

    @classmethod
    def from_run_context(cls, context):
        return cls(
            target_grid_path=context.output_path("fishnet", "lishui_target_grid.gpkg"),
            top_soil_path=context.base_data_path / "soil/usda_top.img",
            sub_soil_path=context.base_data_path / "soil/usda_sub.img",
            top_output_path=context.output_path("soil", "top_soil.txt"),
            sub_output_path=context.output_path("soil", "sub_soil.txt"),
        )


def read_resampled_soil(raster_path, bounds, fine_res):
    left, bottom, right, top = bounds
    width = math.ceil((right - left) / fine_res)
    height = math.ceil((top - bottom) / fine_res)
    right = left + width * fine_res
    bottom = top - height * fine_res
    transform = from_origin(left, top, fine_res, fine_res)

    with rasterio.open(raster_path) as src:
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        data = src.read(
            1,
            window=window,
            out_shape=(height, width),
            resampling=Resampling.nearest,
            boundless=True,
            fill_value=src.nodata if src.nodata is not None else 0,
        )
        nodata = src.nodata

    return data, transform, nodata


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


def majority_soil_by_grid(target_grid, soil_data, transform, nodata):
    results = []

    for row in target_grid.itertuples(index=False):
        row_start, row_stop, col_start, col_stop = grid_window(
            row.geometry.bounds, transform, soil_data.shape
        )
        values = soil_data[row_start:row_stop, col_start:col_stop]
        if values.size == 0:
            results.append((int(row.vic_id), 0))
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

        grid_values = grid_values[(grid_values >= 1) & (grid_values <= SOIL_CLASS_MAX)]
        if grid_values.size == 0:
            results.append((int(row.vic_id), 0))
            continue

        counts = np.bincount(grid_values.astype(np.int16), minlength=SOIL_CLASS_MAX + 1)
        majority_class = int(np.argmax(counts[1:]) + 1)
        results.append((int(row.vic_id), majority_class))

    return results


def write_soil_file(output_path, results):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("vic_id,soil_type\n")
        for vic_id, soil_type in results:
            f.write(f"{vic_id},{soil_type}\n")


def create_soil_majority(
    target_grid_path,
    top_soil_path,
    sub_soil_path,
    top_output_path,
    sub_output_path,
    cell_size=0.081,
    resample_ratio=0.01,
):
    target_grid = gpd.read_file(target_grid_path)
    target_grid = target_grid.sort_values("vic_id").reset_index(drop=True)

    fine_res = cell_size * resample_ratio
    xmin, ymin, xmax, ymax = target_grid.total_bounds
    pad = cell_size / 2
    bounds = (xmin - pad, ymin - pad, xmax + pad, ymax + pad)

    top_data, top_transform, top_nodata = read_resampled_soil(top_soil_path, bounds, fine_res)
    sub_data, sub_transform, sub_nodata = read_resampled_soil(sub_soil_path, bounds, fine_res)

    top_results = majority_soil_by_grid(target_grid, top_data, top_transform, top_nodata)
    sub_results = majority_soil_by_grid(target_grid, sub_data, sub_transform, sub_nodata)

    write_soil_file(top_output_path, top_results)
    write_soil_file(sub_output_path, sub_results)

    return top_results, sub_results


def create_soil_majority_from_config(config):
    return create_soil_majority(
        target_grid_path=config.target_grid_path,
        top_soil_path=config.top_soil_path,
        sub_soil_path=config.sub_soil_path,
        top_output_path=config.top_output_path,
        sub_output_path=config.sub_output_path,
        cell_size=config.cell_size,
        resample_ratio=config.resample_ratio,
    )
