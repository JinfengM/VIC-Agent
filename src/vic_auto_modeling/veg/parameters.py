import csv
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


CLASSES = 11

VEG_LAI = [
    [3.400, 3.400, 3.500, 3.700, 4.000, 4.400, 4.400, 4.300, 4.200, 3.700, 3.500, 3.400],
    [3.400, 3.400, 3.500, 3.700, 4.000, 4.400, 4.400, 4.300, 4.200, 3.700, 3.500, 3.400],
    [1.680, 1.520, 1.680, 2.900, 4.900, 5.000, 5.000, 4.600, 3.440, 3.040, 2.160, 2.000],
    [1.680, 1.520, 1.680, 2.900, 4.900, 5.000, 5.000, 4.600, 3.440, 3.040, 2.160, 2.000],
    [1.680, 1.520, 1.680, 2.900, 4.900, 5.000, 5.000, 4.600, 3.440, 3.040, 2.160, 2.000],
    [1.680, 1.520, 1.680, 2.900, 4.900, 5.000, 5.000, 4.600, 3.440, 3.040, 2.160, 2.000],
    [2.000, 2.250, 2.950, 3.850, 3.750, 3.500, 3.550, 3.200, 3.300, 2.850, 2.600, 2.200],
    [2.000, 2.250, 2.950, 3.850, 3.750, 3.500, 3.550, 3.200, 3.300, 2.850, 2.600, 2.200],
    [2.000, 2.250, 2.950, 3.850, 3.750, 3.500, 3.550, 3.200, 3.300, 2.850, 2.600, 2.200],
    [2.000, 2.250, 2.950, 3.850, 3.750, 3.500, 3.550, 3.200, 3.300, 2.850, 2.600, 2.200],
    [0.500, 0.500, 0.500, 0.500, 1.500, 3.000, 4.500, 5.000, 2.500, 0.500, 0.500, 0.500],
]


@dataclass
class VegParamConfig:
    target_grid_path: str | Path = "output/fishnet/lishui_target_grid.gpkg"
    lucc_path: str | Path = "data/lucc"
    root_fraction_path: str | Path = "data/static/vegetation/fixed_veg_fra.txt"
    output_path: str | Path = "output/veg/output_area_veg.txt"
    cell_size: float = 0.081
    resample_ratio: float = 0.01

    @classmethod
    def from_run_context(cls, context):
        return cls(
            target_grid_path=context.output_path("fishnet", "lishui_target_grid.gpkg"),
            lucc_path=context.base_data_path / "lucc",
            root_fraction_path=context.base_data_path / "static/vegetation/fixed_veg_fra.txt",
            output_path=context.output_path("veg", "output_area_veg.txt"),
        )


def matlab_g(value):
    return f"{value:.5g}"


def read_root_fractions(path):
    root_fractions = []
    with Path(path).open("r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if row:
                root_fractions.append([float(value) for value in row[1:7]])

    if len(root_fractions) != CLASSES:
        raise ValueError(f"{path} must contain {CLASSES} vegetation classes.")

    return root_fractions


def read_resampled_lucc(lucc_path, bounds, fine_res):
    left, bottom, right, top = bounds
    width = math.ceil((right - left) / fine_res)
    height = math.ceil((top - bottom) / fine_res)
    right = left + width * fine_res
    bottom = top - height * fine_res
    transform = from_origin(left, top, fine_res, fine_res)

    with rasterio.open(lucc_path) as src:
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        data = src.read(
            1,
            window=window,
            out_shape=(height, width),
            resampling=Resampling.nearest,
            boundless=True,
            fill_value=src.nodata,
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


def calculate_vegetation_weights(target_grid, lucc_data, transform, nodata):
    weights_by_grid = {}

    for row in target_grid.itertuples(index=False):
        row_start, row_stop, col_start, col_stop = grid_window(
            row.geometry.bounds, transform, lucc_data.shape
        )
        values = lucc_data[row_start:row_stop, col_start:col_stop]
        if values.size == 0:
            weights_by_grid[int(row.vic_id)] = []
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

        total = grid_values.size
        if total == 0:
            weights_by_grid[int(row.vic_id)] = []
            continue

        counts = np.bincount(grid_values.astype(np.int16), minlength=CLASSES + 1)
        weights = []
        for veg_class in range(1, CLASSES + 1):
            ratio = counts[veg_class] / total
            if ratio > 0:
                weights.append((veg_class, ratio))
        weights_by_grid[int(row.vic_id)] = weights

    return weights_by_grid


def write_vic_vegetation_file(output_path, weights_by_grid, root_fractions):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="\n") as f:
        for vic_id in sorted(weights_by_grid):
            weights = weights_by_grid[vic_id]
            f.write(f"{vic_id}\t{len(weights)}\n")
            for veg_class, ratio in weights:
                class_index = veg_class - 1
                root_values = "\t".join(matlab_g(value) for value in root_fractions[class_index])
                lai_values = "\t".join(matlab_g(value) for value in VEG_LAI[class_index])
                f.write(f"\t{veg_class}\t{ratio:.6f}\t{root_values}\n")
                f.write(f"\t{lai_values}\n")


def create_veg_param(
    target_grid_path,
    lucc_path,
    root_fraction_path,
    output_path,
    cell_size=0.081,
    resample_ratio=0.01,
):
    target_grid = gpd.read_file(target_grid_path)
    target_grid = target_grid.sort_values(["vic_id"]).reset_index(drop=True)

    fine_res = cell_size * resample_ratio
    xmin, ymin, xmax, ymax = target_grid.total_bounds
    pad = cell_size / 2
    bounds = (xmin - pad, ymin - pad, xmax + pad, ymax + pad)

    lucc_data, transform, nodata = read_resampled_lucc(lucc_path, bounds, fine_res)
    root_fractions = read_root_fractions(root_fraction_path)
    weights_by_grid = calculate_vegetation_weights(target_grid, lucc_data, transform, nodata)
    write_vic_vegetation_file(output_path, weights_by_grid, root_fractions)

    return weights_by_grid


def create_veg_param_from_config(config):
    return create_veg_param(
        target_grid_path=config.target_grid_path,
        lucc_path=config.lucc_path,
        root_fraction_path=config.root_fraction_path,
        output_path=config.output_path,
        cell_size=config.cell_size,
        resample_ratio=config.resample_ratio,
    )
