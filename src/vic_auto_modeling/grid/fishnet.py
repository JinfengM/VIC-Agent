import math
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box


@dataclass
class FishnetConfig:
    boundary_path: str | Path = "data/boundary/Lishui-boundary.shp"
    full_grid_path: str | Path = "output/fishnet/lishui_full_grid.gpkg"
    target_grid_path: str | Path = "output/fishnet/lishui_target_grid.gpkg"
    full_grid_shp_path: str | Path = "output/fishnet/lishui_full_grid.shp"
    target_grid_shp_path: str | Path = "output/fishnet/lishui_target_grid.shp"
    fraction_path: str | Path = "output/fraction/fraction.txt"
    cell_size: float = 0.081
    grid_crs: str = "EPSG:4326"
    area_crs: str = "EPSG:6933"

    @classmethod
    def from_run_context(cls, context):
        return cls(
            boundary_path=context.input_path("boundary", "boundary.shp"),
            full_grid_path=context.output_path("fishnet", "lishui_full_grid.gpkg"),
            target_grid_path=context.output_path("fishnet", "lishui_target_grid.gpkg"),
            full_grid_shp_path=context.output_path("fishnet", "lishui_full_grid.shp"),
            target_grid_shp_path=context.output_path("fishnet", "lishui_target_grid.shp"),
            fraction_path=context.output_path("fraction", "fraction.txt"),
        )


def _aligned_bounds(bounds, cell_size):
    xmin, ymin, xmax, ymax = bounds
    ncols = math.ceil((xmax - xmin) / cell_size)
    nrows = math.ceil((ymax - ymin) / cell_size)
    return (
        xmin,
        ymin,
        xmin + ncols * cell_size,
        ymin + nrows * cell_size,
    )


def _write_layer(gdf, output_path, layer):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, layer=layer, driver="GPKG")


def _write_shapefile(gdf, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="ESRI Shapefile")


def _write_fraction_ascii(full_grid, output_path, cell_size):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ncols = int(full_grid["col"].max()) + 1
    nrows = int(full_grid["row"].max()) + 1
    xllcorner, yllcorner, _, _ = full_grid.total_bounds

    matrix = [["0" for _ in range(ncols)] for _ in range(nrows)]
    for row in full_grid.itertuples(index=False):
        matrix[int(row.row)][int(row.col)] = f"{row.fraction:.9g}"

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"ncols         {ncols}\n")
        f.write(f"nrows         {nrows}\n")
        f.write(f"xllcorner     {xllcorner:.12g}\n")
        f.write(f"yllcorner     {yllcorner:.12g}\n")
        f.write(f"cellsize      {cell_size:.12g}\n")
        f.write("NODATA_value  0\n")
        for values in matrix:
            f.write(" ".join(values) + " \n")


def create_fishnets(
    boundary_path,
    full_grid_path,
    target_grid_path,
    full_grid_shp_path,
    target_grid_shp_path,
    fraction_path,
    cell_size=0.081,
    grid_crs="EPSG:4326",
    area_crs="EPSG:6933",
):
    boundary = gpd.read_file(boundary_path)
    if boundary.crs is None:
        raise ValueError("Boundary file has no CRS. Define it before creating VIC grids.")

    boundary = boundary.to_crs(grid_crs)
    basin_geom = boundary.geometry.union_all()
    xmin, ymin, xmax, ymax = _aligned_bounds(boundary.total_bounds, cell_size)

    polygons = []
    rows = []
    cols = []
    grid_ids = []
    lons = []
    lats = []

    nrows = round((ymax - ymin) / cell_size)
    ncols = round((xmax - xmin) / cell_size)

    grid_id = 1
    for row in range(nrows):
        y_top = ymax - row * cell_size
        y_bottom = y_top - cell_size
        for col in range(ncols):
            x_left = xmin + col * cell_size
            x_right = x_left + cell_size
            geom = box(x_left, y_bottom, x_right, y_top)
            centroid = geom.centroid

            polygons.append(geom)
            rows.append(row)
            cols.append(col)
            grid_ids.append(grid_id)
            lons.append(centroid.x)
            lats.append(centroid.y)

            grid_id += 1

    full_grid = gpd.GeoDataFrame(
        {
            "grid_id": grid_ids,
            "row": rows,
            "col": cols,
            "lon": lons,
            "lat": lats,
        },
        geometry=polygons,
        crs=grid_crs,
    )

    area_grid = full_grid.to_crs(area_crs)
    area_basin = gpd.GeoSeries([basin_geom], crs=grid_crs).to_crs(area_crs).iloc[0]
    intersections = area_grid.geometry.intersection(area_basin)
    full_grid["fraction"] = (intersections.area / area_grid.geometry.area).clip(0, 1)
    full_grid["run_cell"] = (full_grid["fraction"] > 0).astype(int)

    target_grid = full_grid.loc[full_grid["run_cell"] == 1].copy()
    full_grid = full_grid.sort_values(["row", "col"]).reset_index(drop=True)
    target_grid = target_grid.sort_values(["row", "col"]).reset_index(drop=True)
    full_grid["vic_id"] = range(1, len(full_grid) + 1)
    target_grid["vic_id"] = range(1, len(target_grid) + 1)

    _write_layer(full_grid, full_grid_path, "full_grid")
    _write_layer(target_grid, target_grid_path, "target_grid")
    _write_shapefile(full_grid, full_grid_shp_path)
    _write_shapefile(target_grid, target_grid_shp_path)
    _write_fraction_ascii(full_grid, fraction_path, cell_size)

    return full_grid, target_grid


def create_fishnets_from_config(config):
    return create_fishnets(
        boundary_path=config.boundary_path,
        full_grid_path=config.full_grid_path,
        target_grid_path=config.target_grid_path,
        full_grid_shp_path=config.full_grid_shp_path,
        target_grid_shp_path=config.target_grid_shp_path,
        fraction_path=config.fraction_path,
        cell_size=config.cell_size,
        grid_crs=config.grid_crs,
        area_crs=config.area_crs,
    )
