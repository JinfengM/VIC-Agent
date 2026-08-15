from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd


@dataclass
class StationLocationConfig:
    outlet_path: str | Path = "data/outlets/outlet.shp"
    target_grid_path: str | Path = "output/fishnet/lishui_target_grid.gpkg"
    full_grid_path: str | Path = "output/fishnet/lishui_full_grid.gpkg"
    output_path: str | Path = "output/flow/area_stnloc.txt"

    @classmethod
    def from_run_context(cls, context):
        return cls(
            outlet_path=context.input_path("outlets", "outlet.shp"),
            target_grid_path=context.output_path("fishnet", "lishui_target_grid.gpkg"),
            full_grid_path=context.output_path("fishnet", "lishui_full_grid.gpkg"),
            output_path=context.output_path("flow", "area_stnloc.txt"),
        )


def station_name(index, station_count):
    if station_count == 1:
        return "luanx"
    return f"{index:05d}"


def find_containing_cells(points, target_grid, full_grid):
    target_union = target_grid.geometry.union_all()
    valid_points = points[points.geometry.within(target_union)].copy()

    records = []
    full_max_row = int(full_grid["row"].max())
    full_sindex = full_grid.sindex

    for point_index, point in enumerate(valid_points.geometry, start=1):
        candidate_indexes = list(full_sindex.query(point, predicate="intersects"))
        containing = full_grid.iloc[candidate_indexes]
        containing = containing[containing.geometry.contains(point) | containing.geometry.touches(point)]
        if containing.empty:
            continue

        cell = containing.sort_values(["row", "col"]).iloc[0]
        col_number = int(cell["col"]) + 1
        row_number = full_max_row - int(cell["row"]) + 1
        records.append((point_index, col_number, row_number))

    return records


def write_station_location(output_path, records):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    station_count = len(records)

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for index, col_number, row_number in records:
            f.write(f"1 {station_name(index, station_count):<5s}   {col_number}  {row_number}   -9999\n")
        f.write("NONE\n")


def create_station_location(outlet_path, target_grid_path, full_grid_path, output_path):
    target_grid = gpd.read_file(target_grid_path)
    full_grid = gpd.read_file(full_grid_path).to_crs(target_grid.crs)
    outlets = gpd.read_file(outlet_path).to_crs(target_grid.crs)

    records = find_containing_cells(outlets, target_grid, full_grid)
    write_station_location(output_path, records)
    return records


def create_station_location_from_config(config):
    return create_station_location(
        outlet_path=config.outlet_path,
        target_grid_path=config.target_grid_path,
        full_grid_path=config.full_grid_path,
        output_path=config.output_path,
    )
