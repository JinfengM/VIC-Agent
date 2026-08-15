import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd


INITIAL_DATA = [0.1, 0.02, 30, 0.95, 2, 0.1, 0.26, 0.76]


@dataclass
class SoilDatabaseConfig:
    target_grid_path: str | Path = "output/fishnet/lishui_target_grid.gpkg"
    top_soil_path: str | Path = "output/soil/top_soil.txt"
    sub_soil_path: str | Path = "output/soil/sub_soil.txt"
    elevation_path: str | Path = "output/elevation/area_elev.txt"
    soil_table_path: str | Path | None = "data/static/soil/soil_properties.xlsx"
    forcing_dir: str | Path = "output/forcing/forcing"
    output_path: str | Path = "output/soil/output_area_soil.txt"

    @classmethod
    def from_run_context(cls, context):
        return cls(
            target_grid_path=context.output_path("fishnet", "lishui_target_grid.gpkg"),
            top_soil_path=context.output_path("soil", "top_soil.txt"),
            sub_soil_path=context.output_path("soil", "sub_soil.txt"),
            elevation_path=context.output_path("elevation", "area_elev.txt"),
            soil_table_path=context.base_data_path / "static/soil/soil_properties.xlsx",
            forcing_dir=context.output_path("forcing", "forcing"),
            output_path=context.output_path("soil", "output_area_soil.txt"),
        )


def matlab_g(value):
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{float(value):.6g}"


def read_id_value_csv(path, value_name):
    df = pd.read_csv(path)
    return dict(zip(df["vic_id"].astype(int), df[value_name].astype(float)))


def read_target_grid(path):
    grid = gpd.read_file(path).sort_values("vic_id").reset_index(drop=True)
    return [(int(row.vic_id), float(row.lon), float(row.lat)) for row in grid.itertuples()]


def read_soil_table(path):
    df = pd.read_excel(path, header=None)
    soil_data = df.iloc[3:15].reset_index(drop=True)
    if len(soil_data) < 12:
        raise ValueError(f"{path} must contain 12 soil classes in rows 4..15")
    return soil_data


def soil_value(soil_data, soil_class, column):
    class_index = int(soil_class) - 1
    col_index = column - 1
    if class_index < 0 or class_index >= len(soil_data):
        raise ValueError(f"Soil class {int(soil_class)} is outside the property table")
    return float(soil_data.iat[class_index, col_index])


def forcing_path(forcing_dir, lon, lat):
    path = Path(forcing_dir) / f"forcing_{lat:.4f}_{lon:.4f}"
    if not path.exists():
        raise FileNotFoundError(f"Missing forcing file: {path}")
    return path


def annual_prcp(forcing_dir, lon, lat):
    values = []
    with forcing_path(forcing_dir, lon, lat).open("r") as f:
        for line in f:
            parts = line.split()
            if parts:
                values.append(float(parts[0]))
    if not values:
        raise ValueError(f"Forcing file is empty for lon={lon:.4f}, lat={lat:.4f}")
    return statistics.fmean(values) * 365


def default_row(vic_id, lon, lat, elevation, prcp):
    values = [
        "1",
        str(vic_id),
        f"{lat:.4f}",
        f"{lon:.4f}",
        *[matlab_g(value) for value in INITIAL_DATA[:5]],
        *[matlab_g(value) for value in [19.04, 19.04, 19.04]],
        *[matlab_g(value) for value in [424.8, 424.8, 424.8]],
        *[matlab_g(value) for value in [-9999, -9999, -9999]],
        *[matlab_g(value) for value in [10, 20, 50]],
        matlab_g(elevation),
        *[matlab_g(value) for value in INITIAL_DATA[5:8]],
        *[matlab_g(value) for value in [13, 4]],
        *[matlab_g(value) for value in [20, 20, 20]],
        *[matlab_g(value) for value in [0.3, 0.3, 0.3]],
        *[matlab_g(value) for value in [1430, 1430, 1430]],
        *[matlab_g(value) for value in [2685, 2685, 2685]],
        matlab_g(8),
        *[matlab_g(value) for value in [0.7, 0.7, 0.7]],
        *[matlab_g(value) for value in [0.3, 0.3, 0.3]],
        *[matlab_g(value) for value in [0.001, 0.0005]],
        f"{prcp:.4f}",
        *[matlab_g(value) for value in [0.0001, 0.0001, 0.0001, 0, 0]],
    ]
    return values


def soil_row(vic_id, lon, lat, elevation, prcp, top_class, sub_class, soil_data):
    col = lambda soil_class, column: soil_value(soil_data, soil_class, column)
    numeric_values = [
        *INITIAL_DATA[:5],
        col(top_class, 10),
        col(top_class, 11),
        col(sub_class, 12),
        col(top_class, 13),
        col(top_class, 14),
        col(sub_class, 15),
        col(top_class, 16),
        col(top_class, 17),
        col(sub_class, 18),
        col(top_class, 19),
        col(top_class, 20),
        col(sub_class, 21),
        elevation,
        *INITIAL_DATA[5:8],
        col(top_class, 26),
        col(top_class, 27),
        col(top_class, 28),
        col(top_class, 29),
        col(sub_class, 30),
        col(top_class, 31),
        col(top_class, 32),
        col(sub_class, 33),
        col(top_class, 34),
        col(top_class, 35),
        col(sub_class, 36),
        col(top_class, 37),
        col(top_class, 38),
        col(sub_class, 39),
        col(top_class, 40),
        col(top_class, 41),
        col(top_class, 42),
        col(sub_class, 43),
        col(top_class, 44),
        col(top_class, 45),
        col(sub_class, 46),
        col(top_class, 47),
        col(top_class, 48),
    ]

    return [
        "1",
        str(vic_id),
        f"{lat:.4f}",
        f"{lon:.4f}",
        *[matlab_g(value) for value in numeric_values],
        f"{prcp:.4f}",
        *[matlab_g(col(top_class, column)) for column in [50, 51, 52, 53]],
        "0",
    ]


def create_soil_database(
    target_grid_path,
    top_soil_path,
    sub_soil_path,
    elevation_path,
    soil_table_path,
    forcing_dir,
    output_path,
):
    target_rows = read_target_grid(target_grid_path)
    top_classes = read_id_value_csv(top_soil_path, "soil_type")
    sub_classes = read_id_value_csv(sub_soil_path, "soil_type")
    elevations = read_id_value_csv(elevation_path, "elevation")
    soil_data = read_soil_table(soil_table_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for vic_id, lon, lat in target_rows:
            top_class = top_classes[vic_id]
            sub_class = sub_classes[vic_id]
            elevation = elevations[vic_id]
            prcp = annual_prcp(forcing_dir, lon, lat)

            if top_class == 0 or sub_class == 0:
                row = default_row(vic_id, lon, lat, elevation, prcp)
            else:
                row = soil_row(
                    vic_id, lon, lat, elevation, prcp, top_class, sub_class, soil_data
                )
            f.write("\t".join(row) + "\t\n")

    return len(target_rows)


def default_soil_table():
    path = Path("data/static/soil/soil_properties.xlsx")
    if not path.exists():
        raise FileNotFoundError(f"No soil property .xlsx file found at {path}")
    return path


def create_soil_database_from_config(config):
    soil_table = Path(config.soil_table_path) if config.soil_table_path else default_soil_table()
    return create_soil_database(
        target_grid_path=config.target_grid_path,
        top_soil_path=config.top_soil_path,
        sub_soil_path=config.sub_soil_path,
        elevation_path=config.elevation_path,
        soil_table_path=soil_table,
        forcing_dir=config.forcing_dir,
        output_path=config.output_path,
    )
