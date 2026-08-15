import csv
import time
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np


THRESHOLD = 0.0405
STATION_COUNT = 3
START_YEAR = 2008
END_YEAR = 2016


@dataclass
class ForcingConfig:
    target_grid_path: str | Path = "output/fishnet/lishui_target_grid.gpkg"
    cmads_path: str | Path = "data/forcing/CMADS1.1.shp"
    meteo_root: str | Path = "data/static/meteo"
    output_dir: str | Path = "output/forcing"
    start_year: int = START_YEAR
    end_year: int = END_YEAR
    station_count: int = STATION_COUNT

    @classmethod
    def from_run_context(cls, context):
        return cls(
            target_grid_path=context.output_path("fishnet", "lishui_target_grid.gpkg"),
            cmads_path=context.base_data_path / "forcing/CMADS1.1.shp",
            meteo_root=context.base_data_path / "static/meteo",
            output_dir=context.output_path("forcing"),
        )


def station_to_file_id(station):
    return str(station).strip().replace("-", ".", 1)


def expected_day_count(start_year, end_year):
    return sum(366 if year % 4 == 0 else 365 for year in range(start_year, end_year + 1))


def first_nonempty_line(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return line
    raise ValueError(f"{path} is empty")


def load_numeric(path):
    path = Path(path)
    first_line = first_nonempty_line(path)
    delimiter = "," if "," in first_line else None
    skip_header = 1 if any(ch.isalpha() for ch in first_line) else 0
    data = np.genfromtxt(path, delimiter=delimiter, skip_header=skip_header)
    if data.size == 0:
        raise ValueError(f"{path} has no numeric data")
    return data


def load_station_table(path):
    path = Path(path)
    first_line = first_nonempty_line(path)
    delimiter = "," if "," in first_line else None
    skip_header = 1 if any(ch.isalpha() for ch in first_line) else 0
    data = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=skip_header,
        dtype=str,
        encoding="utf-8",
    )
    data = np.atleast_2d(data)
    if data.size == 0 or data.shape[1] < 3:
        raise ValueError(f"{path} must have station_id, lon, lat columns")
    return {
        "ids": data[:, 0],
        "lons": data[:, 1].astype(float),
        "lats": data[:, 2].astype(float),
    }


def extract_stations(cmads_path, target_grid_path, output_station_path):
    stations = gpd.read_file(cmads_path)
    target_grid = gpd.read_file(target_grid_path).to_crs(stations.crs)
    selected = stations[stations.geometry.within(target_grid.geometry.union_all())].copy()
    selected = selected.sort_values(["Latitude", "Longitude"]).reset_index(drop=True)
    selected.insert(0, "FID", range(len(selected)))

    output_station_path = Path(output_station_path)
    output_station_path.parent.mkdir(parents=True, exist_ok=True)
    selected[["FID", "Station", "Latitude", "Longitude", "Elevation"]].to_csv(
        output_station_path, index=False, lineterminator="\n"
    )
    return selected


def write_station_files(stations, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        [
            station_to_file_id(row.Station),
            f"{float(row.Longitude):.15f}",
            f"{float(row.Latitude):.15f}",
        ]
        for row in stations.itertuples(index=False)
    ]

    station_files = {}
    for key, filename in {
        "pcp": "1_PrepStation.txt",
        "temp": "1_TempStation.txt",
        "wind": "1_WindStation.txt",
    }.items():
        path = output_dir / filename
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=" ")
            writer.writerows(rows)
        station_files[key] = path
    return station_files


def write_grid_file(target_grid_path, output_grid_path):
    grid = gpd.read_file(target_grid_path).sort_values("vic_id").reset_index(drop=True)
    output_grid_path = Path(output_grid_path)
    output_grid_path.parent.mkdir(parents=True, exist_ok=True)
    with output_grid_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in grid.itertuples(index=False):
            f.write(f"{int(row.vic_id)},{int(row.vic_id)},{row.lon:.6f},{row.lat:.6f}\n")
    return output_grid_path


def station_id_to_filename(station_id):
    return f"{str(station_id).strip()}.txt"


def load_station_series(data_dir, station_id, cache):
    key = (Path(data_dir), str(station_id))
    if key not in cache:
        path = key[0] / station_id_to_filename(station_id)
        if not path.exists():
            raise FileNotFoundError(f"Missing station data file: {path}")
        cache[key] = load_numeric(path)
    return cache[key]


def sort_by_distance(stations, lon, lat):
    distances = np.sqrt((stations["lons"] - lon) ** 2 + (stations["lats"] - lat) ** 2)
    order = np.argsort(distances)
    return {
        "ids": stations["ids"][order],
        "distances": distances[order],
    }


def validate_length(data, expected_days, label):
    if len(data) < expected_days:
        raise ValueError(f"{label} has {len(data)} rows, expected at least {expected_days}")
    return data[:expected_days]


def idw_weights(sorted_stations, station_count):
    selected = {
        "ids": sorted_stations["ids"][:station_count],
        "distances": sorted_stations["distances"][:station_count],
    }
    weights = 1.0 / (selected["distances"] ** 2)
    weights = weights / weights.sum()
    return selected, weights


def interpolate_one_column(sorted_stations, station_count, data_dir, expected_days, label, cache):
    if sorted_stations["distances"][0] < THRESHOLD:
        data = np.ravel(load_station_series(data_dir, sorted_stations["ids"][0], cache))
        return validate_length(data, expected_days, label)

    selected, weights = idw_weights(sorted_stations, station_count)
    columns = []
    for station_id in selected["ids"]:
        data = np.ravel(load_station_series(data_dir, station_id, cache))
        columns.append(validate_length(data, expected_days, label))
    return np.column_stack(columns) @ weights


def interpolate_temperature(sorted_stations, station_count, data_dir, expected_days, cache):
    if sorted_stations["distances"][0] < THRESHOLD:
        data = load_station_series(data_dir, sorted_stations["ids"][0], cache)
        data = np.atleast_2d(data)
        if data.shape[1] < 2:
            raise ValueError("Temperature data must have at least 2 columns")
        data = validate_length(data, expected_days, "temperature")
        return data[:, 0], data[:, 1]

    selected, weights = idw_weights(sorted_stations, station_count)
    tmax_columns = []
    tmin_columns = []
    for station_id in selected["ids"]:
        data = load_station_series(data_dir, station_id, cache)
        data = np.atleast_2d(data)
        if data.shape[1] < 2:
            raise ValueError("Temperature data must have at least 2 columns")
        data = validate_length(data, expected_days, "temperature")
        tmax_columns.append(data[:, 0])
        tmin_columns.append(data[:, 1])
    return np.column_stack(tmax_columns) @ weights, np.column_stack(tmin_columns) @ weights


def forcing_filename(lon, lat):
    return f"forcing_{lat:.4f}_{lon:.4f}"


def interpolate_forcing(
    grid_file,
    station_files,
    meteo_root,
    output_dir,
    start_year=START_YEAR,
    end_year=END_YEAR,
    station_count=STATION_COUNT,
):
    meteo_root = Path(meteo_root)
    data_dirs = {
        "pcp": meteo_root / "pcp_data_vic",
        "temp": meteo_root / "temp_data_vic",
        "wind": meteo_root / "wind_data_vic",
    }
    for label, data_dir in data_dirs.items():
        if not data_dir.exists():
            raise FileNotFoundError(f"Missing {label} data directory: {data_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = np.atleast_2d(load_numeric(grid_file))
    p_stations = load_station_table(station_files["pcp"])
    t_stations = load_station_table(station_files["temp"])
    w_stations = load_station_table(station_files["wind"])
    expected_days = expected_day_count(start_year, end_year)
    cache = {}

    started_at = time.time()
    for index, row in enumerate(grid, start=1):
        lon = row[2]
        lat = row[3]

        sorted_p = sort_by_distance(p_stations, lon, lat)
        sorted_t = sort_by_distance(t_stations, lon, lat)
        sorted_w = sort_by_distance(w_stations, lon, lat)

        pcp = interpolate_one_column(
            sorted_p, station_count, data_dirs["pcp"], expected_days, "precipitation", cache
        )
        tmax, tmin = interpolate_temperature(
            sorted_t, station_count, data_dirs["temp"], expected_days, cache
        )
        wind = interpolate_one_column(
            sorted_w, station_count, data_dirs["wind"], expected_days, "wind", cache
        )

        output_data = np.column_stack((pcp, tmax, tmin, wind))
        output_path = output_dir / forcing_filename(lon, lat)
        np.savetxt(
            output_path,
            output_data,
            fmt=["%.6f", "%.5f", "%.5f", "%.5f"],
            delimiter="\t",
        )
        print(f"{index}/{len(grid)} wrote {output_path} elapsed={time.time() - started_at:.2f}s")

    return len(grid), expected_days


def create_forcing(
    target_grid_path,
    cmads_path,
    meteo_root,
    output_dir,
    start_year=START_YEAR,
    end_year=END_YEAR,
    station_count=STATION_COUNT,
):
    station_path = Path(output_dir) / "station.txt"
    grid_path = Path(output_dir) / "output_area.txt"
    forcing_dir = Path(output_dir) / "forcing"

    stations = extract_stations(cmads_path, target_grid_path, station_path)
    station_files = write_station_files(stations, output_dir)
    write_grid_file(target_grid_path, grid_path)
    if forcing_dir.exists():
        for path in forcing_dir.glob("forcing_*"):
            path.unlink()
    grid_count, day_count = interpolate_forcing(
        grid_file=grid_path,
        station_files=station_files,
        meteo_root=meteo_root,
        output_dir=forcing_dir,
        start_year=start_year,
        end_year=end_year,
        station_count=station_count,
    )
    return len(stations), grid_count, day_count, forcing_dir


def create_forcing_from_config(config):
    return create_forcing(
        target_grid_path=config.target_grid_path,
        cmads_path=config.cmads_path,
        meteo_root=config.meteo_root,
        output_dir=config.output_dir,
        start_year=config.start_year,
        end_year=config.end_year,
        station_count=config.station_count,
    )
