import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vic_auto_modeling.forcing.generation import (  # noqa: E402
    END_YEAR,
    START_YEAR,
    STATION_COUNT,
    ForcingConfig,
    create_forcing_from_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Create VIC forcing files for target grid.")
    parser.add_argument("--target-grid", default="output/fishnet/lishui_target_grid.gpkg")
    parser.add_argument("--cmads", default="data/forcing/CMADS1.1.shp")
    parser.add_argument("--meteo-root", default="data/static/meteo")
    parser.add_argument("--output-dir", default="output/forcing")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--station-count", type=int, default=STATION_COUNT)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ForcingConfig(
        target_grid_path=args.target_grid,
        cmads_path=args.cmads,
        meteo_root=args.meteo_root,
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        station_count=args.station_count,
    )
    station_count, grid_count, day_count, forcing_dir = create_forcing_from_config(config)
    print("forcing processing finished")
    print(f"stations: {station_count}")
    print(f"grids: {grid_count}")
    print(f"days: {day_count}")
    print(f"forcing_dir: {forcing_dir}")


if __name__ == "__main__":
    main()
