import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vic_auto_modeling.flow.station_location import (  # noqa: E402
    StationLocationConfig,
    create_station_location_from_config,
)


def main():
    parser = argparse.ArgumentParser(description="Create VIC routing outlet station location file.")
    parser.add_argument("--outlets", default="data/outlets/outlet.shp")
    parser.add_argument("--target-grid", default="output/fishnet/lishui_target_grid.gpkg")
    parser.add_argument("--full-grid", default="output/fishnet/lishui_full_grid.gpkg")
    parser.add_argument("--output", default="output/flow/area_stnloc.txt")
    args = parser.parse_args()

    config = StationLocationConfig(
        outlet_path=args.outlets,
        target_grid_path=args.target_grid,
        full_grid_path=args.full_grid,
        output_path=args.output,
    )
    records = create_station_location_from_config(config)

    print("station location processing finished")
    print(f"valid outlets: {len(records)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
