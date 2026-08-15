import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vic_auto_modeling.soil.majority import (  # noqa: E402
    SoilMajorityConfig,
    create_soil_majority_from_config,
)


def main():
    parser = argparse.ArgumentParser(description="Create top/sub soil majority files by VIC target grid.")
    parser.add_argument(
        "--target-grid",
        default="output/fishnet/lishui_target_grid.gpkg",
        help="Input target grid containing vic_id.",
    )
    parser.add_argument("--top-soil", default="data/soil/usda_top.img")
    parser.add_argument("--sub-soil", default="data/soil/usda_sub.img")
    parser.add_argument("--top-output", default="output/soil/top_soil.txt")
    parser.add_argument("--sub-output", default="output/soil/sub_soil.txt")
    parser.add_argument("--cell-size", type=float, default=0.081)
    parser.add_argument("--resample-ratio", type=float, default=0.01)
    args = parser.parse_args()

    config = SoilMajorityConfig(
        target_grid_path=args.target_grid,
        top_soil_path=args.top_soil,
        sub_soil_path=args.sub_soil,
        top_output_path=args.top_output,
        sub_output_path=args.sub_output,
        cell_size=args.cell_size,
        resample_ratio=args.resample_ratio,
    )
    top_results, sub_results = create_soil_majority_from_config(config)

    top_values = [soil_type for _, soil_type in top_results]
    sub_values = [soil_type for _, soil_type in sub_results]
    print(f"soil grids: {len(top_results)}")
    print(f"top soil classes: {sorted(set(top_values))}")
    print(f"sub soil classes: {sorted(set(sub_values))}")
    print(f"top output: {args.top_output}")
    print(f"sub output: {args.sub_output}")


if __name__ == "__main__":
    main()
