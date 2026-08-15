import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vic_auto_modeling.veg.parameters import VegParamConfig, create_veg_param_from_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Create VIC vegetation parameter file.")
    parser.add_argument(
        "--target-grid",
        default="output/fishnet/lishui_target_grid.gpkg",
        help="Input target grid containing vic_id.",
    )
    parser.add_argument(
        "--lucc",
        default="data/lucc",
        help="Input LUCC raster dataset.",
    )
    parser.add_argument(
        "--root-fraction",
        default="data/static/vegetation/fixed_veg_fra.txt",
        help="Root fraction lookup table.",
    )
    parser.add_argument(
        "--output",
        default="output/veg/output_area_veg.txt",
        help="Output VIC vegetation parameter file.",
    )
    parser.add_argument("--cell-size", type=float, default=0.081)
    parser.add_argument("--resample-ratio", type=float, default=0.01)
    args = parser.parse_args()

    config = VegParamConfig(
        target_grid_path=args.target_grid,
        lucc_path=args.lucc,
        root_fraction_path=args.root_fraction,
        output_path=args.output,
        cell_size=args.cell_size,
        resample_ratio=args.resample_ratio,
    )
    weights_by_grid = create_veg_param_from_config(config)

    empty = sum(1 for weights in weights_by_grid.values() if not weights)
    partial = sum(1 for weights in weights_by_grid.values() if sum(r for _, r in weights) < 0.999999)
    print(f"vegetation grids: {len(weights_by_grid)}")
    print(f"empty vegetation grids: {empty}")
    print(f"grids with summed 1..11 ratio < 1: {partial}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
