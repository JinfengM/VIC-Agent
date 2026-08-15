import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vic_auto_modeling.grid.fishnet import FishnetConfig, create_fishnets_from_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Create EPSG:4326 VIC fishnet grids.")
    parser.add_argument(
        "--boundary",
        default="data/boundary/Lishui-boundary.shp",
        help="Input basin or region boundary vector file.",
    )
    parser.add_argument(
        "--full-grid",
        default="output/fishnet/lishui_full_grid.gpkg",
        help="Output full bbox fishnet GeoPackage.",
    )
    parser.add_argument(
        "--target-grid",
        default="output/fishnet/lishui_target_grid.gpkg",
        help="Output target fishnet GeoPackage where fraction > 0.",
    )
    parser.add_argument(
        "--full-grid-shp",
        default="output/fishnet/lishui_full_grid.shp",
        help="Output full bbox fishnet shapefile.",
    )
    parser.add_argument(
        "--target-grid-shp",
        default="output/fishnet/lishui_target_grid.shp",
        help="Output target fishnet shapefile where fraction > 0.",
    )
    parser.add_argument(
        "--fraction",
        default="output/fraction/fraction.txt",
        help="Output fraction ASCII grid.",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=0.081,
        help="Grid cell size in degrees.",
    )
    args = parser.parse_args()

    config = FishnetConfig(
        boundary_path=args.boundary,
        full_grid_path=args.full_grid,
        target_grid_path=args.target_grid,
        full_grid_shp_path=args.full_grid_shp,
        target_grid_shp_path=args.target_grid_shp,
        fraction_path=args.fraction,
        cell_size=args.cell_size,
    )
    full_grid, target_grid = create_fishnets_from_config(config)

    partial = ((full_grid["fraction"] > 0) & (full_grid["fraction"] < 1)).sum()
    print(f"full_grid: {len(full_grid)} cells")
    print(f"target_grid: {len(target_grid)} cells")
    print(f"fraction=0: {(full_grid['fraction'] == 0).sum()} cells")
    print(f"0<fraction<1: {partial} cells")
    print(f"fraction=1: {(full_grid['fraction'] == 1).sum()} cells")
    print(f"fraction_ascii: {args.fraction}")


if __name__ == "__main__":
    main()
