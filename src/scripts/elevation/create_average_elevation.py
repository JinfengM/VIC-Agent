import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vic_auto_modeling.elevation.average import (  # noqa: E402
    AverageElevationConfig,
    create_average_elevation_from_config,
)


def main():
    parser = argparse.ArgumentParser(description="Create average elevation file by VIC target grid.")
    parser.add_argument(
        "--target-grid",
        default="output/fishnet/lishui_target_grid.gpkg",
        help="Input target grid containing vic_id.",
    )
    parser.add_argument(
        "--dem",
        default="data/static/dem/cndemalb30.tif",
        help="Input DEM raster.",
    )
    parser.add_argument(
        "--output",
        default="output/elevation/area_elev.txt",
        help="Output CSV file: vic_id,elevation.",
    )
    parser.add_argument("--cell-size", type=float, default=0.081)
    args = parser.parse_args()

    config = AverageElevationConfig(
        target_grid_path=args.target_grid,
        dem_path=args.dem,
        output_path=args.output,
        cell_size=args.cell_size,
    )
    results = create_average_elevation_from_config(config)

    values = np.array([value for _, value in results], dtype=float)
    valid = values[np.isfinite(values)]
    print(f"elevation grids: {len(results)}")
    print(f"valid elevations: {valid.size}")
    print(f"nan elevations: {len(results) - valid.size}")
    print(f"min elevation: {valid.min():.6f}")
    print(f"max elevation: {valid.max():.6f}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
