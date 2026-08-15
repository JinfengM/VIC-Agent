import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vic_auto_modeling.flow.routing import (  # noqa: E402
    FILL_ALGORITHMS,
    FlowConfig,
    create_flow_from_config,
)


def main():
    parser = argparse.ArgumentParser(description="Create D8 flow direction and routing distance files.")
    parser.add_argument("--target-grid", default="output/fishnet/lishui_target_grid.gpkg")
    parser.add_argument("--full-grid", default="output/fishnet/lishui_full_grid.gpkg")
    parser.add_argument("--elevation", default="output/elevation/area_elev.txt")
    parser.add_argument("--flow-output", default="output/flow/flow_1_8.txt")
    parser.add_argument("--distance-output", default="output/flow/output_area_mask.txt")
    parser.add_argument("--fishnet-prj", default="output/flow/fishnet_prj.txt")
    parser.add_argument(
        "--fill-algorithm",
        choices=sorted(FILL_ALGORITHMS),
        default="priority-flood",
        help="Depression filling algorithm. Defaults to priority-flood.",
    )
    parser.add_argument("--cellsize", type=float, default=0.081)
    args = parser.parse_args()

    config = FlowConfig(
        target_grid_path=args.target_grid,
        full_grid_path=args.full_grid,
        elevation_path=args.elevation,
        flow_output_path=args.flow_output,
        distance_output_path=args.distance_output,
        fishnet_prj_path=args.fishnet_prj,
        fill_algorithm=args.fill_algorithm,
        cellsize=args.cellsize,
    )
    flow, distances = create_flow_from_config(config)

    nonzero_flow = int(np.count_nonzero(flow))
    nonzero_distance = int(np.count_nonzero(distances))
    print("flow processing finished")
    print(f"shape: {flow.shape[0]} rows x {flow.shape[1]} cols")
    print(f"nonzero flow cells: {nonzero_flow}")
    print(f"nonzero distance cells: {nonzero_distance}")
    print(f"total distance km: {distances.sum() / 1000:.6f}")
    print(f"fill algorithm: {args.fill_algorithm}")
    print(f"flow output: {args.flow_output}")
    print(f"distance output: {args.distance_output}")
    print(f"fishnet prj: {args.fishnet_prj}")


if __name__ == "__main__":
    main()
