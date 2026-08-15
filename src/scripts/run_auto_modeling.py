import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vic_auto_modeling.automation import create_modeling_inputs  # noqa: E402
from vic_auto_modeling.flow.routing import FILL_ALGORITHMS  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create VIC runoff and routing configuration files from boundary/outlet zips."
    )
    parser.add_argument("--run-id", required=True, help="Run directory name under runs/.")
    parser.add_argument("--boundary-zip", required=True, help="Zip file containing boundary shapefile.")
    parser.add_argument("--outlets-zip", required=True, help="Zip file containing outlet shapefile.")
    parser.add_argument("--project-root", default=PROJECT_ROOT, help="Repository root.")
    parser.add_argument(
        "--fill-algorithm",
        choices=sorted(FILL_ALGORITHMS),
        default="priority-flood",
        help="Depression filling algorithm for flow generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = create_modeling_inputs(
        run_id=args.run_id,
        boundary_zip=args.boundary_zip,
        outlets_zip=args.outlets_zip,
        project_root=args.project_root,
        fill_algorithm=args.fill_algorithm,
    )

    print("auto modeling input generation finished")
    print(f"run dir: {result.run_dir}")
    print(f"boundary: {result.boundary_path}")
    print(f"outlet: {result.outlet_path}")
    print(f"chanliu input: {result.chanliu_input}")
    print(f"rout input: {result.rout_input}")
    print(f"fill algorithm: {args.fill_algorithm}")


if __name__ == "__main__":
    main()
