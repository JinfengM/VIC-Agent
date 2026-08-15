import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vic_auto_modeling.model.inputs import (  # noqa: E402
    END_DAY,
    END_MONTH,
    END_YEAR,
    START_DAY,
    START_MONTH,
    START_YEAR,
    ModelInputConfig,
    create_model_inputs,
)


def main():
    parser = argparse.ArgumentParser(description="Generate VIC runoff and routing input files.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--chanliu-template", default="data/input/chanliu_input.txt")
    parser.add_argument("--rout-template", default="data/input/rout_input.txt")
    parser.add_argument("--output-dir", default="output/model")
    parser.add_argument("--forcing-prefix", default="output/forcing/forcing/forcing_")
    parser.add_argument("--soil", default="output/soil/output_area_soil.txt")
    parser.add_argument("--veglib", default="data/static/veglib.LDAS")
    parser.add_argument("--veg-param", default="output/veg/output_area_veg.txt")
    parser.add_argument("--flow-direction", default="output/flow/flow_1_8.txt")
    parser.add_argument("--xmask", default="output/flow/output_area_mask.txt")
    parser.add_argument("--fraction", default="output/fraction/fraction.txt")
    parser.add_argument("--station", default="output/flow/area_stnloc.txt")
    parser.add_argument("--unit-hydrograph", default="data/static/UH.all")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--start-month", type=int, default=START_MONTH)
    parser.add_argument("--start-day", type=int, default=START_DAY)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--end-month", type=int, default=END_MONTH)
    parser.add_argument("--end-day", type=int, default=END_DAY)
    args = parser.parse_args()

    config = ModelInputConfig(
        project_root=args.project_root,
        chanliu_template=args.chanliu_template,
        rout_template=args.rout_template,
        output_dir=args.output_dir,
        forcing_prefix=args.forcing_prefix,
        soil=args.soil,
        veglib=args.veglib,
        veg_param=args.veg_param,
        flow_direction=args.flow_direction,
        xmask=args.xmask,
        fraction=args.fraction,
        station=args.station,
        unit_hydrograph=args.unit_hydrograph,
        start_year=args.start_year,
        start_month=args.start_month,
        start_day=args.start_day,
        end_year=args.end_year,
        end_month=args.end_month,
        end_day=args.end_day,
    )
    chanliu_output, rout_output, missing_required = create_model_inputs(config)

    print("model input generation finished")
    print(f"chanliu input: {chanliu_output}")
    print(f"rout input: {rout_output}")
    if missing_required:
        print(f"missing required files: {', '.join(missing_required)}")


if __name__ == "__main__":
    main()
