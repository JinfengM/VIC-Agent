import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from vic_auto_modeling.core.run_context import RunContext
from vic_auto_modeling.elevation.average import (
    AverageElevationConfig,
    create_average_elevation_from_config,
)
from vic_auto_modeling.flow.routing import FlowConfig, create_flow_from_config
from vic_auto_modeling.flow.station_location import (
    StationLocationConfig,
    create_station_location_from_config,
)
from vic_auto_modeling.forcing.generation import ForcingConfig, create_forcing_from_config
from vic_auto_modeling.grid.fishnet import FishnetConfig, create_fishnets_from_config
from vic_auto_modeling.model.inputs import ModelInputConfig, create_model_inputs
from vic_auto_modeling.soil.database import (
    SoilDatabaseConfig,
    create_soil_database_from_config,
)
from vic_auto_modeling.soil.majority import (
    SoilMajorityConfig,
    create_soil_majority_from_config,
)
from vic_auto_modeling.veg.parameters import VegParamConfig, create_veg_param_from_config


@dataclass
class AutoModelResult:
    run_dir: Path
    boundary_path: Path
    outlet_path: Path
    chanliu_input: Path
    rout_input: Path
    missing_required: list[str]


def _safe_extract_zip(zip_path, output_dir):
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file does not exist: {zip_path}")
    if zip_path.suffix.lower() != ".zip":
        raise ValueError(f"Only .zip archives are supported. Got: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"File is not a valid zip archive: {zip_path}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    root = output_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe zip member path: {member.filename}")
        archive.extractall(output_dir)


def _find_single_shapefile(directory):
    directory = Path(directory)
    shapefiles = sorted(
        path
        for path in directory.rglob("*.shp")
        if "__MACOSX" not in path.parts and not path.name.startswith(".")
    )
    if not shapefiles:
        raise FileNotFoundError(f"No .shp file found in {directory}")
    if len(shapefiles) > 1:
        names = ", ".join(str(path.relative_to(directory)) for path in shapefiles)
        raise ValueError(f"Expected one .shp file in {directory}, found: {names}")
    return shapefiles[0]


def prepare_run_inputs(context, boundary_zip, outlets_zip):
    boundary_dir = context.input_path("boundary")
    outlets_dir = context.input_path("outlets")
    _safe_extract_zip(boundary_zip, boundary_dir)
    _safe_extract_zip(outlets_zip, outlets_dir)
    return _find_single_shapefile(boundary_dir), _find_single_shapefile(outlets_dir)


def create_modeling_inputs(
    run_id,
    boundary_zip,
    outlets_zip,
    project_root=".",
    fill_algorithm="priority-flood",
):
    context = RunContext(run_id, project_root=project_root).ensure_dirs()
    boundary_path, outlet_path = prepare_run_inputs(context, boundary_zip, outlets_zip)

    fishnet_config = FishnetConfig.from_run_context(context)
    fishnet_config.boundary_path = boundary_path
    create_fishnets_from_config(fishnet_config)

    create_veg_param_from_config(VegParamConfig.from_run_context(context))
    create_average_elevation_from_config(AverageElevationConfig.from_run_context(context))
    create_soil_majority_from_config(SoilMajorityConfig.from_run_context(context))
    create_forcing_from_config(ForcingConfig.from_run_context(context))
    create_soil_database_from_config(SoilDatabaseConfig.from_run_context(context))
    flow_config = FlowConfig.from_run_context(context)
    flow_config.fill_algorithm = fill_algorithm
    create_flow_from_config(flow_config)

    station_config = StationLocationConfig.from_run_context(context)
    station_config.outlet_path = outlet_path
    create_station_location_from_config(station_config)

    chanliu_input, rout_input, missing_required = create_model_inputs(
        ModelInputConfig.from_run_context(context)
    )
    if missing_required:
        raise FileNotFoundError(
            "Model input generation is missing required files: "
            + ", ".join(missing_required)
        )

    return AutoModelResult(
        run_dir=context.run_dir,
        boundary_path=boundary_path,
        outlet_path=outlet_path,
        chanliu_input=chanliu_input,
        rout_input=rout_input,
        missing_required=missing_required,
    )
