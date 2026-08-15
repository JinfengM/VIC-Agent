from dataclasses import dataclass
from pathlib import Path


START_YEAR = 2008
START_MONTH = 1
START_DAY = 1
END_YEAR = 2016
END_MONTH = 12
END_DAY = 31


@dataclass
class ModelInputConfig:
    project_root: str | Path = "."
    chanliu_template: str | Path = "data/input/chanliu_input.txt"
    rout_template: str | Path = "data/input/rout_input.txt"
    output_dir: str | Path = "output/model"
    forcing_prefix: str | Path = "output/forcing/forcing/forcing_"
    soil: str | Path = "output/soil/output_area_soil.txt"
    veglib: str | Path = "data/static/veglib.LDAS"
    veg_param: str | Path = "output/veg/output_area_veg.txt"
    flow_direction: str | Path = "output/flow/flow_1_8.txt"
    xmask: str | Path = "output/flow/output_area_mask.txt"
    fraction: str | Path = "output/fraction/fraction.txt"
    station: str | Path = "output/flow/area_stnloc.txt"
    unit_hydrograph: str | Path = "data/static/UH.all"
    start_year: int = START_YEAR
    start_month: int = START_MONTH
    start_day: int = START_DAY
    end_year: int = END_YEAR
    end_month: int = END_MONTH
    end_day: int = END_DAY

    @classmethod
    def from_run_context(cls, context):
        return cls(
            project_root=context.project_path,
            chanliu_template=context.base_data_path / "input/chanliu_input.txt",
            rout_template=context.base_data_path / "input/rout_input.txt",
            output_dir=context.output_path("model"),
            forcing_prefix=context.output_path("forcing", "forcing", "forcing_"),
            soil=context.output_path("soil", "output_area_soil.txt"),
            veglib=context.base_data_path / "static/veglib.LDAS",
            veg_param=context.output_path("veg", "output_area_veg.txt"),
            flow_direction=context.output_path("flow", "flow_1_8.txt"),
            xmask=context.output_path("flow", "output_area_mask.txt"),
            fraction=context.output_path("fraction", "fraction.txt"),
            station=context.output_path("flow", "area_stnloc.txt"),
            unit_hydrograph=context.base_data_path / "static/UH.all",
        )


def config_path(path):
    return Path(path).resolve().as_posix()


def split_comment(line):
    if "#" not in line:
        return line.rstrip("\r\n"), ""
    body, comment = line.rstrip("\r\n").split("#", 1)
    return body.rstrip(), " #" + comment


def replace_keyword_line(line, replacements):
    stripped = line.lstrip()
    if not stripped or stripped.startswith("#"):
        return line

    keyword = stripped.split()[0]
    if keyword not in replacements:
        return line

    _, comment = split_comment(line)
    return f"{keyword:<16}{replacements[keyword]}{comment}\n"


def render_chanliu(template_path, output_path, values):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = Path(template_path).read_text(encoding="utf-8", errors="replace").splitlines(True)
    rendered = [replace_keyword_line(line, values) for line in lines]
    output_path.write_text("".join(rendered), encoding="utf-8")


def render_rout(template_path, output_path, values):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = Path(template_path).read_text(encoding="utf-8", errors="replace").splitlines(True)
    rendered = []
    pending = None
    skip_active_flag = False

    marker_to_key = {
        "# NAME OF FLOW DIRECTION FILE": ("flow_direction", False),
        "# NAME OF XMASK FILE": ("xmask", True),
        "# NAME OF FRACTION FILE": ("fraction", True),
        "# NAME OF STATION FILE": ("station", False),
        "# PATH OF INPUT FILES AND PRECISION": ("input_prefix", False),
        "# PATH OF OUTPUT FILES": ("output_dir", False),
        "# NAME OF UNIT HYDROGRAPH FILE": ("unit_hydrograph", False),
    }

    for line in lines:
        stripped = line.strip()
        matched_marker = next((value for marker, value in marker_to_key.items() if stripped.startswith(marker)), None)
        if matched_marker:
            pending, skip_active_flag = matched_marker
            rendered.append(line)
            continue

        if stripped.startswith("# YEAR AND MONTH OF MODEL OUTPUT"):
            pending = "route_dates_1"
            rendered.append(line)
            continue

        if pending is not None and stripped and not stripped.startswith("#"):
            if skip_active_flag and stripped.lower() in {".true.", ".false."}:
                rendered.append(line)
                skip_active_flag = False
                continue

            if pending == "route_dates_1":
                rendered.append(values["route_period"] + "\n")
                pending = "route_dates_2"
            elif pending == "route_dates_2":
                rendered.append(values["route_period"] + "\n")
                pending = None
            else:
                rendered.append(values[pending] + "\n")
                pending = None
                skip_active_flag = False
            continue

        rendered.append(line)

    output_path.write_text("".join(rendered), encoding="utf-8")


def check_paths(paths):
    missing = [label for label, path in paths.items() if not Path(path).exists()]
    return missing


def check_forcing_prefix(prefix):
    matches = list(Path(prefix).parent.glob(Path(prefix).name + "*"))
    return bool(matches)


def create_model_inputs(config):
    project_root = Path(config.project_root).resolve()
    output_model_dir = project_root / config.output_dir
    chanliu_result_dir = output_model_dir / "chanliu_result"
    rout_result_dir = output_model_dir / "chanliu_result"
    chanliu_result_dir.mkdir(parents=True, exist_ok=True)
    rout_result_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "forcing_prefix": project_root / config.forcing_prefix,
        "soil": project_root / config.soil,
        "veglib": project_root / config.veglib,
        "veg_param": project_root / config.veg_param,
        "flow_direction": project_root / config.flow_direction,
        "xmask": project_root / config.xmask,
        "fraction": project_root / config.fraction,
        "station": project_root / config.station,
        "unit_hydrograph": project_root / config.unit_hydrograph,
    }

    chanliu_values = {
        "STARTYEAR": str(config.start_year),
        "STARTMONTH": f"{config.start_month:02d}",
        "STARTDAY": f"{config.start_day:02d}",
        "ENDYEAR": str(config.end_year),
        "ENDMONTH": f"{config.end_month:02d}",
        "ENDDAY": f"{config.end_day:02d}",
        "FORCING1": config_path(paths["forcing_prefix"]),
        "FORCEYEAR": str(config.start_year),
        "FORCEMONTH": f"{config.start_month:02d}",
        "FORCEDAY": f"{config.start_day:02d}",
        "SOIL": config_path(paths["soil"]),
        "VEGLIB": config_path(paths["veglib"]),
        "VEGPARAM": config_path(paths["veg_param"]),
        "RESULT_DIR": config_path(chanliu_result_dir),
    }

    route_period = (
        f"{config.start_year} {config.start_month:02d} {config.end_year} {config.end_month:02d}"
    )
    rout_values = {
        "flow_direction": config_path(paths["flow_direction"]),
        "xmask": config_path(paths["xmask"]),
        "fraction": config_path(paths["fraction"]),
        "station": config_path(paths["station"]),
        "input_prefix": config_path(chanliu_result_dir / "fluxes_"),
        "output_dir": config_path(rout_result_dir),
        "route_period": route_period,
        "unit_hydrograph": config_path(paths["unit_hydrograph"]),
    }

    chanliu_output = output_model_dir / "chanliu_input.txt"
    rout_output = output_model_dir / "rout_input.txt"
    render_chanliu(project_root / config.chanliu_template, chanliu_output, chanliu_values)
    render_rout(project_root / config.rout_template, rout_output, rout_values)

    required_paths = {
        "soil": paths["soil"],
        "veglib": paths["veglib"],
        "veg_param": paths["veg_param"],
        "flow_direction": paths["flow_direction"],
        "xmask": paths["xmask"],
        "fraction": paths["fraction"],
        "station": paths["station"],
        "unit_hydrograph": paths["unit_hydrograph"],
    }
    missing_required = check_paths(required_paths)
    if not check_forcing_prefix(paths["forcing_prefix"]):
        missing_required.append("forcing_prefix")

    return chanliu_output, rout_output, missing_required
