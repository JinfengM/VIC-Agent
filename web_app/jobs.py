import csv
import json
import shutil
import threading
import time
import traceback
from pathlib import Path

from vic_auto_modeling.automation import create_modeling_inputs
from vic_auto_modeling.core.run_context import RunContext
from vic_auto_modeling.vic_bo import VicCalibrationConfig, run_calibration
from vic_auto_modeling.vic_runner import run_vic_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_JOBS = {}


def context_for(run_id, project_root=PROJECT_ROOT):
    return RunContext(run_id, project_root=project_root).ensure_dirs()


def status_path(context, name):
    return context.log_path(f"{name}_status.json")


def history_path(context):
    return context.log_path("calibration_history.csv")


def read_status(context, name):
    path = status_path(context, name)
    if not path.exists():
        return {"state": "idle"}
    return json.loads(path.read_text(encoding="utf-8"))


def write_status(context, name, **values):
    path = status_path(context, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "state": values.pop("state", "running"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    status.update(values)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def save_uploaded_file(uploaded_file, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as fp:
        fp.write(uploaded_file.getbuffer())
    return target


def has_model_inputs(run_id, project_root=PROJECT_ROOT):
    context = RunContext(run_id, project_root=project_root)
    return (
        context.output_path("model", "chanliu_input.txt").exists()
        and context.output_path("model", "rout_input.txt").exists()
    )


def has_model_output(run_id, station_name="luanx", project_root=PROJECT_ROOT):
    context = RunContext(run_id, project_root=project_root)
    return context.output_path("model", "chanliu_result", f"{station_name}.month").exists()


def start_job(key, target, *args, **kwargs):
    thread = _JOBS.get(key)
    if thread and thread.is_alive():
        return False
    thread = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
    _JOBS[key] = thread
    thread.start()
    return True


def run_auto_modeling_job(run_id, boundary_zip, outlets_zip, fill_algorithm, project_root=PROJECT_ROOT):
    context = context_for(run_id, project_root=project_root)
    write_status(context, "auto_modeling", state="running", message="自动建模运行中")
    try:
        result = create_modeling_inputs(
            run_id=run_id,
            boundary_zip=boundary_zip,
            outlets_zip=outlets_zip,
            project_root=project_root,
            fill_algorithm=fill_algorithm,
        )
        write_status(
            context,
            "auto_modeling",
            state="complete",
            message="自动建模完成",
            run_dir=str(result.run_dir),
            boundary=str(result.boundary_path),
            outlet=str(result.outlet_path),
            chanliu_input=str(result.chanliu_input),
            rout_input=str(result.rout_input),
        )
    except Exception as exc:
        write_status(
            context,
            "auto_modeling",
            state="failed",
            message=str(exc),
            traceback=traceback.format_exc(),
        )


def run_vic_job(run_id, source_dir, processes, project_root=PROJECT_ROOT):
    context = context_for(run_id, project_root=project_root)
    write_status(context, "vic_run", state="running", message="VIC 模型运行中")
    try:
        result = run_vic_model(
            run_id=run_id,
            project_root=project_root,
            source_dir=source_dir,
            processes=processes,
            stream_output=False,
        )
        state = "complete" if result.returncode == 0 else "failed"
        write_status(
            context,
            "vic_run",
            state=state,
            message="VIC 模型运行完成" if result.returncode == 0 else "VIC 模型运行失败",
            returncode=result.returncode,
            stdout=str(result.stdout_path),
            stderr=str(result.stderr_path),
        )
    except Exception as exc:
        write_status(
            context,
            "vic_run",
            state="failed",
            message=str(exc),
            traceback=traceback.format_exc(),
        )


def run_calibration_job(
    run_id,
    observation_file,
    iterations,
    source_dir,
    processes,
    station_name,
    random_state,
    xi,
    project_root=PROJECT_ROOT,
):
    context = context_for(run_id, project_root=project_root)
    history = history_path(context)
    history.parent.mkdir(parents=True, exist_ok=True)
    if history.exists():
        history.unlink()

    write_status(
        context,
        "calibration",
        state="running",
        message="参数率定运行中",
        iteration=0,
        iterations=iterations,
    )

    def on_iteration(result):
        row = {
            "iteration": result.iteration,
            "nse": result.nse,
            "best_nse": result.best_nse,
            **result.params,
        }
        write_header = not history.exists()
        with history.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        write_status(
            context,
            "calibration",
            state="running",
            message="参数率定运行中",
            iteration=result.iteration,
            iterations=iterations,
            nse=result.nse,
            best_nse=result.best_nse,
            params=result.params,
            aligned_csv=str(result.aligned_csv),
            plot_path=str(result.plot_path) if result.plot_path else None,
        )

    try:
        config = VicCalibrationConfig(
            run_id=run_id,
            project_root=project_root,
            source_dir=source_dir,
            processes=processes,
            observation_file=observation_file,
            station_name=station_name,
            make_plot=True,
            stream_output=False,
        )
        best = run_calibration(
            config,
            iterations=iterations,
            random_state=random_state,
            xi=xi,
            on_iteration=on_iteration,
        )
        best_params = {key: float(value) for key, value in best["params"].items()}
        write_status(
            context,
            "calibration",
            state="complete",
            message="参数率定完成",
            iteration=iterations,
            iterations=iterations,
            best_nse=float(best["target"]),
            best_params=best_params,
            history=str(history),
            plot_path=str(context.output_path("model", "chanliu_result", f"{station_name}.png")),
        )
    except Exception as exc:
        write_status(
            context,
            "calibration",
            state="failed",
            message=str(exc),
            traceback=traceback.format_exc(),
        )


def copy_observation_to_run(uploaded_file, run_id, project_root=PROJECT_ROOT):
    context = context_for(run_id, project_root=project_root)
    return save_uploaded_file(uploaded_file, context.input_path("observation.csv"))


def save_boundary_and_outlet(boundary_file, outlet_file, run_id, project_root=PROJECT_ROOT):
    context = context_for(run_id, project_root=project_root)
    upload_dir = context.input_path("uploads")
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    boundary_zip = save_uploaded_file(boundary_file, upload_dir / "boundary.zip")
    outlet_zip = save_uploaded_file(outlet_file, upload_dir / "outlet.zip")
    return boundary_zip, outlet_zip
