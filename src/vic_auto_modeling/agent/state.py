import json
from pathlib import Path

import pandas as pd

from vic_auto_modeling.core.run_context import RunContext


STAGES = ("auto_modeling", "vic_run", "calibration")


def _read_json(path):
    path = Path(path)
    if not path.exists():
        return {"state": "idle"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"state": "failed", "message": f"Invalid JSON in {path}: {exc}"}


def _status(context, name):
    return _read_json(context.log_path(f"{name}_status.json"))


def _exists(path):
    return Path(path).exists()


def _safe_tail(path, max_chars=4000):
    path = Path(path)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]


def _calibration_summary(history_path):
    history_path = Path(history_path)
    if not history_path.exists():
        return {
            "history_exists": False,
            "iterations": 0,
            "best_nse": None,
            "best_params": None,
            "latest_nse": None,
        }

    try:
        df = pd.read_csv(history_path)
    except Exception as exc:
        return {
            "history_exists": True,
            "iterations": 0,
            "best_nse": None,
            "best_params": None,
            "latest_nse": None,
            "error": str(exc),
        }

    if df.empty or "best_nse" not in df.columns:
        return {
            "history_exists": True,
            "iterations": 0,
            "best_nse": None,
            "best_params": None,
            "latest_nse": None,
        }

    best_index = df["best_nse"].astype(float).idxmax()
    best_row = df.loc[best_index]
    latest_row = df.iloc[-1]
    params = {
        name: float(best_row[name])
        for name in ("x1", "x2", "x3", "x4", "x5", "x6")
        if name in df.columns
    }
    return {
        "history_exists": True,
        "iterations": int(len(df)),
        "best_nse": float(best_row["best_nse"]),
        "best_params": params,
        "latest_nse": float(latest_row["nse"]) if "nse" in df.columns else None,
    }


def get_run_summary(run_id, project_root="."):
    context = RunContext(run_id, project_root=project_root)
    model_dir = context.output_path("model")
    result_dir = model_dir / "chanliu_result"
    history = context.log_path("calibration_history.csv")
    calibration = _calibration_summary(history)

    statuses = {stage: _status(context, stage) for stage in STAGES}
    latest_errors = {
        stage: (
            statuses[stage].get("message")
            if statuses[stage].get("state") == "failed"
            else None
        )
        for stage in STAGES
    }

    return {
        "run_id": run_id,
        "paths": {
            "run_dir": str(context.run_dir),
            "input_dir": str(context.input_dir),
            "output_dir": str(context.output_dir),
            "log_dir": str(context.log_dir),
            "boundary_upload": str(context.input_path("uploads", "boundary.zip")),
            "outlet_upload": str(context.input_path("uploads", "outlet.zip")),
            "observation": str(context.input_path("observation.csv")),
            "chanliu_input": str(model_dir / "chanliu_input.txt"),
            "rout_input": str(model_dir / "rout_input.txt"),
            "monthly_flow": str(result_dir / "luanx.month"),
            "calibration_history": str(history),
        },
        "inputs": {
            "boundary_uploaded": _exists(context.input_path("uploads", "boundary.zip"))
            or _exists(context.input_path("boundary")),
            "outlet_uploaded": _exists(context.input_path("uploads", "outlet.zip"))
            or _exists(context.input_path("outlets")),
            "observation_uploaded": _exists(context.input_path("observation.csv")),
        },
        "stages": {
            stage: statuses[stage].get("state", "idle")
            for stage in STAGES
        },
        "status": statuses,
        "outputs": {
            "model_inputs_ready": _exists(model_dir / "chanliu_input.txt")
            and _exists(model_dir / "rout_input.txt"),
            "monthly_flow_exists": _exists(result_dir / "luanx.month"),
            "calibration_history_exists": calibration["history_exists"],
            "best_nse": calibration["best_nse"],
            "best_params": calibration["best_params"],
            "latest_nse": calibration["latest_nse"],
            "calibration_iterations": calibration["iterations"],
        },
        "latest_errors": latest_errors,
        "logs": {
            "vic_stdout_tail": _safe_tail(model_dir / "vic_stdout.log", max_chars=2000),
            "vic_stderr_tail": _safe_tail(model_dir / "vic_stderr.log", max_chars=2000),
        },
    }
