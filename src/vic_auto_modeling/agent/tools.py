import json
import os
import time
import uuid
from pathlib import Path

import pandas as pd

from vic_auto_modeling.core.run_context import RunContext
from vic_auto_modeling.agent.state import get_run_summary


CALIBRATION_PARAMETERS = {
    "x1": {
        "name": "b_infilt",
        "meaning": "可变下渗曲线参数，控制降水在地表径流与入渗之间的分配。",
        "range": "0.01 - 1.00",
    },
    "x2": {
        "name": "Ds",
        "meaning": "非线性基流参数，表示达到 Ws 阈值时的基流系数占 Dsmax 的比例。",
        "range": "0.01 - 1.00",
    },
    "x3": {
        "name": "Dsmax",
        "meaning": "最大基流速度，控制深层土壤向河道贡献基流的最大能力。",
        "range": "0.10 - 30.00",
    },
    "x4": {
        "name": "Ws",
        "meaning": "非线性基流启动阈值，表示土壤含水量达到最大含水量的该比例后基流快速增加。",
        "range": "0.01 - 1.00",
    },
    "x5": {
        "name": "depth[1]",
        "meaning": "第二层土壤厚度，影响中层土壤蓄水、蒸散和径流调节。",
        "range": "0.10 - 1.50 m",
    },
    "x6": {
        "name": "depth[2]",
        "meaning": "第三层土壤厚度，影响深层蓄水和基流过程。",
        "range": "0.10 - 1.50 m",
    },
}

READONLY_TOOLS = {
    "inspect_run",
    "read_stage_logs",
    "summarize_calibration",
    "explain_calibration_parameters",
    "generate_report_context",
}
ACTION_NAMES = {"start_auto_modeling", "start_vic_run", "start_calibration"}


def _context(run_id, project_root="."):
    return RunContext(run_id, project_root=project_root).ensure_dirs()


def _read_text(path, max_chars=6000):
    path = Path(path)
    if not path.exists():
        return {"exists": False, "path": str(path), "content": ""}
    return {
        "exists": True,
        "path": str(path),
        "content": path.read_text(encoding="utf-8", errors="replace")[-max_chars:],
    }


def inspect_run(run_id, project_root="."):
    return get_run_summary(run_id, project_root=project_root)


def read_stage_logs(run_id, stage="vic_run", project_root="."):
    context = _context(run_id, project_root)
    if stage == "vic_run":
        return {
            "stage": stage,
            "stdout": _read_text(context.output_path("model", "vic_stdout.log")),
            "stderr": _read_text(context.output_path("model", "vic_stderr.log")),
            "status": _read_text(context.log_path("vic_run_status.json")),
        }
    if stage in {"auto_modeling", "calibration"}:
        return {
            "stage": stage,
            "status": _read_text(context.log_path(f"{stage}_status.json")),
        }
    return {"stage": stage, "error": "Unsupported stage"}


def summarize_calibration(run_id, project_root="."):
    context = _context(run_id, project_root)
    history = context.log_path("calibration_history.csv")
    if not history.exists():
        return {"history_exists": False, "message": "No calibration history found."}

    df = pd.read_csv(history)
    if df.empty:
        return {"history_exists": True, "iterations": 0}

    best_index = df["best_nse"].astype(float).idxmax()
    best_row = df.loc[best_index]
    latest_row = df.iloc[-1]
    params = {
        key: float(best_row[key])
        for key in ("x1", "x2", "x3", "x4", "x5", "x6")
        if key in df.columns
    }
    return {
        "history_exists": True,
        "iterations": int(len(df)),
        "latest_nse": float(latest_row["nse"]) if "nse" in df.columns else None,
        "best_nse": float(best_row["best_nse"]),
        "best_iteration": int(best_row["iteration"]),
        "best_params": params,
        "parameter_definitions": CALIBRATION_PARAMETERS,
        "recent_rows": df.tail(5).to_dict(orient="records"),
    }


def explain_calibration_parameters(run_id=None, project_root="."):
    return CALIBRATION_PARAMETERS


def generate_report_context(run_id, project_root="."):
    return {
        "run_summary": get_run_summary(run_id, project_root=project_root),
        "calibration": summarize_calibration(run_id, project_root=project_root),
        "parameters": explain_calibration_parameters(),
    }


def execute_readonly_tool(tool, args, run_id, project_root="."):
    args = dict(args or {})
    args["run_id"] = run_id
    if tool == "inspect_run":
        return inspect_run(args["run_id"], project_root=project_root)
    if tool == "read_stage_logs":
        return read_stage_logs(
            args["run_id"],
            stage=args.get("stage", "vic_run"),
            project_root=project_root,
        )
    if tool == "summarize_calibration":
        return summarize_calibration(args["run_id"], project_root=project_root)
    if tool == "explain_calibration_parameters":
        return explain_calibration_parameters()
    if tool == "generate_report_context":
        return generate_report_context(args["run_id"], project_root=project_root)
    return {"error": f"Unsupported readonly tool: {tool}"}


def _pending_path(context):
    return context.log_path("agent_pending_action.json")


def _write_pending(context, payload):
    path = _pending_path(context)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def read_pending_action(run_id, project_root="."):
    context = _context(run_id, project_root)
    path = _pending_path(context)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_pending_action(run_id, action, args=None, reason="", project_root="."):
    context = _context(run_id, project_root)
    args = dict(args or {})
    token = uuid.uuid4().hex
    payload = {
        "token": token,
        "run_id": run_id,
        "action": action,
        "args": args,
        "reason": reason,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "pending",
    }
    _write_pending(context, payload)
    return payload


def validate_action(run_id, action, args=None, project_root="."):
    args = dict(args or {})
    summary = get_run_summary(run_id, project_root=project_root)
    paths = summary["paths"]

    if action not in ACTION_NAMES:
        return False, f"Unsupported action: {action}", args

    if action == "start_auto_modeling":
        boundary = Path(paths["boundary_upload"])
        outlet = Path(paths["outlet_upload"])
        if not boundary.exists() or not outlet.exists():
            return False, "请先在自动建模页面上传 boundary.zip 和 outlet.zip。", args
        args.setdefault("fill_algorithm", "priority-flood")
        return True, "", args

    if action == "start_vic_run":
        if not summary["outputs"]["model_inputs_ready"]:
            return False, "模型输入尚未生成，不能运行 VIC。", args
        source_dir = args.get("source_dir") or os.getenv("VIC_SOURCE_DIR")
        if not source_dir:
            return False, "请在界面中填写 VIC runtime directory，或设置 VIC_SOURCE_DIR。", args
        args["source_dir"] = source_dir
        args.setdefault("processes", 12)
        return True, "", args

    if action == "start_calibration":
        if not summary["outputs"]["monthly_flow_exists"]:
            return False, "尚未找到月尺度模拟径流输出，不能启动率定。", args
        if not summary["inputs"]["observation_uploaded"]:
            return False, "请先在参数率定页面上传 observation.csv。", args
        source_dir = args.get("source_dir") or os.getenv("VIC_SOURCE_DIR")
        if not source_dir:
            return False, "请在界面中填写 VIC runtime directory，或设置 VIC_SOURCE_DIR。", args
        args["source_dir"] = source_dir
        args.setdefault("processes", 12)
        args.setdefault("station_name", "luanx")
        args.setdefault("iterations", 10)
        args.setdefault("random_state", 1)
        args.setdefault("xi", 0.1)
        return True, "", args

    return False, "Unsupported action", args


def confirm_pending_action(run_id, token, project_root="."):
    from web_app.jobs import (
        run_auto_modeling_job,
        run_calibration_job,
        run_vic_job,
        start_job,
    )

    context = _context(run_id, project_root)
    pending = read_pending_action(run_id, project_root=project_root)
    if not pending or pending.get("token") != token:
        return {"ok": False, "message": "未找到匹配的待确认动作。"}
    if pending.get("status") != "pending":
        return {"ok": False, "message": "该动作已经被处理，不能重复执行。"}

    action = pending["action"]
    args = dict(pending.get("args") or {})
    ok, message, args = validate_action(run_id, action, args, project_root=project_root)
    if not ok:
        pending["status"] = "rejected"
        pending["message"] = message
        _write_pending(context, pending)
        return {"ok": False, "message": message, "pending_action": pending}

    if action == "start_auto_modeling":
        boundary_zip = context.input_path("uploads", "boundary.zip")
        outlet_zip = context.input_path("uploads", "outlet.zip")
        started = start_job(
            f"{run_id}:auto_modeling",
            run_auto_modeling_job,
            run_id,
            boundary_zip,
            outlet_zip,
            args["fill_algorithm"],
            Path(project_root).resolve(),
        )
    elif action == "start_vic_run":
        started = start_job(
            f"{run_id}:vic_run",
            run_vic_job,
            run_id,
            args["source_dir"],
            int(args["processes"]),
            Path(project_root).resolve(),
        )
    elif action == "start_calibration":
        started = start_job(
            f"{run_id}:calibration",
            run_calibration_job,
            run_id,
            context.input_path("observation.csv"),
            int(args["iterations"]),
            args["source_dir"],
            int(args["processes"]),
            args["station_name"],
            int(args["random_state"]),
            float(args["xi"]),
            Path(project_root).resolve(),
        )
    else:
        return {"ok": False, "message": f"Unsupported action: {action}"}

    pending["status"] = "consumed" if started else "already_running"
    pending["consumed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    pending["args"] = args
    _write_pending(context, pending)
    return {
        "ok": True,
        "message": "任务已启动。" if started else "同类任务正在运行，未重复启动。",
        "pending_action": pending,
    }
