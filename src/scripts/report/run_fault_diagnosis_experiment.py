import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vic_auto_modeling.vic_bo import (  # noqa: E402
    align_monthly_series,
    read_observed_monthly,
    read_simulated_monthly,
)
from vic_auto_modeling.agent.evidence_validation import derive_diagnosis  # noqa: E402
from vic_auto_modeling.vic_runner import run_vic_model  # noqa: E402


PARAMETER_COLUMNS = ("x1", "x2", "x3", "x4", "x5", "x6")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def log_excerpt(path, patterns, max_lines=12):
    path = Path(path)
    if not path.exists():
        return []
    selected = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if any(pattern.lower() in line.lower() for pattern in patterns):
            selected.append(line.strip())
            if len(selected) >= max_lines:
                break
    return selected


def finalize_case(record, expected_stage, expected_target):
    record["reference"] = {
        "failed_stage": expected_stage,
        "correction_target": expected_target,
    }
    record["scores"] = {
        "stage_correct": record["diagnosis"]["failed_stage"] == expected_stage,
        "target_correct": record["diagnosis"]["correction_target"]["object"]
        == expected_target,
        "evidence_complete": bool(record["diagnosis"]["evidence"]),
        "unsupported_modification": bool(
            record["diagnosis"].get("unsupported_modifications")
        ),
    }
    record["outcome"] = (
        "PASS"
        if record["scores"]["stage_correct"]
        and record["scores"]["target_correct"]
        and record["scores"]["evidence_complete"]
        and not record["scores"]["unsupported_modification"]
        else "FAIL"
    )
    return record


def latest_parameters(history_path):
    row = pd.read_csv(history_path).iloc[-1]
    return [f"{float(row[name]):.10g}" for name in PARAMETER_COLUMNS]


def case_d1_missing_forcing(run_dir, output_root, vic_args, processes, source_dir):
    case_dir = output_root / "cases" / "D1"
    case_dir.mkdir(parents=True, exist_ok=True)
    forcing_dir = run_dir / "output" / "forcing" / "forcing"
    forcing_files = sorted(path for path in forcing_dir.iterdir() if path.is_file())
    original_count = len(forcing_files)
    missing_path = forcing_files[0]
    held_path = case_dir / missing_path.name
    missing_hash = file_sha256(missing_path)

    missing_path.rename(held_path)
    try:
        count_after_injection = sum(1 for path in forcing_dir.iterdir() if path.is_file())
        result = run_vic_model(
            run_id=run_dir.name,
            project_root=run_dir.parents[1],
            source_dir=source_dir,
            processes=processes,
            vic_args=vic_args,
            stream_output=False,
        )
        stdout_copy = case_dir / "vic_stdout.log"
        stderr_copy = case_dir / "vic_stderr.log"
        shutil.copy2(result.stdout_path, stdout_copy)
        shutil.copy2(result.stderr_path, stderr_copy)
        excerpts = log_excerpt(
            stderr_copy,
            ["forcing", "open", "error", "failed", "no such file"],
        ) + log_excerpt(
            stdout_copy,
            ["forcing", "open", "error", "failed", "no such file"],
        )
    finally:
        held_path.rename(missing_path)

    record = {
        "case_id": "D1",
        "injected_fault": "one active-cell forcing file removed",
        "injected_object": str(missing_path),
        "injected_object_sha256": missing_hash,
        "observed_symptom": (
            f"forcing inventory decreased from {original_count} to {count_after_injection}; "
            f"VIC return code {result.returncode}"
        ),
        "run_evidence": {
            "forcing_files_before": original_count,
            "forcing_files_after_injection": count_after_injection,
            "missing_file": str(missing_path),
            "vic_returncode": int(result.returncode),
            "log_excerpts": excerpts,
            "stdout_sha256": file_sha256(stdout_copy),
            "stderr_sha256": file_sha256(stderr_copy),
        },
    }
    record["diagnosis"] = derive_diagnosis(
        record["run_evidence"], correction_path=missing_path
    )
    return finalize_case(record, "forcing_preparation", "missing active-cell forcing file")


def case_d2_temporal_mismatch(run_dir, output_root, baseline_monthly):
    case_dir = output_root / "cases" / "D2"
    case_dir.mkdir(parents=True, exist_ok=True)
    observation_source = run_dir / "input" / "observation.csv"
    shifted_path = case_dir / "observation_shifted.csv"
    shifted = pd.read_csv(observation_source)
    shifted["year"] = shifted["year"].astype(int) + 20
    shifted.to_csv(shifted_path, index=False)

    observed = read_observed_monthly(shifted_path)
    simulated = read_simulated_monthly(baseline_monthly)
    try:
        align_monthly_series(observed, simulated)
        exception = None
    except ValueError as exc:
        exception = str(exc)
    if exception is None:
        raise ValueError("D2 did not trigger the expected temporal-alignment failure")

    observed_range = f"{observed['date'].min():%Y-%m}..{observed['date'].max():%Y-%m}"
    simulated_range = f"{simulated['date'].min():%Y-%m}..{simulated['date'].max():%Y-%m}"
    record = {
        "case_id": "D2",
        "injected_fault": "observation period shifted beyond the simulation period",
        "injected_object": str(shifted_path),
        "injected_object_sha256": file_sha256(shifted_path),
        "observed_symptom": exception,
        "run_evidence": {
            "observation_period": observed_range,
            "simulation_period": simulated_range,
            "common_months": 0,
            "alignment_exception": exception,
        },
    }
    record["diagnosis"] = derive_diagnosis(
        record["run_evidence"], correction_path=shifted_path
    )
    return finalize_case(
        record,
        "temporal_alignment",
        "observation dates or declared evaluation window",
    )


def case_d3_wrong_station(run_dir, output_root):
    case_dir = output_root / "cases" / "D3"
    case_dir.mkdir(parents=True, exist_ok=True)
    requested_station = "wrong_station"
    result_dir = run_dir / "output" / "model" / "chanliu_result"
    requested_path = result_dir / f"{requested_station}.month"
    station_path = run_dir / "output" / "flow" / "area_stnloc.txt"
    available_monthly = sorted(path.name for path in result_dir.glob("*.month"))
    station_lines = station_path.read_text(encoding="utf-8", errors="replace").splitlines()
    declared_stations = [
        line.split()[1]
        for line in station_lines
        if line.strip() and line.strip().upper() != "NONE" and len(line.split()) >= 2
    ]
    try:
        read_simulated_monthly(requested_path)
        exception = None
    except FileNotFoundError as exc:
        exception = str(exc)
    if exception is None:
        raise ValueError("D3 did not trigger the expected missing-station-output failure")

    record = {
        "case_id": "D3",
        "injected_fault": "a non-existent routing station name requested for calibration",
        "injected_object": requested_station,
        "observed_symptom": exception,
        "run_evidence": {
            "requested_station": requested_station,
            "declared_stations": declared_stations,
            "available_monthly_outputs": available_monthly,
            "requested_output_exists": requested_path.exists(),
            "read_exception": exception,
        },
    }
    record["diagnosis"] = derive_diagnosis(
        record["run_evidence"], correction_path=station_path
    )
    return finalize_case(record, "routing_output_selection", "station_name selection")


def read_ascii_grid(path):
    return np.loadtxt(path, skiprows=6)


def inactive_station(mask_path):
    mask = read_ascii_grid(mask_path)
    center_row = (mask.shape[0] - 1) / 2
    center_col = (mask.shape[1] - 1) / 2
    candidates = np.argwhere(mask == 0)
    row, col = min(
        candidates,
        key=lambda item: (item[0] - center_row) ** 2 + (item[1] - center_col) ** 2,
    )
    routing_col = int(col) + 1
    routing_row = int(mask.shape[0] - row)
    return routing_col, routing_row, int(row), int(col)


def case_d4_invalid_outlet(run_dir, output_root, vic_args, processes, source_dir):
    case_dir = output_root / "cases" / "D4"
    case_dir.mkdir(parents=True, exist_ok=True)
    station_path = run_dir / "output" / "flow" / "area_stnloc.txt"
    model_station_path = run_dir / "output" / "model" / "area_stnloc.txt"
    mask_path = run_dir / "output" / "flow" / "output_area_mask.txt"
    original_station = station_path.read_text(encoding="utf-8")
    original_hash = file_sha256(station_path)
    routing_col, routing_row, array_row, array_col = inactive_station(mask_path)
    injected_content = f"1 luanx   {routing_col}  {routing_row}   -9999\nNONE\n"
    station_path.write_text(injected_content, encoding="utf-8")
    injected_hash = file_sha256(station_path)

    try:
        result = run_vic_model(
            run_id=run_dir.name,
            project_root=run_dir.parents[1],
            source_dir=source_dir,
            processes=processes,
            vic_args=vic_args,
            stream_output=False,
        )
        stdout_copy = case_dir / "vic_stdout.log"
        stderr_copy = case_dir / "vic_stderr.log"
        shutil.copy2(result.stdout_path, stdout_copy)
        shutil.copy2(result.stderr_path, stderr_copy)
        excerpts = log_excerpt(
            stdout_copy,
            ["not found", "station", "outlet", "upstream"],
        ) + log_excerpt(
            stderr_copy,
            ["not found", "station", "outlet", "upstream", "error"],
        )
    finally:
        station_path.write_text(original_station, encoding="utf-8")
        if model_station_path.exists():
            shutil.copy2(station_path, model_station_path)

    record = {
        "case_id": "D4",
        "injected_fault": "outlet station assigned to an inactive routing cell",
        "injected_object": str(station_path),
        "original_object_sha256": original_hash,
        "injected_object_sha256": injected_hash,
        "observed_symptom": (
            f"station luanx mapped to inactive routing cell col={routing_col}, "
            f"row={routing_row}; VIC/routing return code {result.returncode}"
        ),
        "run_evidence": {
            "routing_column": routing_col,
            "routing_row": routing_row,
            "mask_array_row": array_row,
            "mask_array_column": array_col,
            "mask_value": 0,
            "vic_returncode": int(result.returncode),
            "log_excerpts": excerpts,
            "stdout_sha256": file_sha256(stdout_copy),
            "stderr_sha256": file_sha256(stderr_copy),
        },
    }
    record["diagnosis"] = derive_diagnosis(
        record["run_evidence"], correction_path=station_path
    )
    return finalize_case(record, "outlet_grid_mapping", "outlet-to-grid assignment")


def write_summary(output_root, run_id, source_run_id, cases):
    metrics = {
        "cases": len(cases),
        "stage_attribution_correct": sum(case["scores"]["stage_correct"] for case in cases),
        "correction_target_correct": sum(case["scores"]["target_correct"] for case in cases),
        "evidence_complete": sum(case["scores"]["evidence_complete"] for case in cases),
        "unsupported_modifications": sum(
            case["scores"]["unsupported_modification"] for case in cases
        ),
        "cases_passed": sum(case["outcome"] == "PASS" for case in cases),
    }
    summary = {
        "run_id": run_id,
        "source_run_id": source_run_id,
        "diagnostic_engine": "deterministic evidence rules",
        "cases": cases,
        "metrics": metrics,
    }
    write_json(output_root / "diagnosis_summary.json", summary)

    fieldnames = [
        "case_id",
        "injected_fault",
        "observed_symptom",
        "diagnosed_stage",
        "correction_target",
        "evidence_items",
        "unrelated_action_avoided",
        "outcome",
    ]
    with (output_root / "diagnosis_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "injected_fault": case["injected_fault"],
                    "observed_symptom": case["observed_symptom"],
                    "diagnosed_stage": case["diagnosis"]["failed_stage"],
                    "correction_target": case["diagnosis"]["correction_target"]["object"],
                    "evidence_items": len(case["diagnosis"]["evidence"]),
                    "unrelated_action_avoided": "; ".join(
                        case["diagnosis"]["do_not_modify"]
                    ),
                    "outcome": case["outcome"],
                }
            )

    table_lines = [
        "# Controlled fault-injection evaluation of evidence-grounded diagnosis",
        "",
        "| Case | Injected fault | Diagnosed stage | Correction target | Evidence | Outcome |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for case in cases:
        table_lines.append(
            "| {case_id} | {fault} | `{stage}` | {target} | {evidence} items | {outcome} |".format(
                case_id=case["case_id"],
                fault=case["injected_fault"],
                stage=case["diagnosis"]["failed_stage"],
                target=case["diagnosis"]["correction_target"]["object"],
                evidence=len(case["diagnosis"]["evidence"]),
                outcome=case["outcome"],
            )
        )
    table_lines.extend(
        [
            "",
            f"- Failed-stage attribution: {metrics['stage_attribution_correct']}/{metrics['cases']}",
            f"- Correction-target accuracy: {metrics['correction_target_correct']}/{metrics['cases']}",
            f"- Evidence-complete diagnoses: {metrics['evidence_complete']}/{metrics['cases']}",
            f"- Unsupported modifications: {metrics['unsupported_modifications']}/{metrics['cases']}",
        ]
    )
    (output_root / "diagnosis_table.md").write_text(
        "\n".join(table_lines) + "\n", encoding="utf-8"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Inject D1-D4 faults and create machine-readable diagnosis records."
    )
    parser.add_argument("--run-id", default="diagnosis_demo")
    parser.add_argument("--source-run-id", default="web_demo")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=12)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    run_dir = project_root / "runs" / args.run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Independent diagnosis run is missing: {run_dir}")
    if args.run_id == args.source_run_id:
        raise ValueError("Fault injection must not target the source run")

    output_root = run_dir / "output" / "diagnosis_audit"
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_dir = output_root / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_monthly = baseline_dir / "luanx.month"
    shutil.copy2(run_dir / "output" / "model" / "chanliu_result" / "luanx.month", baseline_monthly)
    vic_args = latest_parameters(run_dir / "logs" / "calibration_history.csv")

    cases = []
    cases.append(case_d2_temporal_mismatch(run_dir, output_root, baseline_monthly))
    cases.append(case_d3_wrong_station(run_dir, output_root))
    cases.append(
        case_d1_missing_forcing(
            run_dir, output_root, vic_args, args.processes, args.source_dir
        )
    )
    cases.append(
        case_d4_invalid_outlet(
            run_dir, output_root, vic_args, args.processes, args.source_dir
        )
    )
    cases.sort(key=lambda case: case["case_id"])
    for case in cases:
        write_json(output_root / "cases" / case["case_id"] / "diagnosis.json", case)
    metrics = write_summary(output_root, args.run_id, args.source_run_id, cases)
    print(json.dumps(metrics, indent=2))
    print(f"Evidence: {output_root / 'diagnosis_summary.json'}")
    print(f"Table: {output_root / 'diagnosis_table.md'}")


if __name__ == "__main__":
    main()
