import argparse
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

from vic_auto_modeling.vic_bo import VicCalibrationConfig, evaluate_parameters  # noqa: E402
from vic_auto_modeling.agent.evidence_validation import (  # noqa: E402
    validate_lineage_audit,
)


PARAMETER_COLUMNS = ("x1", "x2", "x3", "x4", "x5", "x6")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_record(row):
    values = {name: f"{float(row[name]):.10g}" for name in PARAMETER_COLUMNS}
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return values, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_evaluation(evaluation_id, row, config, output_root):
    params, parameter_hash = parameter_record(row)
    calculated_nse = evaluate_parameters(
        **{name: float(row[name]) for name in PARAMETER_COLUMNS},
        config=config,
    )
    expected_nse = float(row["nse"])
    if not np.isclose(calculated_nse, expected_nse, rtol=0, atol=1e-10):
        raise ValueError(
            f"Evaluation {evaluation_id} did not reproduce its recorded NSE: "
            f"{calculated_nse} vs {expected_nse}"
        )

    evaluation_dir = output_root / f"evaluation_{evaluation_id:03d}"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    monthly_source = config.result_dir / f"{config.station_name}.month"
    aligned_source = config.result_dir / f"{config.station_name}_aligned_monthly.csv"
    monthly_target = evaluation_dir / monthly_source.name
    aligned_target = evaluation_dir / aligned_source.name
    shutil.copy2(monthly_source, monthly_target)
    shutil.copy2(aligned_source, aligned_target)

    record = {
        "run_id": config.run_id,
        "evaluation_id": int(evaluation_id),
        "parameters": params,
        "parameter_sha256": parameter_hash,
        "monthly_path": str(monthly_target),
        "monthly_sha256": file_sha256(monthly_target),
        "aligned_path": str(aligned_target),
        "aligned_sha256": file_sha256(aligned_target),
        "expected_nse": expected_nse,
        "recomputed_nse": float(calculated_nse),
        "lineage_decision": "PASS",
    }
    (evaluation_dir / "lineage.json").write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Replay the best and latest evaluations and audit their lineage."
    )
    parser.add_argument("--source-run-id", default="web_demo")
    parser.add_argument("--experiment-run-id", default="lineage_demo")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=12)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    if args.source_run_id == args.experiment_run_id:
        raise ValueError("The lineage experiment must not overwrite the source run")

    source_history_path = (
        project_root / "runs" / args.source_run_id / "logs" / "calibration_history.csv"
    )
    history = pd.read_csv(source_history_path)
    best_row = history.loc[history["nse"].astype(float).idxmax()]
    latest_row = history.iloc[-1]
    best_id = int(best_row["iteration"])
    latest_id = int(latest_row["iteration"])
    if best_id == latest_id:
        raise ValueError("Best and latest evaluations are identical; no contrast is available")

    experiment_run = project_root / "runs" / args.experiment_run_id
    config = VicCalibrationConfig(
        run_id=args.experiment_run_id,
        project_root=project_root,
        source_dir=args.source_dir,
        processes=args.processes,
        observation_file=experiment_run / "input" / "observation.csv",
        station_name="luanx",
        make_plot=False,
        stream_output=False,
    )
    output_root = experiment_run / "output" / "lineage_audit"
    output_root.mkdir(parents=True, exist_ok=True)

    best_record = save_evaluation(best_id, best_row, config, output_root)
    latest_record = save_evaluation(latest_id, latest_row, config, output_root)

    mismatch = {
        "parameter_evaluation_id": best_id,
        "simulation_evaluation_id": latest_id,
        "aligned_evaluation_id": latest_id,
        "metric_evaluation_id": best_id,
        "parameter_sha256": best_record["parameter_sha256"],
        "simulation_sha256": latest_record["monthly_sha256"],
        "aligned_sha256": latest_record["aligned_sha256"],
        "reason": "parameter, simulation, and metric evaluation identifiers differ",
    }
    mismatch_ids = {
        mismatch["parameter_evaluation_id"],
        mismatch["simulation_evaluation_id"],
        mismatch["aligned_evaluation_id"],
        mismatch["metric_evaluation_id"],
    }
    mismatch["decision"] = "PASS" if len(mismatch_ids) == 1 else "BLOCK"
    audit = {
        "source_run_id": args.source_run_id,
        "experiment_run_id": args.experiment_run_id,
        "history_sha256": file_sha256(source_history_path),
        "observation_path": str(config.observation_path),
        "observation_sha256": file_sha256(config.observation_path),
        "valid_chains": [best_record, latest_record],
        "controlled_mismatch": mismatch,
    }
    validation = validate_lineage_audit(
        audit,
        expected_source_run_id=args.source_run_id,
        expected_experiment_run_id=args.experiment_run_id,
    )
    if not validation["ok"]:
        raise ValueError("Lineage validation failed: " + "; ".join(validation["errors"]))
    for key in (
        "valid_chains_accepted",
        "valid_chains_tested",
        "mismatches_blocked",
        "mismatches_tested",
        "unsafe_mismatches_accepted",
    ):
        audit[key] = validation[key]
    audit_path = output_root / "lineage_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Audit: {audit_path}")
    print(
        f"PASS E{best_id}: NSE={best_record['recomputed_nse']:.10f}; "
        f"PASS E{latest_id}: NSE={latest_record['recomputed_nse']:.10f}"
    )
    print(f"BLOCK E{best_id} parameters + E{latest_id} simulation")


if __name__ == "__main__":
    main()
