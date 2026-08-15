import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PARAMETER_COLUMNS = ("x1", "x2", "x3", "x4", "x5", "x6")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_sha256(parameters):
    values = {
        name: f"{float(parameters[name]):.10g}" for name in PARAMETER_COLUMNS
    }
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def calculate_nse(observed, simulated):
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    denominator = np.sum((observed - observed.mean()) ** 2)
    if denominator == 0:
        raise ValueError("NSE is undefined for a constant observed series")
    return float(1 - np.sum((observed - simulated) ** 2) / denominator)


def validate_evaluation_record(record, experiment_run_id):
    errors = []
    evaluation_id = int(record.get("evaluation_id", -1))
    if record.get("run_id") != experiment_run_id:
        errors.append("evaluation run_id does not match experiment_run_id")

    try:
        actual_parameter_hash = parameter_sha256(record["parameters"])
        if actual_parameter_hash != record.get("parameter_sha256"):
            errors.append("parameter hash mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid parameter record: {exc}")

    monthly_path = Path(record.get("monthly_path", ""))
    aligned_path = Path(record.get("aligned_path", ""))
    if not monthly_path.is_file():
        errors.append("monthly artefact is missing")
    elif file_sha256(monthly_path) != record.get("monthly_sha256"):
        errors.append("monthly artefact hash mismatch")
    if not aligned_path.is_file():
        errors.append("aligned artefact is missing")
    elif file_sha256(aligned_path) != record.get("aligned_sha256"):
        errors.append("aligned artefact hash mismatch")

    recomputed_nse = None
    if monthly_path.is_file() and aligned_path.is_file():
        try:
            aligned = pd.read_csv(aligned_path)
            required = {"date", "observed", "simulated"}
            if not required.issubset(aligned.columns):
                raise ValueError("aligned series lacks date, observed, or simulated")
            if not np.isfinite(aligned[["observed", "simulated"]].to_numpy(float)).all():
                raise ValueError("aligned series contains non-finite values")
            recomputed_nse = calculate_nse(aligned["observed"], aligned["simulated"])
            expected_nse = float(record["expected_nse"])
            if not np.isclose(recomputed_nse, expected_nse, rtol=0, atol=1e-10):
                errors.append("recomputed NSE does not match expected_nse")

            monthly = pd.read_csv(
                monthly_path,
                sep=r"\s+",
                header=None,
                names=["year", "month", "simulated"],
            )
            monthly["date"] = pd.to_datetime(
                dict(year=monthly["year"], month=monthly["month"], day=1)
            )
            comparison = aligned[["date", "simulated"]].copy()
            comparison["date"] = pd.to_datetime(comparison["date"])
            comparison = comparison.merge(
                monthly[["date", "simulated"]],
                on="date",
                how="left",
                suffixes=("_aligned", "_monthly"),
                validate="one_to_one",
            )
            if comparison["simulated_monthly"].isna().any() or not np.allclose(
                comparison["simulated_aligned"],
                comparison["simulated_monthly"],
                rtol=0,
                atol=1e-10,
            ):
                errors.append("aligned simulation is not derived from monthly artefact")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"metric or alignment validation failed: {exc}")

    return {
        "evaluation_id": evaluation_id,
        "ok": not errors,
        "errors": errors,
        "recomputed_nse": recomputed_nse,
    }


def validate_lineage_audit(audit, expected_source_run_id, expected_experiment_run_id):
    errors = []
    if audit.get("source_run_id") != expected_source_run_id:
        errors.append("source_run_id does not match the requested run")
    if audit.get("experiment_run_id") != expected_experiment_run_id:
        errors.append("experiment_run_id does not match the selected lineage run")
    observation_path = Path(audit.get("observation_path", ""))
    if not observation_path.is_file():
        errors.append("observation artefact is missing")
    elif file_sha256(observation_path) != audit.get("observation_sha256"):
        errors.append("observation artefact hash mismatch")

    records = audit.get("valid_chains") or []
    validations = [
        validate_evaluation_record(record, expected_experiment_run_id)
        for record in records
    ]
    evaluation_ids = [item["evaluation_id"] for item in validations]
    if len(evaluation_ids) != len(set(evaluation_ids)):
        errors.append("evaluation identifiers are not unique")
    errors.extend(
        f"E{item['evaluation_id']}: {error}"
        for item in validations
        for error in item["errors"]
    )

    mismatch = audit.get("controlled_mismatch") or {}
    identities = {
        int(mismatch.get("parameter_evaluation_id", -1)),
        int(mismatch.get("simulation_evaluation_id", -1)),
        int(
            mismatch.get(
                "aligned_evaluation_id",
                mismatch.get("simulation_evaluation_id", -1),
            )
        ),
        int(mismatch.get("metric_evaluation_id", -1)),
    }
    derived_mismatch_decision = "PASS" if len(identities) == 1 else "BLOCK"
    if mismatch.get("decision") != derived_mismatch_decision:
        errors.append("controlled mismatch decision is inconsistent with evaluation identities")

    valid_accepted = sum(item["ok"] for item in validations)
    mismatch_tested = 1 if mismatch else 0
    mismatches_blocked = int(bool(mismatch) and derived_mismatch_decision == "BLOCK")
    unsafe = int(bool(mismatch) and derived_mismatch_decision != "BLOCK")
    return {
        "ok": not errors,
        "errors": errors,
        "record_validations": validations,
        "valid_chains_accepted": valid_accepted,
        "valid_chains_tested": len(validations),
        "mismatches_blocked": mismatches_blocked,
        "mismatches_tested": mismatch_tested,
        "unsafe_mismatches_accepted": unsafe,
    }


def derive_diagnosis(run_evidence, correction_path=None):
    evidence = dict(run_evidence or {})
    forcing_before = evidence.get("forcing_files_before")
    forcing_after = evidence.get("forcing_files_after_injection")
    if (
        forcing_before is not None
        and forcing_after is not None
        and forcing_after < forcing_before
        and evidence.get("missing_file")
        and int(evidence.get("vic_returncode", 0)) != 0
    ):
        return {
            "failed_stage": "forcing_preparation",
            "root_cause": "an expected active-cell forcing artefact is absent",
            "evidence": [
                f"forcing inventory: {evidence['forcing_files_after_injection']}/{evidence['forcing_files_before']}",
                f"missing artefact: {Path(evidence['missing_file']).name}",
                f"VIC return code: {evidence['vic_returncode']}",
            ],
            "correction_target": {
                "object": "missing active-cell forcing file",
                "path": str(correction_path or evidence["missing_file"]),
            },
            "do_not_modify": [
                "VIC calibration parameters",
                "observation record",
                "Bayesian-optimization budget",
            ],
            "unsupported_modifications": [],
            "rule_id": "missing_forcing_artefact",
        }
    if int(evidence.get("common_months", -1)) == 0 and evidence.get(
        "observation_period"
    ) and evidence.get("simulation_period"):
        return {
            "failed_stage": "temporal_alignment",
            "root_cause": "observed and simulated monthly series have no common dates",
            "evidence": [
                f"observation period: {evidence['observation_period']}",
                f"simulation period: {evidence['simulation_period']}",
                "common monthly dates: 0",
            ],
            "correction_target": {
                "object": "observation dates or declared evaluation window",
                "path": str(correction_path or ""),
            },
            "do_not_modify": [
                "VIC calibration parameters",
                "routing topology",
                "forcing files",
            ],
            "unsupported_modifications": [],
            "rule_id": "no_temporal_overlap",
        }
    if (
        evidence.get("requested_station")
        and evidence.get("requested_output_exists") is False
        and evidence["requested_station"] not in evidence.get("declared_stations", [])
    ):
        return {
            "failed_stage": "routing_output_selection",
            "root_cause": "the requested station is not declared and has no monthly output",
            "evidence": [
                f"requested station: {evidence['requested_station']}",
                f"declared stations: {', '.join(evidence.get('declared_stations', []))}",
                f"available monthly outputs: {', '.join(evidence.get('available_monthly_outputs', []))}",
            ],
            "correction_target": {
                "object": "station_name selection",
                "path": str(correction_path or ""),
            },
            "do_not_modify": [
                "VIC calibration parameters",
                "forcing files",
                "Bayesian-optimization budget",
            ],
            "unsupported_modifications": [],
            "rule_id": "undeclared_station",
        }
    if int(evidence.get("mask_value", -1)) == 0 and evidence.get(
        "routing_column"
    ) and evidence.get("routing_row"):
        return {
            "failed_stage": "outlet_grid_mapping",
            "root_cause": "the configured outlet cell is inactive in the routing mask",
            "evidence": [
                f"station routing cell: col={evidence['routing_column']}, row={evidence['routing_row']}",
                "routing-mask value at station cell: 0",
                f"VIC/routing return code: {evidence.get('vic_returncode')}",
            ],
            "correction_target": {
                "object": "outlet-to-grid assignment",
                "path": str(correction_path or ""),
            },
            "do_not_modify": [
                "observation dates",
                "VIC calibration parameters",
                "Bayesian-optimization budget",
            ],
            "unsupported_modifications": [],
            "rule_id": "inactive_outlet_cell",
        }
    raise ValueError("No deterministic diagnosis rule matches the supplied evidence")
