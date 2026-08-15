import json
from pathlib import Path

from vic_auto_modeling.agent.evidence_validation import (
    derive_diagnosis,
    validate_lineage_audit,
)


SCIENTIFIC_TOOLS = {
    "deterministic_construction",
    "audit_evaluation_lineage",
    "diagnose_run_evidence",
    "scientific_decision",
}


def _safe_run_id(value, label):
    value = str(value)
    if not value or Path(value).name != value:
        raise ValueError(f"{label} must be a directory name")
    return value


def _read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def deterministic_construction(run_id, project_root="."):
    path = (
        Path(project_root).resolve()
        / "report_assets/figures/deterministic_model_construction_evidence.json"
    )
    evidence = _read_json(path)
    if evidence is None:
        return {
            "ok": False,
            "tool": "deterministic_construction",
            "run_id": run_id,
            "message": f"Construction audit not found: {path}",
            "required_experiment": "scripts/report/create_deterministic_construction_figure.py",
        }
    if evidence.get("run_id") != run_id:
        return {
            "ok": False,
            "tool": "deterministic_construction",
            "run_id": run_id,
            "audit_path": str(path),
            "message": (
                f"Construction audit belongs to run {evidence.get('run_id')!r}, "
                f"not requested run {run_id!r}"
            ),
        }
    active = int(evidence.get("active_cells", 0))
    forcing = int(evidence.get("forcing_files", 0))
    flux = int(evidence.get("flux_files", 0))
    passed = (
        active > 0
        and int(evidence.get("unique_ids", 0)) == active
        and evidence.get("id_sets_equal") is True
        and forcing == active
        and flux == active
        and evidence.get("forcing_flux_names_equal") is True
        and int(evidence.get("forcing_non_finite", -1)) == 0
        and int(evidence.get("flux_non_finite", -1)) == 0
        and evidence.get("forcing_record_counts_match") is True
        and evidence.get("flux_record_counts_match") is True
        and evidence.get("routed_record_counts_match") is True
        and int(evidence.get("returncode", -1)) == 0
        and int(evidence.get("invalid_directions", -1)) == 0
        and int(evidence.get("cycles", -1)) == 0
        and int(evidence.get("not_found", -1)) == 0
        and evidence.get("logged_upstream") == evidence.get("upstream_cells")
        and int(evidence.get("daily_rows", 0)) > 0
        and int(evidence.get("monthly_rows", 0)) > 0
        and int(evidence.get("climatology_rows", 0)) > 0
    )
    return {
        "ok": passed,
        "tool": "deterministic_construction",
        "run_id": run_id,
        "audit_path": str(path),
        "decision": "PASS" if passed else "FAIL",
        "summary": evidence,
    }


def audit_evaluation_lineage(run_id, project_root=".", experiment_run_id="lineage_demo"):
    experiment_run_id = _safe_run_id(experiment_run_id, "experiment_run_id")
    path = (
        Path(project_root).resolve()
        / "runs"
        / experiment_run_id
        / "output/lineage_audit/lineage_audit.json"
    )
    audit = _read_json(path)
    if audit is None:
        return {
            "ok": False,
            "tool": "audit_evaluation_lineage",
            "run_id": run_id,
            "message": f"Lineage audit not found: {path}",
            "required_experiment": "scripts/report/run_lineage_assurance_experiment.py",
        }
    validation = validate_lineage_audit(
        audit,
        expected_source_run_id=run_id,
        expected_experiment_run_id=experiment_run_id,
    )
    valid = validation["valid_chains_accepted"]
    valid_tested = validation["valid_chains_tested"]
    blocked = validation["mismatches_blocked"]
    mismatch_tested = validation["mismatches_tested"]
    unsafe = validation["unsafe_mismatches_accepted"]
    passed = validation["ok"]
    return {
        "ok": passed,
        "tool": "audit_evaluation_lineage",
        "run_id": run_id,
        "experiment_run_id": experiment_run_id,
        "audit_path": str(path),
        "decision": "PASS" if passed else "BLOCK",
        "validation_errors": validation["errors"],
        "record_validations": validation["record_validations"],
        "summary": {
            "valid_chains_accepted": valid,
            "valid_chains_tested": valid_tested,
            "mismatches_blocked": blocked,
            "mismatches_tested": mismatch_tested,
            "unsafe_mismatches_accepted": unsafe,
        },
        "valid_chains": audit.get("valid_chains", []),
        "controlled_mismatch": audit.get("controlled_mismatch"),
    }


def diagnose_run_evidence(
    run_id,
    project_root=".",
    diagnosis_run_id="diagnosis_demo",
    case_id=None,
):
    diagnosis_run_id = _safe_run_id(diagnosis_run_id, "diagnosis_run_id")
    if case_id is not None and case_id not in {"D1", "D2", "D3", "D4"}:
        raise ValueError("case_id must be one of D1, D2, D3, or D4")
    path = (
        Path(project_root).resolve()
        / "runs"
        / diagnosis_run_id
        / "output/diagnosis_audit/diagnosis_summary.json"
    )
    audit = _read_json(path)
    if audit is None:
        return {
            "ok": False,
            "tool": "diagnose_run_evidence",
            "run_id": run_id,
            "message": f"Diagnosis audit not found: {path}",
            "required_experiment": "scripts/report/run_fault_diagnosis_experiment.py",
        }
    if audit.get("source_run_id") != run_id:
        return {
            "ok": False,
            "tool": "diagnose_run_evidence",
            "run_id": run_id,
            "diagnosis_run_id": diagnosis_run_id,
            "audit_path": str(path),
            "decision": "FAIL",
            "message": "Diagnosis evidence does not belong to the requested source run",
        }
    cases = audit.get("cases", [])
    if case_id is not None:
        cases = [case for case in cases if case.get("case_id") == case_id]
        if len(cases) != 1:
            raise ValueError(f"Expected one diagnosis record for {case_id}")
    validated_cases = []
    for case in cases:
        stored = case.get("diagnosis") or {}
        correction_path = (stored.get("correction_target") or {}).get("path")
        try:
            derived = derive_diagnosis(
                case.get("run_evidence"), correction_path=correction_path
            )
            errors = []
            if stored.get("failed_stage") != derived["failed_stage"]:
                errors.append("stored failed_stage differs from rule-derived stage")
            if (stored.get("correction_target") or {}).get("object") != derived[
                "correction_target"
            ]["object"]:
                errors.append("stored correction target differs from rule-derived target")
        except ValueError as exc:
            derived = None
            errors = [str(exc)]
        validated_cases.append(
            {
                **case,
                "diagnosis": derived,
                "validation_errors": errors,
                "outcome": "PASS" if not errors else "FAIL",
            }
        )
    passed = bool(validated_cases) and all(
        case["outcome"] == "PASS" for case in validated_cases
    )
    return {
        "ok": passed,
        "tool": "diagnose_run_evidence",
        "run_id": run_id,
        "diagnosis_run_id": diagnosis_run_id,
        "audit_path": str(path),
        "case_id": case_id,
        "decision": "PASS" if passed else "FAIL",
        "metrics": audit.get("metrics", {}),
        "diagnoses": validated_cases,
    }


def review_scientific_decision(
    run_id,
    project_root=".",
    decision_run_id="decision_demo",
    case_id=None,
):
    decision_run_id = _safe_run_id(decision_run_id, "decision_run_id")
    if case_id is not None and case_id not in {"S1", "S2", "S3"}:
        raise ValueError("case_id must be one of S1, S2, or S3")
    root = Path(project_root).resolve() / "runs" / decision_run_id / "output/decision_audit"
    audit_path = root / "decision_audit.json"
    execution_path = root / "scientific_experiment_execution_audit.json"
    audit = _read_json(audit_path)
    execution = _read_json(execution_path)
    if audit is None:
        return {
            "ok": False,
            "tool": "scientific_decision",
            "run_id": run_id,
            "message": f"Scientific-decision audit not found: {audit_path}",
            "required_experiment": "scripts/report/run_scientific_decision_experiment.py",
        }
    if audit.get("source_run_id") != run_id:
        return {
            "ok": False,
            "tool": "scientific_decision",
            "run_id": run_id,
            "decision_run_id": decision_run_id,
            "audit_path": str(audit_path),
            "decision": "PENDING_OR_INCOMPLETE",
            "message": "Scientific-decision evidence does not belong to the requested source run",
        }
    decisions = []
    for item in audit.get("pending_decisions", []):
        if case_id is not None and item.get("case_id") != case_id:
            continue
        decisions.append(
            {
                key: value
                for key, value in item.items()
                if key != "confirmation_token"
            }
        )
    results = (execution or {}).get("results", {})
    if case_id is not None:
        results = {case_id: results.get(case_id)} if case_id in results else {}
    completed = bool(execution) and bool(decisions) and all(
        item.get("status") == "approved_for_execution"
        and item.get("execution_authorized") is True
        and len(item.get("execution_events", [])) == 1
        and item["execution_events"][0].get("status") == "completed"
        for item in decisions
    ) and all(results.get(item.get("case_id")) for item in decisions)
    return {
        "ok": completed,
        "tool": "scientific_decision",
        "run_id": run_id,
        "decision_run_id": decision_run_id,
        "audit_path": str(audit_path),
        "execution_audit_path": str(execution_path),
        "case_id": case_id,
        "decision": "COMPLETED" if completed else "PENDING_OR_INCOMPLETE",
        "metrics": audit.get("metrics", {}),
        "decisions": decisions,
        "scientific_results": results,
    }


def execute_scientific_tool(tool, args, run_id, project_root="."):
    args = dict(args or {})
    if tool == "deterministic_construction":
        return deterministic_construction(run_id, project_root=project_root)
    if tool == "audit_evaluation_lineage":
        return audit_evaluation_lineage(
            run_id,
            project_root=project_root,
            experiment_run_id=args.get("experiment_run_id", "lineage_demo"),
        )
    if tool == "diagnose_run_evidence":
        return diagnose_run_evidence(
            run_id,
            project_root=project_root,
            diagnosis_run_id=args.get("diagnosis_run_id", "diagnosis_demo"),
            case_id=args.get("case_id"),
        )
    if tool == "scientific_decision":
        return review_scientific_decision(
            run_id,
            project_root=project_root,
            decision_run_id=args.get("decision_run_id", "decision_demo"),
            case_id=args.get("case_id"),
        )
    return {"ok": False, "tool": tool, "message": f"Unsupported scientific tool: {tool}"}
