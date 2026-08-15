import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def cache_count(project_root, run_id):
    cache = project_root / "runs" / run_id / "output/scientific_experiments/simulation_cache"
    return sum(path.is_dir() for path in cache.iterdir())


def main():
    parser = argparse.ArgumentParser(
        description="Validate and aggregate the completed S1-S3 experiment evidence."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--decision-run-id", default="decision_demo")
    parser.add_argument("--s1-run-id", default="decision_execution_demo")
    parser.add_argument("--s2-run-id", default="decision_execution_s2")
    parser.add_argument("--s3-run-id", default="decision_execution_s3")
    args = parser.parse_args()

    root = args.project_root.resolve()
    decision_root = root / "runs" / args.decision_run_id / "output/decision_audit"
    run_ids = {"S1": args.s1_run_id, "S2": args.s2_run_id, "S3": args.s3_run_id}
    paths = {
        "S1": root / "runs" / args.s1_run_id
        / "output/scientific_experiments/S1_parameter_identifiability/summary.json",
        "S2": root / "runs" / args.s2_run_id
        / "output/scientific_experiments/S2_temporal_transferability/summary.json",
        "S3": root / "runs" / args.s3_run_id
        / "output/scientific_experiments/S3_objective_adequacy/summary.json",
    }
    source = {case_id: read_json(path) for case_id, path in paths.items()}

    s1 = source["S1"]
    profiles = {
        (item["profile_parameter"], float(item["fixed_value"])): item
        for item in s1["profiles"]
    }
    required_profiles = {
        (name, value)
        for name in ("x1", "x2", "x4")
        for value in (1.0, 0.8, 0.5)
        if not (name == "x1" and value in (1.0, 0.8, 0.5))
    }
    required_profiles.update({("x1", value) for value in (0.01, 0.1, 0.3)})
    if set(profiles) != required_profiles:
        raise ValueError("S1 does not contain the nine prespecified profiles")
    if s1["local_sensitivity_points"] != 15 or s1["conditional_profile_evaluations"] != 54:
        raise ValueError("S1 evaluation counts do not match the prespecified design")
    s1_result = {
        "design": "15 inward-sensitivity runs and nine anchored conditional profiles",
        "profile_search_evaluations": 54,
        "conditional_nse": {
            "x1_boundary_0.01": profiles[("x1", 0.01)]["conditional_best_nse"],
            "x1_inward_0.30": profiles[("x1", 0.3)]["conditional_best_nse"],
            "x2_boundary_1.00": profiles[("x2", 1.0)]["conditional_best_nse"],
            "x2_inward_0.50": profiles[("x2", 0.5)]["conditional_best_nse"],
            "x4_boundary_1.00": profiles[("x4", 1.0)]["conditional_best_nse"],
            "x4_inward_0.80": profiles[("x4", 0.8)]["conditional_best_nse"],
            "x4_inward_0.50": profiles[("x4", 0.5)]["conditional_best_nse"],
        },
        "conclusion": (
            "Mixed identifiability: x1 and x2 decline inward, whereas x4 is flat near "
            "the upper boundary before declining farther inward."
        ),
        "claim_boundary": (
            "Exploratory anchored conditional profiles; not a formal confidence interval "
            "or proof of global identifiability."
        ),
    }

    s2 = source["S2"]
    directions = {item["direction"]: item for item in s2["directions"]}
    if s2["budget_per_direction"] != 15 or set(directions) != {"early_to_late", "late_to_early"}:
        raise ValueError("S2 does not match the reciprocal 15-evaluation design")
    s2_result = {
        "design": "Reciprocal split-sample calibration with 15 evaluations per direction",
        "early_to_late_validation_nse": directions["early_to_late"]["validation_metrics"]["nse"],
        "late_to_early_validation_nse": directions["late_to_early"]["validation_metrics"]["nse"],
        "conclusion": "Temporal transfer was strongly asymmetric in this case study.",
        "claim_boundary": "The experiment diagnoses transfer failure but does not identify its physical cause.",
    }

    s3 = source["S3"]
    if s3["budget_per_arm"] != 15:
        raise ValueError("S3 arms do not have the prespecified matched budget")
    delta = s3["delta"]
    if not (delta["nse"] < 0 and delta["pbias"] > 0 and delta["low_flow_pbias"] > 0 and delta["log_nse"] > 0):
        raise ValueError("S3 did not produce the expected measurable trade-off")
    s3_result = {
        "design": "Matched 15-evaluation NSE-only and multi-objective calibration arms",
        "delta_multiobjective_minus_control": delta,
        "conclusion": (
            "Changing the objective improved bias and low-flow metrics at a cost in NSE; "
            "large residual errors remained."
        ),
        "claim_boundary": (
            "The result demonstrates an objective trade-off, not that the objective alone "
            "causes all model error."
        ),
    }

    simulation_counts = {case_id: cache_count(root, run_id) for case_id, run_id in run_ids.items()}
    results = {"S1": s1_result, "S2": s2_result, "S3": s3_result}
    for case_id, count in simulation_counts.items():
        results[case_id]["unique_vic_simulations"] = count
    completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    aggregate = {
        "decision_run_id": args.decision_run_id,
        "execution_run_ids": run_ids,
        "completed_at_utc": completed_at,
        "validation": {
            "approved_experiments_completed": 3,
            "unique_vic_simulations": sum(simulation_counts.values()),
            "unique_vic_simulations_by_experiment": simulation_counts,
            "all_prespecified_design_checks_passed": True,
        },
        "results": results,
        "source_summaries": {case_id: str(path) for case_id, path in paths.items()},
    }
    aggregate_path = decision_root / "scientific_experiment_execution_audit.json"
    write_json(aggregate_path, aggregate)

    pending_path = decision_root / "pending_decisions.json"
    audit_path = decision_root / "decision_audit.json"
    pending = read_json(pending_path)
    for item in pending:
        case_id = item["case_id"]
        if item["status"] != "approved_for_execution" or not item["execution_authorized"]:
            raise ValueError(f"{case_id} lacks recorded human authorization")
        item["execution_events"] = [
            {
                "status": "completed",
                "execution_run_id": run_ids[case_id],
                "completed_at_utc": completed_at,
                "summary_path": str(paths[case_id]),
                "result": results[case_id],
            }
        ]
    audit = read_json(audit_path)
    audit["pending_decisions"] = pending
    audit["execution_evidence"] = aggregate
    audit["metrics"].update(
        {
            "approved_experiments_completed": 3,
            "unique_vic_simulations": sum(simulation_counts.values()),
        }
    )
    write_json(pending_path, pending)
    write_json(audit_path, audit)
    print(json.dumps(aggregate["validation"], indent=2))
    print(aggregate_path)


if __name__ == "__main__":
    main()
