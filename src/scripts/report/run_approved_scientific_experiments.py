import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vic_auto_modeling.vic_bo import (  # noqa: E402
    PARAM_BOUNDS,
    align_monthly_series,
    calculate_nse,
    read_observed_monthly,
    read_simulated_monthly,
)
from vic_auto_modeling.vic_runner import run_vic_model  # noqa: E402


PARAMETERS = ("x1", "x2", "x3", "x4", "x5", "x6")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parameter_key(params):
    values = {name: f"{float(params[name]):.10g}" for name in PARAMETERS}
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return values, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def setup_run(project_root, source_run_id, experiment_run_id):
    source = project_root / "runs" / source_run_id
    target = project_root / "runs" / experiment_run_id
    if not target.exists():
        print(f"Copying isolated execution run: {source} -> {target}", flush=True)
        shutil.copytree(source, target)
    config = target / "output/model/chanliu_input.txt"
    text = config.read_text(encoding="utf-8")
    text = text.replace(str(source), str(target))
    text = text.replace(
        str(project_root / "runs" / source_run_id),
        str(project_root / "runs" / experiment_run_id),
    )
    config.write_text(text, encoding="utf-8")
    return target


def pbias(observed, simulated):
    return float(100 * np.sum(simulated - observed) / np.sum(observed))


def log_nse(observed, simulated):
    return float(calculate_nse(np.log1p(observed), np.log1p(np.clip(simulated, 0, None))))


def metrics(aligned):
    observed = aligned["observed"].to_numpy(float)
    simulated = aligned["simulated"].to_numpy(float)
    threshold = float(np.quantile(observed, 0.2))
    low = observed <= threshold
    return {
        "records": int(len(aligned)),
        "start": str(aligned["date"].min().date()),
        "end": str(aligned["date"].max().date()),
        "nse": float(calculate_nse(observed, simulated)),
        "pbias": pbias(observed, simulated),
        "log_nse": log_nse(observed, simulated),
        "low_flow_threshold": threshold,
        "low_flow_pbias": pbias(observed[low], simulated[low]),
    }


class ExperimentRunner:
    def __init__(self, project_root, run_id, source_dir, processes, audit_root):
        self.project_root = project_root
        self.run_id = run_id
        self.source_dir = source_dir
        self.processes = processes
        self.run_dir = project_root / "runs" / run_id
        self.result_dir = self.run_dir / "output/model/chanliu_result"
        self.cache_dir = audit_root / "simulation_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.observed = read_observed_monthly(self.run_dir / "input/observation.csv")
        self.executed = 0
        self.reused = 0

    def simulate(self, params):
        values, digest = parameter_key(params)
        cache = self.cache_dir / digest[:16]
        monthly = cache / "luanx.month"
        if monthly.exists():
            self.reused += 1
            return monthly, digest, False
        result = run_vic_model(
            run_id=self.run_id,
            project_root=self.project_root,
            source_dir=self.source_dir,
            processes=self.processes,
            vic_args=[values[name] for name in PARAMETERS],
            stream_output=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"VIC failed for {digest}: return code {result.returncode}")
        cache.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.result_dir / "luanx.month", monthly)
        write_json(
            cache / "simulation.json",
            {
                "parameter_sha256": digest,
                "parameters": values,
                "monthly_path": str(monthly),
                "vic_returncode": result.returncode,
            },
        )
        self.executed += 1
        print(f"VIC execution {self.executed}: {digest[:8]}", flush=True)
        return monthly, digest, True

    def evaluate(self, params, start_year=2011, end_year=2016):
        monthly, digest, executed = self.simulate(params)
        simulated = read_simulated_monthly(monthly)
        observed = self.observed[self.observed["year"].between(start_year, end_year)]
        aligned = align_monthly_series(observed, simulated)
        result = metrics(aligned)
        result.update(
            {
                "parameter_sha256": digest,
                "parameters": {name: float(params[name]) for name in PARAMETERS},
                "cache_reused": not executed,
            }
        )
        return result


def run_bo(runner, output_path, objective, iterations, random_state, fixed=None):
    from bayes_opt import BayesianOptimization
    from bayes_opt import acquisition

    fixed = dict(fixed or {})
    free_bounds = {name: bound for name, bound in PARAM_BOUNDS.items() if name not in fixed}
    optimizer = BayesianOptimization(
        acquisition_function=acquisition.ProbabilityOfImprovement(xi=0.1),
        f=None,
        pbounds=free_bounds,
        verbose=0,
        random_state=random_state,
    )
    records = []
    if output_path.exists():
        records = json.loads(output_path.read_text(encoding="utf-8"))["evaluations"]
        for record in records:
            free = {name: record["parameters"][name] for name in free_bounds}
            optimizer.register(params=free, target=record["objective"])
    while len(records) < iterations:
        free = {name: float(value) for name, value in optimizer.suggest().items()}
        params = {**free, **fixed}
        result, score = objective(params)
        optimizer.register(params=free, target=score)
        records.append(
            {
                "evaluation": len(records) + 1,
                "parameters": params,
                "objective": float(score),
                "metrics": result,
            }
        )
        write_json(output_path, {"fixed": fixed, "evaluations": records})
    best = max(records, key=lambda record: record["objective"])
    return records, best


def run_s1(runner, audit_root, e19):
    output = audit_root / "S1_parameter_identifiability"
    output.mkdir(parents=True, exist_ok=True)
    levels = {
        "x1": [0.01, 0.03, 0.1, 0.3, 0.6],
        "x2": [1.0, 0.9, 0.75, 0.5, 0.25],
        "x4": [1.0, 0.9, 0.75, 0.5, 0.25],
    }
    local = []
    for name, values in levels.items():
        for value in values:
            params = dict(e19)
            params[name] = value
            result = runner.evaluate(params)
            local.append({"profile_parameter": name, "fixed_value": value, **result})
    pd.DataFrame(local).to_csv(output / "inward_sensitivity.csv", index=False)

    profile = []
    profile_levels = {"x1": [0.01, 0.1, 0.3], "x2": [1.0, 0.8, 0.5], "x4": [1.0, 0.8, 0.5]}
    for parameter, values in profile_levels.items():
        for index, value in enumerate(values):
            path = output / f"profile_{parameter}_{value:.3f}.json"
            def objective(params):
                result = runner.evaluate(params)
                return result, result["nse"]

            records, best = run_bo(
                runner,
                path,
                objective,
                iterations=6,
                random_state=110 + index + 10 * PARAMETERS.index(parameter),
                fixed={parameter: value},
            )
            anchor_parameters = dict(e19)
            anchor_parameters[parameter] = value
            anchor_metrics = runner.evaluate(anchor_parameters)
            anchored_best = {
                "parameters": anchor_parameters,
                "metrics": anchor_metrics,
                "source": "best-evaluation inward anchor",
            }
            if best["metrics"]["nse"] > anchor_metrics["nse"]:
                anchored_best = {
                    "parameters": best["parameters"],
                    "metrics": best["metrics"],
                    "source": "six-evaluation conditional search",
                }
            profile.append(
                {
                    "profile_parameter": parameter,
                    "fixed_value": value,
                    "evaluations": len(records),
                    "search_best_nse": best["metrics"]["nse"],
                    "anchor_nse": anchor_metrics["nse"],
                    "conditional_best_nse": anchored_best["metrics"]["nse"],
                    "conditional_best_source": anchored_best["source"],
                    "best_parameters": anchored_best["parameters"],
                }
            )
    pd.DataFrame(profile).to_csv(output / "conditional_profile.csv", index=False)
    summary = {
        "experiment": "S1",
        "local_sensitivity_points": len(local),
        "conditional_profile_evaluations": sum(item["evaluations"] for item in profile),
        "profiles": profile,
    }
    write_json(output / "summary.json", summary)
    return summary


def run_s2(runner, audit_root):
    output = audit_root / "S2_temporal_transferability"
    output.mkdir(parents=True, exist_ok=True)
    directions = [("early_to_late", 2011, 2013, 2014, 2016), ("late_to_early", 2014, 2016, 2011, 2013)]
    results = []
    for index, (name, cal_start, cal_end, val_start, val_end) in enumerate(directions):
        def objective(params, start=cal_start, end=cal_end):
            result = runner.evaluate(params, start, end)
            return result, result["nse"]

        _, best = run_bo(
            runner,
            output / f"{name}_calibration.json",
            objective,
            iterations=15,
            random_state=210 + index,
        )
        validation = runner.evaluate(best["parameters"], val_start, val_end)
        results.append(
            {
                "direction": name,
                "calibration_period": f"{cal_start}-{cal_end}",
                "validation_period": f"{val_start}-{val_end}",
                "calibration_metrics": best["metrics"],
                "validation_metrics": validation,
                "best_parameters": best["parameters"],
            }
        )
    summary = {"experiment": "S2", "budget_per_direction": 15, "directions": results}
    write_json(output / "summary.json", summary)
    return summary


def run_s3(runner, audit_root):
    output = audit_root / "S3_objective_adequacy"
    output.mkdir(parents=True, exist_ok=True)

    def nse_objective(params):
        result = runner.evaluate(params)
        return result, result["nse"]

    def multi_objective(params):
        result = runner.evaluate(params)
        bias_skill = max(-1.0, 1.0 - abs(result["pbias"]) / 100.0)
        score = (result["nse"] + result["log_nse"] + bias_skill) / 3.0
        result = {**result, "bias_skill": bias_skill, "composite_score": score}
        return result, score

    _, control = run_bo(
        runner,
        output / "nse_only_calibration.json",
        nse_objective,
        iterations=15,
        random_state=310,
    )
    _, treatment = run_bo(
        runner,
        output / "multi_objective_calibration.json",
        multi_objective,
        iterations=15,
        random_state=310,
    )
    summary = {
        "experiment": "S3",
        "budget_per_arm": 15,
        "control": control,
        "multi_objective": treatment,
        "delta": {
            "nse": treatment["metrics"]["nse"] - control["metrics"]["nse"],
            "pbias": treatment["metrics"]["pbias"] - control["metrics"]["pbias"],
            "low_flow_pbias": treatment["metrics"]["low_flow_pbias"] - control["metrics"]["low_flow_pbias"],
            "log_nse": treatment["metrics"]["log_nse"] - control["metrics"]["log_nse"],
        },
    }
    write_json(output / "summary.json", summary)
    return summary


def record_execution(decision_root, summaries, execution_run_id, executed, reused):
    audit_path = decision_root / "decision_audit.json"
    pending_path = decision_root / "pending_decisions.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    for item in pending:
        summary = summaries[item["case_id"]]
        item["execution_events"] = [
            {
                "status": "completed",
                "execution_run_id": execution_run_id,
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "summary_path": str(summary),
            }
        ]
    audit["pending_decisions"] = pending
    audit["metrics"]["approved_experiments_completed"] = 3
    audit["metrics"]["vic_executions"] = executed
    audit["metrics"]["cached_simulations_reused"] = reused
    write_json(pending_path, pending)
    write_json(audit_path, audit)


def main():
    parser = argparse.ArgumentParser(description="Execute the approved S1-S3 experiments.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-run-id", default="web_demo")
    parser.add_argument("--execution-run-id", default="decision_execution_demo")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=12)
    parser.add_argument(
        "--experiment", choices=("S1", "S2", "S3", "all"), default="all"
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    decision_root = project_root / "runs/decision_demo/output/decision_audit"
    pending = json.loads((decision_root / "pending_decisions.json").read_text())
    if not all(item["status"] == "approved_for_execution" and item["execution_authorized"] for item in pending):
        raise ValueError("S1-S3 must all be approved before execution")
    run_dir = setup_run(project_root, args.source_run_id, args.execution_run_id)
    audit_root = run_dir / "output/scientific_experiments"
    audit_root.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(
        project_root / "runs" / args.source_run_id / "logs/calibration_history.csv"
    )
    best = history.loc[history["nse"].astype(float).idxmax()]
    e19 = {name: float(best[name]) for name in PARAMETERS}
    runner = ExperimentRunner(project_root, args.execution_run_id, args.source_dir, args.processes, audit_root)
    results = {}
    if args.experiment in {"S1", "all"}:
        results["S1"] = run_s1(runner, audit_root, e19)
    if args.experiment in {"S2", "all"}:
        results["S2"] = run_s2(runner, audit_root)
    if args.experiment in {"S3", "all"}:
        results["S3"] = run_s3(runner, audit_root)
    overall = {
        "execution_run_id": args.execution_run_id,
        "experiment_selection": args.experiment,
        "vic_executions": runner.executed,
        "cached_simulations_reused": runner.reused,
        **results,
    }
    write_json(audit_root / "execution_summary.json", overall)
    if args.experiment == "all":
        paths = {
            "S1": audit_root / "S1_parameter_identifiability/summary.json",
            "S2": audit_root / "S2_temporal_transferability/summary.json",
            "S3": audit_root / "S3_objective_adequacy/summary.json",
        }
        record_execution(
            decision_root, paths, args.execution_run_id, runner.executed, runner.reused
        )
    print(json.dumps({"vic_executions": runner.executed, "cache_reused": runner.reused}, indent=2))
    print(audit_root / "execution_summary.json")


if __name__ == "__main__":
    main()
