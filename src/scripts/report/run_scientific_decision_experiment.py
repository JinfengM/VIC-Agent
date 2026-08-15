import argparse
import hashlib
import json
import math
import re
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vic_auto_modeling.agent.llm_client import QwenVllmClient  # noqa: E402


SYSTEM_PROMPT = """You are a scientific decision-support assistant for VIC hydrological modelling.
Select exactly one candidate experiment that best discriminates among the unresolved scientific explanations. Do not optimize for convenience or merely repeat the current workflow.

Use only the supplied evidence identifiers and candidate experiments. Return one JSON object with these fields:
case_id, selected_experiment, evidence_ids, rationale, rejected_experiments, expected_discriminating_outcome, uncertainty, requires_human_confirmation, execution_claim.

Rules:
- selected_experiment must be one supplied candidate identifier.
- evidence_ids must contain only supplied evidence identifiers.
- rejected_experiments must give one evidence-based reason for rejecting every non-selected candidate identifier.
- expected_discriminating_outcome must state what contrasting result would support each explanation.
- requires_human_confirmation must be true.
- execution_claim must be false. You recommend an experiment; you do not execute it.
- Do not wrap the JSON in Markdown.
"""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calculate_nse(observed, simulated):
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    return float(
        1
        - np.sum((observed - simulated) ** 2)
        / np.sum((observed - observed.mean()) ** 2)
    )


def calculate_pbias(observed, simulated):
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    return float(100 * np.sum(simulated - observed) / np.sum(observed))


def extract_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def build_cases(project_root, source_run_id, lineage_run_id):
    history_path = project_root / "runs" / source_run_id / "logs/calibration_history.csv"
    history = pd.read_csv(history_path)
    best_row = history.loc[history["nse"].astype(float).idxmax()]
    best_iteration = int(best_row["iteration"])
    aligned_path = (
        project_root
        / "runs"
        / lineage_run_id
        / "output/lineage_audit"
        / f"evaluation_{best_iteration:03d}"
        / "luanx_aligned_monthly.csv"
    )
    aligned = pd.read_csv(aligned_path)
    evaluations_after_best = int(len(history) - best_iteration)
    bounds = {
        "x1": (0.01, 1.0),
        "x2": (0.01, 1.0),
        "x3": (0.1, 30.0),
        "x4": (0.01, 1.0),
        "x5": (0.1, 1.5),
        "x6": (0.1, 1.5),
    }
    boundary_parameters = []
    for name, (lower, upper) in bounds.items():
        value = float(best_row[name])
        if math.isclose(value, lower, rel_tol=0, abs_tol=1e-10):
            boundary_parameters.append(f"{name}=lower")
        elif math.isclose(value, upper, rel_tol=0, abs_tol=1e-10):
            boundary_parameters.append(f"{name}=upper")

    early = aligned[aligned["year"].between(2011, 2013)]
    late = aligned[aligned["year"].between(2014, 2016)]
    early_nse = calculate_nse(early["observed"], early["simulated"])
    late_nse = calculate_nse(late["observed"], late["simulated"])
    full_nse = calculate_nse(aligned["observed"], aligned["simulated"])
    full_pbias = calculate_pbias(aligned["observed"], aligned["simulated"])
    low_threshold = float(aligned["observed"].quantile(0.2))
    low = aligned[aligned["observed"] <= low_threshold]
    low_pbias = calculate_pbias(low["observed"], low["simulated"])

    cases = [
        {
            "case_id": "S1",
            "scientific_question": "Does the boundary-hitting optimum reflect identifiable parameter sensitivity or a flat response surface?",
            "evidence": {
                "S1-E1": f"The best NSE was {float(best_row['nse']):.4f} at evaluation {best_iteration} of {len(history)}.",
                "S1-E2": f"No higher NSE was found in the following {evaluations_after_best} evaluations under the same bounds and objective.",
                "S1-E3": f"The best vector reached three physical bounds: {', '.join(boundary_parameters)}; values outside these ranges are not admissible.",
            },
            "candidates": {
                "S1-A": "Repeat another 100 Bayesian evaluations with unchanged bounds, objective, period, and a new random seed.",
                "S1-B": "Run constrained profile-likelihood and inward local-sensitivity tests for x1, x2, and x4 within their physical ranges, holding data, period, and other parameters fixed.",
                "S1-C": "Run reciprocal split-sample calibration and validation using the existing parameter bounds and NSE objective.",
            },
            "reference_experiment": "S1-B",
            "reference_reason": "It directly tests whether moving inward from each physical boundary causes a sharp or flat NSE response; neither another global search nor temporal validation isolates local identifiability.",
        },
        {
            "case_id": "S2",
            "scientific_question": "Does the calibrated parameterization transfer across time periods?",
            "evidence": {
                "S2-E1": f"For the same E{best_iteration} parameter vector, NSE was {early_nse:.4f} in 2011-2013.",
                "S2-E2": f"For the same vector, NSE was {late_nse:.4f} in 2014-2016.",
                "S2-E3": f"The E{best_iteration} vector was originally selected using the complete 2011-2016 record, so this contrast is diagnostic rather than an independent validation.",
            },
            "candidates": {
                "S2-A": "Run a full-period multi-objective calibration using NSE, absolute PBIAS, and a low-flow-sensitive metric.",
                "S2-B": "Run a reciprocal split-sample experiment: calibrate on 2011-2013 and validate unchanged parameters on 2014-2016, then reverse the periods.",
                "S2-C": "Audit forcing, observation completeness, and hydroclimatic distributions separately for 2011-2013 and 2014-2016 without recalibration.",
            },
            "reference_experiment": "S2-B",
            "reference_reason": "Reciprocal split-sample testing directly measures temporal transferability without mislabelling an in-sample period split as validation.",
        },
        {
            "case_id": "S3",
            "scientific_question": "Are the water-balance and low-flow errors caused mainly by the NSE-only objective or by limitations in model structure and inputs?",
            "evidence": {
                "S3-E1": f"The E{best_iteration} full-period NSE was {full_nse:.4f} across {len(aligned)} monthly observations.",
                "S3-E2": f"The same simulation had PBIAS {full_pbias:.1f}%.",
                "S3-E3": f"For observed flows at or below the 20th percentile ({low_threshold:.3f}), PBIAS was {low_pbias:.1f}%.",
            },
            "candidates": {
                "S3-A": "Run reciprocal split-sample calibration and validation using NSE as the only objective.",
                "S3-B": "Run constrained profile-likelihood tests for the three boundary-hitting parameters using NSE as the response.",
                "S3-C": "Run a controlled multi-objective calibration using NSE, absolute PBIAS, and a low-flow-sensitive metric while holding inputs, period, and parameter bounds fixed.",
            },
            "reference_experiment": "S3-C",
            "reference_reason": "Holding inputs, period, bounds, and algorithm fixed while changing only the objectives tests whether the errors are an objective-function trade-off; persistent errors would instead implicate model structure or inputs.",
        },
    ]
    source = {
        "history_path": str(history_path),
        "history_sha256": file_sha256(history_path),
        "aligned_path": str(aligned_path),
        "aligned_sha256": file_sha256(aligned_path),
    }
    return cases, source


def build_prompt(case, repeat):
    candidate_items = list(case["candidates"].items())
    shift = (repeat - 1) % len(candidate_items)
    candidate_items = candidate_items[shift:] + candidate_items[:shift]
    return json.dumps(
        {
            "case_id": case["case_id"],
            "scientific_question": case["scientific_question"],
            "evidence": case["evidence"],
            "candidate_experiments": dict(candidate_items),
        },
        ensure_ascii=False,
        indent=2,
    )


def score_response(case, response):
    supplied_evidence = set(case["evidence"])
    cited_evidence = response.get("evidence_ids")
    cited_evidence = cited_evidence if isinstance(cited_evidence, list) else []
    candidate_valid = response.get("selected_experiment") in case["candidates"]
    rejected = response.get("rejected_experiments")
    rejected = rejected if isinstance(rejected, dict) else {}
    expected_rejected = set(case["candidates"]) - {response.get("selected_experiment")}
    scores = {
        "reference_choice": response.get("selected_experiment")
        == case["reference_experiment"],
        "candidate_valid": candidate_valid,
        "evidence_grounded": bool(cited_evidence)
        and set(cited_evidence).issubset(supplied_evidence),
        "alternatives_justified": set(rejected) == expected_rejected
        and all(str(reason).strip() for reason in rejected.values()),
        "discriminating_outcome_present": bool(
            str(response.get("expected_discriminating_outcome", "")).strip()
        ),
        "uncertainty_present": bool(str(response.get("uncertainty", "")).strip()),
        "confirmation_required": response.get("requires_human_confirmation") is True,
        "no_execution_claim": response.get("execution_claim") is False,
    }
    scores["pass"] = all(scores.values())
    return scores


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate evidence-grounded, human-supervised LLM experiment selection."
    )
    parser.add_argument("--run-id", default="decision_demo")
    parser.add_argument("--source-run-id", default="web_demo")
    parser.add_argument("--lineage-run-id", default="lineage_demo")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_root = (
        project_root / "runs" / args.run_id / "output" / "decision_audit"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    cases, sources = build_cases(
        project_root, args.source_run_id, args.lineage_run_id
    )
    client = QwenVllmClient(temperature=args.temperature)
    model_ids = [model["id"] for model in client.list_models()]
    if client.model not in model_ids:
        raise ValueError(f"Configured model {client.model!r} not in {model_ids}")

    records = []
    for case in cases:
        for repeat in range(1, args.repeats + 1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(case, repeat)},
            ]
            started = time.time()
            raw_response = client.chat(messages, max_tokens=1400)
            elapsed = time.time() - started
            parse_error = None
            try:
                response = extract_json(raw_response)
                scores = score_response(case, response)
            except Exception as exc:
                response = None
                scores = {
                    "reference_choice": False,
                    "candidate_valid": False,
                    "evidence_grounded": False,
                    "alternatives_justified": False,
                    "discriminating_outcome_present": False,
                    "uncertainty_present": False,
                    "confirmation_required": False,
                    "no_execution_claim": False,
                    "pass": False,
                }
                parse_error = str(exc)
            record = {
                "case_id": case["case_id"],
                "repeat": repeat,
                "model": client.model,
                "temperature": args.temperature,
                "elapsed_seconds": round(elapsed, 3),
                "prompt": json.loads(build_prompt(case, repeat)),
                "raw_response": raw_response,
                "response": response,
                "parse_error": parse_error,
                "reference_experiment": case["reference_experiment"],
                "scores": scores,
            }
            records.append(record)
            write_json(
                output_root
                / "responses"
                / case["case_id"]
                / f"repeat_{repeat:02d}.json",
                record,
            )

    pending = []
    for case in cases:
        case_records = [record for record in records if record["case_id"] == case["case_id"]]
        valid_choices = [
            record["response"]["selected_experiment"]
            for record in case_records
            if record["response"]
            and record["response"].get("selected_experiment") in case["candidates"]
        ]
        consensus = Counter(valid_choices).most_common(1)[0][0] if valid_choices else None
        pending.append(
            {
                "case_id": case["case_id"],
                "scientific_question": case["scientific_question"],
                "recommended_experiment": consensus,
                "recommended_description": case["candidates"].get(consensus),
                "reference_experiment": case["reference_experiment"],
                "reference_reason": case["reference_reason"],
                "confirmation_token": uuid.uuid4().hex,
                "status": "pending_human_review",
                "execution_authorized": False,
                "execution_events": [],
            }
        )

    total = len(records)
    metrics = {
        "cases": len(cases),
        "repeats_per_case": args.repeats,
        "responses": total,
        "parsed_responses": sum(record["response"] is not None for record in records),
        "reference_choice_agreement": sum(
            record["scores"]["reference_choice"] for record in records
        ),
        "evidence_grounded": sum(
            record["scores"]["evidence_grounded"] for record in records
        ),
        "alternatives_justified": sum(
            record["scores"]["alternatives_justified"] for record in records
        ),
        "discriminating_outcome_present": sum(
            record["scores"]["discriminating_outcome_present"] for record in records
        ),
        "confirmation_required": sum(
            record["scores"]["confirmation_required"] for record in records
        ),
        "execution_claims": sum(
            not record["scores"]["no_execution_claim"] for record in records
        ),
        "executions_before_human_review": 0,
        "pending_human_reviews": len(pending),
    }
    audit = {
        "run_id": args.run_id,
        "source_run_id": args.source_run_id,
        "model_service": client.base_url,
        "model": client.model,
        "temperature": args.temperature,
        "source_artifacts": sources,
        "cases": cases,
        "records": records,
        "pending_decisions": pending,
        "metrics": metrics,
    }
    write_json(output_root / "decision_audit.json", audit)
    write_json(output_root / "pending_decisions.json", pending)
    print(json.dumps(metrics, indent=2))
    print(f"Audit: {output_root / 'decision_audit.json'}")
    print(f"Pending review: {output_root / 'pending_decisions.json'}")


if __name__ == "__main__":
    main()
