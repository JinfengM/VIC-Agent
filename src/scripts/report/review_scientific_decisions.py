import argparse
import json
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Record an explicit human review of a pending scientific decision."
    )
    parser.add_argument("--run-id", default="decision_demo")
    parser.add_argument("--case-id", required=True, choices=("S1", "S2", "S3"))
    parser.add_argument("--token", required=True)
    parser.add_argument("--decision", required=True, choices=("approve", "reject"))
    parser.add_argument("--reviewer-note", default="")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    output_root = (
        args.project_root.resolve()
        / "runs"
        / args.run_id
        / "output"
        / "decision_audit"
    )
    pending_path = output_root / "pending_decisions.json"
    audit_path = output_root / "decision_audit.json"
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    matches = [item for item in pending if item["case_id"] == args.case_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one pending record for {args.case_id}")
    item = matches[0]
    if item["confirmation_token"] != args.token:
        raise ValueError("Confirmation token does not match the pending decision")
    if item["status"] != "pending_human_review":
        raise ValueError(f"Decision has already been reviewed: {item['status']}")

    approved = args.decision == "approve"
    item["status"] = "approved_for_execution" if approved else "rejected_by_human"
    item["execution_authorized"] = approved
    item["human_review"] = {
        "decision": args.decision,
        "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reviewer_note": args.reviewer_note,
    }
    audit["pending_decisions"] = pending
    metrics = audit["metrics"]
    metrics["pending_human_reviews"] = sum(
        decision["status"] == "pending_human_review" for decision in pending
    )
    metrics["human_reviews_completed"] = sum(
        decision["status"] != "pending_human_review" for decision in pending
    )
    metrics["human_approved"] = sum(
        decision["status"] == "approved_for_execution" for decision in pending
    )
    metrics["human_rejected"] = sum(
        decision["status"] == "rejected_by_human" for decision in pending
    )
    write_json(pending_path, pending)
    write_json(audit_path, audit)
    print(
        json.dumps(
            {
                "case_id": args.case_id,
                "status": item["status"],
                "execution_authorized": item["execution_authorized"],
                "execution_events": len(item["execution_events"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
