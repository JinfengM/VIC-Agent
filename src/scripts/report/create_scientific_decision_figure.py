import argparse
import hashlib
import json
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wrap(value, width):
    return "\n".join(
        textwrap.wrap(
            str(value), width=width, break_long_words=False, break_on_hyphens=False
        )
    )


def draw_cell(ax, x, y, width, height, facecolor, edgecolor="#d2dce2", lw=0.9):
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            transform=ax.transAxes,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=lw,
        )
    )


def compact_rows(audit):
    records_by_case = {
        case_id: [
            record for record in audit["records"] if record["case_id"] == case_id
        ]
        for case_id in ("S1", "S2", "S3")
    }
    pending = {item["case_id"]: item for item in audit["pending_decisions"]}
    cases = {case["case_id"]: case for case in audit["cases"]}
    evidence_text = {
        "S1": "Best E19/100, NSE 0.8879; no improvement in 81 later evaluations; x1, x2 and x4 at physical bounds",
        "S2": "Same E19 vector: NSE 0.9362 in 2011–2013 versus 0.1772 in 2014–2016",
        "S3": "NSE 0.8879, but PBIAS −36.2% and low-flow PBIAS −93.9%",
    }
    question_text = {
        "S1": "Parameter identifiability",
        "S2": "Temporal transferability",
        "S3": "Objective adequacy",
    }
    decision_text = {
        "S1": "Anchored conditional profiles and inward sensitivity",
        "S2": "Reciprocal split-sample calibration and validation",
        "S3": "Controlled multi-objective calibration",
    }
    rows = []
    for case_id in ("S1", "S2", "S3"):
        records = records_by_case[case_id]
        choices = [record["response"]["selected_experiment"] for record in records]
        consensus, count = Counter(choices).most_common(1)[0]
        item = pending[case_id]
        if consensus != item["recommended_experiment"]:
            raise ValueError(f"Consensus mismatch for {case_id}")
        if len(item["execution_events"]) != 1:
            raise ValueError(f"Expected one completed execution event for {case_id}")
        event = item["execution_events"][0]
        if event["status"] != "completed":
            raise ValueError(f"Execution is not complete for {case_id}")
        result = event["result"]
        if case_id == "S1":
            nse = result["conditional_nse"]
            result_text = (
                f"{result['unique_vic_simulations']} VIC runs; conditional NSE: x1 {nse['x1_boundary_0.01']:.4f}→"
                f"{nse['x1_inward_0.30']:.4f}; x2 {nse['x2_boundary_1.00']:.4f}→"
                f"{nse['x2_inward_0.50']:.4f}; x4 {nse['x4_boundary_1.00']:.4f}→"
                f"{nse['x4_inward_0.80']:.4f} at 0.8. Mixed identifiability."
            )
        elif case_id == "S2":
            result_text = (
                f"{result['unique_vic_simulations']} VIC runs; validation NSE = {result['early_to_late_validation_nse']:.4f} "
                f"(early→late) and {result['late_to_early_validation_nse']:.4f} "
                "(late→early). Asymmetric transfer."
            )
        else:
            delta = result["delta_multiobjective_minus_control"]
            result_text = (
                f"{result['unique_vic_simulations']} unique VIC runs; ΔPBIAS +{delta['pbias']:.2f} pp; "
                f"Δlow-flow PBIAS +{delta['low_flow_pbias']:.2f} pp; "
                f"ΔlogNSE +{delta['log_nse']:.2f}; ΔNSE {delta['nse']:.3f}."
            )
        rows.append(
            {
                "case": case_id,
                "question": question_text[case_id],
                "evidence": evidence_text[case_id],
                "decision": decision_text[case_id],
                "choice": consensus,
                "consensus": f"{count}/{len(records)} repeats",
                "reference_match": consensus == cases[case_id]["reference_experiment"],
                "status": item["status"],
                "authorized": item["execution_authorized"],
                "execution_events": len(item["execution_events"]),
                "execution_run_id": event["execution_run_id"],
                "result": result_text,
                "conclusion": result["conclusion"],
            }
        )
    return rows


def create_figure(audit, rows, output_png, output_pdf):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(17.6, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.025,
        0.96,
        "Evidence-grounded, human-supervised scientific decisions followed by execution",
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#000000",
    )
    ax.text(
        0.025,
        0.913,
        "Run evidence  →  scientific question  →  LLM-selected experiment  →  human authorization  →  measured outcome",
        ha="left",
        va="top",
        fontsize=11,
        color="#000000",
    )

    x0, table_width = 0.025, 0.95
    widths = [0.105, 0.22, 0.225, 0.29, 0.16]
    headers = [
        "Scientific\nquestion",
        "Observed run\nevidence",
        "LLM-selected next\nexperiment",
        "Executed result",
        "Authorization\nand status",
    ]
    header_y, header_h = 0.815, 0.075
    positions = [x0]
    for width in widths[:-1]:
        positions.append(positions[-1] + width * table_width)
    for x, width, header in zip(positions, widths, headers):
        width *= table_width
        draw_cell(ax, x, header_y, width, header_h, "#314b5d", "white", 0.7)
        ax.text(
            x + 0.009,
            header_y + header_h / 2,
            header,
            ha="left",
            va="center",
            fontsize=9.3,
            fontweight="bold",
            color="white",
            linespacing=1.15,
        )

    row_h = 0.175
    fills = ["#f7fafb", "#ffffff"]
    for index, row in enumerate(rows):
        y = header_y - (index + 1) * row_h
        gate_status = row["status"]
        if gate_status == "pending_human_review":
            gate_text = "BLOCKED\npending human review"
            gate_face = "#fff6df"
        elif gate_status == "approved_for_execution":
            gate_text = "BLOCKED before review\n→ APPROVED by human"
            gate_face = "#eaf6ef"
        else:
            gate_text = "BLOCKED\nrejected by human"
            gate_face = "#fdecec"
        if row["execution_events"]:
            gate_text += "\n→ COMPLETED"
        values = [
            f"{row['case']}\n{row['question']}",
            row["evidence"],
            f"{row['choice']}  {row['decision']}\n{row['consensus']}; reference match",
            row["result"],
            gate_text,
        ]
        wrap_widths = [17, 37, 38, 52, 22]
        for col, (x, width, value) in enumerate(zip(positions, widths, values)):
            width *= table_width
            face = fills[index % 2]
            color = "#000000"
            weight = "normal"
            if col == 2:
                face, color, weight = "#edf7f2", "#000000", "bold"
            elif col == 3:
                face, color = "#f7fafb", "#000000"
            elif col == 4:
                face, color, weight = gate_face, "#000000", "bold"
            draw_cell(ax, x, y, width, row_h, face)
            ax.text(
                x + 0.009,
                y + row_h / 2,
                wrap(value, wrap_widths[col]),
                ha="left",
                va="center",
                fontsize=9.0 if col in {0, 4} else 9.4,
                fontweight=weight,
                color=color,
                linespacing=1.25,
            )

    metrics = audit["metrics"]
    total = metrics["responses"]
    metric_values = [
        ("Reference-choice agreement", f"{metrics['reference_choice_agreement']}/{total}"),
        ("Human-approved plans", f"{metrics['human_approved']}/{metrics['cases']}"),
        ("Experiments completed", f"{metrics['approved_experiments_completed']}/{metrics['cases']}"),
        ("Unique VIC simulations", str(metrics["unique_vic_simulations"])),
    ]
    metric_y, metric_h = 0.095, 0.1
    gap = 0.012
    metric_width = (table_width - 3 * gap) / 4
    for index, (label, value) in enumerate(metric_values):
        x = x0 + index * (metric_width + gap)
        face = "#eef6f2"
        edge = "#4b8a70"
        ax.add_patch(
            FancyBboxPatch(
                (x, metric_y),
                metric_width,
                metric_h,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                transform=ax.transAxes,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.1,
            )
        )
        ax.text(
            x + 0.012,
            metric_y + metric_h * 0.62,
            label,
            ha="left",
            va="center",
            fontsize=9.2,
            color="#000000",
        )
        ax.text(
            x + metric_width - 0.012,
            metric_y + metric_h * 0.46,
            value,
            ha="right",
            va="center",
            fontsize=17,
            fontweight="bold",
            color="#000000",
        )

    ax.text(
        x0,
        0.045,
        wrap(
            "Three reordered-candidate LLM repeats per case. S1 used 15 inward tests plus nine six-evaluation conditional searches anchored by E19; S2 and S3 used matched 15-evaluation budgets. Results are case-specific.",
            155,
        ),
        ha="left",
        va="center",
        fontsize=9.0,
        color="#000000",
        style="italic",
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create the human-supervised scientific-decision evidence figure."
    )
    parser.add_argument("--run-id", default="decision_demo")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("report_assets/figures")
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    audit_path = (
        project_root
        / "runs"
        / args.run_id
        / "output"
        / "decision_audit"
        / "decision_audit.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rows = compact_rows(audit)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_png = output_dir / "human_supervised_scientific_decision.png"
    output_pdf = output_dir / "human_supervised_scientific_decision.pdf"
    evidence_path = output_dir / "human_supervised_scientific_decision_evidence.json"
    create_figure(audit, rows, output_png, output_pdf)
    evidence_path.write_text(
        json.dumps(
            {
                "source": str(audit_path),
                "source_sha256": file_sha256(audit_path),
                "rows": rows,
                "metrics": audit["metrics"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_png)
    print(output_pdf)
    print(evidence_path)


if __name__ == "__main__":
    main()
