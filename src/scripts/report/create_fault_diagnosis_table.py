import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render the controlled fault-diagnosis evidence as Table 2."
    )
    parser.add_argument("--run-id", default="diagnosis_demo")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("report_assets/figures")
    )
    return parser.parse_args()


def compact_rows(summary):
    cases = {case["case_id"]: case for case in summary["cases"]}
    expected = {"D1", "D2", "D3", "D4"}
    if set(cases) != expected:
        raise ValueError(f"Expected {sorted(expected)}, found {sorted(cases)}")

    d1 = cases["D1"]
    d2 = cases["D2"]
    d3 = cases["D3"]
    d4 = cases["D4"]
    rows = [
        {
            "case": "D1",
            "scope": "Input completeness",
            "fault": "Remove one active-cell forcing file",
            "evidence": (
                f"Inventory {d1['run_evidence']['forcing_files_before']} → "
                f"{d1['run_evidence']['forcing_files_after_injection']}; "
                f"VIC return code {d1['run_evidence']['vic_returncode']}; "
                "missing-file error"
            ),
            "stage": d1["diagnosis"]["failed_stage"],
            "target": d1["diagnosis"]["correction_target"]["object"],
            "avoided": "Calibration parameters",
        },
        {
            "case": "D2",
            "scope": "Temporal support",
            "fault": "Shift observations outside the simulation period",
            "evidence": (
                f"Observed {d2['run_evidence']['observation_period'].replace('..', '–')}; "
                f"simulated {d2['run_evidence']['simulation_period'].replace('..', '–')}; "
                f"common months = {d2['run_evidence']['common_months']}"
            ),
            "stage": d2["diagnosis"]["failed_stage"],
            "target": d2["diagnosis"]["correction_target"]["object"],
            "avoided": "Calibration parameters",
        },
        {
            "case": "D3",
            "scope": "Station identity",
            "fault": "Request a non-existent calibration station",
            "evidence": (
                f"Requested {d3['run_evidence']['requested_station']}; "
                f"declared {', '.join(d3['run_evidence']['declared_stations'])}; "
                "monthly output absent"
            ),
            "stage": d3["diagnosis"]["failed_stage"],
            "target": d3["diagnosis"]["correction_target"]["object"],
            "avoided": "Forcing and parameters",
        },
        {
            "case": "D4",
            "scope": "Spatial mapping",
            "fault": "Assign the outlet to an inactive routing cell",
            "evidence": (
                f"Cell ({d4['run_evidence']['routing_column']}, "
                f"{d4['run_evidence']['routing_row']}); mask = "
                f"{d4['run_evidence']['mask_value']}; one flux missing; "
                f"return code {d4['run_evidence']['vic_returncode']}"
            ),
            "stage": d4["diagnosis"]["failed_stage"],
            "target": d4["diagnosis"]["correction_target"]["object"],
            "avoided": "Parameters and observations",
        },
    ]
    if not all(cases[row["case"]]["outcome"] == "PASS" for row in rows):
        raise ValueError("At least one controlled diagnosis case did not pass")
    return rows


def wrap(value, width):
    return "\n".join(
        textwrap.wrap(
            str(value), width=width, break_long_words=False, break_on_hyphens=False
        )
    )


def draw_cell(ax, x, y, width, height, facecolor, edgecolor="#d5dde2", lw=0.8):
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


def create_table(rows, metrics, output_png, output_pdf):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(16.2, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.025,
        0.965,
        "Evidence-grounded fault diagnosis under controlled injections",
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#000000",
    )
    ax.text(
        0.025,
        0.922,
        "Injected fault  →  observed run evidence  →  failed-stage attribution  →  corrective object",
        ha="left",
        va="top",
        fontsize=11,
        color="#000000",
    )

    x0, table_width = 0.025, 0.95
    column_widths = [0.10, 0.17, 0.225, 0.16, 0.19, 0.155]
    headers = [
        "Case",
        "Controlled injection",
        "Observed evidence",
        "Attributed stage",
        "Corrective object",
        "Unwarranted action avoided",
    ]
    header_y, header_h = 0.833, 0.071
    x_positions = [x0]
    for width in column_widths[:-1]:
        x_positions.append(x_positions[-1] + width * table_width)

    for x, width, header in zip(x_positions, column_widths, headers):
        width *= table_width
        draw_cell(ax, x, header_y, width, header_h, "#314b5d", "white", 0.7)
        ax.text(
            x + 0.008,
            header_y + header_h / 2,
            wrap(header, 20),
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white",
            linespacing=1.15,
        )

    row_h = 0.138
    body_top = header_y
    fills = ["#f7fafb", "#ffffff"]
    wrap_widths = [10, 25, 37, 21, 27, 24]
    for index, row in enumerate(rows):
        y = body_top - (index + 1) * row_h
        row_fill = fills[index % 2]
        values = [
            f"{row['case']}\n{row['scope']}",
            row["fault"],
            row["evidence"],
            row["stage"].replace("_", "\n"),
            f"PASS\n{row['target'].replace('_', '-')}",
            row["avoided"],
        ]
        for col, (x, width, value) in enumerate(
            zip(x_positions, column_widths, values)
        ):
            width *= table_width
            facecolor = row_fill
            if col == 3:
                facecolor = "#fff8e8"
            elif col == 4:
                facecolor = "#eef8f2"
            draw_cell(ax, x, y, width, row_h, facecolor)
            color = "#000000"
            fontweight = "bold" if col in {0, 4} else "normal"
            if row["case"] == "D4" and col == 2:
                color = "#000000"
                fontweight = "bold"
            ax.text(
                x + 0.009,
                y + row_h / 2,
                wrap(value, wrap_widths[col]),
                ha="left",
                va="center",
                fontsize=8.8 if col == 0 else 9.2,
                fontweight=fontweight,
                color=color,
                linespacing=1.25,
            )
        if row["case"] == "D4":
            ax.add_patch(
                Rectangle(
                    (x0, y),
                    table_width,
                    row_h,
                    transform=ax.transAxes,
                    facecolor="none",
                    edgecolor="#d97728",
                    linewidth=2.0,
                )
            )

    metric_y, metric_h = 0.11, 0.095
    metric_values = [
        ("Stage attribution", f"{metrics['stage_attribution_correct']}/{metrics['cases']}"),
        ("Corrective target", f"{metrics['correction_target_correct']}/{metrics['cases']}"),
        ("Evidence complete", f"{metrics['evidence_complete']}/{metrics['cases']}"),
        ("Unsupported modifications", f"{metrics['unsupported_modifications']}/{metrics['cases']}"),
    ]
    gap = 0.012
    metric_width = (table_width - gap * 3) / 4
    for index, (label, value) in enumerate(metric_values):
        x = x0 + index * (metric_width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, metric_y),
                metric_width,
                metric_h,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                transform=ax.transAxes,
                facecolor="#eef6f2" if index < 3 else "#f4f6f7",
                edgecolor="#4b8a70" if index < 3 else "#80909a",
                linewidth=1.1,
            )
        )
        ax.text(
            x + 0.012,
            metric_y + metric_h * 0.63,
            label,
            ha="left",
            va="center",
            fontsize=9.2,
            color="#000000",
        )
        ax.text(
            x + metric_width - 0.012,
            metric_y + metric_h * 0.48,
            value,
            ha="right",
            va="center",
            fontsize=17,
            fontweight="bold",
            color="#000000",
        )

    ax.text(
        x0,
        0.057,
        "PASS denotes agreement with the prespecified controlled fixture; it is not an estimate of accuracy for unseen or compound faults.",
        ha="left",
        va="center",
        fontsize=9.2,
        color="#000000",
        style="italic",
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    summary_path = (
        project_root
        / "runs"
        / args.run_id
        / "output"
        / "diagnosis_audit"
        / "diagnosis_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = compact_rows(summary)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_png = output_dir / "table2_fault_diagnosis.png"
    output_pdf = output_dir / "table2_fault_diagnosis.pdf"
    evidence_path = output_dir / "table2_fault_diagnosis_evidence.json"
    create_table(rows, summary["metrics"], output_png, output_pdf)
    evidence_path.write_text(
        json.dumps(
            {
                "source": str(summary_path),
                "source_sha256": file_sha256(summary_path),
                "rows": rows,
                "metrics": summary["metrics"],
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
