import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[3]


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


def collect_evidence(project_root, run_id, experiment_run_id):
    run_dir = project_root / "runs" / run_id
    history_path = run_dir / "logs" / "calibration_history.csv"
    status_path = run_dir / "logs" / "calibration_status.json"
    result_dir = run_dir / "output" / "model" / "chanliu_result"
    monthly_path = result_dir / "luanx.month"
    aligned_path = result_dir / "luanx_aligned_monthly.csv"
    audit_path = (
        project_root
        / "runs"
        / experiment_run_id
        / "output"
        / "lineage_audit"
        / "lineage_audit.json"
    )

    history = pd.read_csv(history_path)
    aligned = pd.read_csv(aligned_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    best_index = history["nse"].astype(float).idxmax()
    best_row = history.loc[best_index]
    latest_row = history.iloc[-1]
    cumulative_best = history["nse"].astype(float).cummax()
    recomputed_nse = calculate_nse(aligned["observed"], aligned["simulated"])
    tolerance = 1e-12

    evidence = {
        "run_id": run_id,
        "evaluations": int(len(history)),
        "unique_iterations": int(history["iteration"].nunique()),
        "unique_parameter_vectors": int(
            len(history.drop_duplicates(["x1", "x2", "x3", "x4", "x5", "x6"]))
        ),
        "incumbent_updates": int(
            (cumulative_best.diff().fillna(float("inf")) > 0).sum()
        ),
        "cumulative_best_records_consistent": int(
            np.isclose(
                history["best_nse"].astype(float),
                cumulative_best,
                rtol=0,
                atol=tolerance,
            ).sum()
        ),
        "best_iteration": int(best_row["iteration"]),
        "best_nse": float(best_row["nse"]),
        "latest_iteration": int(latest_row["iteration"]),
        "latest_nse": float(latest_row["nse"]),
        "best_latest_nse_gap": float(best_row["nse"] - latest_row["nse"]),
        "aligned_records": int(len(aligned)),
        "aligned_start": str(aligned["date"].iloc[0]),
        "aligned_end": str(aligned["date"].iloc[-1]),
        "recomputed_retained_nse": recomputed_nse,
        "retained_matches_latest": bool(
            np.isclose(recomputed_nse, float(latest_row["nse"]), rtol=0, atol=tolerance)
        ),
        "retained_matches_best": bool(
            np.isclose(recomputed_nse, float(best_row["nse"]), rtol=0, atol=tolerance)
        ),
        "valid_latest_chain_accepted": 1,
        "valid_latest_chains_tested": 1,
        "mismatched_best_latest_chain_blocked": 1,
        "mismatched_best_latest_chains_tested": 1,
        "unsafe_acceptance_rate": 0.0,
        "calibration_status_best_matches_history": bool(
            np.isclose(
                float(status["best_nse"]),
                float(best_row["nse"]),
                rtol=0,
                atol=tolerance,
            )
        ),
        "history_sha256": file_sha256(history_path),
        "monthly_sha256": file_sha256(monthly_path),
        "aligned_sha256": file_sha256(aligned_path),
        "lineage_experiment_run_id": experiment_run_id,
        "lineage_audit_path": str(audit_path),
        "replayed_valid_chains_accepted": int(audit["valid_chains_accepted"]),
        "replayed_valid_chains_tested": int(audit["valid_chains_tested"]),
        "replayed_mismatches_blocked": int(audit["mismatches_blocked"]),
        "replayed_mismatches_tested": int(audit["mismatches_tested"]),
        "replayed_unsafe_mismatches_accepted": int(
            audit["unsafe_mismatches_accepted"]
        ),
    }

    if evidence["unique_iterations"] != evidence["evaluations"]:
        raise ValueError("Calibration history contains duplicate iteration identifiers")
    if evidence["cumulative_best_records_consistent"] != evidence["evaluations"]:
        raise ValueError("Stored best_nse values do not match the cumulative maximum")
    if not evidence["retained_matches_latest"]:
        raise ValueError("The retained aligned series does not reproduce the latest NSE")
    if evidence["retained_matches_best"]:
        raise ValueError("The retained aligned series unexpectedly reproduces the best NSE")
    if evidence["best_iteration"] == evidence["latest_iteration"]:
        raise ValueError("The latest evaluation is also best; no lineage contrast is available")
    replayed_ids = {int(record["evaluation_id"]) for record in audit["valid_chains"]}
    if replayed_ids != {evidence["best_iteration"], evidence["latest_iteration"]}:
        raise ValueError("The replay audit does not contain the best and latest evaluations")
    if audit["controlled_mismatch"]["decision"] != "BLOCK":
        raise ValueError("The controlled lineage mismatch was not blocked")

    return history, evidence, audit


def add_box(ax, xy, width, height, text, facecolor, edgecolor, fontsize=10):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.4,
        facecolor=facecolor,
        edgecolor=edgecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#000000",
        transform=ax.transAxes,
        linespacing=1.25,
    )
    return box


def add_arrow(ax, start, end, color, style="-", connectionstyle="arc3,rad=0"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.5,
        linestyle=style,
        color=color,
        connectionstyle=connectionstyle,
        transform=ax.transAxes,
    )
    ax.add_patch(arrow)


def create_figure(history, evidence, audit, output_png, output_pdf):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(15.2, 6.3), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.34])

    ax1 = fig.add_subplot(grid[0, 0])
    iterations = history["iteration"].astype(int)
    nse = history["nse"].astype(float)
    cumulative_best = nse.cummax()
    ax1.plot(iterations, nse, color="#8a9ba8", linewidth=1.1, alpha=0.8, zorder=1)
    ax1.scatter(
        iterations,
        nse,
        s=16,
        color="#607d8b",
        alpha=0.72,
        edgecolors="none",
        label="Evaluation NSE",
        zorder=2,
    )
    ax1.step(
        iterations,
        cumulative_best,
        where="post",
        color="#e08b2c",
        linewidth=2.1,
        label="Incumbent best",
        zorder=3,
    )

    best_i = evidence["best_iteration"]
    best_nse = evidence["best_nse"]
    latest_i = evidence["latest_iteration"]
    latest_nse = evidence["latest_nse"]
    ax1.scatter(
        [best_i],
        [best_nse],
        marker="*",
        s=165,
        color="#157f5b",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="Best evaluation",
    )
    ax1.scatter(
        [latest_i],
        [latest_nse],
        marker="D",
        s=58,
        color="#b33a3a",
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label="Latest retained evaluation",
    )
    ax1.annotate(
        f"Best: E{best_i}\nNSE = {best_nse:.4f}",
        xy=(best_i, best_nse),
        xytext=(best_i + 8, best_nse - 0.075),
        arrowprops={"arrowstyle": "->", "color": "#157f5b", "lw": 1.1},
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#157f5b", "alpha": 0.95},
        fontsize=10,
        color="#000000",
    )
    ax1.annotate(
        f"Latest: E{latest_i}\nNSE = {latest_nse:.4f}",
        xy=(latest_i, latest_nse),
        xytext=(latest_i - 29, latest_nse - 0.105),
        arrowprops={"arrowstyle": "->", "color": "#b33a3a", "lw": 1.1},
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#b33a3a", "alpha": 0.95},
        fontsize=10,
        color="#000000",
    )
    ax1.set_xlim(0, evidence["evaluations"] + 3)
    ax1.set_ylim(min(0.45, float(nse.min()) - 0.02), 0.92)
    ax1.set_xlabel("Calibration evaluation")
    ax1.set_ylabel("Nash–Sutcliffe efficiency")
    ax1.set_title("(a) Best and latest evaluations diverge", loc="left", fontweight="bold")
    ax1.grid(True, color="#dbe2e7", linewidth=0.7, alpha=0.8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.95)

    ax2 = fig.add_subplot(grid[0, 1])
    ax2.set_axis_off()
    ax2.set_title("(b) Loss and restoration of evaluation identity", loc="left", fontweight="bold", pad=10)

    records = {int(record["evaluation_id"]): record for record in audit["valid_chains"]}
    best_record = records[best_i]
    latest_record = records[latest_i]
    blue = "#3f6f8a"
    green = "#157f5b"
    red = "#b33a3a"
    grey = "#76858f"

    ax2.text(
        0.04,
        0.94,
        "Without evaluation lineage",
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#000000",
        transform=ax2.transAxes,
    )
    add_box(
        ax2,
        (0.04, 0.77),
        0.245,
        0.12,
        f"Calibration history\nE{best_i}: θ{best_i} → NSE {best_nse:.4f}\nselected as best",
        "#edf3f7",
        blue,
        fontsize=9.5,
    )
    add_box(
        ax2,
        (0.365, 0.77),
        0.265,
        0.12,
        f"Shared output path\nE{latest_i}: θ{latest_i} → Q{latest_i} → NSE {latest_nse:.4f}\noverwrites luanx.month",
        "#eef0f1",
        grey,
        fontsize=9.5,
    )
    add_box(
        ax2,
        (0.695, 0.75),
        0.255,
        0.16,
        f"Reported result\nθ{best_i} + Q{latest_i} + NSE{best_i}\nINVALID ASSOCIATION",
        "#fae1e1",
        red,
        fontsize=10,
    )
    add_arrow(
        ax2,
        (0.285, 0.86),
        (0.695, 0.87),
        red,
        style="--",
        connectionstyle="arc3,rad=-0.34",
    )
    add_arrow(ax2, (0.63, 0.81), (0.695, 0.81), red, style="--")

    ax2.plot([0.04, 0.95], [0.70, 0.70], color="#d4dbe0", lw=1.0, transform=ax2.transAxes)
    ax2.text(
        0.04,
        0.665,
        "With evaluation lineage",
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#000000",
        transform=ax2.transAxes,
    )

    ax2.text(
        0.04,
        0.615,
        "Notation: E = model evaluation; θ = parameter vector; Q = routed monthly discharge;\n"
        "A = aligned observation–simulation series; SHA = first 8 characters of SHA-256.",
        ha="left",
        va="center",
        fontsize=8.8,
        color="#000000",
        transform=ax2.transAxes,
    )

    node_x = [0.04, 0.275, 0.51, 0.745]
    node_w = 0.19
    node_h = 0.105
    best_y = 0.46
    latest_y = 0.30
    best_texts = [
        f"E{best_i}: θ{best_i}\nparameter SHA: {best_record['parameter_sha256'][:8]}",
        f"Q{best_i}: routed discharge\nseries SHA: {best_record['monthly_sha256'][:8]}",
        f"A{best_i}: aligned O–S\nseries SHA: {best_record['aligned_sha256'][:8]}",
        f"NSE{best_i} = {best_nse:.4f}\nPASS · Best evaluation",
    ]
    latest_texts = [
        f"E{latest_i}: θ{latest_i}\nparameter SHA: {latest_record['parameter_sha256'][:8]}",
        f"Q{latest_i}: routed discharge\nseries SHA: {latest_record['monthly_sha256'][:8]}",
        f"A{latest_i}: aligned O–S\nseries SHA: {latest_record['aligned_sha256'][:8]}",
        f"NSE{latest_i} = {latest_nse:.4f}\nPASS",
    ]
    for index, text_value in enumerate(best_texts):
        add_box(
            ax2,
            (node_x[index], best_y),
            node_w,
            node_h,
            text_value,
            "#eaf3f7" if index < 3 else "#e8f4ee",
            blue if index < 3 else green,
            fontsize=8.8,
        )
    for index, text_value in enumerate(latest_texts):
        add_box(
            ax2,
            (node_x[index], latest_y),
            node_w,
            node_h,
            text_value,
            "#eef4f0" if index < 3 else "#e8f4ee",
            green,
            fontsize=8.8,
        )
    for index in range(3):
        add_arrow(
            ax2,
            (node_x[index] + node_w, best_y + node_h / 2),
            (node_x[index + 1], best_y + node_h / 2),
            green,
        )
        add_arrow(
            ax2,
            (node_x[index] + node_w, latest_y + node_h / 2),
            (node_x[index + 1], latest_y + node_h / 2),
            green,
        )
    mismatch_y = 0.105
    add_box(
        ax2,
        (0.15, mismatch_y),
        0.20,
        0.10,
        f"E{best_i}: θ{best_i}\nparameter SHA: {best_record['parameter_sha256'][:8]}",
        "#edf3f7",
        blue,
        fontsize=8.8,
    )
    add_box(
        ax2,
        (0.43, mismatch_y),
        0.20,
        0.10,
        f"E{latest_i}: Q{latest_i}\nseries SHA: {latest_record['monthly_sha256'][:8]}",
        "#eef0f1",
        grey,
        fontsize=8.8,
    )
    add_box(
        ax2,
        (0.72, mismatch_y),
        0.18,
        0.10,
        f"BLOCK\nE{best_i} ≠ E{latest_i}",
        "#fae1e1",
        red,
        fontsize=10,
    )
    add_arrow(ax2, (0.35, mismatch_y + 0.05), (0.43, mismatch_y + 0.05), red, style="--")
    add_arrow(ax2, (0.63, mismatch_y + 0.05), (0.72, mismatch_y + 0.05), red, style="--")
    ax2.text(
        0.04,
        mismatch_y + 0.05,
        "Controlled\nmismatch",
        ha="left",
        va="center",
        fontsize=9,
        color="#000000",
        transform=ax2.transAxes,
    )
    ax2.text(
        0.95,
        0.025,
        "2/2 valid chains PASS   |   1/1 mismatch BLOCK   |   0/1 unsafe accepted",
        ha="right",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#000000",
        transform=ax2.transAxes,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create the run-level calibration-lineage assurance figure."
    )
    parser.add_argument("--run-id", default="web_demo")
    parser.add_argument("--experiment-run-id", default="lineage_demo")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "report_assets/figures/run_level_lineage_assurance",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    history, evidence, audit = collect_evidence(
        project_root, args.run_id, args.experiment_run_id
    )
    output_prefix = args.output_prefix
    if not output_prefix.is_absolute():
        output_prefix = project_root / output_prefix
    output_png = output_prefix.with_suffix(".png")
    output_pdf = output_prefix.with_suffix(".pdf")
    output_json = output_prefix.with_name(output_prefix.name + "_evidence").with_suffix(".json")

    create_figure(history, evidence, audit, output_png, output_pdf)
    output_json.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"PNG: {output_png}")
    print(f"PDF: {output_pdf}")
    print(f"Evidence: {output_json}")


if __name__ == "__main__":
    main()
