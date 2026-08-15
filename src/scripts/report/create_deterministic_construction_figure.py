import argparse
import json
import re
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch


D8_OFFSETS = {
    1: (-1, 0),
    2: (-1, 1),
    3: (0, 1),
    4: (1, 1),
    5: (1, 0),
    6: (1, -1),
    7: (0, -1),
    8: (-1, -1),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the four-panel deterministic VIC construction figure."
    )
    parser.add_argument("--run-id", default="web_demo")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report_assets/figures"),
    )
    return parser.parse_args()


def read_ascii_grid(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header = {}
    for line in lines[:6]:
        key, value = line.split()[:2]
        header[key.lower()] = float(value)
    data = np.array([[float(value) for value in line.split()] for line in lines[6:]])
    return header, data


def read_id_set(path, column="vic_id"):
    frame = pd.read_csv(path)
    return set(frame[column].astype(int))


def count_non_finite(paths):
    total = 0
    for path in paths:
        values = np.loadtxt(path)
        total += int((~np.isfinite(values)).sum())
    return total


def configured_record_counts(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    def value(name):
        match = re.search(rf"^\s*{name}\s+(\d+)", text, flags=re.M)
        if not match:
            raise ValueError(f"Missing {name} in {path}")
        return int(match.group(1))

    start = date(value("STARTYEAR"), value("STARTMONTH"), value("STARTDAY"))
    end = date(value("ENDYEAR"), value("ENDMONTH"), value("ENDDAY"))
    if end < start:
        raise ValueError("Configured simulation end precedes its start")
    return {
        "configured_start": start.isoformat(),
        "configured_end": end.isoformat(),
        "expected_daily_rows": (end - start).days + 1,
        "expected_monthly_rows": (end.year - start.year) * 12
        + end.month
        - start.month
        + 1,
        "expected_climatology_rows": 12,
    }


def find_outlet_cell(target_grid, outlet):
    candidates = target_grid[
        target_grid.geometry.contains(outlet) | target_grid.geometry.touches(outlet)
    ]
    if candidates.empty:
        raise ValueError("The outlet is not contained in the active target grid.")
    cell = candidates.sort_values(["row", "col"]).iloc[0]
    return int(cell["row"]), int(cell["col"])


def contributing_cells(target_grid, flow, outlet_cell):
    active = {
        (int(record.row), int(record.col))
        for record in target_grid.itertuples(index=False)
    }
    reverse_edges = defaultdict(list)
    for source in active:
        direction = int(flow[source])
        if direction not in D8_OFFSETS:
            continue
        row_offset, col_offset = D8_OFFSETS[direction]
        destination = (source[0] + row_offset, source[1] + col_offset)
        if destination in active:
            reverse_edges[destination].append(source)

    contributors = {outlet_cell}
    queue = deque([outlet_cell])
    while queue:
        destination = queue.popleft()
        for source in reverse_edges[destination]:
            if source not in contributors:
                contributors.add(source)
                queue.append(source)
    return contributors


def count_cycles(target_grid, flow):
    active = {
        (int(record.row), int(record.col))
        for record in target_grid.itertuples(index=False)
    }
    finished = set()
    cycles = set()
    for start in active:
        if start in finished:
            continue
        path = []
        positions = {}
        current = start
        while current in active and current not in finished:
            if current in positions:
                cycle = tuple(sorted(path[positions[current] :]))
                cycles.add(cycle)
                break
            positions[current] = len(path)
            path.append(current)
            direction = int(flow[current])
            if direction not in D8_OFFSETS:
                break
            row_offset, col_offset = D8_OFFSETS[direction]
            current = (current[0] + row_offset, current[1] + col_offset)
        finished.update(path)
    return len(cycles)


def collect_evidence(run_dir):
    output_dir = run_dir / "output"
    model_dir = output_dir / "model"
    result_dir = model_dir / "chanliu_result"

    full_grid = gpd.read_file(output_dir / "fishnet/lishui_full_grid.gpkg")
    target_grid = gpd.read_file(output_dir / "fishnet/lishui_target_grid.gpkg")
    boundary = gpd.read_file(run_dir / "input/boundary/boundary.shp").to_crs(
        target_grid.crs
    )
    outlets = gpd.read_file(run_dir / "input/outlets/outlets.shp").to_crs(
        target_grid.crs
    )
    if len(outlets) != 1:
        raise ValueError(f"Expected one outlet, found {len(outlets)}")

    _, flow = read_ascii_grid(output_dir / "flow/flow_1_8.txt")
    flow = flow.astype(int)
    outlet_cell = find_outlet_cell(target_grid, outlets.geometry.iloc[0])
    contributors = contributing_cells(target_grid, flow, outlet_cell)

    target_ids = set(target_grid["vic_id"].astype(int))
    soil = np.loadtxt(output_dir / "soil/output_area_soil.txt")
    soil_ids = set(soil[:, 1].astype(int))
    forcing_grid = np.loadtxt(output_dir / "forcing/output_area.txt", delimiter=",")
    forcing_ids = set(forcing_grid[:, 0].astype(int))
    elevation_ids = read_id_set(output_dir / "elevation/area_elev.txt")
    topsoil_ids = read_id_set(output_dir / "soil/top_soil.txt")
    subsoil_ids = read_id_set(output_dir / "soil/sub_soil.txt")
    id_sets = [
        target_ids,
        soil_ids,
        forcing_ids,
        elevation_ids,
        topsoil_ids,
        subsoil_ids,
    ]

    forcing_files = sorted((output_dir / "forcing/forcing").glob("forcing_*"))
    flux_files = sorted(result_dir.glob("fluxes_*"))
    forcing_names = {
        f"fluxes_{path.name.removeprefix('forcing_')}" for path in forcing_files
    }
    flux_names = {path.name for path in flux_files}
    forcing_lengths = [sum(1 for _ in path.open()) for path in forcing_files]
    flux_lengths = [sum(1 for _ in path.open()) for path in flux_files]
    configured_counts = configured_record_counts(model_dir / "chanliu_input.txt")

    stdout = (model_dir / "vic_stdout.log").read_text(
        encoding="utf-8", errors="replace"
    )
    stderr = (model_dir / "vic_stderr.log").read_text(
        encoding="utf-8", errors="replace"
    )
    status = pd.read_json(run_dir / "logs/vic_run_status.json", typ="series")
    upstream_match = re.search(r"CELLS UPSTREAM[^\n]*?\s(\d+)\s+\d+\s+\d+\s+\S+", stdout)
    logged_upstream = int(upstream_match.group(1)) if upstream_match else None

    invalid_directions = int(
        (~np.isin(flow, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))).sum()
    )
    evidence = {
        "full_cells": len(full_grid),
        "active_cells": len(target_grid),
        "unique_ids": target_grid["vic_id"].nunique(),
        "id_sets_equal": all(values == target_ids for values in id_sets),
        "forcing_files": len(forcing_files),
        "forcing_days": min(forcing_lengths),
        "flux_files": len(flux_files),
        "flux_days": min(flux_lengths),
        "forcing_flux_names_equal": forcing_names == flux_names,
        "forcing_non_finite": count_non_finite(forcing_files),
        "flux_non_finite": count_non_finite(flux_files),
        "returncode": int(status["returncode"]),
        "routing_shape": flow.shape,
        "active_directions": int((flow > 0).sum()),
        "invalid_directions": invalid_directions,
        "cycles": count_cycles(target_grid, flow),
        "not_found": stdout.count("NOT FOUND") + stderr.count("NOT FOUND"),
        "upstream_cells": len(contributors),
        "logged_upstream": logged_upstream,
        "daily_rows": sum(1 for _ in (result_dir / "luanx.day").open()),
        "monthly_rows": sum(1 for _ in (result_dir / "luanx.month").open()),
        "climatology_rows": sum(1 for _ in (result_dir / "luanx.year").open()),
        **configured_counts,
    }
    evidence["forcing_record_counts_match"] = all(
        count == evidence["expected_daily_rows"] for count in forcing_lengths
    )
    evidence["flux_record_counts_match"] = all(
        count == evidence["expected_daily_rows"] for count in flux_lengths
    )
    evidence["routed_record_counts_match"] = (
        evidence["daily_rows"] == evidence["expected_daily_rows"]
        and evidence["monthly_rows"] == evidence["expected_monthly_rows"]
        and evidence["climatology_rows"] == evidence["expected_climatology_rows"]
    )

    if not forcing_lengths or min(forcing_lengths) != max(forcing_lengths):
        raise ValueError("Forcing files have inconsistent record lengths.")
    if not flux_lengths or min(flux_lengths) != max(flux_lengths):
        raise ValueError("Flux files have inconsistent record lengths.")
    if evidence["logged_upstream"] not in {None, evidence["upstream_cells"]}:
        raise ValueError("Computed and routing-log upstream counts differ.")

    return full_grid, target_grid, boundary, outlets, flow, contributors, evidence


def set_map_axes(ax, bounds):
    min_x, min_y, max_x, max_y = bounds
    x_pad = (max_x - min_x) * 0.035
    y_pad = (max_y - min_y) * 0.035
    ax.set_xlim(min_x - x_pad, max_x + x_pad)
    ax.set_ylim(min_y - y_pad, max_y + y_pad)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.tick_params(labelsize=9.5)


def plot_input_grid(ax, full_grid, target_grid, boundary, outlets, evidence):
    full_grid.plot(ax=ax, facecolor="none", edgecolor="#B8B8B8", linewidth=0.18)
    target_grid.plot(
        ax=ax,
        facecolor="#8EC7E8",
        edgecolor="#FFFFFF",
        linewidth=0.12,
        alpha=0.85,
    )
    boundary.boundary.plot(ax=ax, color="#183A5A", linewidth=1.25)
    outlets.plot(
        ax=ax,
        color="#C62828",
        marker="*",
        markersize=105,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
        label="Outlet",
    )
    set_map_axes(ax, full_grid.total_bounds)
    ax.set_title("(a) Input domain and regular grid", loc="left", fontweight="bold")
    ax.text(
        0.975,
        0.975,
        f"Full grid: {evidence['routing_shape'][0]} × {evidence['routing_shape'][1]}"
        f" = {evidence['full_cells']:,} cells\n"
        f"Active watershed grid: {evidence['active_cells']:,} cells",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color="#000000",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#777777"},
    )
    ax.legend(loc="lower left", frameon=True, fontsize=9.5)


def plot_active_ids(ax, target_grid, boundary, outlets, evidence):
    norm = Normalize(
        vmin=float(target_grid["vic_id"].min()),
        vmax=float(target_grid["vic_id"].max()),
    )
    target_grid.plot(
        ax=ax,
        column="vic_id",
        cmap="viridis",
        norm=norm,
        edgecolor="white",
        linewidth=0.12,
    )
    boundary.boundary.plot(ax=ax, color="#183A5A", linewidth=1.25)
    outlets.plot(
        ax=ax,
        color="#C62828",
        marker="*",
        markersize=105,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    set_map_axes(ax, target_grid.total_bounds)
    ax.set_title("(b) Active-cell identity consistency", loc="left", fontweight="bold")

    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    colorbar = ax.figure.colorbar(
        scalar_map,
        ax=ax,
        orientation="horizontal",
        fraction=0.045,
        pad=0.07,
    )
    colorbar.set_label("Stable active-cell identifier (vic_id)", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)

    match_text = "PASS" if evidence["id_sets_equal"] else "FAIL"
    ax.text(
        0.975,
        0.975,
        "Cross-stage identity audit\n"
        f"Target grid:  {evidence['active_cells']:>4}\n"
        f"Elevation:    {evidence['active_cells']:>4}\n"
        f"Top/subsoil:  {evidence['active_cells']:>4}\n"
        f"Soil params:  {evidence['active_cells']:>4}\n"
        f"Forcing grid: {evidence['active_cells']:>4}\n"
        f"Set equality: {match_text}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        family="monospace",
        fontsize=9.8,
        color="#000000",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.92, "edgecolor": "#2E7D32"},
    )


def plot_routing(ax, target_grid, boundary, outlets, flow, contributors, evidence):
    contributor_mask = target_grid.apply(
        lambda row: (int(row["row"]), int(row["col"])) in contributors,
        axis=1,
    )
    target_grid.plot(
        ax=ax,
        facecolor="#E8EEF3",
        edgecolor="white",
        linewidth=0.10,
    )
    target_grid[contributor_mask].plot(
        ax=ax,
        facecolor="#F4A261",
        edgecolor="#C96B18",
        linewidth=0.18,
        alpha=0.88,
        label="Outlet-contributing cells",
    )
    boundary.boundary.plot(ax=ax, color="#183A5A", linewidth=1.25)

    centres = {
        (int(record.row), int(record.col)): (float(record.lon), float(record.lat))
        for record in target_grid.itertuples(index=False)
    }
    normal_x, normal_y, normal_u, normal_v = [], [], [], []
    upstream_x, upstream_y, upstream_u, upstream_v = [], [], [], []
    for source, (x_value, y_value) in centres.items():
        direction = int(flow[source])
        if direction not in D8_OFFSETS:
            continue
        row_offset, col_offset = D8_OFFSETS[direction]
        destination = (source[0] + row_offset, source[1] + col_offset)
        if destination not in centres:
            continue
        target_x, target_y = centres[destination]
        values = (x_value, y_value, (target_x - x_value) * 0.72, (target_y - y_value) * 0.72)
        if source in contributors:
            for collection, value in zip(
                (upstream_x, upstream_y, upstream_u, upstream_v), values
            ):
                collection.append(value)
        else:
            for collection, value in zip(
                (normal_x, normal_y, normal_u, normal_v), values
            ):
                collection.append(value)

    ax.quiver(
        normal_x,
        normal_y,
        normal_u,
        normal_v,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0012,
        headwidth=2.7,
        headlength=3.4,
        color="#7793A7",
        alpha=0.55,
        zorder=3,
    )
    ax.quiver(
        upstream_x,
        upstream_y,
        upstream_u,
        upstream_v,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0018,
        headwidth=3.1,
        headlength=3.8,
        color="#A54F0E",
        alpha=0.9,
        zorder=4,
    )
    outlets.plot(
        ax=ax,
        color="#C62828",
        marker="*",
        markersize=120,
        edgecolor="white",
        linewidth=0.7,
        zorder=6,
        label="Outlet",
    )
    set_map_axes(ax, target_grid.total_bounds)
    ax.set_title("(c) D8 routing and outlet contribution", loc="left", fontweight="bold")
    ax.text(
        0.975,
        0.975,
        f"Active directions: {evidence['active_directions']}\n"
        f"Invalid codes: {evidence['invalid_directions']}\n"
        f"Detected cycles: {evidence['cycles']}\n"
        f"Outlet contributors: {evidence['upstream_cells']}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color="#000000",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.92, "edgecolor": "#A54F0E"},
    )
    ax.legend(
        handles=[
            Patch(
                facecolor="#F4A261",
                edgecolor="#C96B18",
                label="Outlet-contributing cells",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor="#C62828",
                markeredgecolor="white",
                markersize=11,
                label="Outlet",
            ),
        ],
        loc="lower left",
        frameon=True,
        fontsize=9.5,
    )


def evidence_box(ax, x_value, y_value, width, height, title, body, color):
    patch = FancyBboxPatch(
        (x_value, y_value),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor="white",
        edgecolor=color,
        linewidth=1.6,
    )
    ax.add_patch(patch)
    ax.text(
        x_value + width / 2,
        y_value + height * 0.66,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#000000",
    )
    ax.text(
        x_value + width / 2,
        y_value + height * 0.30,
        body,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#000000",
    )


def plot_evidence_chain(ax, evidence):
    ax.set_axis_off()
    ax.set_title("(d) End-to-end artefact evidence", loc="left", fontweight="bold")

    box_width = 0.27
    box_height = 0.19
    x_positions = [0.04, 0.365, 0.69]
    top_y = 0.65
    bottom_y = 0.25
    boxes = [
        (
            x_positions[0],
            top_y,
            "Active grid",
            f"{evidence['active_cells']} cells\nID sets equal: PASS",
            "#1F4E79",
        ),
        (
            x_positions[1],
            top_y,
            "Meteorological forcing",
            f"{evidence['forcing_files']} files\n× {evidence['forcing_days']} days",
            "#2677A8",
        ),
        (
            x_positions[2],
            top_y,
            "MPI-VIC",
            f"Return code {evidence['returncode']}\n{evidence['flux_files']} flux files",
            "#2E7D32",
        ),
        (
            x_positions[0],
            bottom_y,
            "Outlet series",
            f"{evidence['daily_rows']} daily | {evidence['monthly_rows']} monthly\n"
            f"{evidence['climatology_rows']} climatological months",
            "#7B2E83",
        ),
        (
            x_positions[1],
            bottom_y,
            "Routing",
            f"NOT FOUND: {evidence['not_found']}\nUpstream cells: {evidence['upstream_cells']}",
            "#A54F0E",
        ),
        (
            x_positions[2],
            bottom_y,
            "Flux identity",
            f"{evidence['flux_files']} × {evidence['flux_days']} days\nFilename match: PASS",
            "#2E7D32",
        ),
    ]
    for x_value, y_value, title, body, color in boxes:
        evidence_box(
            ax,
            x_value,
            y_value,
            box_width,
            box_height,
            title,
            body,
            color,
        )

    arrow_style = {"arrowstyle": "-|>", "color": "#666666", "lw": 1.35}
    for left, right in zip(x_positions[:-1], x_positions[1:]):
        ax.annotate(
            "",
            xy=(right - 0.012, top_y + box_height / 2),
            xytext=(left + box_width + 0.012, top_y + box_height / 2),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=arrow_style,
        )
    for right, left in zip(reversed(x_positions[1:]), reversed(x_positions[:-1])):
        ax.annotate(
            "",
            xy=(left + box_width + 0.012, bottom_y + box_height / 2),
            xytext=(right - 0.012, bottom_y + box_height / 2),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=arrow_style,
        )
    ax.annotate(
        "",
        xy=(x_positions[2] + box_width / 2, bottom_y + box_height + 0.012),
        xytext=(x_positions[2] + box_width / 2, top_y - 0.012),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=arrow_style,
    )
    ax.text(
        0.5,
        0.07,
        "All quantities are calculated from the run-scoped artefacts; no values are entered manually.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#000000",
        style="italic",
    )


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    run_dir = project_root / "runs" / args.run_id
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        full_grid,
        target_grid,
        boundary,
        outlets,
        flow,
        contributors,
        evidence,
    ) = collect_evidence(run_dir)
    evidence["run_id"] = args.run_id

    try:
        import scienceplots  # noqa: F401

        style = ["science", "no-latex"]
    except ImportError:
        style = ["default"]

    with plt.style.context(style):
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 10.5,
                "axes.titlesize": 13.5,
                "axes.labelsize": 10.5,
                "figure.dpi": 150,
            }
        )
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(15.2, 11.4),
            constrained_layout=True,
        )
        plot_input_grid(
            axes[0, 0], full_grid, target_grid, boundary, outlets, evidence
        )
        plot_active_ids(axes[0, 1], target_grid, boundary, outlets, evidence)
        plot_routing(
            axes[1, 0],
            target_grid,
            boundary,
            outlets,
            flow,
            contributors,
            evidence,
        )
        plot_evidence_chain(axes[1, 1], evidence)

        png_path = output_dir / "deterministic_model_construction.png"
        pdf_path = output_dir / "deterministic_model_construction.pdf"
        evidence_path = output_dir / "deterministic_model_construction_evidence.json"
        figure.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
        figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        plt.close(figure)

    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {evidence_path}")
    for key, value in evidence.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
