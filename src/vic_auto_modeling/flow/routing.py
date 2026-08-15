import math
import heapq
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


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

LOWER_DIRECTION_ORDER = [7, 1, 5, 3, 2, 4, 6, 8]
FLAT_DIRECTION_ORDER = [1, 5, 3, 7, 2, 4, 6, 8]
ARCGIS_TIE_DIRECTION_ORDER = [3, 4, 5, 6, 7, 8, 1, 2]
CARDINAL_DIRECTIONS = {1, 3, 5, 7}
FILL_ALGORITHMS = {
    "priority-flood",
    "planchon-darboux",
    "planchon-darboux-gradient",
}


@dataclass
class FlowConfig:
    target_grid_path: str | Path = "output/fishnet/lishui_target_grid.gpkg"
    full_grid_path: str | Path = "output/fishnet/lishui_full_grid.gpkg"
    elevation_path: str | Path = "output/elevation/area_elev.txt"
    flow_output_path: str | Path = "output/flow/flow_1_8.txt"
    distance_output_path: str | Path = "output/flow/output_area_mask.txt"
    fishnet_prj_path: str | Path = "output/flow/fishnet_prj.txt"
    fill_algorithm: str = "priority-flood"
    cellsize: float = 0.081

    @classmethod
    def from_run_context(cls, context):
        return cls(
            target_grid_path=context.output_path("fishnet", "lishui_target_grid.gpkg"),
            full_grid_path=context.output_path("fishnet", "lishui_full_grid.gpkg"),
            elevation_path=context.output_path("elevation", "area_elev.txt"),
            flow_output_path=context.output_path("flow", "flow_1_8.txt"),
            distance_output_path=context.output_path("flow", "output_area_mask.txt"),
            fishnet_prj_path=context.output_path("flow", "fishnet_prj.txt"),
        )


def matlab_g(value):
    return f"{value:.6g}"


def ascii_header_g(value):
    return f"{value:.12g}"


def read_elevations(path):
    df = pd.read_csv(path)
    if {"vic_id", "elevation"}.issubset(df.columns):
        id_column = "vic_id"
        elevation_column = "elevation"
    elif {"ID", "MEAN"}.issubset(df.columns):
        id_column = "ID"
        elevation_column = "MEAN"
    else:
        raise ValueError(
            "Elevation file must contain either vic_id/elevation columns "
            "or ArcGIS ID/MEAN columns."
        )
    return dict(zip(df[id_column].astype(int), df[elevation_column].astype(float)))


def routing_extent(target_grid):
    min_row = int(target_grid["row"].min())
    max_row = int(target_grid["row"].max())
    min_col = int(target_grid["col"].min())
    max_col = int(target_grid["col"].max())
    if min_row != 0 or min_col != 0:
        raise ValueError("This workflow expects target grid row/col to start from 0")
    return max_row + 1, max_col + 1


def build_projected_centers(full_grid, nrows, ncols, crs="EPSG:3857"):
    point_grid = gpd.GeoDataFrame(
        full_grid.drop(columns="geometry"),
        geometry=gpd.points_from_xy(full_grid["lon"], full_grid["lat"]),
        crs=full_grid.crs,
    )
    projected = point_grid.to_crs(crs).copy()
    projected["x"] = projected.geometry.x
    projected["y"] = projected.geometry.y

    centers = {}
    for row in projected.itertuples(index=False):
        row_id = int(row.row)
        col_id = int(row.col)
        if row_id < nrows and col_id < ncols:
            centers[(row_id, col_id)] = (float(row.x), float(row.y))

    expected = nrows * ncols
    if len(centers) != expected:
        raise ValueError(f"Projected fishnet has {len(centers)} cells, expected {expected}")
    return centers


def distance_between(centers, src_key, dst_key):
    x1, y1 = centers[src_key]
    x2, y2 = centers[dst_key]
    return math.hypot(x2 - x1, y2 - y1)


def d8_step_distance(row_offset, col_offset):
    if row_offset != 0 and col_offset != 0:
        return math.sqrt(2.0)
    return 1.0


def is_drainage_edge(row, col, valid_cells, nrows, ncols):
    for row_offset, col_offset in D8_OFFSETS.values():
        next_row = row + row_offset
        next_col = col + col_offset
        if not (0 <= next_row < nrows and 0 <= next_col < ncols):
            return True
        if not valid_cells[next_row, next_col]:
            return True
    return False


def fill_depressions(elevation_grid, valid_cells):
    nrows, ncols = elevation_grid.shape
    filled = elevation_grid.copy()
    visited = np.zeros(valid_cells.shape, dtype=bool)
    queue = []

    for row in range(nrows):
        for col in range(ncols):
            if valid_cells[row, col] and is_drainage_edge(row, col, valid_cells, nrows, ncols):
                visited[row, col] = True
                heapq.heappush(queue, (filled[row, col], row, col))

    while queue:
        elevation, row, col = heapq.heappop(queue)
        for row_offset, col_offset in D8_OFFSETS.values():
            next_row = row + row_offset
            next_col = col + col_offset
            if not (0 <= next_row < nrows and 0 <= next_col < ncols):
                continue
            if visited[next_row, next_col] or not valid_cells[next_row, next_col]:
                continue

            visited[next_row, next_col] = True
            if filled[next_row, next_col] < elevation:
                filled[next_row, next_col] = elevation
            heapq.heappush(queue, (filled[next_row, next_col], next_row, next_col))

    return filled


def fill_depressions_planchon_darboux(
    elevation_grid,
    valid_cells,
    epsilon=0.0,
    max_iterations=10000,
):
    nrows, ncols = elevation_grid.shape
    max_elevation = np.nanmax(elevation_grid[valid_cells])
    water = np.where(valid_cells, max_elevation + 1.0, np.nan)

    for row in range(nrows):
        for col in range(ncols):
            if valid_cells[row, col] and is_drainage_edge(row, col, valid_cells, nrows, ncols):
                water[row, col] = elevation_grid[row, col]

    for _ in range(max_iterations):
        changed = False
        for row in range(nrows):
            for col in range(ncols):
                if not valid_cells[row, col] or is_drainage_edge(row, col, valid_cells, nrows, ncols):
                    continue

                min_neighbor = math.inf
                for row_offset, col_offset in D8_OFFSETS.values():
                    next_row = row + row_offset
                    next_col = col + col_offset
                    if not (0 <= next_row < nrows and 0 <= next_col < ncols):
                        continue
                    if not valid_cells[next_row, next_col]:
                        continue
                    spill_elevation = water[next_row, next_col] + (
                        epsilon * d8_step_distance(row_offset, col_offset)
                    )
                    if spill_elevation < min_neighbor:
                        min_neighbor = spill_elevation

                if min_neighbor == math.inf:
                    continue

                new_elevation = max(elevation_grid[row, col], min_neighbor)
                if new_elevation < water[row, col] - 1e-12:
                    water[row, col] = new_elevation
                    changed = True

        if not changed:
            return water

    raise RuntimeError(
        f"Planchon-Darboux fill did not converge after {max_iterations} iterations."
    )


def fill_elevation_grid(elevation_grid, valid_cells, fill_algorithm):
    if fill_algorithm == "priority-flood":
        return fill_depressions(elevation_grid, valid_cells)
    if fill_algorithm == "planchon-darboux":
        return fill_depressions_planchon_darboux(elevation_grid, valid_cells, epsilon=0.0)
    if fill_algorithm == "planchon-darboux-gradient":
        return fill_depressions_planchon_darboux(elevation_grid, valid_cells, epsilon=1e-6)
    raise ValueError(
        f"Unsupported fill algorithm: {fill_algorithm}. "
        f"Choose one of: {', '.join(sorted(FILL_ALGORITHMS))}."
    )


def best_lower_direction(elevation_grid, valid_cells, row, col, direction_order):
    nrows, ncols = elevation_grid.shape
    best_slope = 0.0
    best_directions = []

    for direction in direction_order:
        row_offset, col_offset = D8_OFFSETS[direction]
        next_row = row + row_offset
        next_col = col + col_offset
        if not (0 <= next_row < nrows and 0 <= next_col < ncols):
            continue
        if not valid_cells[next_row, next_col]:
            continue

        drop = elevation_grid[row, col] - elevation_grid[next_row, next_col]
        if not np.isfinite(drop) or drop <= 0:
            continue

        slope = drop / d8_step_distance(row_offset, col_offset)
        if slope > best_slope + 1e-12:
            best_slope = slope
            best_directions = [direction]
        elif abs(slope - best_slope) <= 1e-12:
            best_directions.append(direction)

    if not best_directions:
        return 0

    return select_tied_direction(best_directions, direction_order)


def select_tied_direction(best_directions, direction_order):
    if len(best_directions) >= 3 and set(best_directions).issubset(CARDINAL_DIRECTIONS):
        for direction in ARCGIS_TIE_DIRECTION_ORDER:
            if direction in best_directions:
                return direction

    for direction in direction_order:
        if direction in best_directions:
            return direction

    return best_directions[0]


def best_outward_boundary_direction(row, col, valid_cells):
    nrows, ncols = valid_cells.shape
    outside_vectors = []

    for direction, (row_offset, col_offset) in D8_OFFSETS.items():
        next_row = row + row_offset
        next_col = col + col_offset
        if not (0 <= next_row < nrows and 0 <= next_col < ncols):
            outside_vectors.append((row_offset, col_offset))
        elif not valid_cells[next_row, next_col]:
            outside_vectors.append((row_offset, col_offset))

    if not outside_vectors:
        return 0

    row_vector = sum(vector[0] for vector in outside_vectors)
    col_vector = sum(vector[1] for vector in outside_vectors)
    best_direction = 0
    best_alignment = -math.inf

    for direction, (row_offset, col_offset) in D8_OFFSETS.items():
        length = d8_step_distance(row_offset, col_offset)
        alignment = (row_offset * row_vector + col_offset * col_vector) / length
        if alignment > best_alignment:
            best_alignment = alignment
            best_direction = direction

    return best_direction


def resolve_flat_directions(elevation_grid, valid_cells, flow, direction_order):
    nrows, ncols = elevation_grid.shape
    seen = set()

    for start_row in range(nrows):
        for start_col in range(ncols):
            start_key = (start_row, start_col)
            if not valid_cells[start_key] or flow[start_key] != 0 or start_key in seen:
                continue

            elevation = elevation_grid[start_key]
            flat_cells = []
            stack = [start_key]
            seen.add(start_key)

            for row, col in stack:
                flat_cells.append((row, col))
                for row_offset, col_offset in D8_OFFSETS.values():
                    next_key = (row + row_offset, col + col_offset)
                    next_row, next_col = next_key
                    if not (0 <= next_row < nrows and 0 <= next_col < ncols):
                        continue
                    if next_key in seen or not valid_cells[next_key]:
                        continue
                    if abs(elevation_grid[next_key] - elevation) < 1e-9:
                        seen.add(next_key)
                        stack.append(next_key)

            flat_set = set(flat_cells)
            assigned = set()

            for row, col in flat_cells:
                if flow[row, col] != 0:
                    assigned.add((row, col))
                    continue

                direction = best_outward_boundary_direction(row, col, valid_cells)
                if direction != 0:
                    flow[row, col] = direction
                    assigned.add((row, col))

            while len(assigned) < len(flat_set):
                wave = []
                for row, col in flat_cells:
                    cell = (row, col)
                    if cell in assigned:
                        continue
                    for direction in direction_order:
                        row_offset, col_offset = D8_OFFSETS[direction]
                        next_key = (row + row_offset, col + col_offset)
                        if next_key in assigned:
                            wave.append((cell, direction))
                            break

                if not wave:
                    break

                for (row, col), direction in wave:
                    cell = (row, col)
                    if cell in assigned:
                        continue
                    flow[row, col] = direction
                    assigned.add(cell)

    return flow


def calculate_flow_direction(
    target_grid,
    elevations,
    projected_centers,
    nrows,
    ncols,
    fill_algorithm="priority-flood",
):
    elevation_grid = np.full((nrows, ncols), np.nan, dtype=float)
    valid_cells = np.zeros((nrows, ncols), dtype=bool)

    for row in target_grid.itertuples(index=False):
        key = (int(row.row), int(row.col))
        elevation_grid[key] = elevations[int(row.vic_id)]
        valid_cells[key] = True

    use_gradient_tie_break = fill_algorithm == "planchon-darboux-gradient"
    direction_order = ARCGIS_TIE_DIRECTION_ORDER if use_gradient_tie_break else LOWER_DIRECTION_ORDER
    flat_direction_order = ARCGIS_TIE_DIRECTION_ORDER if use_gradient_tie_break else FLAT_DIRECTION_ORDER
    filled_elevation = fill_elevation_grid(elevation_grid, valid_cells, fill_algorithm)

    flow = np.zeros((nrows, ncols), dtype=int)
    for row in range(nrows):
        for col in range(ncols):
            if valid_cells[row, col]:
                flow[row, col] = best_lower_direction(
                    filled_elevation,
                    valid_cells,
                    row,
                    col,
                    direction_order,
                )

    flow = resolve_flat_directions(
        filled_elevation,
        valid_cells,
        flow,
        flat_direction_order,
    )

    return flow


def calculate_flow_distance(flow, projected_centers):
    distances = np.zeros(flow.shape, dtype=float)
    nrows, ncols = flow.shape

    for row in range(nrows):
        for col in range(ncols):
            direction = int(flow[row, col])
            if direction == 0:
                continue
            row_offset, col_offset = D8_OFFSETS[direction]
            dst_key = (row + row_offset, col + col_offset)
            if not (0 <= dst_key[0] < nrows and 0 <= dst_key[1] < ncols):
                continue
            distances[row, col] = distance_between(projected_centers, (row, col), dst_key)

    return distances


def write_flow_grid(path, data, xllcorner, yllcorner, cellsize):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nrows, ncols = data.shape

    with path.open("w", encoding="utf-8", newline="\r\n") as f:
        f.write(f"ncols\t{ncols}\n")
        f.write(f"nrows\t{nrows}\n")
        f.write(f"xllcorner\t{ascii_header_g(xllcorner)}\n")
        f.write(f"yllcorner\t{ascii_header_g(yllcorner)}\n")
        f.write(f"cellsize\t{ascii_header_g(cellsize)}\n")
        f.write("NODATA_value\t0\n")
        for row in data:
            f.write("\t".join(str(int(value)) for value in row) + "\n")


def write_distance_grid(path, data, xllcorner, yllcorner, cellsize):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nrows, ncols = data.shape

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"ncols          {ncols}\n")
        f.write(f"nrows          {nrows}\n")
        f.write(f"xllcorner   {ascii_header_g(xllcorner)}\n")
        f.write(f"yllcorner   {ascii_header_g(yllcorner)}\n")
        f.write(f"cellsize       {ascii_header_g(cellsize)}\n")
        f.write("NODATA_value     0\n")
        for row in data:
            f.write("\t".join(matlab_g(float(value)) for value in row) + "\n")


def write_fishnet_prj(path, full_grid, projected_centers, nrows, ncols):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid_ids = {
        (int(row.row), int(row.col)): int(row.vic_id) if hasattr(row, "vic_id") else int(row.grid_id)
        for row in full_grid.itertuples(index=False)
        if int(row.row) < nrows and int(row.col) < ncols
    }

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("FID,Id,X,Y\n")
        fid = 0
        for row in range(nrows):
            for col in range(ncols):
                x, y = projected_centers[(row, col)]
                f.write(f"{fid},{grid_ids[(row, col)]},{x:.15f},{y:.15f}\n")
                fid += 1


def create_flow(
    target_grid_path,
    full_grid_path,
    elevation_path,
    flow_output_path,
    distance_output_path,
    fishnet_prj_path,
    fill_algorithm="priority-flood",
    cellsize=0.081,
):
    target_grid = gpd.read_file(target_grid_path).sort_values(["row", "col"]).reset_index(drop=True)
    full_grid = gpd.read_file(full_grid_path).sort_values(["row", "col"]).reset_index(drop=True)
    elevations = read_elevations(elevation_path)
    nrows, ncols = routing_extent(target_grid)

    projected_centers = build_projected_centers(full_grid, nrows, ncols)
    flow = calculate_flow_direction(
        target_grid,
        elevations,
        projected_centers,
        nrows,
        ncols,
        fill_algorithm=fill_algorithm,
    )
    distances = calculate_flow_distance(flow, projected_centers)

    xllcorner, yllcorner, _, _ = target_grid.total_bounds
    write_flow_grid(
        flow_output_path,
        flow,
        xllcorner=xllcorner,
        yllcorner=yllcorner,
        cellsize=cellsize,
    )
    write_distance_grid(
        distance_output_path,
        distances,
        xllcorner=xllcorner,
        yllcorner=yllcorner,
        cellsize=cellsize,
    )
    write_fishnet_prj(fishnet_prj_path, full_grid, projected_centers, nrows, ncols)

    return flow, distances


def create_flow_from_config(config):
    return create_flow(
        target_grid_path=config.target_grid_path,
        full_grid_path=config.full_grid_path,
        elevation_path=config.elevation_path,
        flow_output_path=config.flow_output_path,
        distance_output_path=config.distance_output_path,
        fishnet_prj_path=config.fishnet_prj_path,
        fill_algorithm=config.fill_algorithm,
        cellsize=config.cellsize,
    )
