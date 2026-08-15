import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vic_auto_modeling.core.run_context import RunContext
from vic_auto_modeling.vic_runner import DEFAULT_VIC_ARGS, run_vic_model

PARAM_BOUNDS = {
    "x1": (0.01, 1),   # b_infilt
    "x2": (0.01, 1.0),   # Ds
    "x3": (0.1, 30.0),   # Dsmax
    "x4": (0.01, 1.0),   # Ws
    "x5": (0.1, 1.5),    # depth[1]
    "x6": (0.1, 1.5),    # depth[2]
}


@dataclass
class VicCalibrationConfig:
    run_id: str
    project_root: str | Path = "."
    source_dir: str | Path | None = None
    processes: int = 12
    observation_file: str | Path = "data/static/observation.csv"
    station_name: str = "luanx"
    make_plot: bool = True
    stream_output: bool = False

    @classmethod
    def from_run_context(
        cls,
        context,
        source_dir=None,
        processes=12,
        observation_file=None,
        station_name="luanx",
        make_plot=True,
        stream_output=False,
    ):
        observation = (
            observation_file
            if observation_file is not None
            else context.input_path("observation.csv")
        )
        return cls(
            run_id=context.run_id,
            project_root=context.project_path,
            source_dir=source_dir,
            processes=processes,
            observation_file=observation,
            station_name=station_name,
            make_plot=make_plot,
            stream_output=stream_output,
        )

    @property
    def context(self):
        return RunContext(self.run_id, project_root=self.project_root)

    @property
    def model_dir(self):
        return self.context.output_path("model")

    @property
    def result_dir(self):
        return self.model_dir / "chanliu_result"

    @property
    def observation_path(self):
        path = Path(self.observation_file)
        if path.is_absolute():
            return path
        return self.context.project_path / path


@dataclass
class CalibrationIteration:
    iteration: int
    params: dict
    nse: float
    best_nse: float
    aligned_csv: Path
    plot_path: Path | None


DEFAULT_PARAMS = tuple(float(value) for value in DEFAULT_VIC_ARGS)


def read_observed_monthly(path):
    """Read observation.csv: year, month, observed(m3/s)."""
    df = pd.read_csv(path)
    required = {"year", "month"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain year and month columns")

    value_columns = [col for col in df.columns if col not in ("year", "month")]
    if len(value_columns) != 1:
        raise ValueError(f"{path} must contain exactly one observed value column")

    df = df.rename(columns={value_columns[0]: "observed"})
    df["date"] = pd.to_datetime(
        {"year": df["year"].astype(int), "month": df["month"].astype(int), "day": 1}
    )
    return df[["date", "year", "month", "observed"]].sort_values("date")


def read_simulated_monthly(path):
    """Read luanx.month: year month simulated."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["year", "month", "simulated"],
        usecols=[0, 1, 2],
        engine="python",
    )
    df["date"] = pd.to_datetime(
        {"year": df["year"].astype(int), "month": df["month"].astype(int), "day": 1}
    )
    return df[["date", "year", "month", "simulated"]].sort_values("date")


def align_monthly_series(observed, simulated):
    """Keep only the exact common monthly dates before plotting and NSE."""
    overlap_start = max(observed["date"].min(), simulated["date"].min())
    overlap_end = min(observed["date"].max(), simulated["date"].max())
    if overlap_start > overlap_end:
        raise ValueError(
            "Observed and simulated series have no overlapping period: "
            f"observed {observed['date'].min():%Y-%m}..{observed['date'].max():%Y-%m}, "
            f"simulated {simulated['date'].min():%Y-%m}..{simulated['date'].max():%Y-%m}"
        )

    observed_overlap = observed[
        (observed["date"] >= overlap_start) & (observed["date"] <= overlap_end)
    ]
    simulated_overlap = simulated[
        (simulated["date"] >= overlap_start) & (simulated["date"] <= overlap_end)
    ]
    aligned = observed_overlap.merge(
        simulated_overlap[["date", "simulated"]], on="date", how="inner"
    )
    aligned = aligned.dropna(subset=["observed", "simulated"]).sort_values("date")

    if aligned.empty:
        raise ValueError(
            f"No matching monthly records within {overlap_start:%Y-%m}..{overlap_end:%Y-%m}"
        )
    return aligned


def calculate_nse(observed, simulated):
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    if denominator == 0:
        raise ValueError("Cannot calculate NSE because observed values have zero variance")
    return 1 - np.sum((observed - simulated) ** 2) / denominator


def plot_aligned_series(aligned, output_path):
    import scienceplots  # noqa: F401

    with plt.style.context(["science", "nature", "no-latex"]):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(aligned["date"], aligned["observed"], marker="o", label="Observed")
        ax.plot(aligned["date"], aligned["simulated"], marker="s", label="Simulated")
        ax.set_title("Observed vs Simulated Monthly Streamflow")
        ax.set_xlabel("Time")
        ax.set_ylabel("Flow (m3/s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


def evaluate_parameters(x1, x2, x3, x4, x5, x6, config):
    params = (x1, x2, x3, x4, x5, x6)
    result = run_vic_model(
        run_id=config.run_id,
        project_root=config.project_root,
        source_dir=config.source_dir,
        processes=config.processes,
        vic_args=[f"{value:.10g}" for value in params],
        stream_output=config.stream_output,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "VIC model failed with return code "
            f"{result.returncode}\nSTDOUT log: {result.stdout_path}\nSTDERR log: {result.stderr_path}"
        )

    result_dir = config.result_dir
    observed = read_observed_monthly(config.observation_path)
    simulated = read_simulated_monthly(result_dir / f"{config.station_name}.month")
    aligned = align_monthly_series(observed, simulated)

    nse = calculate_nse(aligned["observed"], aligned["simulated"])
    aligned.to_csv(result_dir / f"{config.station_name}_aligned_monthly.csv", index=False)

    if config.make_plot:
        plot_aligned_series(aligned, result_dir / f"{config.station_name}.png")

    print(
        "Aligned period: "
        f"{aligned['date'].min():%Y-%m}..{aligned['date'].max():%Y-%m}; "
        f"records={len(aligned)}; NSE={nse:.6f}"
    )
    return float(nse)


def run_calibration(config, iterations, random_state=1, xi=0.1, on_iteration=None):
    from bayes_opt import BayesianOptimization
    from bayes_opt import acquisition

    pi = acquisition.ProbabilityOfImprovement(xi=xi)
    optimizer = BayesianOptimization(
        acquisition_function=pi,
        f=None,
        pbounds=PARAM_BOUNDS,
        verbose=2,
        random_state=random_state,
    )

    for i in range(iterations):
        start_time = time.time()
        print(f"\n###VIC模型参数自动率定### 第 {i + 1} 次迭代:")

        next_point = optimizer.suggest()
        params = {name: float(value) for name, value in next_point.items()}
        print("建议采样点:", params)

        target = evaluate_parameters(**params, config=config)
        print("目标函数值:", target)

        optimizer.register(params=params, target=target)
        print(f"迭代时间: {time.time() - start_time:.2f} 秒")

        if on_iteration:
            best = optimizer.max
            on_iteration(
                CalibrationIteration(
                    iteration=i + 1,
                    params=params,
                    nse=float(target),
                    best_nse=float(best["target"]),
                    aligned_csv=config.result_dir / f"{config.station_name}_aligned_monthly.csv",
                    plot_path=(
                        config.result_dir / f"{config.station_name}.png"
                        if config.make_plot
                        else None
                    ),
                )
            )

    return optimizer.max


def build_optimizer(args):
    from bayes_opt import BayesianOptimization
    from bayes_opt import acquisition

    pi = acquisition.ProbabilityOfImprovement(xi=args.xi)
    return BayesianOptimization(
        acquisition_function=pi,
        f=None,
        pbounds=PARAM_BOUNDS,
        verbose=2,
        random_state=args.random_state,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate VIC parameters with Bayesian optimization.")
    parser.add_argument("--run-id", required=True, help="Existing RunContext run id.")
    parser.add_argument("--project-root", default=Path.cwd(), help="Project root containing runs/ and data/.")
    parser.add_argument("--source-dir", required=True, help="Directory with MAC_MPI_VIC.X and rout.so.")
    parser.add_argument("--iterations", type=int, default=500, help="Number of BO iterations.")
    parser.add_argument("--processes", type=int, default=12, help="Total MPI processes: 1 master + workers.")
    parser.add_argument("--observation-file", default="data/static/observation.csv", help="Monthly observation CSV.")
    parser.add_argument("--station-name", default="luanx", help="Routing station output prefix.")
    parser.add_argument("--random-state", type=int, default=1, help="Bayesian optimization random seed.")
    parser.add_argument("--xi", type=float, default=0.1, help="Probability-of-improvement xi.")
    parser.add_argument(
        "--evaluate-default",
        action="store_true",
        help="Evaluate the default parameter set once, without Bayesian optimization.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable PNG plot generation.")
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Stream VIC stdout/stderr while also writing logs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = VicCalibrationConfig(
        run_id=args.run_id,
        project_root=args.project_root,
        source_dir=args.source_dir,
        processes=args.processes,
        observation_file=args.observation_file,
        station_name=args.station_name,
        make_plot=not args.no_plot,
        stream_output=args.stream_output,
    )

    if args.evaluate_default:
        nse = evaluate_parameters(
            *DEFAULT_PARAMS,
            config=config,
        )
        print(f"Default parameter NSE = {nse:.6f}")
        return

    best = run_calibration(
        config,
        iterations=args.iterations,
        random_state=args.random_state,
        xi=args.xi,
    )

    print("\n最优解:")
    for name in sorted(best["params"]):
        print(f"{name} = {best['params'][name]:.4f}")
    print(f"目标函数最大值 = {best['target']:.4f}")


if __name__ == "__main__":
    main()
