from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunContext:
    run_id: str
    project_root: str | Path = "."
    runs_dir: str | Path = "runs"
    base_data_dir: str | Path = "data"

    def __post_init__(self):
        if not self.run_id or Path(self.run_id).name != self.run_id:
            raise ValueError("run_id must be a non-empty directory name")

    @property
    def project_path(self):
        return Path(self.project_root).resolve()

    @property
    def base_data_path(self):
        return self.project_path / self.base_data_dir

    @property
    def run_dir(self):
        return self.project_path / self.runs_dir / self.run_id

    @property
    def input_dir(self):
        return self.run_dir / "input"

    @property
    def output_dir(self):
        return self.run_dir / "output"

    @property
    def log_dir(self):
        return self.run_dir / "logs"

    @property
    def manifest_path(self):
        return self.run_dir / "manifest.json"

    def input_path(self, *parts):
        return self.input_dir.joinpath(*parts)

    def output_path(self, *parts):
        return self.output_dir.joinpath(*parts)

    def log_path(self, *parts):
        return self.log_dir.joinpath(*parts)

    def ensure_dirs(self):
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self
