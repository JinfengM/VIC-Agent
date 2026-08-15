import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path


DEFAULT_VIC_ARGS = [
    "0.910654",
    "0.436423",
    "24.482024",
    "0.502105",
    "0.988413",
    "0.413380",
]


@dataclass
class VicRunResult:
    model_dir: Path
    command: list[str]
    returncode: int
    stdout_path: Path
    stderr_path: Path


def copy_vic_binaries(model_dir, source_dir):
    model_dir = Path(model_dir)
    source_dir = Path(source_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for filename in ["MAC_MPI_VIC.X", "rout.so"]:
        source = source_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Missing VIC runtime file: {source}")
        target = model_dir / filename
        shutil.copy2(source, target)
        if filename == "MAC_MPI_VIC.X":
            target.chmod(target.stat().st_mode | 0o111)
        copied.append(target)
    return copied


def _replace_value_after_marker(lines, marker, replacement, skip_active_flag=False):
    for index, line in enumerate(lines):
        if line.strip().startswith(marker):
            next_index = index + 1
            if (
                skip_active_flag
                and next_index < len(lines)
                and lines[next_index].strip().lower() in {".true.", ".false."}
            ):
                next_index += 1
            if next_index >= len(lines):
                raise ValueError(f"Missing value after marker: {marker}")
            lines[next_index] = replacement + "\n"
            return
    raise ValueError(f"Missing marker in rout_input.txt: {marker}")


def prepare_local_routing_inputs(model_dir):
    model_dir = Path(model_dir)
    output_dir = model_dir.parent
    project_root = model_dir.parents[3]
    copies = {
        output_dir / "flow/flow_1_8.txt": model_dir / "flow_1_8.txt",
        output_dir / "flow/output_area_mask.txt": model_dir / "output_area_mask.txt",
        output_dir / "fraction/fraction.txt": model_dir / "fraction.txt",
        output_dir / "flow/area_stnloc.txt": model_dir / "area_stnloc.txt",
        project_root / "data/static/UH.all": model_dir / "UH.all",
    }
    for source, target in copies.items():
        if not source.exists():
            raise FileNotFoundError(f"Missing routing input file: {source}")
        shutil.copy2(source, target)

    rout_input = model_dir / "rout_input.txt"
    backup = model_dir / "rout_input.original.txt"
    if not backup.exists():
        shutil.copy2(rout_input, backup)
    lines = rout_input.read_text(encoding="utf-8", errors="replace").splitlines(True)
    _replace_value_after_marker(lines, "# NAME OF FLOW DIRECTION FILE", "flow_1_8.txt")
    _replace_value_after_marker(
        lines, "# NAME OF XMASK FILE", "output_area_mask.txt", skip_active_flag=True
    )
    _replace_value_after_marker(
        lines, "# NAME OF FRACTION FILE", "fraction.txt", skip_active_flag=True
    )
    _replace_value_after_marker(lines, "# NAME OF STATION FILE", "area_stnloc.txt")
    _replace_value_after_marker(lines, "# PATH OF INPUT FILES AND PRECISION", "chanliu_result/fluxes_")
    _replace_value_after_marker(lines, "# PATH OF OUTPUT FILES", "chanliu_result/")
    _replace_value_after_marker(lines, "# NAME OF UNIT HYDROGRAPH FILE", "UH.all")
    rout_input.write_text("".join(lines), encoding="utf-8")


def run_vic_model(
    run_id,
    project_root=".",
    source_dir=None,
    processes=12,
    vic_args=None,
    stream_output=True,
):
    if source_dir is None:
        raise ValueError("source_dir must point to MAC_MPI_VIC.X and rout.so")
    project_root = Path(project_root).resolve()
    model_dir = project_root / "runs" / run_id / "output" / "model"
    global_param = model_dir / "chanliu_input.txt"
    if not global_param.exists():
        raise FileNotFoundError(f"Missing VIC global parameter file: {global_param}")

    copy_vic_binaries(model_dir, source_dir=source_dir)
    prepare_local_routing_inputs(model_dir)
    args = list(vic_args) if vic_args is not None else DEFAULT_VIC_ARGS
    command = [
        "mpirun",
        "-np",
        str(processes),
        "./MAC_MPI_VIC.X",
        "-g",
        "chanliu_input.txt",
        *args,
    ]

    env = os.environ.copy()
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")

    stdout_path = model_dir / "vic_stdout.log"
    stderr_path = model_dir / "vic_stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.Popen(
            command,
            cwd=model_dir,
            env=env,
            stdout=subprocess.PIPE if stream_output else stdout,
            stderr=subprocess.PIPE if stream_output else stderr,
            text=True,
            bufsize=1,
        )
        if stream_output:
            threads = [
                threading.Thread(
                    target=_stream_to_log,
                    args=(completed.stdout, stdout, sys.stdout),
                ),
                threading.Thread(
                    target=_stream_to_log,
                    args=(completed.stderr, stderr, sys.stderr),
                ),
            ]
            for thread in threads:
                thread.start()
            returncode = completed.wait()
            for thread in threads:
                thread.join()
        else:
            returncode = completed.wait()

    return VicRunResult(
        model_dir=model_dir,
        command=command,
        returncode=returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def _stream_to_log(source, log_file, console):
    for line in source:
        log_file.write(line)
        log_file.flush()
        console.write(line)
        console.flush()
