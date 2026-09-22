#!/usr/bin/env python3
"""Write a compact, automatic completion audit for one controlled Slurm job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone


MARKERS = (
    "QWEN35_RUNTIME_VALIDATION",
    "CONTROLLED_SFT_CONFIG",
    "FORMAL_PROFILE_CHECKPOINT",
    "FORMAL_PROFILE_RESUME",
    "FORMAL_PROFILE_COMPLETE",
    "EVAL_SMOKE_CANDIDATE",
    "EVAL_SMOKE_COMPLETE",
    "train_runtime",
    "Traceback",
    "CUDA out of memory",
    "Error",
)


def log_edges(path: Path, limit: int = 256 * 1024) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        head = handle.read(limit)
        handle.seek(0, 2)
        size = handle.tell()
        if size > limit:
            handle.seek(max(0, size - limit))
            tail = handle.read(limit)
        else:
            tail = b""
    return (head + b"\n" + tail).decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--slurm-log", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()

    fields = "JobID,JobName,State,ExitCode,Elapsed,Start,End,AllocTRES"
    command = ["sacct", "-j", args.job_id, f"--format={fields}", "-P", "-n"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = [line.split("|", 7) for line in result.stdout.splitlines() if line]
    matching = [row for row in rows if row[0] == args.job_id]
    if len(matching) != 1 or len(matching[0]) != 8:
        raise RuntimeError(f"Expected one sacct parent row for {args.job_id}: {rows[:4]}")

    markers = [line[-1000:] for line in log_edges(args.slurm_log).splitlines()
               if any(marker in line for marker in MARKERS)]
    report = {
        "reported_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "slurm": dict(zip(fields.split(","), matching[0])),
        "slurm_log": str(args.slurm_log),
        "slurm_log_exists": args.slurm_log.is_file(),
        "markers": markers[-30:],
    }
    if args.artifact_dir:
        report["artifact_dir"] = str(args.artifact_dir)
        report["checkpoint_count"] = len(list(args.artifact_dir.glob("checkpoint-*")))
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / f"{args.job_id}-{args.label}.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"SLURM_COMPLETION_AUDIT output={output} state={report['slurm']['State']} exit={report['slurm']['ExitCode']}")


if __name__ == "__main__":
    main()
