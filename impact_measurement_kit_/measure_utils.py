import csv
import json
import os
import platform
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Any, Optional

import psutil

try:
    from codecarbon import EmissionsTracker
except Exception:
    EmissionsTracker = None

SAMPLE_EVERY_SEC = 0.5


def detect_gpus():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        gpus = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            idx, name, mem, driver = [p.strip() for p in line.split(",", 3)]
            gpus.append(
                {
                    "index": idx,
                    "name": name,
                    "memory_total": mem,
                    "driver_version": driver,
                }
            )
        return gpus
    except Exception:
        return []


def _monitor_process(stop_flag: Dict[str, bool], out: Dict[str, Any]):
    proc = psutil.Process(os.getpid())
    max_rss = 0
    max_cpu_percent = 0.0
    cpu_samples = []
    io_read_bytes = 0
    io_write_bytes = 0

    while not stop_flag["stop"]:
        try:
            rss = proc.memory_info().rss
            max_rss = max(max_rss, rss)
            cpu = psutil.cpu_percent(interval=None)
            max_cpu_percent = max(max_cpu_percent, cpu)
            cpu_samples.append(cpu)
            io = proc.io_counters()
            io_read_bytes = max(io_read_bytes, getattr(io, "read_bytes", 0))
            io_write_bytes = max(io_write_bytes, getattr(io, "write_bytes", 0))
        except Exception:
            pass
        time.sleep(SAMPLE_EVERY_SEC)

    out["max_rss_bytes"] = max_rss
    out["max_cpu_percent"] = max_cpu_percent
    out["avg_cpu_percent"] = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    out["process_read_bytes"] = io_read_bytes
    out["process_write_bytes"] = io_write_bytes


@dataclass
class RunMetadata:
    run_name: str
    hostname: str
    platform: str
    python_version: str
    cpu_count_logical: int
    cpu_count_physical: int
    ram_total_gb: float
    gpus: list
    start_time_epoch: float
    end_time_epoch: float
    elapsed_seconds: float
    emissions_kg_co2eq: Optional[float]
    max_rss_gb: float
    max_cpu_percent: float
    avg_cpu_percent: float
    process_read_gb: float
    process_write_gb: float
    status: str
    notes: Optional[str] = None


def run_measured(fn: Callable[[], Any], run_name: str, output_dir: str = "impact_runs", notes: str = "") -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    monitor_stats: Dict[str, Any] = {}
    stop_flag = {"stop": False}
    monitor_thread = threading.Thread(target=_monitor_process, args=(stop_flag, monitor_stats), daemon=True)
    monitor_thread.start()

    tracker = None
    if EmissionsTracker is not None:
        tracker = EmissionsTracker(
            output_dir=str(out_dir),
            output_file=f"{run_name}_codecarbon.csv",
            save_to_file=True,
            measure_power_secs=1,
            log_level="error",
        )

    start_time_epoch = time.time()
    t0 = time.perf_counter()
    emissions = None
    status = "success"
    error_text = None

    if tracker is not None:
        tracker.start()

    try:
        fn()
    except Exception as exc:
        status = "failed"
        error_text = repr(exc)
        raise
    finally:
        elapsed = time.perf_counter() - t0
        if tracker is not None:
            try:
                emissions = tracker.stop()
            except Exception:
                emissions = None
        stop_flag["stop"] = True
        monitor_thread.join(timeout=5)

    end_time_epoch = time.time()

    meta = RunMetadata(
        run_name=run_name,
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        cpu_count_logical=psutil.cpu_count(logical=True) or 0,
        cpu_count_physical=psutil.cpu_count(logical=False) or 0,
        ram_total_gb=round(psutil.virtual_memory().total / 1024**3, 3),
        gpus=detect_gpus(),
        start_time_epoch=start_time_epoch,
        end_time_epoch=end_time_epoch,
        elapsed_seconds=round(elapsed, 4),
        emissions_kg_co2eq=emissions,
        max_rss_gb=round(monitor_stats.get("max_rss_bytes", 0) / 1024**3, 4),
        max_cpu_percent=round(monitor_stats.get("max_cpu_percent", 0.0), 2),
        avg_cpu_percent=round(monitor_stats.get("avg_cpu_percent", 0.0), 2),
        process_read_gb=round(monitor_stats.get("process_read_bytes", 0) / 1024**3, 4),
        process_write_gb=round(monitor_stats.get("process_write_bytes", 0) / 1024**3, 4),
        status=status,
        notes=notes or None,
    )

    meta_dict = asdict(meta)
    if error_text:
        meta_dict["error"] = error_text

    with open(out_dir / f"{run_name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)

    return meta_dict


def append_manual_measurement_csv(csv_path: str, row: Dict[str, Any]):
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_file.exists()
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
