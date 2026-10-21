"""Measure release simulation and RGB costs under the host's current shared load."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np

from vera20k_agent import VeraBatch


def gpu_sample():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv"
    ], text=True).strip()


def process_gpu_memory():
    root = ET.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True))
    return [{"pci_bus": gpu.attrib["id"], "used_memory": process.findtext("used_memory")}
            for gpu in root.findall("gpu") for process in gpu.findall("./processes/process_info")
            if process.findtext("pid") == str(os.getpid())]


def process_memory():
    rows = Path("/proc/self/smaps_rollup").read_text().splitlines()[1:]
    values = {key: int(value.split()[0]) / 1024 for key, value in (line.split(":", 1) for line in rows)}
    return {"rss_mib": values["Rss"], "pss_mib": values["Pss"],
            "private_mib": values["Private_Clean"] + values["Private_Dirty"]}


def measure(assets, slots, render, threads, steps, stride, gpu_index):
    initial_gpu = gpu_sample()
    started = time.perf_counter()
    batch = VeraBatch(assets, slots, render=render, threads=threads, gpu_index=gpu_index)
    batch.reset_slots({slot: slot for slot in range(slots)})
    reset_sec = time.perf_counter() - started
    actions = {slot: [] for slot in range(slots)}
    for _ in range(5):
        batch.step_slots(actions)
    loaded_gpu = gpu_sample()
    cpu_started = time.process_time()
    started = time.perf_counter()
    image_latencies = []
    observations = 0
    try:
        for step in range(steps):
            batch.step_slots(actions)
            if render and ((step + 1) % stride == 0 or step + 1 == steps):
                image_started = time.perf_counter()
                batch.observe_slots(list(actions))
                image_latencies.append(time.perf_counter() - image_started)
                observations += slots
        elapsed = time.perf_counter() - started
        cpu = time.process_time() - cpu_started
        result = {
            "slots": slots, "render": render, "threads": threads, "gpu_index": gpu_index,
            "steps_per_slot": steps, "observation_stride": stride, "reset_sec": reset_sec,
            "wall_sec": elapsed, "cpu_sec": cpu, "cpu_cores_mean": cpu / elapsed,
            "environment_steps_per_sec": steps * slots / elapsed,
            "rgb_observations_per_sec": observations / elapsed,
            "observation_batch_ms_p50": float(np.median(image_latencies) * 1000) if image_latencies else None,
            "observation_batch_ms_p95": float(np.percentile(image_latencies, 95) * 1000) if image_latencies else None,
            "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "gpu_before": initial_gpu, "gpu_loaded": loaded_gpu, "gpu_after": gpu_sample(),
            "process_gpu_memory": process_gpu_memory(), "memory": process_memory(),
            "observation_batches": len(image_latencies), "warmup_steps": 5,
            "workload": "initial 1v1 skirmish with no player commands",
            "drawn_frames_per_sec": steps * slots / elapsed if render else None,
        }
    finally:
        batch.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--slots", required=True, type=int)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--threads", default=8, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--stride", default=10, type=int)
    parser.add_argument("--gpu-index", default=0, type=int)
    args = parser.parse_args()
    result = measure(args.assets, args.slots, args.render, args.threads, args.steps, args.stride, args.gpu_index)
    result.update({"build": "release", "shared_gpu_load": True, "pid": os.getpid(), "evidence": "new_simulation"})
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
