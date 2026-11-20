"""Qualify two same-GPU Hard evaluation processes using isolated smoke replays."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

if __package__:
    from .hard_native import PILOT_TASKS
    from .panel import MIN_PAIR_FREE_MIB, free_gpu_memory_mib
else:
    from hard_native import PILOT_TASKS
    from panel import MIN_PAIR_FREE_MIB, free_gpu_memory_mib


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def signature(job, profile):
    runtime = json.loads((job / "runtime.json").read_bytes())
    uuids = json.loads((job / "allocated_gpu_uuids.json").read_bytes())
    if len(uuids) != 1:
        raise ValueError("Hard pilot requires exactly one Slurm-assigned GPU")
    uuid = f"GPU-{uuids[0].removeprefix('GPU-')}"
    query = subprocess.check_output(["nvidia-smi", f"--id={uuid}",
                                    "--query-gpu=uuid,name,driver_version,memory.total",
                                    "--format=csv,noheader,nounits"], text=True, timeout=3)
    rows = list(csv.reader(io.StringIO(query)))
    if len(rows) != 1 or len(rows[0]) != 4 or rows[0][0].strip().lower() != uuid.lower():
        raise ValueError("Hard pilot telemetry did not identify the assigned GPU")
    _, name, driver, total = (part.strip() for part in rows[0])
    result = {"profile_fingerprint": profile["protocol_fingerprint"],
              "pilot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "hardware": {"name": name, "driver_version": driver, "memory_total_mib": int(total)},
              "environment": {key: runtime.get(key) for key in
                              ("environment", "python", "cuda", "prefix", "pip_freeze_sha256", "install_patches",
                               "code", "configs", "camera", "embodiment", "model")}}
    return result, uuid


def episodes(root, lane, task):
    shard = root / "pilot" / lane / "smoke" / task / "hard"
    manifest = json.loads((shard / "seed_manifest.json").read_bytes())
    expected = {row["episode_id"]: row for row in manifest["episodes"]}
    if manifest.get("counts_toward_formal") is not False or len(expected) != 5:
        raise ValueError("Hard pilot requires five isolated smoke initial states")
    path = shard / "episodes.jsonl"
    latest = {}
    technical = 0
    for line in path.read_bytes().splitlines(keepends=True) if path.is_file() else []:
        if not line.endswith(b"\n"):
            raise ValueError("Hard pilot contains a partial episode journal")
        row = json.loads(line)
        key = row["episode_id"]
        seed = expected.get(key)
        if (seed is None or any(row.get(field) != seed[field] for field in
                               ("actual_seed", "policy_seed", "scene_signature", "source_episode_id"))
                or row.get("protocol_fingerprint") != manifest["protocol_fingerprint"]):
            raise ValueError("Hard pilot episode differs from the frozen smoke manifest")
        if latest.get(key, {}).get("status") in {"success", "failure", "unresolved"}:
            raise ValueError("Hard pilot retried a terminal episode")
        technical += int(row["status"] in {"technical_error", "unresolved"})
        latest[key] = row
    complete = len(latest) == 5 and all(row["status"] in {"success", "failure"} for row in latest.values())
    fields = ("source_episode_id", "actual_seed", "policy_seed", "scene_signature", "status", "reason")
    counters = ("queries", "anchors_sent", "anchors", "physics_ticks", "settling_ticks", "step_lim",
                "left_plan_failures", "right_plan_failures", "simulated_seconds", "wall_seconds")
    return {"complete": complete, "technical_attempts": technical,
            "episodes": {key: {field: row[field] for field in fields} for key, row in latest.items()},
            "diagnostics": {key: {field: row.get(field) for field in counters} for key, row in latest.items()}}


def scenario(args, lane, job, uuid, total_mib):
    stop = threading.Event()
    readings, errors = [], []

    def monitor():
        with (job / f"hard-pilot-{lane}-memory.jsonl").open("a", encoding="utf-8") as stream:
            while not stop.is_set():
                try:
                    used = int(subprocess.check_output(["nvidia-smi", f"--id={uuid}", "--query-gpu=memory.used",
                                                        "--format=csv,noheader,nounits"], text=True, timeout=3).strip())
                    readings.append(used)
                    stream.write(json.dumps({"time": time.time(), "gpu_uuid": uuid, "used_mib": used}) + "\n")
                    stream.flush()
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    errors.append(str(error))
                stop.wait(2)

    def worker(task):
        log = job / f"hard-pilot-{lane}-{task}.log"
        cache = job / "hard-pilot-cache" / lane / task
        cache.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", str(Path(__file__).with_name("hard_native.py")), "--eval",
                   "--pilot-lane", lane, "--task", task, "--pinned-evaluation-root", str(args.evaluation_root),
                   "--source-panel", str(Path(os.environ["FYP_ROBOTWIN_DATA"]) / "panels" / args.evaluation_root.name),
                   "--output-root", str(args.output_root), "--robotwin-root", os.environ["FYP_ROBOTWIN_ROOT"],
                   "--xvla-root", os.environ["FYP_ROBOTWIN_XVLA"], "--checkpoint", os.environ["FYP_ROBOTWIN_MODEL"]]
        started = time.perf_counter()
        with log.open("ab") as stream:
            try:
                code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                      cwd=args.evaluation_root,
                                      env=dict(os.environ, PYTHONPATH=str(args.evaluation_root),
                                               WARP_CACHE_PATH=str(cache))).returncode
            except OSError as error:
                stream.write(str(error).encode("utf-8"))
                code = 1
        return {"task": task, "returncode": code, "seconds": time.perf_counter() - started,
                "log": log.relative_to(job).as_posix()}

    started = time.perf_counter()
    sampler = threading.Thread(target=monitor, daemon=True)
    sampler.start()
    try:
        with ThreadPoolExecutor(max_workers=1 if lane == "serial" else 2) as pool:
            futures = [pool.submit(worker, task) for task in PILOT_TASKS]
            results = [future.result() for future in futures]
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        sampler.join(timeout=4)
    return {"seconds": elapsed, "workers": results,
            "memory_peak_mib": max(readings) if readings else None,
            "sampled_headroom_mib": total_mib - max(readings) if readings else None,
            "memory_samples": len(readings), "memory_sample_errors": len(errors), "memory_interval_seconds": 2}


def qualify(args):
    """Return eval_workers=1 or 2, raising on technical pilot failure before formal evaluation."""
    args.evaluation_root = Path(args.evaluation_root).resolve()
    args.output_root = Path(args.output_root).resolve()
    job = Path(os.environ["FYP_ROBOTWIN_JOB"])
    profile = json.loads((args.output_root / "profile.json").read_bytes())
    receipt = args.output_root / "pilot/qualification.json"
    previous = json.loads(receipt.read_bytes()) if receipt.is_file() else None
    if previous and previous.get("status") == "failed":
        raise RuntimeError("A previous Hard qualification failed technically; inspect its receipt before resuming")
    current, uuid = signature(job, profile)
    result = {"profile_fingerprint": profile["protocol_fingerprint"], "signature": current,
              "gpu_uuid": uuid, "job_id": os.environ["SLURM_JOB_ID"], "eval_workers": 1,
              "qualified": False, "status": "incomplete", "scenarios": {}}
    if previous:
        if previous.get("signature") == current and previous.get("status") == "complete":
            return {**previous, "cached": True}
    prior = list((args.output_root / "pilot").glob("*/smoke/*/hard/episodes.jsonl"))
    if any(path.stat().st_size for path in prior):
        result.update(status="complete", reason="prior_pilot_rows_without_matching_complete_timing")
        write_json(receipt, result)
        return result
    write_json(receipt, result)
    try:
        result["scenarios"]["serial"] = scenario(args, "serial", job, uuid, current["hardware"]["memory_total_mib"])
        if any(row["returncode"] for row in result["scenarios"]["serial"]["workers"]):
            raise RuntimeError("Serial Hard qualification child failed")
        serial = {task: episodes(args.output_root, "serial", task) for task in PILOT_TASKS}
        result["results"] = {"serial": serial}
        if not all(row["complete"] and row["technical_attempts"] == 0 for row in serial.values()):
            raise RuntimeError("Serial Hard qualification has incomplete or technically retried episodes")
        free = free_gpu_memory_mib(job)
        result["pair_launch_free_mib"] = free
        if free is None or free < MIN_PAIR_FREE_MIB:
            result.update(status="complete", reason="insufficient_or_unavailable_pair_memory")
            return result
        result["scenarios"]["parallel"] = scenario(args, "parallel", job, uuid, current["hardware"]["memory_total_mib"])
        if any(row["returncode"] for row in result["scenarios"]["parallel"]["workers"]):
            raise RuntimeError("Parallel Hard qualification child failed")
        parallel = {task: episodes(args.output_root, "parallel", task) for task in PILOT_TASKS}
        result["results"]["parallel"] = parallel
        if not all(row["complete"] and row["technical_attempts"] == 0 for row in parallel.values()):
            raise RuntimeError("Parallel Hard qualification has incomplete or technically retried episodes")
        matched = all(serial[task]["episodes"] == parallel[task]["episodes"] for task in PILOT_TASKS)
        before, after = result["scenarios"]["serial"], result["scenarios"]["parallel"]
        gain = before["seconds"] / after["seconds"]
        headroom = after["sampled_headroom_mib"]
        result.update(status="complete", outcomes_and_identities_match=matched, serial_over_parallel=gain,
                      reason="outcome_mismatch" if not matched else "no_speed_gain" if gain <= 1 else
                      "insufficient_sampled_headroom" if headroom is None or headroom < 1024 else
                      "unreliable_memory_samples" if after["memory_sample_errors"] else "qualified")
        result["qualified"] = result["reason"] == "qualified"
        result["eval_workers"] = 2 if result["qualified"] else 1
        return result
    except Exception as error:
        result.update(status="failed", reason=str(error))
        raise
    finally:
        write_json(receipt, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    print(json.dumps(qualify(parser.parse_args()), allow_nan=False))
