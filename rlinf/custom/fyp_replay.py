"""Manifest-only, bounded expert memory for FYP experiments.

This module has no Torch/MuJoCo dependency so eligibility can be checked before
loading a model. Files belonging to inactive tasks are never opened or hashed.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import random


FORMAT = "fyp-libero-replay-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def non_noop_indices(actions, threshold=1e-4):
    """Match bundled regenerate_libero_dataset.is_noop (previous KEPT action)."""
    kept = []
    previous = None
    for index, action in enumerate(actions):
        if len(action) != 7 or not all(math.isfinite(float(x)) for x in action):
            raise ValueError("Replay action must contain seven finite values")
        stationary = math.sqrt(sum(float(x) ** 2 for x in action[:-1])) < threshold
        noop = stationary and (previous is None or action[-1] == previous[-1])
        if not noop:
            kept.append(index)
            previous = action
    return kept


def raw_gripper_to_model(value):
    """OXE libero_dataset_transform: invert(clip(raw gripper, 0, 1))."""
    return 1.0 - min(1.0, max(0.0, float(value)))


def validate_action_statistics(expected, actual):
    """Require the runtime tokenizer's quantiles/mask to equal the manifest."""
    for key in ("q01", "q99"):
        wanted = tuple(float(value) for value in expected.get(key, []))
        loaded = tuple(float(value) for value in actual.get(key, []))
        if len(wanted) != 7 or not all(math.isfinite(value) for value in wanted):
            raise ValueError(f"Replay manifest has invalid action {key}")
        if wanted != loaded:
            raise ValueError(f"Runtime model action {key} differs from replay manifest")
    wanted_mask = tuple(bool(value) for value in expected.get("mask", [True] * 7))
    loaded_mask = tuple(bool(value) for value in actual.get("mask", [True] * 7))
    if len(wanted_mask) != 7 or wanted_mask != loaded_mask:
        raise ValueError("Runtime model action normalization mask differs from replay manifest")


def load_manifest(path, allowed_task_ids):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("replay_manifest_path must be absolute")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("Unsupported replay manifest format")
    allowed = {int(task) for task in allowed_task_ids}
    if not allowed:
        raise ValueError("A nonempty replay_allowed_task_ids hard allowlist is required")
    if manifest.get("preprocessing") != {
        "image_rotation_degrees": 180,
        "action_transform": "oxe_libero_raw_gripper",
        "noop_filter": "openvla_oft_prev_kept_1e-4",
    }:
        raise ValueError("Replay preprocessing contract does not match this implementation")
    entries = manifest.get("demos", [])
    identities = set()
    for entry in entries:
        task = int(entry["task_id"])
        if task not in allowed:
            raise ValueError(f"Manifest task {task} is outside the experiment allowlist")
        source = Path(entry["file"])
        if not source.is_absolute() or len(entry["sha256"]) != 64:
            raise ValueError("Manifest requires an absolute HDF5 path and SHA256")
        identity = (str(source), entry["demo"])
        if identity in identities:
            raise ValueError("A demo may appear only once in the replay manifest")
        identities.add(identity)
        steps = entry["step_indices"]
        if (not steps or steps != sorted(set(steps)) or steps[0] < 0
                or steps[-1] >= int(entry["num_steps"])):
            raise ValueError("Invalid non-noop source indices in manifest")
        if not entry.get("task_description"):
            raise ValueError("Missing task language in replay manifest")
    return manifest


def select_memory(manifest, active_task_ids, allowed_task_ids, capacity, chunk_size, seed):
    """Deterministic, balanced fixed-total-capacity selection of action chunks.

    Returns (entry-index, start-in-filtered-demo) pairs. The source files are
    deliberately not inspected here, including files from future tasks.
    """
    active = sorted({int(task) for task in active_task_ids})
    allowed = {int(task) for task in allowed_task_ids}
    if not set(active) <= allowed:
        raise ValueError("Requested replay task is outside replay_allowed_task_ids")
    if capacity <= 0 or chunk_size <= 0:
        raise ValueError("Replay capacity and action chunk size must be positive")
    if active and capacity < len(active):
        raise ValueError("Replay capacity must cover at least one chunk per active task")
    pools = {task: [] for task in active}
    for entry_index, entry in enumerate(manifest["demos"]):
        task = int(entry["task_id"])
        if task in pools:
            pools[task].extend((entry_index, offset) for offset in
                               range(max(0, len(entry["step_indices"]) - chunk_size + 1)))
    for task, pool in pools.items():
        if not pool:
            raise ValueError(f"No usable replay chunks for active task {task}")
        random.Random(f"{int(seed)}:memory:{task}").shuffle(pool)
    # Round-robin allocation also redistributes capacity from very short tasks.
    selected = []
    offsets = {task: 0 for task in active}
    while len(selected) < capacity:
        progress = False
        for task in active:
            offset = offsets[task]
            if offset < len(pools[task]):
                selected.append(pools[task][offset])
                offsets[task] += 1
                progress = True
                if len(selected) == capacity:
                    break
        if not progress:
            break
    return selected


def memory_summary(manifest, selected, chunk_size):
    counts = {}
    for entry_index, _ in selected:
        task = str(manifest["demos"][entry_index]["task_id"])
        counts[task] = counts.get(task, 0) + 1
    fingerprint = hashlib.sha256(json.dumps(selected, separators=(",", ":")).encode()).hexdigest()
    return {"chunks": len(selected), "action_steps": len(selected) * chunk_size,
            "chunks_per_task": counts, "selection_sha256": fingerprint}


class ManifestReplayMemory:
    """Small HDF5-backed dataset with no directory discovery or implicit tasks."""

    def __init__(self, manifest, active_task_ids, allowed_task_ids, capacity,
                 chunk_size, seed, verified_files=None):
        self.manifest = manifest
        self.chunk_size = int(chunk_size)
        self.selected = select_memory(manifest, active_task_ids, allowed_task_ids,
                                      int(capacity), self.chunk_size, seed)
        self.summary = memory_summary(manifest, self.selected, self.chunk_size)
        self.file_handles = {}
        self.verified_files = verified_files if verified_files is not None else {}
        payload_bytes = 0
        # Validate only admitted source files; future tasks remain unopened.
        for entry_index in sorted({index for index, _ in self.selected}):
            entry = manifest["demos"][entry_index]
            path = entry["file"]
            signature = (os.path.getsize(path), os.stat(path).st_mtime_ns)
            if signature[0] != int(entry["file_size"]):
                raise ValueError(f"Replay file size mismatch: {path}")
            verified = self.verified_files.get(path)
            if verified != (signature, entry["sha256"]):
                if sha256_file(path) != entry["sha256"]:
                    raise ValueError(f"Replay file SHA256 mismatch: {path}")
                self.verified_files[path] = (signature, entry["sha256"])
            import h5py
            with h5py.File(path, "r") as stream:
                demo = stream["data"][entry["demo"]]
                if demo["actions"].shape != (entry["num_steps"], 7):
                    raise ValueError("Replay demo actions do not match manifest")
                if (demo["obs"]["agentview_rgb"].shape[0] != entry["num_steps"]
                        or demo["obs"]["agentview_rgb"].shape[-1] != 3):
                    raise ValueError("Replay image/action alignment does not match manifest")
                image_dataset = demo["obs"]["agentview_rgb"]
                chunks = sum(index == entry_index for index, _ in self.selected)
                payload_bytes += chunks * (math.prod(image_dataset.shape[1:]) * image_dataset.dtype.itemsize
                                            + self.chunk_size * 7 * 4)
        self.summary["retained_sample_payload_bytes"] = payload_bytes

    def __len__(self):
        return len(self.selected)

    def __getitem__(self, index):
        import h5py
        import numpy as np

        entry_index, start = self.selected[index]
        entry = self.manifest["demos"][entry_index]
        path = entry["file"]
        if path not in self.file_handles:
            self.file_handles[path] = h5py.File(path, "r")
        demo = self.file_handles[path]["data"][entry["demo"]]
        steps = entry["step_indices"][start:start + self.chunk_size]
        image = np.array(demo["obs"]["agentview_rgb"][steps[0]])[::-1, ::-1].copy()
        actions = np.asarray(demo["actions"][steps], dtype=np.float32)
        actions[:, -1] = 1.0 - np.clip(actions[:, -1], 0.0, 1.0)
        if not np.isfinite(actions).all():
            raise ValueError("Non-finite expert replay actions")
        return {"obs_rgb": image, "task_desc": entry["task_description"],
                "actions": actions, "replay_task_id": int(entry["task_id"]),
                "replay_sample_id": f"{entry['task_id']}:{entry['demo']}:{start}"}

    def close(self):
        for stream in getattr(self, "file_handles", {}).values():
            stream.close()
        if hasattr(self, "file_handles"):
            self.file_handles.clear()

    def __del__(self):
        self.close()
