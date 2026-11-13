"""Combine three independently audited LIBERO-Spatial conversions for Cseq SFT."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from .sft import load_manifest, sha256_file


TASKS = (0, 1, 2)
DEMOS_PER_TASK = 10
SELECTION_SEED = 1234
SPLIT_PATH = Path(__file__).resolve().parents[1] / "configs/xvla/protocol_20261110/split.json"
SPLIT_SHA256 = "fa4ad56a5fd7f5b08b9612b7fefa1b3e156d63feb0df1d8aa60522fe062ce651"
RAW_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
RAW_SHA256 = {
    0: "ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead",
    1: "bf1090dd4826eacb49eb052f17d04798e2a1eb2f74f8d024bc3b9c5d234c25f9",
    2: "75ede0cf5fbfc925093671b55032ad80f1b1f1cf35442ec6c663460d37d3e0b3",
}
TASK0_MANIFEST_SHA256 = "0a3d6d871939e3365cd8719e94f2c6946cf40a594643c579cc82f73cb245bd77"
CONVERSION_PROTOCOL = {
    "absolute_replay_required": "success_final",
    "main_view_orientation": "rotate_180",
    "wrist_orientation": "unmodified",
    "settling_steps_per_demo": 10,
    "domain_id": 3,
    "sft_source_frequency": 30.0,
    "sft_query_duration_seconds": 1.0,
}


def _read_constituent(path, task, suite):
    manifest = load_manifest(path)
    task_id = task["task_id"]
    episodes = manifest["episodes"]
    raw_path = Path(manifest["source_hdf5"])
    expected_raw_name = f'{task["task_name"]}_demo.hdf5'
    if ((raw_path.parent.name, raw_path.name) != (suite, expected_raw_name)
            or manifest.get("data_revision") != RAW_REVISION
            or manifest.get("source_sha256") != RAW_SHA256[task_id]):
        raise ValueError(f"Task {task_id} conversion does not match the pinned raw dataset")
    if task_id == 0 and manifest["manifest_sha256"] != TASK0_MANIFEST_SHA256:
        raise ValueError("Task 0 must reuse the frozen ten-demonstration conversion")
    if (manifest.get("selection_seed") != SELECTION_SEED
            or manifest.get("requested_demonstrations") != DEMOS_PER_TASK
            or len(episodes) != DEMOS_PER_TASK
            or any(manifest.get(key) != value for key, value in CONVERSION_PROTOCOL.items())
            or manifest.get("selected_demo_ids") != [episode["demo_id"] for episode in episodes]):
        raise ValueError(f"Task {task_id} conversion does not match the ten-demonstration replay protocol")
    for episode in episodes:
        if ((episode["task_suite"], episode["task_id"], episode["task_name"], episode["language"])
                != (suite, task_id, task["task_name"], task["runtime_language"])
                or not re.fullmatch(r"demo_[0-9]+", episode["demo_id"])):
            raise ValueError(f"Task {task_id} conversion contains a demonstration outside its frozen task")
        replay = episode.get("absolute_replay") or {}
        if (episode.get("controller_replay_verified") is not True or episode.get("success_final") is not True
                or replay.get("success_final") is not True
                or episode["frames"] != replay.get("action_env_steps")
                or episode.get("control_freq") != 20 or replay.get("control_freq") != 20):
            raise ValueError(f"Task {task_id} demonstration lacks successful paired controller replay")
    return manifest


def build_cseq_manifest(task_manifests, output):
    """Create a new standard SFT manifest, leaving constituent data in place."""
    if set(task_manifests) != set(TASKS):
        raise ValueError("Cseq requires converted manifests for exactly tasks 0, 1 and 2")
    if sha256_file(SPLIT_PATH) != SPLIT_SHA256:
        raise ValueError("Cseq requires the frozen Spatial split")
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    suite = split["suite"]
    tasks = {task["task_id"]: task for task in split["tasks"]}
    output = Path(output)
    output = output.parent.resolve() / output.name
    if output.suffix != ".json":
        raise ValueError("Cseq output must be a JSON manifest")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    manifests = []
    episodes = []
    seen = set()
    for task_id in TASKS:
        manifest = _read_constituent(task_manifests[task_id], tasks[task_id], suite)
        manifests.append({
            "task_id": task_id,
            "path": Path(os.path.relpath(manifest["manifest_path"], output.parent)).as_posix(),
            "sha256": manifest["manifest_sha256"],
            "source_hdf5": manifest["source_hdf5"],
            "source_sha256": manifest["source_sha256"],
        })
        for episode in manifest["episodes"]:
            episode = dict(episode)
            converted = episode.pop("resolved_path")
            if converted in seen:
                raise ValueError(f"Cseq cannot reuse a converted episode file: {converted}")
            seen.add(converted)
            episode["path"] = Path(os.path.relpath(converted, output.parent)).as_posix()
            episodes.append(episode)

    payload = {
        "schema": manifest["schema"],
        "source_commit": manifest["source_commit"],
        "absolute_source": manifest["absolute_source"],
        "image_alignment": manifest["image_alignment"],
        **CONVERSION_PROTOCOL,
        "split_sha256": SPLIT_SHA256,
        "data_revision": RAW_REVISION,
        "selection_seed": SELECTION_SEED,
        "constituent_manifests": manifests,
        "requested_demonstrations": DEMOS_PER_TASK * len(TASKS),
        "demonstrations": len(episodes),
        "training_frames_before_static_filter": sum(episode["frames"] for episode in episodes),
        "episodes": episodes,
    }
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for task_id in TASKS:
        parser.add_argument(f"--task{task_id}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = build_cseq_manifest({task: getattr(args, f"task{task}") for task in TASKS}, args.output)
    print(f"XVLA_CSEQ_MANIFEST {output} sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()
