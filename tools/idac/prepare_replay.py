"""Build a provenance manifest for explicitly selected raw LIBERO expert demos.

Does not run a simulator, load model weights, scan unspecified suites, or cache
logits. Run on the server with source/output paths in personal storage.
"""

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("fyp_replay", REPO / "rlinf/custom/fyp_replay.py")
_replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_replay)
FORMAT, non_noop_indices, sha256_file = _replay.FORMAT, _replay.non_noop_indices, _replay.sha256_file


def task_map():
    source = REPO / "LIBERO/libero/libero/benchmark/libero_suite_task_map.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "libero_task_map"
                              for target in node.targets))
    return ast.literal_eval(assignment.value)


def build_manifest(source_dir, output, suite, task_ids, demo_ids, model_path,
                   source_repo, source_revision):
    import h5py
    import numpy as np

    source_dir = Path(source_dir).resolve()
    output = Path(output).resolve()
    model_path = Path(model_path).resolve()
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    norm_key = f"{suite}_no_noops"
    stats = config.get("norm_stats", {}).get(norm_key)
    stats_source = config_path
    if stats is None:
        stats_source = model_path / "dataset_statistics.json"
        stats = json.loads(stats_source.read_text(encoding="utf-8"))[norm_key]
    if "action" not in stats or "q01" not in stats["action"] or "q99" not in stats["action"]:
        raise ValueError("Frozen model action quantiles are missing")
    tasks = task_map()[suite]
    task_ids = sorted(set(int(task) for task in task_ids))
    if not task_ids or any(task < 0 or task >= len(tasks) for task in task_ids):
        raise ValueError("Invalid task IDs")
    if not demo_ids or len(set(demo_ids)) != len(demo_ids):
        raise ValueError("Explicit, unique demo IDs are required")
    entries = []
    for task_id in task_ids:
        name = tasks[task_id]
        path = source_dir / suite / f"{name}_demo.hdf5"
        if not path.exists():
            path = source_dir / f"{name}_demo.hdf5"
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        with h5py.File(path, "r") as stream:
            for demo_id in demo_ids:
                demo = stream["data"][demo_id]
                actions = np.asarray(demo["actions"], dtype=np.float32)
                images = demo["obs"]["agentview_rgb"]
                if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                    raise ValueError(f"Invalid actions: {path}:{demo_id}")
                if images.shape[0] != len(actions) or len(images.shape) != 4 or images.shape[-1] != 3:
                    raise ValueError(f"Unaligned images: {path}:{demo_id}")
                if images.dtype != np.uint8:
                    raise ValueError("Raw LIBERO replay requires uint8 RGB observations")
                steps = non_noop_indices(actions)
                if len(steps) < 8:
                    raise ValueError(f"Too few non-noop actions: {path}:{demo_id}")
                if np.max(np.abs(actions[:, -1])) > 1.00001:
                    raise ValueError("Unexpected raw gripper units")
                if "rewards" in demo and not bool(np.asarray(demo["rewards"]).max() > 0):
                    raise ValueError(f"Expert demo contains no recorded success: {path}:{demo_id}")
                entries.append({"task_id": task_id, "task_name": name,
                                "task_description": name.replace("_", " "),
                                "file": str(path), "sha256": digest,
                                "file_size": path.stat().st_size, "demo": demo_id,
                                "num_steps": len(actions), "step_indices": steps,
                                "noops_removed": len(actions) - len(steps),
                                "image_shape": list(images.shape[1:]),
                                "sft_overlap": "unknown"})
    manifest = {
        "format": FORMAT, "suite": suite,
        "source": {"repo": source_repo, "revision": source_revision,
                   "kind": "original_libero_hdf5_expert_demonstrations",
                   "caveat": "Raw expert BC; not the authors' regenerated SFT pixels or audited SFT demo IDs"},
        "preprocessing": {"image_rotation_degrees": 180,
                          "action_transform": "oxe_libero_raw_gripper",
                          "noop_filter": "openvla_oft_prev_kept_1e-4"},
        "normalization": {"unnorm_key": norm_key,
                          "source": str(stats_source), "source_sha256": sha256_file(stats_source),
                          "action_statistics": stats["action"],
                          "future_statistics_recomputed": False},
        "allowed_task_ids": task_ids, "demos": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(output.suffix + ".tmp")
    pending.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(output)
    print(json.dumps({"manifest": str(output), "sha256": sha256_file(output),
                      "task_ids": task_ids, "demos": len(entries),
                      "non_noop_steps": sum(len(entry["step_indices"]) for entry in entries)}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-ids", required=True, help="Explicit comma-separated train IDs only")
    parser.add_argument("--demo-ids", default="demo_0", help="Explicit comma-separated expert demo names")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-repo", default="yifengzhu-hf/LIBERO-datasets")
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    build_manifest(args.source_dir, args.output, args.suite,
                   args.task_ids.split(","), args.demo_ids.split(","), args.model_path,
                   args.source_repo, args.source_revision)


if __name__ == "__main__":
    main()
