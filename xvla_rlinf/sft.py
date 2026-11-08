"""Controller-verified LIBERO SFT data and a bounded RLinf loader.

Raw relative actions are never interpreted as absolute targets.  Conversion
replays each selected demonstration through LIBERO's delta controller and
records controller.goal_pos/goal_ori, following the pinned X-VLA rel2abs code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .transforms import absolute_actions_to_6d, absolute_sft_window


SCHEMA = "xvla_libero_controller_abs_v1"
SOURCE_COMMIT = "6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2"
PT_ID = "2toINF/X-VLA-Pt"
PT_REVISION = "c1c4a64a7e03ac5b95c468bf1578f3d03651b53b"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: str | Path, *, verify_files: bool = True) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.suffix != ".json":
        raise ValueError("SFT requires a controller-verified JSON manifest, not raw relative-action HDF5")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("Unsupported/unverified SFT conversion provenance")
    if manifest.get("absolute_source") != "replayed_controller_goal_pos_goal_ori":
        raise ValueError("Absolute targets must come from replayed controller goals, not accumulated deltas")
    if manifest.get("image_alignment") != "post_action_drop_first":
        raise ValueError("Unknown image/absolute-action alignment")
    episodes = manifest.get("episodes", [])
    if not episodes or manifest.get("demonstrations") != len(episodes):
        raise ValueError("Manifest demonstration count is empty or inconsistent")
    identities = set()
    for episode in episodes:
        identity = (episode["task_suite"], int(episode["task_id"]), episode["demo_id"])
        if identity in identities:
            raise ValueError("Duplicated demonstration identity")
        identities.add(identity)
        if not episode.get("success_final") or not episode.get("controller_replay_verified"):
            raise ValueError("Failed or unverified replay cannot enter controlled SFT")
        episode_path = (path.parent / episode["path"]).resolve()
        if not episode_path.is_file():
            raise FileNotFoundError(episode_path)
        if verify_files and sha256_file(episode_path) != episode["sha256"]:
            raise ValueError(f"Converted demonstration hash differs: {episode_path}")
        episode["resolved_path"] = str(episode_path)
    manifest["manifest_path"] = str(path)
    manifest["manifest_sha256"] = sha256_file(path)
    return manifest


def _default_env(task_suite: str, task_id: int, camera_size: int):
    # Local import: importing data schema does not instantiate a simulator.
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[task_suite]()
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=camera_size,
                             camera_widths=camera_size)
    return env, task


def convert_demonstrations(
    source: str | Path,
    output_dir: str | Path,
    *,
    task_suite: str,
    task_id: int,
    data_revision: str,
    demonstrations: int = 1,
    seed: int = 1234,
    camera_size: int = 256,
    expected_source_sha256: str | None = None,
    env_factory=None,
) -> Path:
    """Replay full selected demos and publish a provenance-checked manifest.

    The CLI enforces Slurm; ``env_factory`` permits unit-testing orchestration
    without confusing a mock environment with real simulation evidence.
    """
    import h5py
    from robosuite.utils import transform_utils as transform

    if demonstrations < 1 or not data_revision:
        raise ValueError("A positive demonstration count and explicit data revision are required")
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    if (output_dir / "manifest.json").exists():
        raise FileExistsError("Use a fresh conversion directory; manifests are immutable")
    output_dir.mkdir(parents=True, exist_ok=True)
    make_env = env_factory or _default_env
    manifest = {
        "schema": SCHEMA, "source_commit": SOURCE_COMMIT,
        "source_hdf5": str(source), "source_sha256": sha256_file(source),
        "data_revision": data_revision, "selection_seed": seed,
        "requested_demonstrations": demonstrations,
        "absolute_source": "replayed_controller_goal_pos_goal_ori",
        "image_alignment": "post_action_drop_first",
        "main_view_orientation": "rotate_180", "wrist_orientation": "unmodified",
        "settling_steps_per_demo": 10, "domain_id": 3,
        "sft_source_frequency": 30.0, "sft_query_duration_seconds": 1.0,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "episodes": [],
    }
    if expected_source_sha256 is not None and manifest["source_sha256"] != expected_source_sha256:
        raise ValueError("Raw HDF5 differs from its pinned download SHA256 receipt")
    with h5py.File(source, "r") as original:
        if "data" not in original:
            raise ValueError("Expected raw LIBERO /data/demo_* hierarchy")
        names = sorted((key for key in original["data"] if key.startswith("demo_")),
                       key=lambda value: int(value.split("_")[-1]))
        if len(names) < demonstrations:
            raise ValueError("Requested more independent demonstrations than present")
        selected = np.random.default_rng(seed).choice(names, demonstrations, replace=False).tolist()
        manifest["selected_demo_ids"] = selected
        for demo_id in selected:
            demo = original["data"][demo_id]
            actions, states = np.asarray(demo["actions"]), np.asarray(demo["states"])
            if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) <= 10 or len(states) == 0:
                raise ValueError("Expected full relative-action7 demonstration with initial simulator state")
            env, task = make_env(task_suite, task_id, camera_size)
            target = output_dir / f"{task_suite}_task{task_id}_{demo_id}.hdf5"
            if target.exists():
                env.close()
                raise FileExistsError(target)
            try:
                env.seed(0)
                env.reset()
                obs = env.set_init_state(states[0])
                for _ in range(10):
                    obs, _, _, _ = env.step(np.asarray([0, 0, 0, 0, 0, 0, -1.0]))
                for robot in env.env.robots:
                    robot.controller.use_delta = True
                main = [np.asarray(obs["agentview_image"])[::-1, ::-1].copy()]
                wrist = [np.asarray(obs["robot0_eye_in_hand_image"]).copy()]
                absolute, ever, final = [], False, False
                for relative in actions:
                    obs, _, done, _ = env.step(relative)
                    controller = env.env.robots[0].controller
                    position = np.asarray(controller.goal_pos).copy()
                    axis_angle = transform.quat2axisangle(transform.mat2quat(controller.goal_ori))
                    absolute.append(np.concatenate((position, axis_angle, relative[-1:])))
                    main.append(np.asarray(obs["agentview_image"])[::-1, ::-1].copy())
                    wrist.append(np.asarray(obs["robot0_eye_in_hand_image"]).copy())
                    ever, final = ever or bool(done), bool(done)
                if not final:
                    raise RuntimeError(f"Replay did not finish successfully: {task_suite}/{task_id}/{demo_id}")
                raw10 = absolute_actions_to_6d(np.asarray(absolute))
                with h5py.File(target, "x") as converted:
                    converted.attrs.update(schema=SCHEMA, source_commit=SOURCE_COMMIT,
                                           controller_replay_verified=True, success_final=final,
                                           domain_id=3, task_id=task_id, task_suite=task_suite,
                                           source_demo_id=demo_id, language=task.language)
                    converted.create_dataset("abs_action_6d", data=raw10.astype(np.float32))
                    converted.create_dataset("images/main", data=np.stack(main), compression="lzf")
                    converted.create_dataset("images/wrist", data=np.stack(wrist), compression="lzf")
                manifest["episodes"].append({
                    "path": target.name, "sha256": sha256_file(target),
                    "task_suite": task_suite, "task_id": task_id, "task_name": task.name,
                    "language": task.language, "demo_id": demo_id, "frames": len(actions),
                    "success_ever": ever, "success_final": final,
                    "controller_replay_verified": True,
                    "conversion_env_steps": len(actions), "settling_env_steps": 10,
                    "control_freq": getattr(env.env, "control_freq", None),
                })
                print("XVLA_SFT_CONVERTED " + json.dumps(manifest["episodes"][-1]), flush=True)
            finally:
                env.close()
    manifest["demonstrations"] = len(manifest["episodes"])
    manifest["training_frames_before_static_filter"] = sum(ep["frames"] for ep in manifest["episodes"])
    target_manifest = output_dir / "manifest.json"
    target_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target_manifest


class ConvertedLiberoDataset:
    """Map-style, independently counted demos with official SFT windows.

    The exact pinned LiberoHandler timing, image-index drop and static filter
    are used. Image augmentation is applied by the collator, not native_eval's
    Florence processor image pipeline.
    """

    def __init__(self, manifest_path: str | Path, *, horizon: int, verify_files: bool = True):
        import h5py
        if horizon < 1:
            raise ValueError("A positive checkpoint prediction horizon is required")
        self.manifest = load_manifest(manifest_path, verify_files=verify_files)
        self.horizon = int(horizon)
        self.samples: list[tuple[int, int]] = []
        self.actions: list[np.ndarray] = []
        for episode_index, episode in enumerate(self.manifest["episodes"]):
            with h5py.File(episode["resolved_path"], "r") as data:
                if data.attrs.get("schema") != SCHEMA or not data.attrs.get("controller_replay_verified"):
                    raise ValueError("Converted HDF5 provenance attributes are missing")
                absolute = np.asarray(data["abs_action_6d"], dtype=np.float32)
                if absolute.shape != (episode["frames"], 10) or not np.isfinite(absolute).all():
                    raise ValueError("Expected finite controller-derived abs_action_6d[T,10]")
                for key in ("images/main", "images/wrist"):
                    if data[key].shape[0] != len(absolute) + 1:
                        raise ValueError("Official drop-first alignment requires T+1 rendered frames")
            self.actions.append(absolute)
            for frame in range(max(0, len(absolute) - 10)):
                window = absolute_sft_window(absolute, frame, num_actions=self.horizon)
                if not window["is_static"]:
                    self.samples.append((episode_index, frame))
        if not self.samples:
            raise ValueError("No usable SFT windows after the official static-segment filter")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import h5py
        episode_index, frame = self.samples[index]
        episode = self.manifest["episodes"][episode_index]
        window = absolute_sft_window(self.actions[episode_index], frame, num_actions=self.horizon)
        image_index = window["image_index"]
        with h5py.File(episode["resolved_path"], "r") as data:
            images = [np.asarray(data[key][image_index]) for key in ("images/main", "images/wrist")]
        return {"images": images, "language_instruction": episode["language"],
                "domain_id": 3, "proprio": window["proprio"], "action": window["action"]}


class OfficialSftCollator:
    """Pinned training image transform + official processor language encoder."""

    def __init__(self, processor, *, training: bool = True):
        import torchvision.transforms as transform
        self.processor = processor
        self.image_aug = transform.Compose([
            transform.Resize((224, 224), interpolation=transform.InterpolationMode.BICUBIC),
            transform.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=0.)
            if training else transform.Lambda(lambda image: image),
            transform.ToTensor(),
            transform.Normalize((.485, .456, .406), (.229, .224, .225), inplace=True),
        ])

    def __call__(self, samples):
        import torch
        from PIL import Image
        pixels, masks = [], []
        for sample in samples:
            images = [self.image_aug(Image.fromarray(image)) for image in sample["images"]]
            if not 1 <= len(images) <= 3:
                raise ValueError("SFT sample requires one to three valid image views")
            mask = torch.zeros(3, dtype=torch.bool)
            mask[:len(images)] = True
            while len(images) < 3:
                images.append(torch.zeros_like(images[0]))
            pixels.append(torch.stack(images))
            masks.append(mask)
        language = self.processor.encode_language([sample["language_instruction"] for sample in samples])
        return {**language, "image_input": torch.stack(pixels), "image_mask": torch.stack(masks),
                "domain_id": torch.tensor([sample["domain_id"] for sample in samples], dtype=torch.long),
                "proprio": torch.from_numpy(np.stack([sample["proprio"] for sample in samples])).float(),
                "action": torch.from_numpy(np.stack([sample["action"] for sample in samples])).float()}


def build_xvla_sft_dataloader(cfg, world_size: int, rank: int, data_paths, eval_dataset: bool = False):
    import torch
    from torch.utils.data import DistributedSampler
    from torchdata.stateful_dataloader import StatefulDataLoader
    from .vendor.xvla.processing_xvla import XVLAProcessor

    if eval_dataset:
        raise NotImplementedError("SFT validation metrics are not implemented; use native_eval for policy evaluation")
    if isinstance(data_paths, (list, tuple)) or not isinstance(data_paths, (str, Path)):
        raise ValueError("Provide one immutable converted JSON manifest as data.train_data_paths")
    model_path = Path(cfg.actor.model.model_path)
    receipt = json.loads((model_path / "verified.json").read_text())
    if (receipt["repo_id"], receipt["revision"]) != (PT_ID, PT_REVISION):
        raise ValueError("Controlled finite-demo SFT must initialize from the fixed X-VLA-Pt checkpoint")
    model_cfg = json.loads((model_path / "config.json").read_text())
    dataset = ConvertedLiberoDataset(data_paths, horizon=int(model_cfg["num_actions"]))
    processor = XVLAProcessor.from_pretrained(str(model_path), trust_remote_code=False, local_files_only=True)
    seed = int(cfg.actor.seed)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    batch = int(cfg.actor.micro_batch_size)
    if len(sampler) < batch:
        raise ValueError("Too few usable windows for one microbatch")
    workers = int(cfg.data.get("num_workers", 0))
    if workers < 0 or workers > 4:
        raise ValueError("SFT data workers must be bounded to 0..4 for the smoke allocation")
    loader = StatefulDataLoader(dataset, batch_size=batch, sampler=sampler, drop_last=True,
                               num_workers=workers, collate_fn=OfficialSftCollator(processor),
                               generator=torch.Generator().manual_seed(seed + rank))
    config = {"schema": SCHEMA, "manifest_sha256": dataset.manifest["manifest_sha256"],
              "demonstrations": dataset.manifest["demonstrations"],
              "usable_windows": len(dataset), "seed": seed, "rank": rank, "world_size": world_size,
              "prediction_horizon": dataset.horizon, "source_frequency": 30.0,
              "query_duration": 1.0, "transform": "official_resize224_bicubic_colorjitter_imagenet"}
    print("XVLA_SFT_DATA " + json.dumps(config), flush=True)
    return loader, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="Replay raw LIBERO delta demonstrations under Slurm")
    convert.add_argument("--source-hdf5", required=True)
    convert.add_argument("--output-dir", required=True)
    convert.add_argument("--task-suite", default="libero_spatial")
    convert.add_argument("--task-id", type=int, required=True)
    convert.add_argument("--data-revision", required=True)
    convert.add_argument("--source-sha256", help="Expected SHA256 from the pinned download receipt")
    convert.add_argument("--demonstrations", type=int, default=1)
    convert.add_argument("--seed", type=int, default=1234)
    convert.add_argument("--camera-size", type=int, default=256)
    check = commands.add_parser("inspect", help="Validate a converted dataset without starting simulation")
    check.add_argument("manifest")
    args = parser.parse_args()
    if args.command == "convert":
        if not os.environ.get("SLURM_JOB_ID"):
            parser.error("Controller replay is simulation: submit through this project's Slurm launcher")
        manifest = convert_demonstrations(args.source_hdf5, args.output_dir, task_suite=args.task_suite,
                                         task_id=args.task_id, data_revision=args.data_revision,
                                         demonstrations=args.demonstrations, seed=args.seed,
                                         camera_size=args.camera_size,
                                         expected_source_sha256=args.source_sha256)
        print(str(manifest))
    else:
        manifest = load_manifest(args.manifest)
        print(json.dumps({"manifest": manifest["manifest_path"], "demonstrations": manifest["demonstrations"],
                          "frames": manifest["training_frames_before_static_filter"]}))


if __name__ == "__main__":
    main()
