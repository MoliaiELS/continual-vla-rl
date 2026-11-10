"""Measure paired teacher-forced action errors for Pt and audited SFT weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .benchmark import CHUNK, _load_policy
from .sft import ConvertedLiberoDataset, OfficialSftCollator
from .transforms import axis_angle_to_matrix, rotation_6d_to_axis_angle


def action_errors(prediction, target):
    if prediction.shape != (CHUNK, 20) or target.shape != prediction.shape:
        raise ValueError("Expected matching full-horizon 20D action chunks")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Action comparison requires finite values")
    predicted_rot = axis_angle_to_matrix(rotation_6d_to_axis_angle(prediction[:, 3:9]))
    target_rot = axis_angle_to_matrix(rotation_6d_to_axis_angle(target[:, 3:9]))
    # The Frobenius inner product equals trace(predicted_rot.T @ target_rot).
    cosine = (np.sum(predicted_rot * target_rot, axis=(-1, -2)) - 1) / 2
    grip_pred, grip_target = prediction[:, 9], target[:, 9]
    return {
        "position_l2_m": np.linalg.norm(prediction[:, :3] - target[:, :3], axis=-1),
        "rotation_geodesic_deg": np.degrees(np.arccos(np.clip(cosine, -1, 1))),
        "gripper_probability_abs_error": np.abs(grip_pred - grip_target),
        "gripper_command_mismatch": (grip_pred > .5) != (grip_target > .5),
    }


def evaluate(dataset, indices, model_name, checkpoint, seed):
    import torch

    from .sampler import clean_action_prediction

    model, provenance = _load_policy(model_name, checkpoint)
    collate = OfficialSftCollator(model.processor, training=False)
    noise_rng = torch.Generator(device="cpu").manual_seed(seed)
    errors = []
    with torch.inference_mode():
        for index in indices:
            sample = collate([dataset[index]])
            inputs = model._device_inputs(sample)
            noise = torch.randn((1, CHUNK, 20), generator=noise_rng).to(inputs["proprio"])
            raw, _ = clean_action_prediction(model.core, inputs, noise, steps=model.steps)
            prediction = model.core.action_space.postprocess(raw)[0].float().cpu().numpy()
            errors.append(action_errors(prediction, sample["action"][0].numpy()))

    def summarize(actions):
        result = {}
        for name in errors[0]:
            values = np.stack([window[name][actions] for window in errors]).astype(float)
            result[name] = {"mean": float(np.mean(values)), "p90": float(np.quantile(values, .9))}
        return result

    return {
        "checkpoint": provenance,
        "first_action": summarize(slice(0, 1)),
        "full_chunk": summarize(slice(None)),
        "per_window": [{name: float(np.mean(values)) for name, values in window.items()}
                       for window in errors],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[],
                        help="Audited SFT actor directory. Repeat to compare several checkpoints.")
    parser.add_argument("--demo-id", help="Restrict the paired windows to one demonstration ID")
    parser.add_argument("--windows", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.windows <= 128:
        parser.error("--windows must be between 1 and 128")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not mask or mask != os.environ.get("RLINF_XVLA_ALLOCATION_CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Action diagnostics require the unchanged Slurm CUDA visibility mask")
    from .checkpoint_check import assert_context

    runtime = assert_context()
    output = args.output or Path(os.environ["XVLA_RUN_DIR"]) / "sft_action_check.json"
    if output.exists():
        raise FileExistsError(output)
    dataset = ConvertedLiberoDataset(args.manifest, horizon=CHUNK)
    episodes = dataset.manifest["episodes"]
    available = [index for index, (episode, _) in enumerate(dataset.samples)
                 if args.demo_id is None or episodes[episode]["demo_id"] == args.demo_id]
    if not available:
        parser.error("No usable windows match the requested demonstration")
    indices = [available[index] for index in np.linspace(
        0, len(available) - 1, num=min(args.windows, len(available)), dtype=int
    )]
    windows = []
    for index in indices:
        episode, frame = dataset.samples[index]
        identity = {key: episodes[episode][key] for key in ("task_suite", "task_id", "demo_id")}
        windows.append({**identity, "source_action_index": frame, "image_index": frame + 1})

    report = {
        "schema_version": 1,
        "runtime": runtime,
        "manifest": dataset.manifest["manifest_path"],
        "manifest_sha256": dataset.manifest["manifest_sha256"],
        "protocol": {"conditioning": "teacher_forced_demonstration_proprio", "image_augmentation": False,
                     "noise_seed": args.seed, "chunk": CHUNK},
        "windows": windows,
        "models": [evaluate(dataset, indices, "pt", None, args.seed)],
    }
    for checkpoint in args.checkpoint:
        report["models"].append(evaluate(dataset, indices, "sft-fewshot", checkpoint, args.seed))
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("XVLA_SFT_ACTION_CHECK " + str(output.resolve()), flush=True)


if __name__ == "__main__":
    main()
