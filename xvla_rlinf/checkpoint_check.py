"""Rebuild a saved X-VLA model in a fresh, allocated Python process.

This checks RLinf's exported full model state, the configured row/LoRA/value
structure, and an optional fixed-input/noise probe. It does not resume an RLinf
worker or validate optimizer, scheduler, dataloader, distributed or RNG state.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any


def read_evidence(checkpoint_dir: Path) -> tuple[dict[str, Any], Path]:
    checkpoint_dir = checkpoint_dir.resolve()
    path = checkpoint_dir / "xvla_evidence.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("xvla_evidence.json must be a JSON object")
    config = evidence.get("model_config")
    if not isinstance(config, dict) or config.get("model_type") != "xvla":
        raise ValueError("Evidence must include the exact resolved X-VLA model_config")
    if config.get("precision") != "fp32":
        raise ValueError("Independent checkpoint smoke currently requires fp32")
    if config.get("is_lora") is not False:
        raise ValueError("RLinf factory LoRA must be false; the X-VLA extension owns its parameter scope")
    digest = evidence.get("digest")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Evidence must include a lowercase SHA256 model-state digest")
    if evidence.get("probe") is not None and not isinstance(evidence["probe"], dict):
        raise ValueError("Optional probe must be a JSON object")
    weights = checkpoint_dir / "model_state_dict" / "full_weights.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Actual RLinf full-model export is missing: {weights}")
    return evidence, weights


def compare_probe(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    keys = {"raw", "actions", "value"}
    if set(expected) != keys or set(actual) != keys:
        raise ValueError("Fixed probe must contain exactly raw/actions/value")
    comparisons = {}
    for key in sorted(keys):
        before, after = np.asarray(expected[key]), np.asarray(actual[key])
        if before.shape != after.shape or before.size == 0:
            raise ValueError(f"Checkpoint probe shape mismatch or empty data: {key}")
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError(f"Non-finite checkpoint probe: {key}")
        np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-6, equal_nan=False)
        comparisons[key] = {"shape": list(before.shape),
                            "max_abs_difference": float(np.max(np.abs(after - before)))}
    return comparisons


def assert_context() -> dict[str, Any]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Independent checkpoint verification requires a Slurm allocation")
    if os.environ.get("RLINF_XVLA_ISOLATED_RAY") != "1":
        raise RuntimeError("Use the project's isolated X-VLA Slurm launcher")
    if os.environ.get("RLINF_EXT_MODULE") != "xvla_rlinf":
        raise RuntimeError("RLINF_EXT_MODULE must be xvla_rlinf for the checkpoint tool process")
    from .sft_entry import assert_pinned_runtime
    assert_pinned_runtime()
    import rlinf
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Independent checkpoint smoke requires exactly one CUDA-visible allocated GPU")
    if torch.distributed.is_initialized():
        raise RuntimeError("Run this checker in a fresh process without an existing process group")
    return {"python": sys.executable, "rlinf": str(Path(rlinf.__file__).resolve()),
            "torch": torch.__version__, "pid": os.getpid(),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "allocation_cuda_visible_devices": os.environ.get("RLINF_XVLA_ALLOCATION_CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0)}


def check_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    identity = assert_context()
    evidence, weights_path = read_evidence(checkpoint_dir)
    import torch
    from omegaconf import OmegaConf
    from rlinf.scheduler.cluster import load_user_extension_module
    from rlinf.models import get_model
    from .audit_workers import fixed_probe, state_digest

    # Invoke the same extension hook used by the pinned driver/worker paths.
    # The hook logs registration errors; a missing builder remains fatal below.
    load_user_extension_module()
    torch.cuda.reset_peak_memory_stats()
    loaded_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded_state, dict):
        raise TypeError("RLinf full_weights.pt must contain an ordinary model state dict")
    exported_digest = state_digest(loaded_state)
    if exported_digest != evidence["digest"]:
        raise RuntimeError("Exported full_weights.pt differs from the pre-save model-state evidence")

    config = OmegaConf.create(evidence["model_config"])
    if OmegaConf.to_container(config, resolve=True) != evidence["model_config"]:
        raise ValueError("model_config must already be resolved; do not reinterpret saved interpolation")
    if config.get("xvla", {}).get("initial_sft_checkpoint") is not None:
        # Rebuild the architecture, then verify the complete RL export directly.
        config.xvla.initial_sft_checkpoint = None
    model = get_model(config)
    if model is None:
        raise RuntimeError("The RLINF_EXT_MODULE hook did not register an X-VLA model builder")
    if type(model).__module__ != "xvla_rlinf.policy" or type(model).__name__ != "XVLAPolicy":
        raise TypeError("Checkpoint must rebuild the registered XVLAPolicy")
    model = model.to(device="cuda:0", dtype=torch.float32).eval()
    model.load_state_dict(loaded_state, strict=True)
    del loaded_state
    restored_digest = state_digest(model.state_dict())
    if restored_digest != evidence["digest"]:
        raise RuntimeError("Freshly rebuilt model parameters/buffers differ after strict state loading")

    probe_result: dict[str, Any] = {"status": "NOT RUN", "reason": "No pre-save probe supplied; model-state digest checked"}
    if evidence.get("probe") is not None:
        probe_result = {"status": "PASS", "comparisons": compare_probe(evidence["probe"], fixed_probe(model)),
                        "rtol": 1e-5, "atol": 1e-6}
    torch.cuda.synchronize()
    return {"status": "PASS", "identity": identity,
            "checkpoint_dir": str(checkpoint_dir.resolve()), "weights_path": str(weights_path),
            "model_config": evidence["model_config"], "digest": restored_digest,
            "exported_digest": exported_digest, "strict_state_load": True,
            "fixed_probe": probe_result, "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "scope": "independent process model rebuild/full-state load and optional fixed observation/noise probe",
            "optimizer_resume_checked": False, "dataloader_rng_resume_checked": False,
            "distributed_worker_resume_checked": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path, help="Actual RLinf actor checkpoint directory")
    parser.add_argument("--output", type=Path, help="Default: <checkpoint-dir>/checkpoint_check.json")
    args = parser.parse_args(argv)
    started = time.time()
    output = args.output or args.checkpoint_dir / "checkpoint_check.json"
    try:
        result = check_checkpoint(args.checkpoint_dir)
        code = 0
    except Exception as error:
        traceback.print_exc()
        result = {"status": "FAIL", "checkpoint_dir": str(args.checkpoint_dir.resolve()),
                  "pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "error_type": type(error).__name__, "error": str(error),
                  "optimizer_resume_checked": False, "distributed_worker_resume_checked": False}
        code = 1
    result.update(started=started, elapsed_seconds=time.time() - started)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print("XVLA_CHECKPOINT_CHECK " + json.dumps(result), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
