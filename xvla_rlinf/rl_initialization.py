"""Initialize a fresh X-VLA RL policy from a strictly audited Pt SFT export."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

from .checkpoint_check import read_evidence
from .sft import PT_ID, PT_REVISION


def _initialization_path(model_cfg):
    value = model_cfg.get("xvla", {}).get("initial_sft_checkpoint")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("initial_sft_checkpoint must be an actor checkpoint directory or null")
    return Path(value).resolve()


def read_sft_source(checkpoint_dir, model_cfg):
    evidence, weights = read_evidence(checkpoint_dir)
    check = json.loads((checkpoint_dir / "checkpoint_check.json").read_text(encoding="utf-8"))
    source_cfg = evidence["model_config"]
    if (check.get("status") != "PASS" or check.get("strict_state_load") is not True
            or check.get("digest") != evidence["digest"] or check.get("exported_digest") != evidence["digest"]
            or check.get("model_config") != source_cfg):
        raise ValueError("SFT initialization requires a matching independent strict full-state audit")
    sft_data = json.loads((checkpoint_dir / "xvla_sft_data_state.json").read_text(encoding="utf-8"))
    ranks = sft_data.get("ranks", [])
    if sft_data.get("version") != 1 or len(ranks) != 1:
        raise ValueError("Expected single-rank SFT data state at the source checkpoint")
    data = ranks[0]
    optimizer_steps = data.get("optimizer_steps")
    manifest_digest = data.get("manifest_sha256")
    if (data.get("rank") != 0 or data.get("world_size") != 1
            or type(optimizer_steps) is not int or optimizer_steps < 1
            or not isinstance(manifest_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_digest)):
        raise ValueError("Source must identify a completed SFT update and its data manifest")

    model_path = Path(model_cfg["model_path"]).resolve()
    if Path(source_cfg["model_path"]).resolve() != model_path:
        raise ValueError("SFT and RL must use the same pinned Pt base path")
    receipt = json.loads((model_path / "verified.json").read_text(encoding="utf-8"))
    if (receipt.get("repo_id"), receipt.get("revision")) != (PT_ID, PT_REVISION):
        raise ValueError("SFT initialization is only supported from the verified Pt base")

    layout = {"model_type": "xvla", "precision": "fp32", "is_lora": False,
              "num_action_chunks": 30, "action_dim": 7}
    domain = {"domain_rows": [3], "domain_id": 3, "steps": 10}
    source_xvla = source_cfg.get("xvla", {})
    target_xvla = model_cfg.get("xvla", {})
    scope = source_xvla.get("update_scope")
    if scope not in ("prompt_heads", "lora_heads") or target_xvla.get("update_scope") != scope:
        raise ValueError("SFT and RL must use the same prompt-head or LoRA-head update scope")
    if scope == "lora_heads":
        # Rank 8 is the default used by the extension model builder.
        source_rank = source_xvla.get("lora_rank", 8)
        target_rank = target_xvla.get("lora_rank", 8)
        if (type(source_rank) is not int or type(target_rank) is not int
                or source_rank < 1 or source_rank != target_rank):
            raise ValueError("SFT and RL must use the same positive X-VLA LoRA rank")
    for cfg in (source_cfg, model_cfg):
        xvla = cfg.get("xvla", {})
        if (any(cfg.get(key) != value for key, value in layout.items())
                or any(xvla.get(key) != value for key, value in domain.items())):
            raise ValueError("SFT and RL must use the same FP32 domain-3 X-VLA layout")
    if source_cfg.get("add_value_head") is not False:
        raise ValueError("SFT source must contain no value head")
    return evidence, weights, check, data


def validate_rl_initialization(cfg):
    actor_path = _initialization_path(cfg.actor.model)
    rollout_path = _initialization_path(cfg.rollout.model)
    if actor_path != rollout_path:
        raise ValueError("Actor and rollout must select the same SFT initialization checkpoint")
    require_sft = cfg.runner.get("require_sft_initialization", False)
    if cfg.runner.get("resume_dir"):
        if actor_path is not None or cfg.runner.get("ckpt_path"):
            raise ValueError("DCP resume requires initial_sft_checkpoint=null and ckpt_path=null")
        return
    if cfg.runner.get("ckpt_path") and (actor_path is not None or require_sft):
        raise ValueError("A fresh SFT-initialized RL job cannot also load runner.ckpt_path")
    if actor_path is None:
        if require_sft:
            raise ValueError("A fresh Pt RL job requires XVLA_INITIAL_SFT_CHECKPOINT")
        return
    if cfg.runner.get("only_eval", False):
        raise ValueError("SFT initialization is for training. Use the paired benchmark for SFT evaluation")
    read_sft_source(actor_path, cfg.actor.model)
    read_sft_source(rollout_path, cfg.rollout.model)


def apply_sft_state(model, state):
    """Load the SFT core while preserving the RL sampler and optional fresh critic."""
    target = model.state_dict()
    head = {key: value for key, value in target.items() if key.startswith("value_head.")}
    if bool(head) != model.add_value_head:
        raise ValueError("Value-head keys disagree with the RL model configuration")
    expected = target.keys() - head.keys()
    core = expected - {"sampler.std"}
    if "sampler.std" not in expected or not core or any(not key.startswith("core.") for key in core):
        raise ValueError("Unexpected RL policy state outside the X-VLA core, sampler and optional value head")
    if state.keys() != expected:
        raise ValueError(f"SFT core state keys differ: missing={sorted(expected - state.keys())} "
                         f"unexpected={sorted(state.keys() - expected)}")
    for key, source in state.items():
        if source.shape != target[key].shape or source.dtype != target[key].dtype:
            raise ValueError(f"SFT core state shape/dtype differs at {key}")
    model.load_state_dict({**state, **head, "sampler.std": target["sampler.std"]}, strict=True)
    return sorted(head)


def initialize_from_sft(model, model_cfg):
    checkpoint = _initialization_path(model_cfg)
    if checkpoint is None:
        return

    import torch
    from .audit_workers import state_digest

    evidence, weights, check, data = read_sft_source(checkpoint, model_cfg)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state_digest(state) != evidence["digest"]:
        raise RuntimeError("SFT full-state export does not match the audited digest")
    sampler_std = model.sampler.std.detach().clone()
    fresh_head = apply_sft_state(model, state)
    source_core = {key: value for key, value in state.items() if key.startswith("core.")}
    restored = {key: value for key, value in model.state_dict().items() if key in source_core}
    source_core_digest = state_digest(source_core)
    restored_digest = state_digest(restored)
    if restored_digest != source_core_digest:
        raise RuntimeError("RL model core differs from the audited SFT state after load")
    if not torch.equal(model.sampler.std, sampler_std):
        raise RuntimeError("SFT initialization modified the configured RL sampler")
    record = {"checkpoint": str(checkpoint), "base_repo_id": PT_ID, "base_revision": PT_REVISION,
              "source_sft_job_id": check.get("identity", {}).get("slurm_job_id"),
              "source_sft_optimizer_steps": data["optimizer_steps"],
              "source_sft_manifest_sha256": data["manifest_sha256"],
              "source_state_sha256": evidence["digest"], "source_core_sha256": source_core_digest,
              "restored_core_sha256": restored_digest, "restored_core_tensors": len(restored),
              "source_sft_sampler_std": state["sampler.std"].tolist(),
              "rl_sampler_std": sampler_std.cpu().tolist(), "sampler_disposition": "retained_rl_config",
              "fresh_value_head_keys": fresh_head,
              "update_scope": model_cfg["xvla"]["update_scope"],
              "job_id": os.environ.get("SLURM_JOB_ID"), "pid": os.getpid()}
    if record["update_scope"] == "lora_heads":
        record["lora_rank"] = model_cfg["xvla"].get("lora_rank", 8)
    print("XVLA_RL_INITIALIZATION " + json.dumps(record), flush=True)
