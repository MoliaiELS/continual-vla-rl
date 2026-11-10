"""Load the audited two-stage source-10 policy for a new SFT optimizer run."""
from __future__ import annotations

from pathlib import Path

from .rl_initialization import read_sft_source


SFT_CONT_STEPS = 2048
SFT_CONT_CHECKPOINT_STEPS = (512, 1024, SFT_CONT_STEPS)


def validate_sft_cont_config(cfg):
    if "sft_cont_checkpoint" not in cfg.runner:
        return False
    path = cfg.runner["sft_cont_checkpoint"]
    if not isinstance(path, str) or not path.strip():
        raise ValueError("SFT-cont requires a 19525 actor checkpoint directory")
    if (cfg.runner.get("initial_prompt_checkpoint") is not None or cfg.runner.get("resume_dir")
            or cfg.actor.model.get("xvla", {}).get("initial_sft_checkpoint") is not None):
        raise ValueError("SFT-cont uses only the source weights and a fresh optimizer")
    if (cfg.runner.max_steps != SFT_CONT_STEPS or cfg.runner.max_epochs != -1
            or cfg.runner.save_interval != SFT_CONT_CHECKPOINT_STEPS[0]
            or cfg.runner.val_check_interval != -1 or cfg.runner.get("early_stop") is not None
            or cfg.runner.get("xvla_audit", True) is not True
            or cfg.actor.optim.total_training_steps != SFT_CONT_STEPS
            or cfg.actor.fsdp_config.grad_scaler.enabled
            or cfg.actor.optim.get("critic_warmup_steps", 0)):
        raise ValueError("SFT-cont requires its fixed 2048-update budget and checkpoint audit")
    requirements = cfg.data.get("manifest_requirements")
    if (requirements is None or requirements.get("demonstrations") != 10
            or requirements.get("selection_seed") != 1234 or requirements.get("task_id") != 0
            or requirements.get("task_suite") != "libero_spatial" or cfg.actor.global_batch_size != 4):
        raise ValueError("SFT-cont requires the original task-0 ten-demo data and global batch 4")
    return True


def read_continuation_source(checkpoint, model_cfg, manifest_sha256):
    xvla = model_cfg["xvla"]
    if (xvla.get("update_scope") != "lora_heads" or xvla.get("lora_rank") != 8
            or model_cfg.get("add_value_head") is not False):
        raise ValueError("SFT-cont requires the source rank-8 LoRA plus domain-3 heads layout")

    checkpoint = Path(checkpoint).resolve()
    evidence, weights, check, data = read_sft_source(checkpoint, model_cfg)
    if str(check.get("identity", {}).get("slurm_job_id")) != "19525":
        raise ValueError("SFT-cont requires independently audited Slurm source job 19525")
    if data["manifest_sha256"] != manifest_sha256:
        raise ValueError("SFT-cont requires the same manifest SHA256 as job 19525")

    warm_start = evidence.get("prompt_warm_start", {})
    if (not warm_start.get("source_checkpoint") or warm_start.get("lora_rank") != 8
            or warm_start.get("lora_alpha") != 16):
        raise ValueError("19525 must carry its audited prompt-stage lineage and LoRA configuration")
    prompt_path = Path(warm_start["source_checkpoint"]).resolve()
    prompt_config = {**model_cfg, "xvla": {**xvla, "update_scope": "prompt_heads"}}
    prompt_evidence, _, prompt_check, prompt_data = read_sft_source(prompt_path, prompt_config)
    prompt_job_id = str(prompt_check.get("identity", {}).get("slurm_job_id"))
    if (prompt_job_id != "19518" or str(warm_start.get("source_sft_job_id")) != prompt_job_id
            or warm_start.get("source_state_sha256") != prompt_evidence["digest"]
            or warm_start.get("source_optimizer_steps") != prompt_data["optimizer_steps"]
            or prompt_data["manifest_sha256"] != manifest_sha256
            or warm_start.get("source_manifest_sha256") != manifest_sha256
            or warm_start.get("manifest_sha256") != manifest_sha256):
        raise ValueError("19525 prompt-stage audit, data or update lineage does not match its source")

    stages = [prompt_data["optimizer_steps"], data["optimizer_steps"]]
    if stages != [1024, 1024]:
        raise ValueError(f"SFT-cont budget requires the audited [1024, 1024] source updates, found {stages}")
    return weights, {
        "source_checkpoint": str(checkpoint),
        "source_sft_job_id": "19525",
        "source_state_sha256": evidence["digest"],
        "source_manifest_sha256": manifest_sha256,
        "source_optimizer_steps_by_stage": stages,
        "source_total_optimizer_steps": sum(stages),
        "new_optimizer_step_budget": SFT_CONT_STEPS,
        "lora_rank": 8,
        "lora_alpha": 16,
    }


def initialize_sft_continuation(model, model_cfg, checkpoint, *, manifest_sha256):
    import torch

    from .audit_workers import state_digest

    weights, provenance = read_continuation_source(checkpoint, model_cfg, manifest_sha256)
    source_state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    expected = provenance["source_state_sha256"]
    if not isinstance(source_state, dict) or state_digest(source_state) != expected:
        raise RuntimeError("19525 export does not match its audited state digest")
    model.load_state_dict(source_state, strict=True)
    del source_state
    if state_digest(model.state_dict()) != expected:
        raise RuntimeError("SFT-cont model state differs after the strict source load")
    return provenance
