"""Warm-start audited X-VLA LoRA SFT with the trained Pt domain rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .parameters import HEADS


def _manifest_relationship(source_digest, target_path, target_digest):
    if source_digest == target_digest:
        return "same_manifest"
    payload = Path(target_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != target_digest:
        raise RuntimeError("Stage-two SFT manifest changed after loader verification")
    if json.loads(payload).get("included_manifest_sha256") != source_digest:
        raise ValueError("Stage two requires the same SFT manifest or its recorded nested extension")
    return "included_manifest"


def _copy_prompt_rows(model, source_state):
    import torch

    base = model.core.get_base_model()
    pairs = []
    for head_name in HEADS:
        head = base.get_submodule(head_name).modules_to_save["default"]
        for name, parameter in head.named_parameters():
            prefix, _, leaf = name.rpartition(".")
            key = f"core.{head_name}.{name}"
            if not parameter.requires_grad or leaf != "selected_weight":
                raise ValueError(f"Unexpected non-row parameter in active LoRA head: {key}")
            row = head.get_submodule(prefix)
            source_prefix = key.removesuffix("selected_weight")
            source = source_state[key]
            if (source.shape != parameter.shape or source.dtype != parameter.dtype
                    or row.row_indices.tolist() != [3]
                    or not torch.equal(source_state[source_prefix + "row_indices"], row.row_indices.cpu())
                    or not torch.equal(source_state[source_prefix + "frozen_weight"], row.frozen_weight.cpu())):
                raise ValueError(f"Prompt checkpoint row or frozen head base differs: {key}")
            if not torch.isfinite(source).all():
                raise ValueError(f"Prompt checkpoint row is non-finite: {key}")
            pairs.append((key, parameter, source))
    if len(pairs) != 5:
        raise RuntimeError("Expected the five domain-3 prompt/action head tensors")
    with torch.no_grad():
        for _, parameter, source in pairs:
            parameter.copy_(source)
    return [key for key, _, _ in pairs]


def initialize_lora_from_prompt(model, model_cfg, checkpoint, *, manifest_path, manifest_sha256):
    import torch

    from .audit_workers import state_digest
    from .rl_initialization import read_sft_source

    scope = dict(model_cfg.get("xvla", {}))
    rank = scope.get("lora_rank", 8)
    if (scope.get("update_scope") != "lora_heads" or type(rank) is not int or rank != 8
            or model_cfg.get("add_value_head") is not False):
        raise ValueError("Prompt warm-start requires X-VLA LoRA plus heads, rank 8, and no value head")
    # Reuse the independent audit, fixed Pt, domain/layout and optimizer-lineage checks.
    expected_prompt = {**model_cfg, "xvla": {**scope, "update_scope": "prompt_heads"}}
    checkpoint = Path(checkpoint).resolve()
    evidence, weights, check, data = read_sft_source(checkpoint, expected_prompt)
    relationship = _manifest_relationship(data["manifest_sha256"], manifest_path, manifest_sha256)
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or state_digest(state) != evidence["digest"]:
        raise RuntimeError("Prompt SFT export changed since its independent audit")
    copied = _copy_prompt_rows(model, state)
    return {"source_checkpoint": str(checkpoint),
            "source_sft_job_id": check.get("identity", {}).get("slurm_job_id"),
            "source_optimizer_steps": data["optimizer_steps"], "source_state_sha256": evidence["digest"],
            "source_manifest_sha256": data["manifest_sha256"], "manifest_sha256": manifest_sha256,
            "manifest_relationship": relationship, "copied_trainable_tensors": copied,
            "lora_rank": rank, "lora_alpha": 2 * rank}
