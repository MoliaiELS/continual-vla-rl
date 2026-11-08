"""Explicit X-VLA trainable scopes, including domain-aware embedding heads.

Torch/PEFT are imported only when an operation actually needs them. Row-only
training uses separate Parameters, not a gradient mask vulnerable to AdamW.
"""
from __future__ import annotations

from typing import Any, Sequence

HEADS = ("transformer.soft_prompt_hub", "transformer.action_encoder", "transformer.action_decoder")


def trainable_report(model: Any) -> dict[str, Any]:
    parameters = []
    total = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            parameters.append({"name": name, "shape": list(parameter.shape),
                               "numel": count, "dtype": str(parameter.dtype)})
    trainable = sum(p["numel"] for p in parameters)
    return {"total_parameters": total, "trainable_parameters": trainable,
            "trainable_fraction": trainable / total if total else 0,
            "parameters": parameters}


def gradient_report(model: Any) -> dict[str, Any]:
    import torch
    records = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        grad = parameter.grad
        records.append({"name": name, "has_gradient": grad is not None,
                        "finite": bool(torch.isfinite(grad).all()) if grad is not None else None,
                        "nonzero": bool(torch.count_nonzero(grad)) if grad is not None else False,
                        "grad_norm": float(grad.float().norm()) if grad is not None else 0.0})
    return {"trainable_tensors": len(records),
            "tensors_with_gradient": sum(p["has_gradient"] for p in records),
            "tensors_with_nonzero_gradient": sum(p["nonzero"] for p in records),
            "all_finite": all(p["finite"] is not False for p in records), "parameters": records}


def _domain_rows_embedding(embedding: Any, rows: Sequence[int]):
    """A frozen full embedding plus a physically separate selected-row weight.

    Frozen rows never enter the optimizer, so both decoupled weight decay and
    momentum leave them bit-identical. Recreate this structure before loading a
    checkpoint; ``domain_rows`` is part of the model configuration.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    if embedding.max_norm is not None or embedding.sparse:
        raise NotImplementedError("Domain-row restriction supports dense non-renormalizing embeddings only")
    indices = sorted(set(int(r) for r in rows))
    if not indices or indices[0] < 0 or indices[-1] >= embedding.num_embeddings:
        raise ValueError("domain_rows must contain valid embedding row indices")

    class DomainRowsEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_embeddings = embedding.num_embeddings
            self.embedding_dim = embedding.embedding_dim
            self.padding_idx = embedding.padding_idx
            self.register_buffer("frozen_weight", embedding.weight.detach().clone())
            self.register_buffer("row_indices", torch.tensor(indices, device=embedding.weight.device, dtype=torch.long))
            self.selected_weight = nn.Parameter(self.frozen_weight[self.row_indices].clone())

        @property
        def weight(self):
            return self.frozen_weight.index_copy(0, self.row_indices, self.selected_weight)

        def forward(self, inputs):
            return functional.embedding(inputs, self.weight, padding_idx=self.padding_idx)

    return DomainRowsEmbedding()


def _restrict_domain_rows(model: Any, rows: Sequence[int]) -> list[str]:
    import torch.nn as nn
    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Embedding) or not module.weight.requires_grad:
            continue
        # A PEFT modules_to_save wrapper also retains a frozen original module;
        # only its active, trainable copy is restricted here.
        parent_name = name.rpartition(".")[0]
        parent = model.get_submodule(parent_name) if parent_name else model
        if any(head in name for head in HEADS) or type(parent).__name__ == "DomainAwareLinear":
            candidates.append((name, module))
    if not candidates:
        raise ValueError("No trainable domain embeddings found for domain_rows")
    replaced = []
    for name, embedding in candidates:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, _domain_rows_embedding(embedding, rows))
        replaced.append(name)
    return replaced


def configure_trainable(model: Any, mode: str, *, lora_rank: int = 8,
                        lora_alpha: int | None = None,
                        domain_rows: Sequence[int] | None = None) -> tuple[Any, dict[str, Any]]:
    """Return ``(possibly_peft_wrapped_model, exact_parameter_report)``.

    full: all model parameters; prompt_heads: official prompt/action subtrees;
    lora_heads: official all-linear LoRA plus modules_to_save for all three
    subtrees. DomainAwareLinear uses Embedding and is explicitly included.
    """
    normalized = mode.lower().replace("-", "_").replace(" ", "")
    aliases = {"full": "full", "lora_heads": "lora_heads",
               "lora+heads": "lora_heads", "prompt_heads": "prompt_heads",
               "prompt/actionheadsonly": "prompt_heads"}
    if normalized not in aliases:
        raise ValueError(f"Unsupported trainable scope: {mode}")
    mode = aliases[normalized]
    if bool(getattr(model, "peft_config", None)):
        raise ValueError("configure_trainable must run once on the unwrapped X-VLA model; avoid double LoRA injection")
    for name in HEADS:
        try:
            model.get_submodule(name)
        except AttributeError as error:
            raise ValueError(f"X-VLA parameter subtree missing: {name}") from error

    for parameter in model.parameters():
        parameter.requires_grad_(mode == "full")
    if mode == "prompt_heads":
        for name in HEADS:
            for parameter in model.get_submodule(name).parameters():
                parameter.requires_grad_(True)
    elif mode == "lora_heads":
        from peft import LoraConfig, get_peft_model
        if lora_rank < 1:
            raise ValueError("lora_rank must be positive")
        model = get_peft_model(model, LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha if lora_alpha is not None else 2 * lora_rank,
            bias="none", target_modules="all-linear", modules_to_save=list(HEADS),
        ))
    restricted = _restrict_domain_rows(model, domain_rows) if domain_rows is not None else []
    report = trainable_report(model)
    if not report["trainable_parameters"]:
        raise RuntimeError("Parameter selection produced no trainable parameters")
    report.update(mode=mode, domain_rows=list(domain_rows) if domain_rows is not None else None,
                  domain_row_modules=restricted, lora_rank=lora_rank if mode == "lora_heads" else None,
                  row_protection="separate_parameter_and_frozen_buffer" if restricted else "not_requested")
    return model, report
