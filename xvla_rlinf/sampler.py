"""Auditable stochastic policies around X-VLA's clean-action predictor.

``terminal_clean_gaussian_v1`` is deliberately *not* the native X-VLA action
marginal.  Given a full native noise tensor z, it samples a 10-D single-arm
raw-action prefix from a diagonal Normal centred on the differentiable native
clean-action output.  See docs/xvla_rl_sampler.md for the probability space.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


SAMPLER_NAME = "terminal_clean_gaussian_v1"
ACTIVE_ACTION_DIM = 10
MODEL_ACTION_DIM = 20


def _check_native_inputs(
    core: nn.Module, model_inputs: Mapping[str, Tensor], initial_noise: Tensor
) -> tuple[int, int]:
    required = {"input_ids", "image_input", "image_mask", "domain_id", "proprio"}
    missing = required.difference(model_inputs)
    if missing:
        raise KeyError(f"Missing native X-VLA inputs: {sorted(missing)}")
    proprio = model_inputs["proprio"]
    if proprio.ndim != 2 or proprio.shape[-1] != MODEL_ACTION_DIM:
        raise ValueError("ee6d X-VLA requires proprio with shape [B,20]")
    batch = proprio.shape[0]
    horizon = int(core.num_actions)
    if int(core.action_space.dim_action) != MODEL_ACTION_DIM:
        raise ValueError("terminal_clean_gaussian_v1 supports ee6d/20-D X-VLA only")
    if initial_noise.shape != (batch, horizon, MODEL_ACTION_DIM):
        raise ValueError(
            f"initial_noise must retain the complete [B,H,20] tensor; "
            f"expected {(batch, horizon, MODEL_ACTION_DIM)}, got {tuple(initial_noise.shape)}"
        )
    if initial_noise.device != proprio.device or initial_noise.dtype != proprio.dtype:
        raise ValueError("initial_noise device/dtype must match proprio")
    if initial_noise.requires_grad:
        raise ValueError("Initial native noise is a fixed auxiliary variable, not a parameter")
    for key in required:
        if model_inputs[key].shape[0] != batch:
            raise ValueError(f"Input {key} has inconsistent batch size")
    return batch, horizon


def clean_action_prediction(
    core: nn.Module,
    model_inputs: Mapping[str, Tensor],
    initial_noise: Tensor,
    *,
    steps: int = 10,
    checkpoint_transformer: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Run the fixed-source native clean-action recurrence with autograd.

    This reproduces the arithmetic *before* action_space.postprocess.  It is not
    a velocity integrator and intentionally does not call the official
    ``@torch.no_grad() generate_actions``.  eval() disables dropout/DropPath,
    without disabling gradients.  Both rollout and recompute must use the same
    parameter dtype and autocast policy supplied by their caller.

    ``enc`` is returned for an observation-conditioned value head.  It must not
    be cached in trajectories in place of the actual observation: actor
    recompute must re-encode with the current parameters.
    """
    _check_native_inputs(core, model_inputs, initial_noise)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    core.eval()
    enc = core.forward_vlm(
        model_inputs["input_ids"],
        model_inputs["image_input"],
        model_inputs["image_mask"],
    )
    proprio = model_inputs["proprio"]
    predictor = core.transformer
    if checkpoint_transformer and torch.is_grad_enabled():
        from functools import partial

        from torch.utils.checkpoint import checkpoint

        predictor = partial(checkpoint, predictor, use_reentrant=False)
    action = torch.zeros_like(initial_noise)
    for i in range(steps, 0, -1):
        t = torch.full(
            (proprio.shape[0],),
            i / steps,
            device=proprio.device,
            dtype=proprio.dtype,
        )
        x_t = initial_noise * t[:, None, None] + action * (1 - t[:, None, None])
        proprio_m, x_t_m = core.action_space.preprocess(proprio, x_t)
        action = predictor(
            domain_id=model_inputs["domain_id"],
            action_with_noise=x_t_m,
            proprio=proprio_m,
            t=t,
            **enc,
        )
    if action.shape != initial_noise.shape:
        raise ValueError("X-VLA predictor changed the native action tensor shape")
    return action, enc


def log_prob_per_action(raw_actions: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Normal log density, SUM over all ten raw single-arm coordinates.

    The final coordinate is the gripper *logit*, not its sigmoid or discrete
    environment command.  The returned [B,C,1] factors permit the RLinf action
    mask to be applied before chunk-level summation.
    """
    if raw_actions.ndim != 3 or raw_actions.shape[-1] != ACTIVE_ACTION_DIM:
        raise ValueError("raw_actions must have shape [B,C,10], never env-action [B,C,7]")
    if mean.shape != raw_actions.shape:
        raise ValueError("mean and sampled raw actions must have identical shapes")
    if std.shape != (ACTIVE_ACTION_DIM,):
        raise ValueError("std must contain exactly ten raw-action standard deviations")
    std = std.to(device=mean.device, dtype=torch.float32)
    if not bool(torch.isfinite(std).all()) or not bool((std > 0).all()):
        raise ValueError("All terminal standard deviations must be finite and positive")
    delta = (raw_actions.float() - mean.float()) / std
    return (-0.5 * (delta.square() + math.log(2 * math.pi)) - std.log()).sum(
        dim=-1, keepdim=True
    )


def strict_joint_logprob(per_action_logprobs: Tensor, executed_mask: Tensor | None = None) -> Tensor:
    """Sum executed action factors once: [B,C,1] -> [B,1].

    This is the log density of sampled terminal variables conditional on the
    full auxiliary z.  The theta-independent density p(z) cancels in likelihood
    ratios.  No averaging over action, chunk or denoising dimensions is used.
    Masking q's unexecuted suffix is valid only with the measured-state RL
    history contract; native_eval uses its separate unmodified history path.
    """
    if per_action_logprobs.ndim != 3 or per_action_logprobs.shape[-1] != 1:
        raise ValueError("Expected already-summed per-action logprobs [B,C,1]")
    if executed_mask is None:
        return per_action_logprobs.float().sum(dim=1)
    if executed_mask.dtype != torch.bool:
        raise ValueError("executed_mask must be boolean")
    if executed_mask.shape == per_action_logprobs.shape[:2]:
        executed_mask = executed_mask.unsqueeze(-1)
    if executed_mask.shape != per_action_logprobs.shape:
        raise ValueError("executed_mask must have shape [B,C] or [B,C,1]")
    masked = torch.where(executed_mask, per_action_logprobs.float(), 0.0)
    return masked.sum(dim=1)


class TerminalCleanGaussianSampler(nn.Module):
    """Fixed-diagonal terminal exploration around a native clean-action mean.

    Every coordinate of z in [B,H,20] remains in the cache, including padded
    model dimensions and the prediction suffix.  The terminal Normal acts only
    on the 10 raw coordinates of the executing single arm.  RL history MUST
    use measured state, so unexecuted terminal q cannot affect future behavior.
    The fixed sigma buffer has no noise-head parameters or entropy gradient.
    """

    name = SAMPLER_NAME

    def __init__(
        self,
        *,
        pos_std: float = 0.003,
        rotation_std: float = 0.01,
        gripper_logit_std: float = 0.15,
        history_mode: str = "measured_state",
    ):
        super().__init__()
        if history_mode != "measured_state":
            raise ValueError(
                "terminal_clean_gaussian_v1 requires measured_state RL history; "
                "native autoregressive history belongs to native_eval"
            )
        values = [pos_std] * 3 + [rotation_std] * 6 + [gripper_logit_std]
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Terminal exploration standard deviations must be finite and >0")
        self.register_buffer("std", torch.tensor(values, dtype=torch.float32))
        self.history_mode = history_mode

    def _entropy(self, mean: Tensor) -> Tensor:
        std = self.std.to(device=mean.device, dtype=torch.float32)
        per_action = (std.log() + 0.5 * math.log(2 * math.pi * math.e)).sum()
        return per_action.expand(mean.shape[0], mean.shape[1], 1)

    @torch.no_grad()
    def sample(
        self,
        core: nn.Module,
        model_inputs: Mapping[str, Tensor],
        *,
        initial_noise: Tensor | None = None,
        terminal_noise: Tensor | None = None,
        execution_steps: int | None = None,
        steps: int = 10,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        proprio = model_inputs["proprio"]
        batch, horizon = proprio.shape[0], int(core.num_actions)
        if execution_steps is None:
            execution_steps = horizon
        if not isinstance(execution_steps, int) or not 1 <= execution_steps <= horizon:
            raise ValueError("execution_steps must be in [1, model prediction horizon]")
        if initial_noise is None:
            initial_noise = torch.randn(
                (batch, horizon, MODEL_ACTION_DIM),
                device=proprio.device,
                dtype=proprio.dtype,
                generator=generator,
            )
        raw20, enc = clean_action_prediction(core, model_inputs, initial_noise, steps=steps)
        mean = raw20[:, :execution_steps, :ACTIVE_ACTION_DIM].float()
        if terminal_noise is None:
            terminal_noise = torch.randn(
                mean.shape, device=mean.device, dtype=torch.float32, generator=generator
            )
        if terminal_noise.shape != mean.shape:
            raise ValueError("terminal_noise must match the [B,C,10] executed prefix")
        terminal_noise = terminal_noise.to(device=mean.device, dtype=torch.float32)
        raw_actions = mean + terminal_noise * self.std.to(mean.device)
        return {
            "raw_actions": raw_actions,
            "initial_noise": initial_noise.detach().clone(),
            "mean": mean,
            "native_raw_actions": raw20,
            "logprobs": log_prob_per_action(raw_actions, mean, self.std),
            "entropy": self._entropy(mean),
            # Immediate observation features for rollout value prediction only.
            # The caller must cache raw inputs, NOT these features, for actor.
            "enc_features": enc,
        }

    def recompute(
        self,
        core: nn.Module,
        model_inputs: Mapping[str, Tensor],
        initial_noise: Tensor,
        raw_actions: Tensor,
        *,
        steps: int = 10,
        checkpoint_transformer: bool = False,
    ) -> dict[str, Any]:
        if raw_actions.ndim != 3 or raw_actions.shape[-1] != ACTIVE_ACTION_DIM:
            raise ValueError("Stored raw actions must have shape [B,C,10]")
        if raw_actions.requires_grad:
            raise ValueError("Stored sampled actions must be detached from their sampling graph")
        raw20, enc = clean_action_prediction(
            core, model_inputs, initial_noise, steps=steps,
            checkpoint_transformer=checkpoint_transformer,
        )
        if raw_actions.shape[0] != raw20.shape[0] or not 1 <= raw_actions.shape[1] <= raw20.shape[1]:
            raise ValueError("Stored action prefix does not match batch or native horizon")
        mean = raw20[:, :raw_actions.shape[1], :ACTIVE_ACTION_DIM].float()
        return {
            "logprobs": log_prob_per_action(raw_actions, mean, self.std),
            "entropy": self._entropy(mean),
            "mean": mean,
            "native_raw_actions": raw20,
            "enc_features": enc,
        }
