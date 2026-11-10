"""Optional evidence around unchanged RLinf training, sync and checkpoint calls.

The smoke launcher uses single-rank no_shard. These classes delegate the entire
algorithm to RLinf; they do not introduce another optimization loop.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from rlinf.runners.embodied_runner import EmbodiedRunner


def record(event, **payload):
    value = dict(event=event, time=time.time(), pid=os.getpid(), **payload)
    print('XVLA_AUDIT '+json.dumps(value), flush=True)
    folder = Path(os.environ['XVLA_RUN_DIR'])
    with (folder / ('audit-'+str(os.getpid())+'.jsonl')).open('a') as stream:
        stream.write(json.dumps(value)+'\n')
    return value


def state_digest(state):
    """Hash every parameter and buffer, one CPU tensor at a time."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not torch.is_tensor(tensor):
            raise TypeError('Smoke state audit requires an ordinary no_shard state dict')
        name = name.replace('_fsdp_wrapped_module.', '')
        digest.update(name.encode())
        tensor = tensor.detach().contiguous().cpu()
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def unwrapped(model):
    while hasattr(model, 'module'):
        model = model.module
    return model


@torch.no_grad()
def fixed_probe(policy):
    """Same observation/z/q in actor and rollout; no history is mutated."""
    from .sampler import clean_action_prediction
    state = np.zeros((1,20), dtype=np.float32)
    state[:,[3,7]] = 1
    obs = dict(main_images=np.full((1,256,256,3), 80, np.uint8),
               wrist_images=np.full((1,256,256,3), 140, np.uint8),
               states=state, task_descriptions=['put the black bowl on the plate'])
    inputs = policy._inputs(obs, native_history=False)
    generator = torch.Generator(device='cpu').manual_seed(717)
    noise = torch.randn((1,policy.core.num_actions,20), generator=generator).to(inputs['proprio'])
    raw, enc = clean_action_prediction(policy.core, inputs, noise, steps=policy.steps)
    return dict(raw=raw.float().cpu().tolist(), actions=policy._actions(raw).tolist(),
                value=policy._values(enc,inputs).cpu().tolist())


class AuditedActor(EmbodiedFSDPActor):
    def optimizer_step(self):
        names = {name: p for name,p in self.model.named_parameters() if p.requires_grad}
        before = {name:p.detach().clone() for name,p in names.items()}
        gradients = {name:float(p.grad.float().norm()) if p.grad is not None else None
                     for name,p in names.items()}
        # The observation-only critic never calls the action transformer, and
        # fixed terminal sigma has no entropy mean gradient. These gradients
        # isolate the actor path even if a full/VLM scope is used later.
        actor_norms = {name:norm for name,norm in gradients.items()
                       if 'core.' in name and 'transformer.' in name}
        result = super().optimizer_step()
        policy=unwrapped(self.model)
        if hasattr(policy,'_audit_initial_optimizer_pending'):
            policy._audit_initial_optimizer_pending=False
        changes = {name:float((p.detach()-before[name]).abs().max()) for name,p in names.items()}
        record('optimizer', rank=self._rank, step=self.optimizer_steps,
               gradients=gradients, actor_transformer_gradient_norms=actor_norms,
               parameter_max_changes=changes, peak_cuda_allocated=torch.cuda.max_memory_allocated(),
               critic_warmup_steps=self.critic_warmup_steps)
        if self.cfg.algorithm.get("xvla_require_actor_update", False):
            if any(not math.isfinite(value) for value in (*gradients.values(), *changes.values())
                   if value is not None):
                raise RuntimeError("Non-finite gradient or parameter update")
            if not any(norm is not None and norm > 0 and changes[name] > 0
                       for name, norm in actor_norms.items()):
                raise RuntimeError("No nonzero action-transformer gradient and parameter update")
        return result

    def run_training(self):
        batch = self.rollout_batch
        stats = {}
        for key in ('rewards','advantages','returns','executed_mask','dones','terminations','truncations'):
            value = batch.get(key)
            if torch.is_tensor(value):
                value = value.detach().float()
                stats[key] = dict(shape=list(value.shape), min=float(value.min()),
                                  max=float(value.max()), sum=float(value.sum()),
                                  std=float(value.std(unbiased=False)))
        record('training_batch', rank=self._rank, stats=stats)
        if self.cfg.algorithm.get("xvla_require_positive_reward", False):
            rewards = batch["rewards"]
            if not bool(torch.isfinite(rewards).all()) or not bool((rewards > 0).any()):
                raise RuntimeError("NO_POSITIVE_PPO_REWARD: no successful rollout for the Pt-SFT PPO update")
        if self.cfg.algorithm.adv_type=='grpo':
            rewards=batch['rewards'].float().sum(dim=(0,2)).reshape(-1,self.cfg.algorithm.group_size)
            variances=rewards.var(dim=1,unbiased=False)
            record('grpo_groups',returns=rewards.tolist(),variance=variances.tolist(),
                   all_failure_fraction=float((rewards==0).all(dim=1).float().mean()),
                   all_success_fraction=float((rewards>0).all(dim=1).float().mean()),
                   informative_fraction=float((variances>0).float().mean()))
            if not bool((variances>0).any()):
                raise RuntimeError('NO_INFORMATIVE_GRPO_GROUP: real rewards have zero within-group variance; no actor update is claimed')
        started = time.monotonic()
        metrics = super().run_training()
        record('training_finished', rank=self._rank, seconds=time.monotonic()-started,
               peak_cuda_allocated=torch.cuda.max_memory_allocated(), metrics=metrics)
        return metrics

    def audit_sync_probe(self):
        if self._world_size != 1:
            raise RuntimeError('Full-state smoke evidence currently requires one no_shard actor rank')
        return dict(digest=state_digest(self.get_model_state_dict(cpu_offload=True, full_state_dict=True)),
                    probe=fixed_probe(unwrapped(self.model)))

    def save_checkpoint(self, save_path, step=0):
        before = self.audit_sync_probe()
        super().save_checkpoint(save_path, step)
        # Prove load restores data, rather than comparing an unchanged object.
        parameter = next(p for p in self.model.parameters() if p.requires_grad)
        with torch.no_grad():
            parameter.view(-1)[0].add_(1.)
        super().load_checkpoint(save_path)
        after = self.audit_sync_probe()
        if before != after:
            raise RuntimeError('Actual RLinf checkpoint did not restore full model/probe exactly')
        policy=unwrapped(self.model)
        if hasattr(policy,'cfg'):
            from omegaconf import OmegaConf
            evidence=dict(model_config=OmegaConf.to_container(policy.cfg,resolve=True),**before)
            (Path(save_path)/'xvla_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
        record('checkpoint_reload', status='PASS', path=save_path, step=step,
               digest=after['digest'], corruption_restored=True)


class AuditedRollout(MultiStepRolloutWorker):
    def audit_sync_probe(self):
        if self.enable_offload:
            raise RuntimeError('Smoke sync evidence expects rollout offload disabled')
        return dict(digest=state_digest(self.hf_model.state_dict()), probe=fixed_probe(self.hf_model))


class AuditedRunner(EmbodiedRunner):
    def update_rollout_weights(self):
        started = time.monotonic()
        super().update_rollout_weights()
        actors = self.actor.audit_sync_probe().wait()
        rollouts = self.rollout.audit_sync_probe().wait()
        if len(actors) != 1 or len(rollouts) != 1:
            raise RuntimeError('Smoke sync audit is restricted to one actor/rollout rank')
        a, r = actors[0], rollouts[0]
        if a['digest'] != r['digest']:
            raise RuntimeError('Actor/rollout full state differs after RLinf weight synchronization')
        for key in ('raw','actions','value'):
            np.testing.assert_allclose(a['probe'][key],r['probe'][key],rtol=1e-5,atol=1e-6)
        record('sync', status='PASS', global_step=self.global_step,
               seconds=time.monotonic()-started, digest=a['digest'], same_noise_probe=True)
