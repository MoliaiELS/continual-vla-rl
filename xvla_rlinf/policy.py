"""Differentiable X-VLA policy; native evaluation and RL are distinct modes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

from .parameters import configure_trainable
from .sampler import TerminalCleanGaussianSampler, clean_action_prediction
from .transforms import NativeProprioHistory, env_actions_from_model, prepare_processor_inputs

INPUT_KEYS = ('input_ids','image_input','image_mask','domain_id','proprio')

class XVLAPolicy(nn.Module, BasePolicy):
    def __init__(self, core, processor, cfg):
        super().__init__()
        self.cfg = cfg
        self.xcfg = cfg.get('xvla', {})
        self.core, self.trainable_manifest = configure_trainable(
            core, mode=self.xcfg.get('update_scope','prompt_heads'),
            lora_rank=int(self.xcfg.get('lora_rank',8)),
            domain_rows=self.xcfg.get('domain_rows',None))
        self.processor = processor
        self.action_dim = 7
        self.model_action_dim = 20
        self.num_action_chunks = int(cfg.get('num_action_chunks',core.num_actions))
        if not 1 <= self.num_action_chunks <= core.num_actions:
            raise ValueError('Execution chunk must be a prefix of the checkpoint prediction horizon')
        self.steps = int(self.xcfg.get('steps',10))
        self.domain_id = int(self.xcfg.get('domain_id',3))
        self.history = NativeProprioHistory(mode='native_history')
        self._history_batch_size = None
        self.global_step = 0
        self.sampler = TerminalCleanGaussianSampler(
            pos_std=float(self.xcfg.get('position_std',.003)),
            rotation_std=float(self.xcfg.get('rotation_std',.01)),
            gripper_logit_std=float(self.xcfg.get('gripper_logit_std',.15)))
        self.add_value_head = bool(cfg.get('add_value_head',False))
        if self.add_value_head:
            feature_dim=int(core.config.florence_config.projection_dim)
            self.value_head=nn.Sequential(nn.Linear(feature_dim+20,256),nn.Tanh(),nn.Linear(256,1))
        self._no_split_modules = ['SoftPromptedTransformer']
        print('XVLA_TRAINABLE '+json.dumps(self.trainable_manifest),flush=True)

    @classmethod
    def from_checkpoint(cls,cfg,torch_dtype):
        from .vendor.xvla.modeling_xvla import XVLA
        from .vendor.xvla.processing_xvla import XVLAProcessor
        folder=Path(cfg.model_path)
        receipt=folder/'verified.json'
        if not receipt.is_file():
            raise FileNotFoundError('Fixed checkpoint must be downloaded and verified: '+str(folder))
        provenance=json.loads(receipt.read_text())
        allowed={
          ('2toINF/X-VLA-Libero','129e71460678b7236cee6fc9707f09d9fa0c3590'),
          ('2toINF/X-VLA-Pt','c1c4a64a7e03ac5b95c468bf1578f3d03651b53b')}
        if (provenance['repo_id'],provenance['revision']) not in allowed:
            raise ValueError('Checkpoint identity is not an approved fixed X-VLA snapshot')
        core=XVLA.from_pretrained(str(folder),torch_dtype=torch_dtype,
                                 trust_remote_code=False,local_files_only=True)
        processor=XVLAProcessor.from_pretrained(str(folder),trust_remote_code=False,local_files_only=True)
        return cls(core,processor,cfg)

    def set_global_step(self,step): self.global_step=int(step)

    def _device_inputs(self,inputs):
        parameter=next(self.core.parameters())
        result={}
        for key in INPUT_KEYS:
            value=inputs[key]
            value=value if torch.is_tensor(value) else torch.as_tensor(value)
            dtype=parameter.dtype if value.is_floating_point() else value.dtype
            result[key]=value.to(device=parameter.device,dtype=dtype)
        return result

    def _inputs(self,obs,native_history):
        processed=prepare_processor_inputs(self.processor,obs,
                    image_orientation=self.xcfg.get('image_orientation','native_raw'))
        states=obs['states']
        if native_history:
            batch=len(states)
            if self._history_batch_size not in (None,batch):
                raise ValueError('Native history requires fixed batch or explicit stable env_ids')
            self._history_batch_size=batch
            proprio=self.history.prepare(states,obs.get('reset_mask'),obs.get('env_ids'))
        else:
            from .transforms import measured_proprio_from_states
            proprio=measured_proprio_from_states(states)
        processed['proprio']=torch.as_tensor(proprio,dtype=torch.float32)
        processed['domain_id']=torch.full((len(proprio),),self.domain_id,dtype=torch.long)
        return self._device_inputs(processed)

    def _values(self,enc,inputs,enabled=True):
        batch=inputs['proprio'].shape[0]
        if not self.add_value_head or not enabled:
            return torch.zeros((batch,1),device=inputs['proprio'].device,dtype=torch.float32)
        # Observation-only features: no sampled action/noise enters the baseline.
        features=torch.cat((enc['vlm_features'].float().mean(dim=1),inputs['proprio'].float()),dim=-1)
        return self.value_head(features).float()

    def _actions(self,raw):
        # Vendor postprocess is in-place. NEVER sigmoid the stored Gaussian q.
        native=raw.detach().clone()
        if native.shape[-1]==10:
            native=torch.cat((native,torch.zeros_like(native)),dim=-1)
        post=self.core.action_space.postprocess(native)
        return env_actions_from_model(post,rotation_backend=self.xcfg.get('rotation_backend','robosuite'))

    @torch.no_grad()
    def predict_action_batch(self,env_obs,mode=None,compute_values=True,**kwargs):
        if mode not in ('train','eval'):
            raise ValueError('X-VLA requires explicit mode from patched rollout worker')
        eval_sampler=self.xcfg.get('eval_sampler','native_eval')
        native=mode=='eval' and eval_sampler=='native_eval'
        if mode=='eval' and eval_sampler not in ('native_eval',self.sampler.name):
            raise NotImplementedError('Unknown X-VLA evaluation sampler: '+str(eval_sampler))
        inputs=self._inputs(env_obs,native_history=native)
        noise=kwargs.get('initial_noise')
        if native:
            if noise is None:
                noise=torch.randn((len(inputs['proprio']),self.core.num_actions,20),
                                  device=inputs['proprio'].device,dtype=inputs['proprio'].dtype)
            raw,enc=clean_action_prediction(self.core,inputs,noise,steps=self.steps)
            self.history.update(raw,env_obs.get('env_ids'))
            actions=self._actions(raw)[:,:self.num_action_chunks]
            return actions,{'prev_logprobs':None,'prev_values':self._values(enc,inputs,compute_values),
                'forward_inputs':{k:v.detach().clone() for k,v in inputs.items()}}
        sampled=self.sampler.sample(self.core,inputs,initial_noise=noise,
                    terminal_noise=kwargs.get('terminal_noise'),execution_steps=self.num_action_chunks,steps=self.steps)
        raw=sampled['raw_actions']
        # Only first ten raw single-arm coordinates are stochastic terminal q.
        # Padding arm coordinates are not fabricated as seven-dimensional actions.
        actions=self._actions(raw)
        cache={k:v.detach().clone() for k,v in inputs.items()}
        cache.update(initial_noise=sampled['initial_noise'],raw_actions=raw.detach().clone(),
                     action=torch.as_tensor(actions).reshape(len(raw),-1),model_action=raw.detach().reshape(len(raw),-1))
        return actions,{'prev_logprobs':sampled['logprobs'].float(),
                       'prev_values':self._values(sampled['enc_features'],inputs,compute_values),
                       'forward_inputs':cache}

    def forward(self,forward_type=ForwardType.DEFAULT,**kwargs):
        if forward_type==ForwardType.DEFAULT: return self.default_forward(**kwargs)
        if forward_type==ForwardType.SFT: return self.sft_forward(**kwargs)
        raise NotImplementedError('X-VLA supports only SFT and on-policy PPO/GRPO')

    def default_forward(self,forward_inputs,compute_logprobs=True,compute_values=False,compute_entropy=False,**kwargs):
        if not compute_logprobs:
            raise NotImplementedError('On-policy recompute requires log probabilities')
        inputs=self._device_inputs(forward_inputs)
        stats=self.sampler.recompute(self.core,inputs,forward_inputs['initial_noise'],
                                     forward_inputs['raw_actions'],steps=self.steps)
        logprobs=stats['logprobs'].float()
        mask=forward_inputs.get('executed_mask')
        if mask is not None:
            if mask.shape!=logprobs.shape[:2]:raise ValueError('Executed mask shape differs from action prefix')
            logprobs=torch.where(mask[...,None].bool(),logprobs,0.)
        return {'logprobs':logprobs,'values':self._values(stats['enc_features'],inputs,compute_values).squeeze(-1),
                'entropy':stats['entropy'].float() if compute_entropy else None}

    def sft_forward(self,data,**kwargs):
        # Official clean-action supervision and loss weights, NOT velocity loss.
        self.core.train()
        inputs=self._device_inputs(data)
        actions=data['action'].to(inputs['proprio'].device,dtype=inputs['proprio'].dtype)
        losses=self.core(**inputs,action=actions)
        return {'loss':sum(losses.values()),**losses}

    def gradient_checkpointing_enable(self,gradient_checkpointing_kwargs=None,**kwargs):
        self.core.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,**kwargs)

    def gradient_checkpointing_disable(self): self.core.gradient_checkpointing_disable()
