"""X-VLA data hook into the pinned RLinf FSDP SFT worker and SFTRunner."""
from __future__ import annotations

import json
import hashlib
import logging
import os
from pathlib import Path
import subprocess

import hydra
import rlinf
import torch.multiprocessing as multiprocessing
import torch
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.sft_runner import SFTRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

from . import register
from .sft import build_xvla_sft_dataloader


class XVLASftWorker(FSDPVlaSftWorker):
    """Only replace data loading; RLinf owns forward/backward/optimizer/save."""

    def build_dataloader(self, data_paths, eval_dataset: bool = False):
        return build_xvla_sft_dataloader(self.cfg, self._world_size, self._rank, data_paths, eval_dataset)

    def _audit_write(self, event, **fields):
        record={"event":event,"rank":self._rank,**fields}
        print('XVLA_SFT_AUDIT '+json.dumps(record),flush=True)
        directory=Path(os.environ['XVLA_RUN_DIR'])
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/f'sft_worker_rank{self._rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')

    def _model_fingerprints(self):
        # FSDP may expose temporary Tensor views through named_parameters after
        # a forward. The native full state has stable canonical keys instead.
        # Capture requires_grad once, keeping every tied-parameter alias.
        if not hasattr(self,'_xvla_trainable_state_names'):
            self._xvla_trainable_state_names={
                name.replace('_fsdp_wrapped_module.','')
                for name,parameter in self.model.named_parameters(remove_duplicate=False)
                if parameter.requires_grad}
        state=self.get_model_state_dict(cpu_offload=True,full_state_dict=True)
        canonical={}
        for name,tensor in state.items():
            key=name.replace('_fsdp_wrapped_module.','')
            if key in canonical:
                raise RuntimeError('Duplicate canonical SFT state key: '+key)
            if not torch.is_tensor(tensor):
                raise TypeError('SFT smoke audit requires ordinary tensor state: '+key)
            canonical[key]=tensor
        keys=frozenset(canonical)
        if not self._xvla_trainable_state_names.issubset(keys):
            raise RuntimeError('Initial trainable aliases absent from full SFT state: '+
                               str(sorted(self._xvla_trainable_state_names-keys)))
        if hasattr(self,'_xvla_state_keys') and keys!=self._xvla_state_keys:
            raise RuntimeError('SFT full state keys changed: missing='+
                               str(sorted(self._xvla_state_keys-keys))+' added='+
                               str(sorted(keys-self._xvla_state_keys)))
        self._xvla_state_keys=keys
        records={}
        for name,tensor in canonical.items():
            contiguous=tensor.detach().cpu().contiguous()
            digest=hashlib.sha256(contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
            records[name]={"sha256":digest,"shape":list(tensor.shape),"dtype":str(tensor.dtype),
                           "trainable":name in self._xvla_trainable_state_names}
        return records

    def init_worker(self):
        super().init_worker()
        self._xvla_audit=bool(self.cfg.runner.get('xvla_audit',True)) and self._world_size==1
        self._xvla_steps=0
        if self._xvla_audit:
            self._xvla_initial=self._model_fingerprints()
            self._audit_write('initialized',parameter_and_buffer_fingerprints=self._xvla_initial,
                              data_config=self.data_config,world_size=self._world_size)
        else:
            self._audit_write('fingerprint_probe_not_run',reason='disabled or multi-rank; no single-rank claim')

    def optimizer_step(self):
        audit=self._xvla_audit and self._xvla_steps==0
        if audit:
            gradients=[]
            for name,parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    gradient=parameter.grad
                    gradients.append({'name':name,'has_gradient':gradient is not None,
                                      'finite':bool(torch.isfinite(gradient).all()) if gradient is not None else True,
                                      'nonzero':bool(torch.count_nonzero(gradient)) if gradient is not None else False})
            if not any(item['nonzero'] for item in gradients) or not all(item['finite'] for item in gradients):
                raise RuntimeError('Real SFT backward did not yield finite nonzero trainable gradients')
        result=super().optimizer_step()
        if audit:
            after=self._model_fingerprints()
            changed=[key for key in self._xvla_initial if after[key]['sha256']!=self._xvla_initial[key]['sha256']]
            changed_trainable=[key for key in changed if self._xvla_initial[key]['trainable']]
            changed_frozen=[key for key in changed if not self._xvla_initial[key]['trainable']]
            self._audit_write('optimizer_update',gradients=gradients,changed_trainable=changed_trainable,
                              changed_frozen=changed_frozen,post_update_fingerprints=after)
            if not changed_trainable or changed_frozen:
                raise RuntimeError('SFT update changed no trainable parameter, or modified frozen weights/buffers')
        self._xvla_steps+=1
        return result

    def _save_resume_counters(self, save_path):
        # Native data.pt owns loader/sampler cursors and rng.pt owns RNGs. These
        # worker counters are also needed when RLinf advances to the next epoch.
        state={'rank':self._rank,'world_size':self._world_size,
               'data_epoch':int(self._data_epoch),'data_iter_offset':int(self._data_iter_offset),
               'optimizer_steps':int(self.optimizer_steps),
               'manifest_sha256':self.data_config['manifest_sha256'],
               'micro_batch_size':int(self.micro_batch_size),
               'sampler_epoch':int(self.data_loader.sampler.epoch)}
        states=[None]*self._world_size
        torch.distributed.all_gather_object(states,state)
        if self._rank==0:
            path=Path(save_path)/'xvla_sft_data_state.json'
            temporary=path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps({'version':1,'ranks':states},indent=2)+'\n')
            temporary.replace(path)
        torch.distributed.barrier()

    def load_checkpoint(self, load_path):
        record=json.loads((Path(load_path)/'xvla_sft_data_state.json').read_text())
        if record.get('version')!=1 or len(record.get('ranks',[]))!=self._world_size:
            raise ValueError('SFT worker resume requires the original distributed topology')
        state=record['ranks'][self._rank]
        expected={'rank':self._rank,'world_size':self._world_size,
                  'manifest_sha256':self.data_config['manifest_sha256'],
                  'micro_batch_size':int(self.micro_batch_size)}
        if any(state.get(key)!=value for key,value in expected.items()):
            raise ValueError('SFT dataset identity, microbatch or rank changed during resume')
        for key in ('data_epoch','data_iter_offset','optimizer_steps','sampler_epoch'):
            if type(state.get(key)) is not int or state[key]<0:
                raise ValueError('Invalid SFT worker resume counter: '+key)
        if state['sampler_epoch']!=state['data_epoch']:
            raise ValueError('Saved SFT worker epoch differs from sampler epoch')
        # The parent immediately creates iter(loader) after loading native
        # data.pt. Restore epoch BEFORE that operation, and never set_epoch
        # afterwards: doing so could discard its just-restored cursor.
        self.data_loader.sampler.set_epoch(state['sampler_epoch'])
        super().load_checkpoint(load_path)
        if self.data_loader.sampler.epoch!=state['sampler_epoch']:
            raise RuntimeError('Native SFT loader state restored a different sampler epoch')
        self._data_epoch=state['data_epoch']
        self._data_iter_offset=state['data_iter_offset']
        self.optimizer_steps=state['optimizer_steps']
        if getattr(self,'_xvla_audit',False) and self._xvla_steps==0:
            self._xvla_initial=self._model_fingerprints()
        self._audit_write('data_resume',checkpoint=str(load_path),**state)

    def save_checkpoint(self, save_path, step=0):
        super().save_checkpoint(save_path,step)
        self._save_resume_counters(save_path)
        if self._xvla_audit:
            expected=self._model_fingerprints()
            # A deliberate temporary perturbation proves load really restores a
            # weight rather than comparing an unchanged model to itself.
            with torch.no_grad():
                for name,parameter in self.model.named_parameters():
                    if parameter.requires_grad and parameter.numel():
                        parameter.reshape(-1)[0].add_(0.125)
                        probe_name=name
                        break
                else:
                    raise RuntimeError('No trainable parameter for native checkpoint roundtrip probe')
            self.load_checkpoint(save_path)
            actual=self._model_fingerprints()
            passed=actual==expected
            self._audit_write('native_checkpoint_roundtrip',passed=passed,checkpoint=str(save_path),
                              perturbed_parameter=probe_name,scope='same-worker native save/load, not independent process')
            if not passed:
                raise RuntimeError('Native RLinf checkpoint did not exactly restore X-VLA parameters/buffers')
            from .audit_workers import state_digest
            digest=state_digest(self.get_model_state_dict(cpu_offload=True,full_state_dict=True))
            evidence={'model_config':OmegaConf.to_container(self.cfg.actor.model,resolve=True),'digest':digest}
            (Path(save_path)/'xvla_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')


def assert_pinned_runtime():
    root = Path(__file__).resolve().parents[1]
    expected = root / "third_party" / "rlinf"
    actual = Path(rlinf.__file__).resolve()
    if actual != expected / "rlinf" / "__init__.py":
        raise RuntimeError(f"Wrong RLinf import: {actual}; expected pinned runtime under {expected}")
    head = subprocess.check_output(["git", "-C", str(expected), "rev-parse", "HEAD"], text=True).strip()
    if head != "1bf78980d9339cbc67497eddceb9a1e3a74e2acc":
        raise RuntimeError(f"Unexpected RLinf baseline commit: {head}")


@hydra.main(version_base="1.1", config_path=None, config_name=None)
def main(cfg):
    assert_pinned_runtime()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("SFT execution must use this project's Slurm launcher")
    os.environ.setdefault("RLINF_EXT_MODULE", "xvla_rlinf")
    register()
    if cfg.runner.task_type != "sft" or cfg.actor.model.model_type != "xvla":
        raise ValueError("This entry only supports X-VLA supervised fine-tuning")
    cfg = validate_cfg(cfg)
    logging.info(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster).get_strategy("actor")
    if cfg.actor.training_backend not in ("fsdp", "fsdp2"):
        raise ValueError("X-VLA SFT currently supports only the pinned FSDP/FSDP2 worker")
    actor = XVLASftWorker.create_group(cfg).launch(cluster, name=cfg.actor.group_name,
                                                 placement_strategy=placement)
    runner = SFTRunner(cfg=cfg, actor=actor)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
