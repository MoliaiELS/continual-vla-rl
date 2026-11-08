"""Launch the pinned RLinf workers/runners with the X-VLA extension."""
import json,os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from . import register
from .sft_entry import assert_pinned_runtime

@hydra.main(version_base='1.1',config_path='../configs/xvla',config_name='ppo_smoke')
def main(cfg):
    assert_pinned_runtime()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use project Slurm launcher')
    register()
    from rlinf.config import validate_cfg
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
    cfg=validate_cfg(cfg)
    print(OmegaConf.to_yaml(cfg,resolve=True),flush=True)
    cluster=Cluster(cluster_cfg=cfg.cluster)
    placement=HybridComponentPlacement(cfg,cluster)
    rollout=MultiStepRolloutWorker.create_group(cfg).launch(cluster,name=cfg.rollout.group_name,
                        placement_strategy=placement.get_strategy('rollout'))
    env=EnvWorker.create_group(cfg).launch(cluster,name=cfg.env.group_name,
                        placement_strategy=placement.get_strategy('env'))
    if cfg.runner.only_eval:
        from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
        runner=EmbodiedEvalRunner(cfg,rollout,env)
    else:
        from rlinf.runners.embodied_runner import EmbodiedRunner
        from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
        actor=EmbodiedFSDPActor.create_group(cfg).launch(cluster,name=cfg.actor.group_name,
                        placement_strategy=placement.get_strategy('actor'))
        runner=EmbodiedRunner(cfg,actor,rollout,env)
    runner.init_workers()
    runner.run()
    print('XVLA_RLINF_RUNNER_FINISHED',flush=True)

if __name__=='__main__':
    mp.set_start_method('spawn',force=True)
    main()
