"""X-VLA data hook into the pinned RLinf FSDP SFT worker and SFTRunner."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess

import hydra
import rlinf
import torch.multiprocessing as multiprocessing
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
