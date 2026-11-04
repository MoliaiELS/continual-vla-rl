"""Compose the real Hydra entrypoint without importing models or starting Ray."""
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

repo = Path(os.environ.get("FYP_CONFIG_REPO", Path(__file__).resolve().parents[2])).resolve()
os.environ["EMBODIED_PATH"] = str(repo / "examples/embodiment")
os.environ["REPO_PATH"] = str(repo)
os.environ.setdefault("FYP_MODEL_PATH", "/not-downloaded")
OmegaConf.register_new_resolver("int_div", lambda x, y: int(x) // int(y), replace=True)
OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
with initialize_config_dir(config_dir=str(repo / "examples/embodiment/config/crl_experiment"), version_base="1.1"):
    for name in ("fyp_spatial_smoke", "fyp_spatial_eval"):
        overrides = [f"runner.logger.log_path={os.environ['LOG_DIR']}"] if os.environ.get("LOG_DIR") else []
        if name.endswith("eval"):
            overrides.extend(sys.argv[1:])
        cfg = compose(config_name=name, overrides=overrides)
        OmegaConf.resolve(cfg)
        assert cfg.env.fixed_task_ids == [0]
        assert cfg.actor.model.num_action_chunks == 8
        assert cfg.rollout.model_dir == os.environ["FYP_MODEL_PATH"]
        if name.endswith("smoke"):
            assert (cfg.algorithm.group_size, cfg.algorithm.num_group_envs, cfg.algorithm.rollout_epoch) == (8, 1, 1)
            assert (cfg.actor.micro_batch_size, cfg.actor.global_batch_size, cfg.runner.max_epochs) == (1, 64, 1)
        else:
            assert cfg.env.eval.num_envs == 2 and cfg.env.eval.eval_per_task == 2
        if os.environ.get("LOG_DIR"):
            kind = "train" if name.endswith("smoke") else "eval"
            Path(os.environ["LOG_DIR"], f"{kind}.resolved.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
        print("CONFIG_OK", name)
