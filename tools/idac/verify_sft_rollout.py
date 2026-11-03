"""Reject incomplete deployments even when the Python evaluation entry point exits 0."""
import json
import os
from pathlib import Path

import numpy as np


root = Path(os.environ["LOG_DIR"])
evidence = root / "evidence"
env = json.loads((evidence / "env_rank_0.json").read_text())
model = json.loads((evidence / "model_rank_0.json").read_text())
metrics = json.loads((evidence / "eval_metrics.json").read_text())
episodes = env["episodes"]
assert len(episodes) == 2, episodes
assert all(ep["task_id"] == 0 and ep["policy_env_steps"] == 512 for ep in episodes)
assert env["policy_env_steps"] == 1024, env
assert model["is_lora"] is False
assert model["model_class"] == "OpenVLAOFTForRLActionPrediction"
assert model["sampling_params"]["temperature_eval"] == 1.6
assert metrics["eval/env_info/task_0_success_total"] == 2, metrics
assert all(np.isfinite(v) for v in metrics.values())
with np.load(root / "action_logs/rank_0_step_eval.npz") as actions:
    assert actions["actions"].shape == (128, 8, 7), actions["actions"].shape
    assert actions["action_tokens"].shape == (128, 8, 7)
    assert np.isfinite(actions["actions"]).all()
videos = [str(p.relative_to(root)) for p in root.rglob("*.mp4")]
assert videos, "Missing rollout video"
summary = {"validation": "SFT rollout connectivity only; no RL update or success-rate estimate",
           "policy_env_steps": env["policy_env_steps"],
           "settling_env_steps": env["settling_env_steps"],
           "post_success_executed_steps": env["post_success_executed_steps"],
           "optimizer_steps": 0, "episodes": episodes, "metrics": metrics,
           "videos": videos, "attention_backend": model["language_attention_backend"]}
(root / "sft_rollout.verified.json").write_text(json.dumps(summary, indent=2) + "\n")
print("FYP_SFT_ROLLOUT_VERIFIED", json.dumps(summary), flush=True)
