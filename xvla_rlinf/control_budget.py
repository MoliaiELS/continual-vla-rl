"""Stop a synchronous source-pool PPO run at its actual control-step budget."""
from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import random
import re


FORMAL_MILESTONES = (50_000, 100_000, 200_000)
FORMAL_SEEDS = (101, 202, 303)


def validate_control_budget(cfg):
    runner = cfg.runner
    budget = runner.get("xvla_control_step_budget")
    milestones = runner.get("xvla_control_step_milestones")
    seed = runner.get("xvla_training_seed")
    if milestones is not None or seed is not None:
        if (not isinstance(milestones, Sequence) or tuple(milestones) != FORMAL_MILESTONES
                or any(type(value) is not int for value in milestones)
                or budget != FORMAL_MILESTONES[-1] or type(seed) is not int or seed not in FORMAL_SEEDS):
            raise ValueError("Formal PPO requires the registered 50k/100k/200k milestones and a registered seed")
    if budget is None:
        return False
    if type(budget) is not int or budget < 1:
        raise ValueError("xvla_control_step_budget must be a positive integer")
    train = cfg.env.train
    if (runner.get("only_eval") or runner.get("resume_dir") or runner.get("use_training_pipeline")
            or runner.get("overlap_env_bootstrap") or runner.get("enable_decoupled_mode")
            or cfg.algorithm.adv_type != "gae" or cfg.algorithm.loss_type != "actor_critic"
            or train.get("xvla_source_reset_pool") is None
            or train.rollout_epoch != 1 or cfg.rollout.pipeline_stage_num != 1):
        raise ValueError("Control-step budget requires fresh synchronous PPO and one source-pool rollout per cycle")
    if not (0 < runner.max_steps <= runner.max_epochs):
        raise ValueError("Control-step budget requires a positive bounded cycle cap in max_steps and max_epochs")
    if runner.save_interval > 0 or runner.val_check_interval > 0:
        raise ValueError("Use budget milestones and independent evaluation for a control-step run")
    return True


def read_budget_ledger(path, *, environments, horizon, plan_sha256, mode, training_seed=None):
    """Validate completed source rows and count physically executed actions."""
    completed = 0
    steps = 0
    settling = 0
    successes = 0
    signal_batches = 0
    pending = None
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                event = row["event"]
                if (row["rollout_index"] != completed or row["plan_sha256"] != plan_sha256
                        or row["mode"] != mode or row["task_id"] != 0 or row["seed_offset"] != 0
                        or row.get("training_seed") != training_seed):
                    raise RuntimeError("Source budget ledger has mismatched rollout or reset plan")
                if event == "source_rollout_started" and pending is None:
                    pending = row["trial_ids"]
                    continue
                if event != "source_rollout_completed" or pending != row["trial_ids"]:
                    raise RuntimeError("Source budget ledger has missing or duplicated rollout boundaries")
                policy_steps = row["policy_env_steps"]
                reset_steps = row["settling_env_steps"]
                success = row["success"]
                if (len(pending) != environments or len(policy_steps) != environments
                        or len(reset_steps) != environments or len(success) != environments
                        or any(type(n) is not int or not 1 <= n <= horizon for n in policy_steps)
                        or any(type(n) is not int or n < 0 for n in reset_steps)
                        or any(type(flag) is not bool for flag in success)):
                    raise RuntimeError("Invalid per-environment evidence in source budget ledger")
                steps += sum(policy_steps)
                settling += sum(reset_steps)
                successes += sum(success)
                signal_batches += int(any(success))
                completed += 1
                pending = None
    return {
        "policy_env_steps": steps,
        "settling_env_steps": settling,
        "completed_rollouts": completed,
        "completed_episodes": completed * environments,
        "successful_episodes": successes,
        "positive_reward_rollouts": signal_batches,
        "partial_rollout": pending is not None,
    }


class _BudgetComplete(Exception):
    """Stop the pinned loop at the post-rollout hook and finish its logger."""


def control_budget_runner(base_runner):
    class ControlBudgetRunner(base_runner):
        def __init__(self, cfg, *args, **kwargs):
            super().__init__(cfg, *args, **kwargs)
            self.budget = cfg.runner.xvla_control_step_budget
            self.training_seed = cfg.runner.get("xvla_training_seed")
            self.targets = tuple(cfg.runner.get("xvla_control_step_milestones") or (self.budget,))
            self.milestone_receipts = []
            self.run_dir = Path(os.environ["XVLA_RUN_DIR"])
            self.ledger_path = self.run_dir / "source_resets_0.jsonl"
            self.summary_path = self.run_dir / "control_step_budget.json"
            self.environments = cfg.env.train.total_num_envs
            self.horizon = cfg.env.train.max_episode_steps
            self.reset_pool = cfg.env.train.xvla_source_reset_pool
            self.max_batch_steps = self.environments * self.horizon
            if self.ledger_path.exists() or self.summary_path.exists():
                raise FileExistsError("Control-step training requires a fresh Slurm run directory")

        def _ledger(self):
            return read_budget_ledger(
                self.ledger_path, environments=self.environments, horizon=self.horizon,
                plan_sha256=self.reset_pool.sha256, mode=self.reset_pool.mode,
                training_seed=self.training_seed,
            )

        def _write_budget(self, ledger, state, **extra):
            summary = {
                "state": state,
                "requested_policy_env_steps": self.budget,
                "max_rollout_batch_control_steps": self.max_batch_steps,
                "overshoot_policy_env_steps": max(0, ledger["policy_env_steps"] - self.budget),
                "rollout_cycle_cap": self.max_steps,
                "reset_plan_sha256": self.reset_pool.sha256,
                "sampling_mode": self.reset_pool.mode,
                **ledger,
                **extra,
            }
            if self.training_seed is not None:
                summary.update(training_seed=self.training_seed, milestones=self.milestone_receipts)
            temporary = self.summary_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.summary_path)

        def _log_step_metrics(self, *args, **kwargs):
            super()._log_step_metrics(*args, **kwargs)
            ledger = self._ledger()
            if ledger["completed_rollouts"] != self.global_step or ledger["partial_rollout"]:
                raise RuntimeError("Source budget ledger does not cover the completed PPO cycles exactly")
            self._write_budget(ledger, "RUNNING")
            target = self.targets[len(self.milestone_receipts)]
            if ledger["policy_env_steps"] < target:
                return
            if ledger["policy_env_steps"] >= target + self.max_batch_steps:
                raise RuntimeError("Actual environment interaction exceeded one rollout batch of overshoot")

            # Distinguish a completed collection with no reward from a learned policy.
            optimizer_updates = 0
            for path in self.run_dir.glob("audit-*.jsonl"):
                with path.open(encoding="utf-8") as stream:
                    optimizer_updates += sum(json.loads(line)["event"] == "optimizer" for line in stream)
            if bool(optimizer_updates) != bool(ledger["positive_reward_rollouts"]):
                raise RuntimeError("Observed successes disagree with actual actor optimizer activity")
            final = target == self.budget
            if final:
                self.update_rollout_weights()
            self._save_checkpoint()
            checkpoint = (
                Path(self.cfg.runner.logger.log_path) / self.cfg.runner.logger.experiment_name
                / "checkpoints" / f"global_step_{self.global_step}" / "actor"
            )
            state = "COMPLETE_VALID" if optimizer_updates else "COMPLETE_NO_SIGNAL"
            if self.training_seed is not None:
                evidence = json.loads((checkpoint / "xvla_evidence.json").read_text(encoding="utf-8"))
                digest = evidence["digest"]
                if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                    raise RuntimeError("Milestone checkpoint is missing its audited model-state digest")
                self.milestone_receipts.append({
                    **ledger,
                    "state": state,
                    "requested_policy_env_steps": target,
                    "overshoot_policy_env_steps": ledger["policy_env_steps"] - target,
                    "optimizer_updates": optimizer_updates,
                    "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_sha256": digest,
                })
            if not final:
                self._write_budget(ledger, "RUNNING")
                return
            self._write_budget(
                ledger, state,
                optimizer_updates=optimizer_updates, checkpoint_path=str(checkpoint.resolve()),
            )
            raise _BudgetComplete

        def run(self):
            try:
                super().run()
            except _BudgetComplete:
                self._finish_run()
                return
            except Exception as error:
                # The per-rollout ledger is primary when a failure precedes an update.
                try:
                    self._write_budget(self._ledger(), "FAILED", error_type=type(error).__name__)
                except (OSError, ValueError, KeyError, RuntimeError):
                    pass
                raise
            self._write_budget(self._ledger(), "BATCH_CAP_REACHED")
            raise RuntimeError("Control-step budget not reached before the explicit rollout cycle cap")

    return ControlBudgetRunner


def seed_training_worker(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def control_budget_rollout(base_rollout):
    class ControlBudgetRollout(base_rollout):
        def init_worker(self):
            seed = self.cfg.runner.xvla_training_seed
            seed_training_worker(seed)
            super().init_worker()
            # Start exploration at a known RNG state regardless of model construction draws.
            seed_training_worker(seed)

    return ControlBudgetRollout


def control_budget_actor(base_actor):
    class ControlBudgetActor(base_actor):
        def init_worker(self):
            seed = self.cfg.runner.get("xvla_training_seed")
            if seed is not None:
                seed_training_worker(seed)
            return super().init_worker()

        def run_training(self):
            import torch

            rewards = self.rollout_batch["rewards"]
            if not bool(torch.isfinite(rewards).all()):
                raise RuntimeError("Nonfinite rewards in budgeted source PPO")
            if bool((rewards > 0).any()):
                return super().run_training()

            from .audit_workers import record

            record("training_skipped", rank=self._rank, reason="NO_POSITIVE_PPO_REWARD",
                   optimizer_steps=self.optimizer_steps)
            return {}

    return ControlBudgetActor
