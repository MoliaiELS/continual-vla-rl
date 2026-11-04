# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from pathlib import Path
import time

import torch
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress
from rlinf.utils.fyp_protocol import FYPExperiment, evidence_delta, merge_evidence
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: EmbodiedFSDPActor,
        rollout: MultiStepRolloutWorker,
        env: EnvWorker,
        critic=None,
        reward=None,
        run_timer=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0
        experiment_cfg = cfg.runner.get("fyp_experiment", None)
        self.fyp_experiment = FYPExperiment.from_config(experiment_cfg) if experiment_cfg else None
        self._experiment_previous_evidence = None
        self._experiment_started_at = None
        self._experiment_replay_metadata = None
        self._experiment_optimizer_steps = 0
        self._experiment_skipped_iterations = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.metric_logger = MetricLogger(cfg)

    def init_workers(self):
        # create worker in order to decrease the maximum memory usage
        self.actor.init_worker().wait()
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

    def update_rollout_weights(self):
        rollout_futures = self.rollout.sync_model_from_actor()
        actor_futures = self.actor.sync_model_to_rollout()
        actor_futures.wait()
        rollout_futures.wait()
        self.actor.preallocate_memory()

    def generate_rollouts(self):
        env_futures = self.env.interact()
        rollout_futures = self.rollout.generate()
        actor_futures = self.actor.recv_rollout_batch()
        env_futures.wait()
        actor_futures.wait()
        rollout_futures.wait()

    def evaluate(self):
        env_futures = self.env.evaluate()
        rollout_futures = self.rollout.evaluate()
        env_futures.wait()
        rollout_results = rollout_futures.wait()
        eval_metrics_list = [
            results for results in rollout_results if results is not None
        ]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        start_step = self.global_step
        if self.fyp_experiment is not None:
            self._start_experiment()
        for _step in tqdm(range(start_step, self.max_steps), ncols=120):
            condition = None
            if self.fyp_experiment is not None:
                condition = self.fyp_experiment.condition(_step)
                self.env.set_experiment_condition(
                    condition["task_id"], condition["init_state_id"], _step
                ).wait()
                if self.cfg.algorithm.get("use_experience_replay", False):
                    self._experiment_replay_metadata = self.actor.set_replay_tasks(
                        self.fyp_experiment.replay_tasks(_step), condition["stage"]
                    ).wait()
                self._append_experiment_record(dict(event="iteration_start", replay=self._experiment_replay_metadata, **condition))
            if (
                _step % self.cfg.runner.val_check_interval == 0
                and self.cfg.runner.val_check_interval > 0
            ):
                with self.timer("eval"):
                    self.update_rollout_weights()
                    eval_metrics = self.evaluate()
                    eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                    self.metric_logger.log(data=eval_metrics, step=_step)

            with self.timer("step"):
                with self.timer("rollout"):
                    self.update_rollout_weights()
                    self.generate_rollouts()

                # compute advantages and returns.
                with self.timer("cal_adv_and_returns"):
                    actor_futures = self.actor.compute_advantages_and_returns()
                    actor_rollout_metrics = actor_futures.wait()

                # actor training.
                with self.timer("actor_training"):
                    is_last_step = (_step == self.max_steps - 1)
                    actor_training_futures = self.actor.run_training(is_last_step=is_last_step)
                    actor_training_metrics = actor_training_futures.wait()

                self.global_step += 1

                run_val, save_model, is_train_end = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                checkpoint_saved = save_model or (condition is not None and condition["stage_end"])
                if checkpoint_saved:
                    self._save_checkpoint()

            time_metrics = self.timer.consume_durations()

            rollout_metrics = {
                f"rollout/{k}": v for k, v in actor_rollout_metrics[0].items()
            }
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            training_metrics = {
                f"train/{k}": v for k, v in actor_training_metrics[0].items()
            }
            self.metric_logger.log(rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)
            if condition is not None:
                self._record_experiment_iteration(
                    condition, actor_rollout_metrics[0], actor_training_metrics[0], checkpoint_saved
                )

        self.metric_logger.finish()
        if self.fyp_experiment is not None:
            if self._experiment_optimizer_steps == 0:
                self._append_experiment_record({
                    "event": "qualification_failed", "reason": "no_effective_rl_update",
                    "outer_step": self.global_step,
                    "policy_env_steps": self._experiment_previous_evidence["totals"]["policy_env_steps"],
                    "skipped_iterations": self._experiment_skipped_iterations,
                })
                raise RuntimeError("Experiment spent its budget without an effective GRPO optimizer update")
            self._append_experiment_record({
                "event": "complete", "outer_step": self.global_step,
                "policy_env_steps": self._experiment_previous_evidence["totals"]["policy_env_steps"],
                "per_task": self._experiment_previous_evidence["per_task"],
                "grpo_optimizer_steps": self._experiment_optimizer_steps,
                "skipped_iterations": self._experiment_skipped_iterations,
                "elapsed_seconds": time.monotonic() - self._experiment_started_at,
            })
        
        # Compute and save EWC data if enabled
        if self.cfg.algorithm.get("use_ewc", False):
            ewc_save_path = os.path.join(
                self.cfg.runner.logger.log_path,
                "ewc_data.pt"
            )

            # Delegate EWC saving to the actor worker group (runs on training ranks)
            if hasattr(self.actor, "compute_and_save_ewc_data"):
                futures = self.actor.compute_and_save_ewc_data(ewc_save_path)
                futures.wait()

    def _save_checkpoint(self):
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        save_futures = self.actor.save_checkpoint(actor_save_path, self.global_step)
        save_futures.wait()
        if self.fyp_experiment is not None:
            metadata = {
                "outer_step": self.global_step,
                "condition": self.fyp_experiment.condition(self.global_step - 1),
                "protocol": self.fyp_experiment.manifest(),
                "optimizer_state_at_task_switch": "retained_in_same_worker",
                "full_resume_supported": False,
            }
            Path(base_output_dir, "fyp_protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        if self.fyp_experiment is not None:
            if (self.cfg.algorithm.group_size != 8 or self.cfg.algorithm.num_group_envs != 1
                    or self.cfg.algorithm.rollout_epoch != 1
                    or self.cfg.rollout.pipeline_stage_num != 1
                    or self.cfg.env.train.simulator_type != "libero"
                    or self.cfg.env.train.auto_reset or self.cfg.env.train.ignore_terminations
                    or not self.cfg.algorithm.use_fixed_reset_state_ids
                    or self.cfg.env.enable_offload):
                raise ValueError("FYP protocol requires non-offloaded LIBERO, G1/K8/R1/stage1 and real done masks")
            if self.cfg.env.train.max_episode_steps % self.cfg.actor.model.num_action_chunks:
                raise ValueError("Protocol horizon must be divisible by action chunk size")
            self.max_steps = self.fyp_experiment.total_iterations
            configured_max = self.cfg.runner.get("max_steps", -1)
            if configured_max >= 0 and configured_max != self.max_steps:
                raise ValueError("runner.max_steps disagrees with the FYP task budget")
            return
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    def _start_experiment(self):
        if self.global_step != 0:
            raise ValueError("FYP full experiment resume is not implemented")
        self._experiment_started_at = time.monotonic()
        self._experiment_journal = Path(self.cfg.runner.logger.log_path) / "experiment.jsonl"
        self._experiment_journal.parent.mkdir(parents=True, exist_ok=True)
        if self._experiment_journal.exists():
            raise FileExistsError(f"Refusing to overwrite existing experiment journal: {self._experiment_journal}")
        self._experiment_previous_evidence = merge_evidence(self.env.get_experiment_evidence().wait())
        self._append_experiment_record({
            "event": "start", "protocol": self.fyp_experiment.manifest(),
            "initialization_evidence": self._experiment_previous_evidence,
            "expected_total_policy_env_steps": self.max_steps * 8 * self.cfg.env.train.max_episode_steps,
        })

    def _append_experiment_record(self, record):
        def scalar(value):
            if hasattr(value, "tolist"):
                return value.tolist()
            raise TypeError(f"Experiment journal cannot serialize {type(value).__name__}")
        record = dict(record, timestamp_unix=time.time())
        with self._experiment_journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=scalar, allow_nan=False) + "\n")
            stream.flush()

    def _record_experiment_iteration(self, condition, rollout_metrics, training_metrics, checkpoint_saved):
        current = merge_evidence(self.env.get_experiment_evidence().wait())
        delta = evidence_delta(current, self._experiment_previous_evidence)
        expected_steps = 8 * self.cfg.env.train.max_episode_steps
        if delta["totals"]["policy_env_steps"] != expected_steps:
            raise RuntimeError(f"Protocol budget mismatch: {delta['totals']} expected {expected_steps} controls")
        episodes = delta["episodes"]
        if len(episodes) != 8 or any(
            episode["task_id"] != condition["task_id"]
            or episode["init_state_id"] != condition["init_state_id"]
            for episode in episodes
        ):
            raise RuntimeError("Collected episodes do not match the prescribed GRPO group")
        optimizer_steps = int(training_metrics["fyp_grpo_optimizer_steps"])
        skipped = int(training_metrics["fyp_grpo_skipped"])
        if optimizer_steps < 0 or skipped not in {0, 1} or bool(optimizer_steps) == bool(skipped):
            raise RuntimeError("Invalid actual GRPO update/skip accounting")
        self._experiment_optimizer_steps += optimizer_steps
        self._experiment_skipped_iterations += skipped
        record = dict(
            event="iteration", method=self.fyp_experiment.method, **condition,
            policy_env_steps=current["totals"]["policy_env_steps"],
            per_task=current["per_task"], budget_delta=delta,
            cumulative={"totals": current["totals"], "per_task": current["per_task"]},
            expected_iteration_policy_env_steps=expected_steps,
            rollout_metrics=rollout_metrics, training_metrics=training_metrics,
            grpo_optimizer_steps=self._experiment_optimizer_steps,
            skipped_iterations=self._experiment_skipped_iterations,
            replay=self._experiment_replay_metadata,
            elapsed_seconds=time.monotonic() - self._experiment_started_at,
            checkpoint=(str(Path(self.cfg.runner.logger.log_path) / "checkpoints" /
                            f"global_step_{self.global_step}" / "actor") if checkpoint_saved else None),
        )
        self._append_experiment_record(record)
        self._experiment_previous_evidence = current

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
