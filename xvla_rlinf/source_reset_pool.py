"""Opt-in reset sampling for standard LIBERO-Spatial task 0.

The pool contains existing benchmark trial IDs. It does not edit simulator
states or change reward, episode termination, evaluation, or the learner.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def _trial_ids(values, name):
    if not isinstance(values, list) or any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"{name} must be a list of nonnegative integer trial IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate trial IDs")
    return tuple(values)


@dataclass(frozen=True)
class SourceResetPlan:
    sha256: str
    seed: int
    train_trial_ids: tuple[int, ...]
    source_dev_trial_ids: tuple[int, ...]
    strata: tuple[tuple[int, ...], ...]
    mode: str

    def sample(self, rng, count):
        if self.mode == "uniform":
            return rng.choice(self.train_trial_ids, size=count, replace=True)
        strata = rng.integers(len(self.strata), size=count)
        return np.asarray([rng.choice(self.strata[index]) for index in strata], dtype=np.int64)

    def validate_trial_count(self, available):
        if max(self.train_trial_ids + self.source_dev_trial_ids) >= available:
            raise ValueError(f"Reset plan trial ID is outside task 0's {available} benchmark trials")


def load_source_reset_plan(path, expected_sha256, mode):
    if mode not in ("uniform", "stratified"):
        raise ValueError("Source reset sampling mode must be uniform or stratified")
    contents = Path(path).read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Source reset plan SHA256 mismatch: {path}")
    data = json.loads(contents)
    if not isinstance(data, dict) or any(
        type(data.get(key)) is not type(value) or data[key] != value
        for key, value in {"schema_version": 1, "task_suite": "libero_spatial", "task_id": 0}.items()
    ):
        raise ValueError("Source reset plan must use schema 1 and LIBERO-Spatial task 0")
    seed = data.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("Source reset plan seed must be a nonnegative integer")
    train = _trial_ids(data.get("train_trial_ids"), "train_trial_ids")
    dev = _trial_ids(data.get("source_dev_trial_ids", []), "source_dev_trial_ids")
    if not train or set(train).intersection(dev):
        raise ValueError("Source reset training pool must be nonempty and disjoint from development trials")
    labels = data.get("strata", {})
    if not isinstance(labels, dict) or any(not label.strip() for label in labels):
        raise ValueError("Reset strata must be a mapping with nonempty labels")
    strata = tuple(_trial_ids(labels[label], f"stratum {label!r}") for label in sorted(labels))
    if strata:
        assigned = [trial for stratum in strata for trial in stratum]
        if not all(strata) or len(assigned) != len(train) or set(assigned) != set(train):
            raise ValueError("Reset strata must be a disjoint, nonempty partition of the training pool")
    if mode == "stratified" and not strata:
        raise ValueError("Stratified sampling requires reset strata")
    return SourceResetPlan(digest, seed, train, dev, strata, mode)


def _read_env_plan(env_cfg):
    options = env_cfg["xvla_source_reset_pool"]
    if (env_cfg.get("env_type") != "libero" or env_cfg.get("task_suite_name") != "libero_spatial"
            or env_cfg.get("libero_variant", "standard") != "standard"
            or os.environ.get("LIBERO_TYPE", "standard").lower() != "standard"
            or list(env_cfg.get("task_id_filter") or ()) != [0]):
        raise ValueError("Source reset pools require standard LIBERO-Spatial with task_id_filter=[0]")
    if (env_cfg.get("is_eval", False) or env_cfg.get("auto_reset", False)
            or env_cfg.get("use_ordered_reset_state_ids", False)
            or env_cfg.get("specific_reset_id") is not None
            or env_cfg.get("use_fixed_reset_state_ids") is not True
            or env_cfg.get("xvla_strict_episodes") is not True):
        raise ValueError("Source reset pools require strict non-auto-reset training with fixed grouped resets")
    return load_source_reset_plan(options["path"], options["sha256"], options["mode"])


def validate_source_reset_pool(cfg):
    if cfg.env.eval.get("xvla_source_reset_pool") is not None:
        raise ValueError("Source reset pools cannot modify evaluation")
    if cfg.env.train.get("xvla_source_reset_pool") is None:
        return False
    if cfg.runner.get("only_eval") or cfg.runner.get("resume_dir"):
        raise ValueError("Source reset sampling requires a fresh training run")
    _read_env_plan(cfg.env.train)
    return True


class _SourceResetEnv:
    """Mix into the pinned LiberoEnv, retaining its grouped reset lifecycle."""

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self._source_plan = _read_env_plan(cfg)
        self._source_rng = np.random.default_rng(np.random.SeedSequence([self._source_plan.seed, seed_offset]))
        self._source_rollout_index = 0
        self._source_reset_pending = None
        self._source_reset_log = Path(os.environ["XVLA_RUN_DIR"]) / f"source_resets_{seed_offset}.jsonl"
        group_size = cfg.group_size
        if type(group_size) is not int or group_size < 1 or num_envs < 1 or num_envs % group_size:
            raise ValueError("Training environment count must be a positive multiple of group_size")
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

    def _compute_total_num_group_envs(self):
        super()._compute_total_num_group_envs()
        self._source_plan.validate_trial_count(self.trial_id_bins[0])

    def _get_random_reset_state_ids(self, num_reset_states):
        # Task 0 trial IDs are the first range in LIBERO's global reset IDs.
        return self._source_plan.sample(self._source_rng, num_reset_states)

    def reset(self, env_idx=None, reset_state_ids=None):
        if not self.is_start or env_idx is not None or reset_state_ids is not None:
            raise RuntimeError("Source reset pools support only complete grouped rollout resets")
        if self._source_reset_pending is not None:
            raise RuntimeError("Source reset was replaced before any policy action executed")
        result = super().reset(env_idx=env_idx, reset_state_ids=reset_state_ids)
        trials = self.trial_ids.tolist()
        if (len(trials) != self.num_envs or not np.all(self.task_ids == 0)
                or trials != self.reset_state_ids.tolist()
                or not set(trials).issubset(self._source_plan.train_trial_ids)
                or any(len(set(trials[start:start + self.group_size])) != 1
                       for start in range(0, len(trials), self.group_size))):
            raise RuntimeError("Applied source resets escaped the task-0 pool or split a group")
        self._source_reset_pending = trials
        return result

    def chunk_step(self, chunk_actions):
        result = super().chunk_step(chunk_actions)
        if self._source_reset_pending is not None:
            # Confirm policy actions actually executed before recording a
            # reset. Initial and end-of-rollout draws alone are not evidence.
            mask = result[-1][-1]["executed_mask"]
            policy_steps = mask.sum(axis=1).tolist()
            if len(policy_steps) != self.num_envs or min(policy_steps) < 1:
                raise RuntimeError("A grouped source reset did not execute any policy action")
            trials = self._source_reset_pending
            row = {
                "event": "source_rollout_started",
                "plan_sha256": self._source_plan.sha256,
                "mode": self._source_plan.mode,
                "plan_seed": self._source_plan.seed,
                "seed_offset": self.seed_offset,
                "rollout_index": self._source_rollout_index,
                "task_id": 0,
                "group_size": self.group_size,
                "trial_ids": trials,
                "group_trial_ids": trials[::self.group_size],
                "first_chunk_policy_env_steps": policy_steps,
            }
            line = json.dumps(row)
            with self._source_reset_log.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
            print("XVLA_SOURCE_RESETS " + line, flush=True)
            self._source_reset_pending = None
            self._source_rollout_index += 1
        return result


def source_reset_worker(base_worker):
    class SourceResetEnvWorker(base_worker):
        def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage):
            if env_cfg.get("xvla_source_reset_pool") is not None:
                class SourceResetLiberoEnv(_SourceResetEnv, env_cls):
                    pass

                env_cls = SourceResetLiberoEnv
            return super()._setup_env_and_wrappers(env_cls, env_cfg, num_envs_per_stage)

    return SourceResetEnvWorker
