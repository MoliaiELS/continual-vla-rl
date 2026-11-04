"""Deterministic small-scale continual-RL protocol, independent of GPU packages."""

from dataclasses import dataclass
import random


COUNTER_KEYS = (
    "policy_env_steps",
    "settling_env_steps",
    "post_done_executed_steps",
    "episodes_collected",
    "success_ever_count",
    "success_final_count",
    "steps_through_first_success_or_horizon",
)


def zero_counters():
    return dict.fromkeys(COUNTER_KEYS, 0)


def merge_evidence(reports):
    """Combine worker/stage reports without counting aggregate rows twice."""
    merged = {"totals": zero_counters(), "per_task": {}, "episodes": []}
    for report in reports:
        if report is None:
            continue
        for key in COUNTER_KEYS:
            merged["totals"][key] += int(report["totals"].get(key, 0))
        for task_id, counters in report.get("per_task", {}).items():
            dest = merged["per_task"].setdefault(str(task_id), zero_counters())
            for key in COUNTER_KEYS:
                dest[key] += int(counters.get(key, 0))
        merged["episodes"].extend(report.get("episodes", []))
    return merged


def evidence_delta(current, previous):
    """Executed interaction costs are monotonic, including discarded groups."""
    delta = {"totals": zero_counters(), "per_task": {}, "episodes": []}
    for key in COUNTER_KEYS:
        value = current["totals"].get(key, 0) - previous["totals"].get(key, 0)
        if value < 0:
            raise ValueError(f"Environment evidence moved backwards: {key}")
        delta["totals"][key] = int(value)
    for task_id, counters in current.get("per_task", {}).items():
        prior = previous.get("per_task", {}).get(task_id, {})
        values = {key: int(counters.get(key, 0) - prior.get(key, 0))
                  for key in COUNTER_KEYS}
        if any(value < 0 for value in values.values()):
            raise ValueError(f"Task {task_id} environment counters moved backwards")
        if any(values.values()):
            delta["per_task"][task_id] = values
    previous_count = len(previous.get("episodes", []))
    delta["episodes"] = current.get("episodes", [])[previous_count:]
    return delta


@dataclass(frozen=True)
class FYPExperiment:
    method: str
    task_ids: tuple
    iterations_per_task: int
    seed: int
    train_init_ids: tuple = tuple(range(40))

    @classmethod
    def from_config(cls, config):
        result = cls(
            method=str(config["method"]),
            task_ids=tuple(int(task) for task in config["task_ids"]),
            iterations_per_task=int(config["iterations_per_task"]),
            seed=int(config.get("seed", 1234)),
            train_init_ids=tuple(int(state) for state in config.get("train_init_ids", range(40))),
        )
        if result.method not in {"seq", "joint"}:
            raise ValueError("fyp_experiment.method must be seq or joint")
        if (not result.task_ids or len(set(result.task_ids)) != len(result.task_ids)
                or any(task < 0 for task in result.task_ids)):
            raise ValueError("fyp_experiment.task_ids must contain distinct nonnegative IDs")
        if result.iterations_per_task < 1:
            raise ValueError("iterations_per_task must be positive")
        if (not result.train_init_ids or len(set(result.train_init_ids)) != len(result.train_init_ids)
                or any(state < 0 for state in result.train_init_ids)):
            raise ValueError("train_init_ids must contain distinct nonnegative IDs")
        return result

    @property
    def total_iterations(self):
        return len(self.task_ids) * self.iterations_per_task

    def condition(self, iteration):
        if not 0 <= iteration < self.total_iterations:
            raise IndexError("Experiment iteration outside prescribed budget")
        if self.method == "seq":
            task_index = iteration // self.iterations_per_task
            visit = iteration % self.iterations_per_task
        else:
            task_index = iteration % len(self.task_ids)
            visit = iteration // len(self.task_ids)
        task_id = self.task_ids[task_index]
        cycle, offset = divmod(visit, len(self.train_init_ids))
        ordered_ids = list(self.train_init_ids)
        # Task-local permutations give every method the same initial-state stream,
        # independent of the number/order of visits to other tasks.
        random.Random(f"fyp:{self.seed}:{task_id}:{cycle}").shuffle(ordered_ids)
        return {
            "iteration": iteration,
            "outer_step": iteration + 1,
            "stage": iteration // self.iterations_per_task + 1,
            "task_id": task_id,
            "task_visit": visit,
            "init_state_id": ordered_ids[offset],
            "stage_end": (iteration + 1) % self.iterations_per_task == 0,
        }

    def replay_tasks(self, iteration):
        if self.method == "joint":
            return list(self.task_ids)
        return list(self.task_ids[:iteration // self.iterations_per_task])

    def manifest(self):
        return {
            "method": self.method,
            "task_ids": list(self.task_ids),
            "iterations_per_task": self.iterations_per_task,
            "total_iterations": self.total_iterations,
            "seed": self.seed,
            "train_init_ids": list(self.train_init_ids),
            "optimizer_state_at_task_switch": "retained_in_same_worker",
            "conditions": [self.condition(i) for i in range(self.total_iterations)],
        }
