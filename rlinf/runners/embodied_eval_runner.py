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

import json
import os
from pathlib import Path

from omegaconf.dictconfig import DictConfig

from rlinf.scheduler import Worker
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: DictConfig,
        rollout: Worker,
        env: Worker,
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_loger = MetricLogger(cfg)

    def evaluate(self):
        # Pool the raw per-episode tensors across batches. Averaging batch-level
        # ratios gives tasks with fewer evaluated episodes too much weight and
        # also averages counts instead of accumulating them.
        aggregated_eval_results = []
        evidence_dir = os.environ.get("FYP_DEPLOYMENT_EVIDENCE")
        before = None
        if evidence_dir:
            before = self.env.get_experiment_evidence(mode="eval").wait()
        completed_batches = 0
        for _ in range(self.cfg.algorithm.eval_rollout_epoch):
            env_futures = self.env.evaluate()
            rollout_futures = self.rollout.evaluate()
            env_futures.wait()
            rollout_results = rollout_futures.wait()
            eval_metrics_list = [
                results for results in rollout_results if results is not None
            ]
            aggregated_eval_results.extend(eval_metrics_list)
            completed_batches += 1

        if not aggregated_eval_results:
            raise RuntimeError("Evaluation produced no rollout results")
        final_metrics = compute_evaluate_metrics(aggregated_eval_results)
        if evidence_dir:
            after = self.env.get_experiment_evidence(mode="eval").wait()
            summary = self._episode_summary(before, after, completed_batches)
            path = Path(evidence_dir) / "eval_summary.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(summary, indent=2) + "\n")
            temporary.replace(path)
            # The goal predicate stored by the wrapper is authoritative for
            # ever/final, even when a generic rollout metric is absent.
            final_metrics["env_info/episodes_total"] = summary["episode_count"]
            for task_id, stats in summary["per_task"].items():
                prefix = f"env_info/task_{task_id}"
                final_metrics[f"{prefix}_success"] = stats["success_ever"] / stats["episodes"]
                final_metrics[f"{prefix}_success_total"] = stats["episodes"]
                final_metrics[f"{prefix}_success_at_end"] = stats["success_final"] / stats["episodes"]
            final_metrics["env_info/success_once"] = summary["success_ever"] / summary["episode_count"]
            final_metrics["env_info/success_at_end"] = summary["success_final"] / summary["episode_count"]
        return final_metrics

    @staticmethod
    def _episode_summary(before, after, completed_batches):
        """Subtract cumulative worker evidence without counting resets as episodes."""
        before_by_rank = {int(item["rank"]): item for item in before}
        episodes = []
        counter_names = ("policy_env_steps", "settling_env_steps", "post_done_executed_steps")
        counters = dict.fromkeys(counter_names, 0)
        for worker in after:
            rank = int(worker["rank"])
            previous = before_by_rank.get(rank, {})
            start = len(previous.get("episodes", []))
            worker_episodes = worker.get("episodes", [])
            if len(worker_episodes) < start:
                raise RuntimeError("Evaluation episode evidence was reset mid-run")
            for index, episode in enumerate(worker_episodes[start:], start):
                episodes.append({**episode, "source_rank": rank, "source_episode_index": index})
            for name in counter_names:
                current_totals = worker.get("totals", worker)
                previous_totals = previous.get("totals", previous)
                delta = int(current_totals.get(name, 0)) - int(previous_totals.get(name, 0))
                if delta < 0:
                    raise RuntimeError(f"Evaluation counter moved backwards: {name}")
                counters[name] += delta
        if not episodes:
            raise RuntimeError("Evaluation completed without recorded full episodes")
        per_task = {}
        for episode in episodes:
            task_id = str(episode["task_id"])
            stats = per_task.setdefault(task_id, {"episodes": 0, "success_ever": 0, "success_final": 0})
            stats["episodes"] += 1
            for name in ("success_ever", "success_final"):
                if not isinstance(episode[name], bool):
                    raise RuntimeError(f"Expected boolean episode {name}")
                stats[name] += int(episode[name])
        episode_steps = sum(int(ep["policy_env_steps"]) for ep in episodes)
        if episode_steps != counters["policy_env_steps"]:
            raise RuntimeError("Completed evaluation episodes do not match executed policy steps")
        return {"schema_version": 1, "completed_batches": completed_batches,
                "episode_count": len(episodes), "episodes": episodes, "per_task": per_task,
                "success_ever": sum(int(ep["success_ever"]) for ep in episodes),
                "success_final": sum(int(ep["success_final"]) for ep in episodes), **counters}

    def run(self):
        eval_metrics = self.evaluate()
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.metric_loger.log(step=0, data=eval_metrics)
        print(f"{eval_metrics=}")
        if os.environ.get("FYP_DEPLOYMENT_EVIDENCE"):
            path = Path(os.environ["FYP_DEPLOYMENT_EVIDENCE"]) / "eval_metrics.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(eval_metrics, indent=2) + "\n")

        self.metric_loger.finish()
