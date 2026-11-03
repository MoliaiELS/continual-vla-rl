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
        aggregated_eval_metrics = []

        for _ in range(self.cfg.algorithm.eval_rollout_epoch):
            env_futures = self.env.evaluate()
            rollout_futures = self.rollout.evaluate()
            env_futures.wait()
            rollout_results = rollout_futures.wait()
            eval_metrics_list = [
                results for results in rollout_results if results is not None
            ]
            eval_metrics = compute_evaluate_metrics(eval_metrics_list)
            aggregated_eval_metrics.append(eval_metrics)

        final_metrics = {}
        for k in aggregated_eval_metrics[0].keys():
            vals = [m[k] for m in aggregated_eval_metrics]
            final_metrics[k] = sum(vals) / len(vals)
        return final_metrics

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
