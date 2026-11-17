"""Resume expert screening in the existing frozen RoboTwin evaluation."""
from __future__ import annotations

import argparse
from functools import wraps
from pathlib import Path
import sys
import traceback

if __package__:
    from .prescreen_failures import expert_failure_reason
else:
    from prescreen_failures import expert_failure_reason


EVALUATION_COMMIT = "9baa17360abe242cb36e8f9c78f0aadbe64b3bee"


def install_recovery(runner, adapter_evidence):
    original_load, original_append = runner.load_native, runner.append
    pending = None

    def load(root, task, condition):
        module, cls, cfg, robot_cfg = original_load(root, task, condition)
        play_once = cls.play_once

        @wraps(play_once)
        def play_expert(env):
            nonlocal pending
            try:
                return play_once(env)
            except Exception as error:
                reason = expert_failure_reason(env, task, error)
                if reason is None:
                    raise
                env.plan_success = False
                pending = dict(reason=reason, expert_exception=type(error).__name__,
                               traceback=traceback.format_exc(), prescreen_recovery=adapter_evidence)

        cls.play_once = play_expert
        return module, cls, cfg, robot_cfg

    def append(path, row):
        nonlocal pending
        if pending is not None:
            if path.name != "prescreen.jsonl" or row.get("reason") != "expert_plan_failed":
                raise RuntimeError("Pinned runner did not record the recovered expert exclusion")
            row.update(pending)
            pending = None
        return original_append(path, row)

    runner.load_native, runner.append = load, append


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinned-evaluation-root", required=True, type=Path)
    args, runner_args = parser.parse_known_args()
    if not runner_args or runner_args[0] != "screen":
        parser.error("Recovery only supports the expert prescreen stage")
    evaluation_root = args.pinned_evaluation_root.resolve(strict=True)
    sys.path.insert(0, str(evaluation_root))
    from tools.xvla.robotwin import runner

    if Path(runner.__file__).resolve() != evaluation_root / "tools/xvla/robotwin/runner.py":
        raise ImportError("Recovery did not import the pinned runner")
    runner.pinned_checkout(evaluation_root, EVALUATION_COMMIT)
    source = Path(__file__).resolve()
    evidence = {"evaluation_commit": EVALUATION_COMMIT,
                "adapter_commit": runner.git(source.parent, "rev-parse", "HEAD"),
                "source_sha256": {path.name: runner.sha256(path) for path in
                                  (source, source.with_name("prescreen_failures.py"))}}
    install_recovery(runner, evidence)
    sys.argv = [sys.argv[0], *runner_args]
    runner.main()


if __name__ == "__main__":
    main()
