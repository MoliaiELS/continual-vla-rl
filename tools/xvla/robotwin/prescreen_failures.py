"""Identify expected RoboTwin expert failures on a prescreen candidate."""

import linecache
from pathlib import Path
from traceback import walk_tb


EXPERT_GRASP_POSE_UNAVAILABLE = "expert_grasp_pose_unavailable"


def _at(frame, path, function):
    return frame.f_code.co_name == function and Path(frame.f_code.co_filename).parts[-len(path):] == path


def expert_failure_reason(env, task, error):
    """Classify only confirmed no-pose exceptions from the pinned expert's play_once."""
    frames = list(walk_tb(error.__traceback__))
    if not frames:
        return None
    tail, lineno = frames[-1]
    if (isinstance(error, AssertionError) and str(error) == "target_pose cannot be None for move action."
            and len(frames) >= 2 and _at(tail, ("envs", "utils", "action.py"), "__init__")):
        caller, _ = frames[-2]
        if (_at(caller, ("envs", "_base_task.py"), "grasp_actor")
                and caller.f_locals.get("self") is env
                and caller.f_locals.get("pre_grasp_pose", False) is None
                and caller.f_locals.get("grasp_pose", False) is None
                and tail.f_locals.get("action") == "move"
                and tail.f_locals.get("target_pose", False) is None):
            return EXPERT_GRASP_POSE_UNAVAILABLE
    if (task == "click_alarmclock" and isinstance(error, TypeError)
            and str(error) == "'NoneType' object is not subscriptable"
            and _at(tail, ("envs", "click_alarmclock.py"), "play_once")
            and tail.f_locals.get("self") is env):
        source = linecache.getline(tail.f_code.co_filename, lineno, tail.f_globals)
        if "self.get_grasp_pose(self.alarm, pre_dis=0.1, contact_point_id=0, arm_tag=arm_tag)[:3]" in source:
            return EXPERT_GRASP_POSE_UNAVAILABLE
    return None
