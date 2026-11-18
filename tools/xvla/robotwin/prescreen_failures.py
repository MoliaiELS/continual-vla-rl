"""Identify expected RoboTwin expert failures on a prescreen candidate."""

import linecache
from pathlib import Path
from traceback import walk_tb


EXPERT_GRASP_POSE_UNAVAILABLE = "expert_grasp_pose_unavailable"
EXPERT_PLAN_FAILED_EMPTY_GRASP = "expert_plan_failed_empty_grasp"

_EMPTY_GRASP_SITES = {
    ("right_action = self.grasp_actor(bottle, arm_tag=arm_tag, pre_grasp_dis=0.1)",
     "right_action[1][0].target_pose[2] += delta_dis"): "right_action",
    ('left_action = self.grasp_actor(bottle, arm_tag="left", pre_grasp_dis=0.1)',
     "left_action[1][0].target_pose[2] -= delta_dis"): "left_action",
}


def _at(frame, path, function):
    return frame.f_code.co_name == function and Path(frame.f_code.co_filename).parts[-len(path):] == path


def expert_failure_reason(env, task, error):
    """Classify only confirmed grasp failures from the pinned expert's play_once."""
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
    if (task == "put_bottles_dustbin" and isinstance(error, IndexError)
            and str(error) == "list index out of range"
            and _at(tail, ("envs", "put_bottles_dustbin.py"), "play_once")
            and tail.f_locals.get("self") is env and getattr(env, "plan_success", None) is False):
        filename = tail.f_code.co_filename
        call = linecache.getline(filename, lineno - 1, tail.f_globals).strip()
        source = linecache.getline(filename, lineno, tail.f_globals).strip()
        action_name = _EMPTY_GRASP_SITES.get((call, source))
        action = tail.f_locals.get(action_name)
        if (type(action) is tuple and len(action) == 2 and action[0] is None
                and type(action[1]) is list and not action[1]):
            return EXPERT_PLAN_FAILED_EMPTY_GRASP
    return None
