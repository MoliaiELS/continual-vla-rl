"""Select a GPU type from a live Slurm node-capacity snapshot."""
from __future__ import annotations

import re
import subprocess


GPU_MODELS = ("4090", "3090")
UNAVAILABLE_FLAGS = (
    "BLOCKED", "COMPLETING", "DOWN", "DRAIN", "FAIL", "INVALID_REG", "MAINT",
    "NO_RESPOND", "NOT_RESPOND", "PLANNED", "POWER", "REBOOT", "RESERVED",
)
GPU_ENTRY = re.compile(r"(?:^|,)gpu(?::([^:,()]+))?:(\d+)(?:\([^)]*\))?(?=,|$)")


def _gpu_counts(value):
    counts = {}
    # Commas inside (IDX:...) and (S:...) are part of a single GRES entry.
    for model, count in GPU_ENTRY.findall(value):
        counts[model] = counts.get(model, 0) + int(count)
    return counts


def _preferred_gpu_model(nodes, partition):
    available = set()
    for line in nodes.splitlines():
        fields = dict(re.findall(r"(?:^|\s)(\w+)=(\S+)", line))
        name, state = fields.get("NodeName"), fields.get("State", "").upper()
        if (not name or name == "4090node1" or partition not in fields.get("Partitions", "").split(",")
                or state.split("+")[0] not in ("IDLE", "MIXED", "ALLOCATED")
                or any(flag in state for flag in UNAVAILABLE_FLAGS)):
            continue
        cpu_total = fields.get("CPUEfctv", fields.get("CPUTot"))
        cpu_allocated = fields.get("CPUAlloc")
        if cpu_total is not None and cpu_allocated is not None:
            # The standard X-VLA job requests eight CPUs on its GPU node.
            if not (cpu_total.isdecimal() and cpu_allocated.isdecimal()
                    and int(cpu_total) - int(cpu_allocated) >= 8):
                continue
        configured = _gpu_counts(fields.get("Gres", ""))
        used = _gpu_counts(fields.get("GresUsed", ""))
        if used.get("", 0):
            continue
        available.update(model for model in GPU_MODELS
                         if model in used and used[model] < configured.get(model, 0))
    return next((model for model in GPU_MODELS if model in available), None)


def available_gpu_model(partition: str) -> str | None:
    """Return the preferred known-free GPU type, or None if capacity is uncertain or absent."""
    try:
        nodes = subprocess.check_output(
            ["scontrol", "show", "node", "-d", "-o"], text=True, stderr=subprocess.PIPE, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return _preferred_gpu_model(nodes, partition)
