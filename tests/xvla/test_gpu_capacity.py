import subprocess
import unittest
from unittest.mock import patch

from tools.xvla import gpu_capacity


def node(name, *, model=None, used=0, state="MIXED", partitions="GPU", cpu_allocated=20):
    model = model or name[:4]
    return (
        f"NodeName={name} CPUTot=96 CPUEfctv=96 CPUAlloc={cpu_allocated} "
        f"Partitions={partitions} State={state} Gres=gpu:{model}:4(S:0-1) "
        f"GresUsed=gpu:{model}:{used}(IDX:0,1)"
    )


class GpuCapacityTests(unittest.TestCase):
    def select(self, *nodes, partition="GPU"):
        with patch.object(gpu_capacity.subprocess, "check_output", return_value="\n".join(nodes)) as command:
            result = gpu_capacity.available_gpu_model(partition)
        command.assert_called_once_with(
            ["scontrol", "show", "node", "-d", "-o"], text=True, stderr=subprocess.PIPE, timeout=5,
        )
        return result

    def test_prefers_4090_then_falls_back_to_3090(self):
        self.assertEqual(self.select(node("3090node2"), node("4090node2", used=3)), "4090")
        self.assertEqual(self.select(node("3090node2"), node("4090node2", used=4)), "3090")
        self.assertIsNone(self.select(node("3090node2", used=4), node("4090node2", used=4)))

    def test_ignores_login_wrong_partition_and_unavailable_states(self):
        for state in ("DOWN", "IDLE+DRAIN", "MIXED+NO_RESPOND", "IDLE+MAINT", "IDLE+POWERING_UP"):
            with self.subTest(state=state):
                self.assertEqual(self.select(
                    node("4090node1"), node("4090node2", partitions="GPU-other"),
                    node("4090node3", state=state), node("3090node2", partitions="other,GPU"),
                ), "3090")

    def test_ignores_nodes_that_cannot_supply_the_requested_cpus(self):
        self.assertEqual(self.select(node("4090node2", cpu_allocated=89), node("3090node2")), "3090")
        self.assertEqual(self.select(node("4090node2", cpu_allocated=88), node("3090node2")), "4090")

    def test_counts_typed_gres_separately_and_preserves_idx_commas(self):
        mixed = (
            "NodeName=mixed Partitions=GPU State=MIXED "
            "Gres=gpu:4090:2(S:0-1),gpu:3090:4(S:2-3),scratch:1 "
            "GresUsed=gpu:4090:2(IDX:0,1),gpu:3090:3(IDX:2,3,4),scratch:0"
        )
        self.assertEqual(self.select(mixed), "3090")
        self.assertIsNone(self.select(mixed.replace("gpu:3090:3(IDX:2,3,4)", "gpu:3(IDX:2,3,4)")))
        self.assertIsNone(self.select(node("4090node2").split(" GresUsed=")[0]))

    def test_query_failure_leaves_the_default_to_the_caller(self):
        for error in (FileNotFoundError(), subprocess.CalledProcessError(1, "scontrol"),
                      subprocess.TimeoutExpired("scontrol", 5)):
            with self.subTest(error=type(error).__name__):
                with patch.object(gpu_capacity.subprocess, "check_output", side_effect=error):
                    self.assertIsNone(gpu_capacity.available_gpu_model("GPU"))
