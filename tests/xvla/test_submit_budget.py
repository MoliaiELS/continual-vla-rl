import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('xvla_submit',Path(__file__).resolve().parents[2]/'tools/xvla/submit.py')
submit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)

class GpuBudgetTests(unittest.TestCase):
    def test_pending_reservation_counts_and_types_are_not_double_counted(self):
        info='JobState=PENDING NumNodes=1 TRES=cpu=8,gres/gpu=2,gres/gpu:4090=2 TresPerNode=gres/gpu:4090:2'
        self.assertEqual(submit.gpu_commitment(info),2)
        self.assertEqual(submit.gpu_commitment('NumNodes=2 TresPerNode=gres/gpu:4090:1'),2)
        self.assertEqual(submit.gpu_commitment('NumNodes=1 TresPerJob=gres/gpu:2'),2)
        self.assertEqual(submit.gpu_commitment('NumNodes=1 TresPerNode=gres/gpu:4090:1,gres/gpu:3090:1'),2)
        self.assertEqual(submit.gpu_commitment('NumNodes=1 TRES=cpu=8'),0)
        self.assertEqual(submit.gpu_commitment('NumNodes=1 TRES=gres/gpu=2'),2)
