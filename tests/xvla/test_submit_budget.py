import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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


class BenchmarkSubmissionTests(unittest.TestCase):
    def arguments(self, **overrides):
        values = {'mode': 'benchmark_libero', 'gpu_model': None, 'gpus': 1,
                  'start_task': 0, 'task_count': 10, 'trials': 5,
                  'checkpoint_job_id': None, 'partition': 'GPU'}
        return SimpleNamespace(**(values | overrides))

    def test_benchmark_shards_are_bounded(self):
        full = submit.benchmark_settings(self.arguments())
        self.assertEqual((full['model'], full['start_task'], full['task_count'], full['trials']),
                         ('libero', 0, 10, 5))
        for overrides in ({'start_task': 9, 'task_count': 2}, {'task_count': 0}, {'trials': 6}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                submit.benchmark_settings(self.arguments(**overrides))
        with self.assertRaises(ValueError):
            submit.benchmark_settings(self.arguments(mode='grpo_stress', gpu_model='3090'))

    def test_sft_benchmark_requires_explicit_checked_checkpoint(self):
        with TemporaryDirectory() as directory:
            data = Path(directory)
            args = self.arguments(mode='benchmark_sft_smoke', data=data)
            with self.assertRaisesRegex(ValueError, 'numeric --checkpoint-job-id'):
                submit.benchmark_settings(args)
            args.checkpoint_job_id = '19407'
            with self.assertRaisesRegex(ValueError, 'independent load PASS'):
                submit.benchmark_settings(args)
            checkpoint = data / 'runs/19407/sft-smoke/checkpoints/global_step_1/actor'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'checkpoint_check.json').write_text('{"status":"PASS"}', encoding='utf-8')
            settings = submit.benchmark_settings(args)
            self.assertEqual(Path(settings['checkpoint']), checkpoint.resolve())
            self.assertEqual(settings['model'], 'sft-smoke')

    def test_benchmark_gpu_type_does_not_bypass_budget(self):
        with patch.dict(submit.os.environ, {'USER': 'benchmark-test'}):
            with patch.object(submit.subprocess, 'check_output', side_effect=[
                '100\n', 'NumNodes=1 TresPerNode=gres/gpu:4090:2',
            ]) as run:
                with self.assertRaisesRegex(RuntimeError, 'BLOCKED_GPU_BUDGET'):
                    submit.submit_locked(self.arguments(gpu_model='3090'), None, [])
                self.assertEqual(run.call_count, 2)

    def test_benchmark_requests_selected_model_and_retains_default(self):
        for mode, selected, expected in (('benchmark_pt', '3090', '3090'), ('ppo', None, '4090')):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                data = Path(directory)
                args = self.arguments(mode=mode, gpu_model=selected, data=data, repo=data)
                benchmark = submit.benchmark_settings(args)
                extra = ['0', '10', '5'] if benchmark else []
                with patch.dict(submit.os.environ, {'USER': 'benchmark-test'}):
                    with patch.object(submit.subprocess, 'check_output', side_effect=['', '101']) as run:
                        submit.submit_locked(args, None, extra, benchmark)
                command = run.call_args.args[0]
                self.assertIn(f'--gres=gpu:{expected}:1', command)
                self.assertIn('--time=01:00:00', command)
                self.assertIn('--exclude=4090node1', command)
                receipt = json.loads((data / 'runs/submission-101.json').read_text())
                self.assertEqual(receipt['requested_gpu_model'], expected)
                self.assertEqual(receipt['benchmark'], benchmark)
