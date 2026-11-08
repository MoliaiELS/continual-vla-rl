"""Audit-observer unit tests; these do not prove FSDP, Ray or DCP execution.

Real Torch arithmetic exercises hashing and gradient/parameter observations.
The mocked worker boundaries below isolate the observer from GPU scheduling;
actual synchronization/checkpoint acceptance remains a Slurm integration test.
"""
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "Real Torch unavailable; use dedicated pinned runtime")
class AuditTests(unittest.TestCase):
    def module(self):
        from xvla_rlinf import audit_workers
        return audit_workers

    def actor(self):
        import torch
        from torch import nn
        module = self.module()
        actor = object.__new__(module.AuditedActor)
        actor.model = nn.Module()
        actor.model.core = nn.Module()
        actor.model.core.transformer = nn.Linear(2, 2)
        actor.model.value_head = nn.Linear(2, 1)
        actor.model.register_buffer("frozen_domain", torch.arange(3))
        actor.optimizer = torch.optim.SGD(actor.model.parameters(), lr=.1)
        actor.optimizer_steps = 0
        actor.critic_warmup_steps = 0
        actor._rank = 0
        return actor

    def test_digest_covers_buffers_scalar_dtypes_and_fsdp_prefix(self):
        import torch
        module = self.module()
        state = {"weight": torch.tensor([.1, .2]), "domain": torch.tensor([3]),
                 "mask": torch.tensor(True), "bf16": torch.ones(2, dtype=torch.bfloat16)}
        normal = module.state_digest(state)
        wrapped = module.state_digest({"_fsdp_wrapped_module." + k: v for k, v in state.items()})
        self.assertEqual(normal, wrapped)
        changed = {key: value.clone() for key, value in state.items()}
        changed["domain"][0] = 2
        self.assertNotEqual(normal, module.state_digest(changed))
        changed = {key: value.clone() for key, value in state.items()}
        changed["weight"][0] += .001
        self.assertNotEqual(normal, module.state_digest(changed))
        with self.assertRaises(TypeError):
            module.state_digest({"not_a_tensor": [1, 2]})

    def test_record_appends_independently_parseable_json_lines(self):
        module = self.module()
        with TemporaryDirectory() as folder, patch.dict(os.environ, {"XVLA_RUN_DIR": folder}):
            module.record("first", status="PASS")
            module.record("second", count=3)
            lines = (Path(folder) / ("audit-" + str(os.getpid()) + ".jsonl")).read_text().splitlines()
            self.assertEqual([json.loads(line)["event"] for line in lines], ["first", "second"])

    def test_unwrapped_removes_multiple_wrappers(self):
        module = self.module()
        core = object()
        self.assertIs(module.unwrapped(SimpleNamespace(module=SimpleNamespace(module=core))), core)

    def check_optimizer_observer(self, *, actor_loss):
        import torch
        module = self.module()
        actor = self.actor()
        x = torch.tensor([[.4, -.3], [.2, .7]])
        loss = actor.model.value_head(x).square().mean()
        if actor_loss:
            loss = loss + actor.model.core.transformer(x).square().mean()
        loss.backward()
        captured = []
        def base_step(worker):
            worker.optimizer_steps += 1
            worker.optimizer.step()
            return 1., [.1]
        with patch.object(module.EmbodiedFSDPActor, "optimizer_step", base_step), \
             patch.object(module, "record", lambda event, **kwargs: captured.append((event, kwargs))), \
             patch.object(torch.cuda, "max_memory_allocated", return_value=0):
            result = actor.optimizer_step()
        self.assertEqual(result, (1., [.1]))
        self.assertEqual(captured[0][0], "optimizer")
        evidence = captured[0][1]
        actor_norms = evidence["actor_transformer_gradient_norms"]
        self.assertTrue(actor_norms)
        self.assertTrue(all(name.startswith("core.transformer.") for name in actor_norms))
        deltas = {key: value for key, value in evidence["parameter_max_changes"].items()
                  if key.startswith("core.transformer.")}
        if actor_loss:
            self.assertTrue(any(value is not None and value > 0 for value in actor_norms.values()))
            self.assertTrue(any(value > 0 for value in deltas.values()))
        else:
            self.assertTrue(all(value is None for value in actor_norms.values()))
            self.assertTrue(all(value == 0 for value in deltas.values()))
            self.assertTrue(any(value > 0 for key, value in evidence["parameter_max_changes"].items()
                                if key.startswith("value_head.")))

    def test_optimizer_observer_sees_actor_gradient_and_actual_delta(self):
        self.check_optimizer_observer(actor_loss=True)

    def test_critic_only_update_does_not_report_actor_gradient_or_delta(self):
        self.check_optimizer_observer(actor_loss=False)

    def test_sync_audit_rejects_state_or_probe_disagreement(self):
        module = self.module()
        runner = object.__new__(module.AuditedRunner)
        runner.global_step = 1
        def handle(value):
            return SimpleNamespace(wait=lambda: [value])
        a = {"digest": "equal", "probe": {"raw": [[.1]], "actions": [[.2]], "value": [[.3]]}}
        b = json.loads(json.dumps(a))
        runner.actor = SimpleNamespace(audit_sync_probe=lambda: handle(a))
        runner.rollout = SimpleNamespace(audit_sync_probe=lambda: handle(b))
        captured = []
        with patch.object(module.EmbodiedRunner, "update_rollout_weights", lambda _: None), \
             patch.object(module, "record", lambda event, **kwargs: captured.append((event, kwargs))):
            runner.update_rollout_weights()
            self.assertEqual(captured[-1][1]["status"], "PASS")
            b["digest"] = "different"
            with self.assertRaises(RuntimeError):
                runner.update_rollout_weights()
            b["digest"] = "equal"
            b["probe"]["raw"][0][0] += .1
            with self.assertRaises(AssertionError):
                runner.update_rollout_weights()

    def test_checkpoint_observer_detects_failed_restore_after_real_corruption(self):
        import torch
        module = self.module()
        actor = self.actor()
        snapshots = {}
        captured = []
        actor.audit_sync_probe = lambda: {"digest": module.state_digest(actor.model.state_dict())}
        def save(worker, path, step):
            snapshots[path] = {key: value.clone() for key, value in worker.model.state_dict().items()}
        def load(worker, path):
            # The observer must actually have changed a parameter before load.
            self.assertNotEqual(module.state_digest(worker.model.state_dict()),
                                module.state_digest(snapshots[path]))
            worker.model.load_state_dict(snapshots[path], strict=True)
        with patch.object(module.EmbodiedFSDPActor, "save_checkpoint", save), \
             patch.object(module.EmbodiedFSDPActor, "load_checkpoint", load), \
             patch.object(module, "record", lambda event, **kwargs: captured.append((event, kwargs))):
            actor.save_checkpoint("unit-path", step=1)
            self.assertTrue(captured[-1][1]["corruption_restored"])
        with patch.object(module.EmbodiedFSDPActor, "save_checkpoint", save), \
             patch.object(module.EmbodiedFSDPActor, "load_checkpoint", lambda worker, path: None):
            with self.assertRaises(RuntimeError):
                actor.save_checkpoint("broken-load", step=2)


if __name__ == "__main__":
    unittest.main()
