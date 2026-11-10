"""Guard the SFT-cont budget and exact shared checkpoint/data lineage."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from xvla_rlinf.sft_continuation import read_continuation_source, validate_sft_cont_config


class SFTContinuationTests(unittest.TestCase):
    def setUp(self):
        self.source = Path("runs/19525/actor").resolve()
        self.prompt = Path("runs/19518/actor").resolve()
        self.manifest = "a" * 64
        self.model_cfg = {"xvla": {"update_scope": "lora_heads", "lora_rank": 8},
                          "add_value_head": False}

    def mock_sources(self, *, steps=1024):
        prompt_evidence = {"digest": "b" * 64}
        warm_start = {
            "source_checkpoint": str(self.prompt),
            "source_sft_job_id": "19518",
            "source_state_sha256": prompt_evidence["digest"],
            "source_optimizer_steps": 1024,
            "source_manifest_sha256": self.manifest,
            "manifest_sha256": self.manifest,
            "lora_rank": 8,
            "lora_alpha": 16,
        }
        source_evidence = {"digest": "c" * 64, "prompt_warm_start": warm_start}
        return [
            (source_evidence, self.source / "full_weights.pt",
             {"identity": {"slurm_job_id": "19525"}},
             {"optimizer_steps": steps, "manifest_sha256": self.manifest}),
            (prompt_evidence, self.prompt / "full_weights.pt",
             {"identity": {"slurm_job_id": "19518"}},
             {"optimizer_steps": 1024, "manifest_sha256": self.manifest}),
        ]

    def test_derives_budget_from_both_audited_stages(self):
        with patch("xvla_rlinf.sft_continuation.read_sft_source",
                   side_effect=self.mock_sources()) as reader:
            weights, lineage = read_continuation_source(self.source, self.model_cfg, self.manifest)
        self.assertEqual(weights, self.source / "full_weights.pt")
        self.assertEqual(lineage["source_optimizer_steps_by_stage"], [1024, 1024])
        self.assertEqual(lineage["new_optimizer_step_budget"], 2048)
        prompt_args = reader.call_args_list[-1].args
        self.assertEqual(prompt_args[0], self.prompt)
        self.assertEqual(prompt_args[1]["xvla"]["update_scope"], "prompt_heads")

    def test_rejects_changed_manifest_and_prior_budget(self):
        with patch("xvla_rlinf.sft_continuation.read_sft_source", side_effect=self.mock_sources()):
            with self.assertRaisesRegex(ValueError, "same manifest"):
                read_continuation_source(self.source, self.model_cfg, "d" * 64)
        with patch("xvla_rlinf.sft_continuation.read_sft_source",
                   side_effect=self.mock_sources(steps=256)):
            with self.assertRaisesRegex(ValueError, "source updates"):
                read_continuation_source(self.source, self.model_cfg, self.manifest)

    def test_requires_fresh_optimizer_and_no_early_stop(self):
        class Options(dict):
            __getattr__ = dict.__getitem__

        runner = Options(sft_cont_checkpoint=str(self.source), max_steps=2048, max_epochs=-1,
                         save_interval=512, val_check_interval=-1)
        actor = SimpleNamespace(model=self.model_cfg, optim=Options(total_training_steps=2048),
                                global_batch_size=4,
                                fsdp_config=SimpleNamespace(grad_scaler=SimpleNamespace(enabled=False)))
        cfg = SimpleNamespace(runner=runner, actor=actor, data={
            "manifest_requirements": {"demonstrations": 10, "selection_seed": 1234,
                                      "task_suite": "libero_spatial", "task_id": 0},
        })
        self.assertTrue(validate_sft_cont_config(cfg))
        runner["resume_dir"] = str(self.source)
        with self.assertRaisesRegex(ValueError, "fresh optimizer"):
            validate_sft_cont_config(cfg)
        del runner["resume_dir"]
        runner["early_stop"] = {"enabled": True}
        with self.assertRaisesRegex(ValueError, "fixed 2048-update budget"):
            validate_sft_cont_config(cfg)
