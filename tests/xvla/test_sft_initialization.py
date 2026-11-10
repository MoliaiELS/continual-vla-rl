"""Check prompt-to-LoRA SFT lineage and PEFT row transfer."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from xvla_rlinf.sft_initialization import _copy_prompt_rows, _manifest_relationship


class ManifestLineageTests(unittest.TestCase):
    def test_accepts_same_manifest_and_verified_nested_extension(self):
        source = "a" * 64
        self.assertEqual(_manifest_relationship(source, None, source), "same_manifest")
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "manifest.json"
            payload = json.dumps({"included_manifest_sha256": source}).encode()
            manifest.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            self.assertEqual(_manifest_relationship(source, manifest, digest), "included_manifest")
            with self.assertRaises(ValueError):
                _manifest_relationship("b" * 64, manifest, digest)
            with self.assertRaises(RuntimeError):
                _manifest_relationship(source, manifest, "c" * 64)


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "peft", "timm")),
                     "Torch/PEFT/timm unavailable; run in the dedicated X-VLA environment")
class PromptWarmStartTests(unittest.TestCase):
    def test_transfers_only_active_rows_and_validates_before_mutating(self):
        from copy import deepcopy
        import torch
        from torch import nn

        from xvla_rlinf.parameters import configure_trainable
        from xvla_rlinf.vendor.xvla.transformer import SoftPromptedTransformer

        core = nn.Module()
        core.config = {}
        core.transformer = SoftPromptedTransformer(hidden_size=16, multi_modal_input_size=8, depth=1,
                                                    num_heads=4, num_domains=4, len_soft_prompts=2)
        prompt, lora = nn.Module(), nn.Module()
        prompt.core, _ = configure_trainable(deepcopy(core), "prompt_heads", domain_rows=[3])
        lora.core, _ = configure_trainable(core, "lora_heads", lora_rank=8, domain_rows=[3])
        with torch.no_grad():
            for parameter in prompt.parameters():
                if parameter.requires_grad:
                    parameter.fill_(0.25)

        source = prompt.state_dict()
        before = {key: value.clone() for key, value in lora.state_dict().items()}
        bad_key = "core.transformer.action_decoder.bias.selected_weight"
        with self.assertRaises(ValueError):
            _copy_prompt_rows(lora, {**source, bad_key: source[bad_key].double()})
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in lora.state_dict().items()))

        copied = _copy_prompt_rows(lora, source)
        self.assertEqual(len(copied), 5)
        changed = [key for key, value in lora.state_dict().items() if not torch.equal(value, before[key])]
        self.assertEqual(len(changed), 5)
        for key in changed:
            self.assertIn(".modules_to_save.default.", key)
            self.assertTrue(key.endswith(".selected_weight"))
            self.assertTrue(torch.all(lora.state_dict()[key] == 0.25))
