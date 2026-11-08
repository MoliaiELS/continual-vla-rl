"""Real Torch/PEFT parameter tests, skipped explicitly where unavailable."""
import importlib.util
import unittest

from xvla_rlinf.parameters import configure_trainable, gradient_report, trainable_report

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_PEFT = importlib.util.find_spec("peft") is not None


@unittest.skipUnless(HAS_TORCH, "Torch unavailable; run in the dedicated X-VLA environment")
class ParameterScopeTests(unittest.TestCase):
    def make_model(self):
        import torch.nn as nn
        from xvla_rlinf.vendor.xvla.transformer import SoftPromptedTransformer
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = {}
                self.vlm = nn.Linear(8, 8)
                self.transformer = SoftPromptedTransformer(hidden_size=16, multi_modal_input_size=8, depth=1,
                    num_heads=4, num_domains=4, len_soft_prompts=2, max_len_seq=50)
        return Model()

    def test_full_scope(self):
        model, report = configure_trainable(self.make_model(), "full")
        self.assertEqual(report["total_parameters"], report["trainable_parameters"])
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_prompt_heads_covers_embedding_encoder_decoder(self):
        model, _ = configure_trainable(self.make_model(), "prompt_heads")
        names = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertIn("transformer.action_encoder.fc.weight", names)
        self.assertIn("transformer.action_decoder.bias.weight", names)
        self.assertIn("transformer.soft_prompt_hub.weight", names)
        self.assertNotIn("vlm.weight", names)
        self.assertNotIn("transformer.blocks.0.attn.qkv.weight", names)

    def test_row_restriction_resists_adamw_decay_and_momentum(self):
        import torch
        model, report = configure_trainable(self.make_model(), "prompt_heads", domain_rows=[3])
        embedding = model.transformer.soft_prompt_hub
        before = embedding.weight.detach().clone()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.1, weight_decay=.5)
        for _ in range(3):
            opt.zero_grad()
            loss = embedding(torch.tensor([3])).square().sum()
            loss.backward()
            opt.step()
        after = embedding.weight.detach()
        self.assertTrue(torch.equal(before[:3], after[:3]))
        self.assertFalse(torch.equal(before[3], after[3]))
        self.assertEqual(report["row_protection"], "separate_parameter_and_frozen_buffer")
        self.assertFalse(model.transformer.soft_prompt_hub.frozen_weight.requires_grad)

    def test_checkpoint_reload_same_row_structure(self):
        import torch
        model, _ = configure_trainable(self.make_model(), "prompt_heads", domain_rows=[3])
        target, _ = configure_trainable(self.make_model(), "prompt_heads", domain_rows=[3])
        target.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(torch.equal(target.transformer.action_encoder.fc.weight, model.transformer.action_encoder.fc.weight))

    @unittest.skipUnless(HAS_PEFT, "PEFT unavailable; actual LoRA test NOT RUN")
    def test_lora_plus_saved_heads(self):
        model, _ = configure_trainable(self.make_model(), "lora_heads", lora_rank=2)
        names = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(any("lora_A" in n for n in names))
        self.assertTrue(any("action_encoder" in n and "modules_to_save" in n and "fc.weight" in n for n in names))
        self.assertTrue(any("soft_prompt_hub" in n and "modules_to_save" in n for n in names))

    def test_invalid_scope_and_rows_fail(self):
        with self.assertRaises(ValueError):
            configure_trainable(self.make_model(), "unimplemented")
        with self.assertRaises(ValueError):
            configure_trainable(self.make_model(), "prompt_heads", domain_rows=[40])

    def test_official_clean_action_loss_weights(self):
        import torch
        from xvla_rlinf.vendor.xvla.action_hub import EE6DActionSpace
        action_space = EE6DActionSpace()
        target, pred = torch.zeros(1, 2, 20), torch.zeros(1, 2, 20, requires_grad=True)
        losses = action_space.compute_loss(pred, target)
        self.assertEqual(set(losses), {"position_loss", "rotate6D_loss", "gripper_loss"})
        self.assertAlmostEqual(float(losses["gripper_loss"]), .69314718, places=6)
        sum(losses.values()).backward()
        self.assertTrue(bool(pred.grad[..., [9, 19]].abs().sum() > 0))
        changed = pred.detach().clone(); changed[..., 0] = 1
        self.assertAlmostEqual(float(action_space.compute_loss(changed, target)["position_loss"]), 500 / 3, places=4)


if __name__ == "__main__":
    unittest.main()
