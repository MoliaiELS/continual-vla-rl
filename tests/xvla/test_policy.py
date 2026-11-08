"""Small real-Torch policy contract tests, not real VLA/robot acceptance.

The tiny core makes numerical, cache and gradient mistakes observable without
loading 0.9B weights. Full-model and simulation checks remain separate evidence.
"""
import importlib.util
import unittest
from types import SimpleNamespace

HAS_TORCH = importlib.util.find_spec("torch") is not None


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


@unittest.skipUnless(HAS_TORCH, "Real Torch unavailable; run in dedicated pinned runtime")
class PolicyTests(unittest.TestCase):
    def make_policy(self, *, value=True, chunk=3, scope="prompt_heads"):
        import torch
        from torch import nn
        from xvla_rlinf.policy import XVLAPolicy
        from xvla_rlinf.vendor.xvla.action_hub import EE6DActionSpace
        from xvla_rlinf.vendor.xvla.transformer import DomainAwareLinear

        class TinyTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.soft_prompt_hub = nn.Embedding(4, 20)
                self.action_encoder = DomainAwareLinear(45, 20, num_domains=4)
                self.action_decoder = DomainAwareLinear(20, 20, num_domains=4)
                identity = torch.zeros(20)
                identity[[3, 7]] = 1
                self.register_buffer("identity", identity)

            def forward(self, domain_id, action_with_noise, proprio, t, vlm_features, aux_visual_inputs):
                batch, horizon = action_with_noise.shape[:2]
                state = proprio[:, None].expand(-1, horizon, -1)
                vision = vlm_features.mean(1)[:, None].expand(-1, horizon, -1)
                time = t[:, None, None].expand(batch, horizon, 1)
                features = torch.cat((action_with_noise, state, time, vision), dim=-1)
                hidden = self.action_encoder(features, domain_id) + self.soft_prompt_hub(domain_id)[:, None]
                return self.identity + self.action_decoder(hidden.tanh(), domain_id) * .02

        class Core(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_actions = 3
                self.config = SimpleNamespace(florence_config=SimpleNamespace(projection_dim=4))
                self.vlm = nn.Linear(4, 4)
                self.transformer = TinyTransformer()
                self.action_space = EE6DActionSpace()

            def forward_vlm(self, input_ids, image_input, image_mask):
                batch = input_ids.shape[0]
                feature = image_input.float().mean(dim=(1, 2, 3, 4))[:, None, None].expand(batch, 2, 4)
                return {"vlm_features": self.vlm(feature), "aux_visual_inputs": feature[:, :1]}

            def forward(self, input_ids, image_input, image_mask, domain_id, proprio, action):
                enc = self.forward_vlm(input_ids, image_input, image_mask)
                t = (torch.rand(1) + torch.arange(len(action)) / len(action)) % (1 - 1e-5)
                noisy = torch.randn_like(action) * t[:, None, None] + action * (1 - t[:, None, None])
                p, x = self.action_space.preprocess(proprio, noisy)
                prediction = self.transformer(domain_id, x, p, t, **enc)
                return self.action_space.compute_loss(prediction, action)

        class Processor:
            def __call__(self, images, language_instruction):
                import numpy as np
                arrays = []
                masks = []
                for sample in images:
                    tensors = [torch.from_numpy(np.asarray(i).copy()).permute(2, 0, 1).float() for i in sample]
                    masks.append([True] * len(tensors) + [False] * (3 - len(tensors)))
                    tensors += [torch.zeros_like(tensors[0])] * (3 - len(tensors))
                    arrays.append(torch.stack(tensors))
                return {"input_ids": torch.ones(len(images), 4, dtype=torch.long),
                        "image_input": torch.stack(arrays), "image_mask": torch.tensor(masks)}

        cfg = Config(num_action_chunks=chunk, add_value_head=value,
                     xvla=Config(update_scope=scope, domain_rows=[3], domain_id=3, steps=2,
                                 image_orientation="native_raw", rotation_backend="numpy",
                                 eval_sampler="native_eval", audit_gradients=False))
        return XVLAPolicy(Core(), Processor(), cfg)

    @staticmethod
    def obs(batch=2):
        import numpy as np
        state = np.zeros((batch, 20), np.float32)
        state[:, :3] = [.3, .1, .8]
        state[:, 3:9] = [1, 0, 0, 0, 1, 0]
        return {"main_images": np.zeros((batch, 4, 4, 3), np.uint8),
                "wrist_images": np.ones((batch, 4, 4, 3), np.uint8),
                "states": state, "task_descriptions": ["put the bowl on the plate"] * batch,
                "reset_mask": np.ones(batch, bool)}

    def test_rollout_cache_recompute_ratio_and_execution_mask(self):
        import torch
        policy = self.make_policy(chunk=2)
        actions, result = policy.predict_action_batch(self.obs(), mode="train",
            initial_noise=torch.randn(2, 3, 20), terminal_noise=torch.randn(2, 2, 10))
        self.assertEqual(actions.shape, (2, 2, 7))
        cache = result["forward_inputs"]
        self.assertEqual(cache["initial_noise"].shape, (2, 3, 20))
        self.assertEqual(cache["raw_actions"].shape, (2, 2, 10))
        before = policy(forward_inputs=cache, compute_values=True)
        torch.testing.assert_close(before["logprobs"], result["prev_logprobs"], atol=1e-5, rtol=1e-6)
        torch.testing.assert_close(before["values"], result["prev_values"].squeeze(-1))
        mask = torch.tensor([[True, False], [True, True]])
        cache["executed_mask"] = mask
        masked = policy(forward_inputs=cache, compute_values=True)
        old = result["prev_logprobs"].masked_fill(~mask[..., None], 0)
        torch.testing.assert_close(masked["logprobs"], old, atol=1e-5, rtol=1e-6)
        self.assertEqual(float(masked["logprobs"][0, 1]), 0)
        torch.testing.assert_close((masked["logprobs"] - old).exp(), torch.ones_like(old), atol=1e-5, rtol=1e-5)

    def test_actor_loss_has_nonzero_core_gradient_without_critic_loss(self):
        import torch
        policy = self.make_policy(value=True)
        _, sampled = policy.predict_action_batch(self.obs(), mode="train")
        new = policy(forward_inputs=sampled["forward_inputs"], compute_values=True)
        advantage = torch.tensor([1., -1.])[:, None, None]
        ratio = (new["logprobs"] - sampled["prev_logprobs"]).exp()
        loss = -(ratio * advantage).mean()  # no critic term and no weight decay
        loss.backward()
        core_grad = [p.grad for p in policy.core.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(any(bool(g.count_nonzero()) for g in core_grad))
        self.assertTrue(all(p.grad is None for p in policy.value_head.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.core.vlm.parameters()))

    def test_actor_microbatch_recompute_uses_cached_history(self):
        import torch
        policy = self.make_policy()
        _, sampled = policy.predict_action_batch(self.obs(), mode="train")
        # Rollout B=2 vs actor micro B=1; no resampling or measured-state lookup.
        for index in range(2):
            cache = {key: value[index:index+1] for key, value in sampled["forward_inputs"].items()}
            out = policy(forward_inputs=cache, compute_values=True)
            torch.testing.assert_close(out["logprobs"], sampled["prev_logprobs"][index:index+1], atol=2e-5, rtol=1e-6)
            torch.testing.assert_close(out["values"], sampled["prev_values"][index:index+1].squeeze(-1))

    def test_trainable_manifest_includes_value_head(self):
        policy = self.make_policy(value=True)
        expected = {name for name, parameter in policy.named_parameters() if parameter.requires_grad}
        actual = {record["name"] for record in policy.trainable_manifest["parameters"]}
        self.assertEqual(actual, expected)
        self.assertTrue(any(name.startswith("value_head.") for name in actual))

    def test_observation_conditioned_values_ignore_sample_noise(self):
        import torch
        policy = self.make_policy()
        inputs = self.obs()
        _, a = policy.predict_action_batch(inputs, mode="train", initial_noise=torch.zeros(2, 3, 20))
        _, b = policy.predict_action_batch(inputs, mode="train", initial_noise=torch.ones(2, 3, 20))
        torch.testing.assert_close(a["prev_values"], b["prev_values"])

    def test_native_history_caches_the_used_proprio_and_resets_one_env(self):
        import numpy as np
        import torch
        policy = self.make_policy(value=False)
        obs = self.obs()
        noise = torch.randn(2, 3, 20)
        _, a = policy.predict_action_batch(obs, mode="eval", initial_noise=noise)
        previous = policy.history.prepare(obs["states"]).copy()
        obs["states"][:, :3] += 10
        obs["reset_mask"] = np.array([False, True])
        _, b = policy.predict_action_batch(obs, mode="eval", initial_noise=noise)
        used = b["forward_inputs"]["proprio"].cpu().numpy()
        np.testing.assert_array_equal(used[0], previous[0])
        np.testing.assert_array_equal(used[1], obs["states"][1])
        self.assertIsNone(a["prev_logprobs"])

    def test_native_rejects_short_execution(self):
        policy = self.make_policy(value=False, chunk=2)
        with self.assertRaises(ValueError):
            policy.predict_action_batch(self.obs(), mode="eval")

    def test_sft_uses_clean_action_loss_and_trainable_scope(self):
        import torch
        from rlinf.models.embodiment.base_policy import ForwardType
        policy = self.make_policy(value=False)
        inputs = policy._inputs(self.obs(), native_history=False)
        target = torch.zeros(2, 3, 20); target[..., [3, 7]] = 1
        inputs["action"] = target
        result = policy(ForwardType.SFT, data=inputs)
        self.assertEqual(set(result), {"loss", "position_loss", "rotate6D_loss", "gripper_loss"})
        result["loss"].backward()
        self.assertTrue(any(p.grad is not None for p in policy.core.parameters() if p.requires_grad))
        self.assertTrue(all(p.grad is None for p in policy.core.vlm.parameters()))

    def test_checkpoint_and_sync_complete_policy_state(self):
        import torch
        a = self.make_policy()
        b = self.make_policy()
        b.load_state_dict(a.state_dict(), strict=True)
        initial = torch.randn(2, 3, 20)
        terminal = torch.randn(2, 3, 10)
        a_action, a_info = a.predict_action_batch(self.obs(), mode="train", initial_noise=initial, terminal_noise=terminal)
        b_action, b_info = b.predict_action_batch(self.obs(), mode="train", initial_noise=initial, terminal_noise=terminal)
        import numpy as np
        np.testing.assert_array_equal(a_action, b_action)
        torch.testing.assert_close(a_info["prev_logprobs"], b_info["prev_logprobs"], rtol=0, atol=0)
        torch.testing.assert_close(a_info["prev_values"], b_info["prev_values"], rtol=0, atol=0)

    def test_explicit_mode_and_unsupported_branches_fail(self):
        from rlinf.models.embodiment.base_policy import ForwardType
        policy = self.make_policy()
        with self.assertRaises(ValueError):
            policy.predict_action_batch(self.obs())
        with self.assertRaises(NotImplementedError):
            policy(ForwardType.SAC)


if __name__ == "__main__":
    unittest.main()
