"""Independent analytic checks for the extended X-VLA terminal policy.

These CPU tests do not prove model loading, simulation or online RL success.
Run with a torch-enabled environment: python -m unittest discover -s tests/xvla
"""

import math
import unittest

import torch
from torch import nn

from xvla_rlinf.sampler import (
    TerminalCleanGaussianSampler,
    clean_action_prediction,
    log_prob_per_action,
    strict_joint_logprob,
)


class ToySpace:
    dim_action = 20

    def preprocess(self, proprio, action):
        proprio, action = proprio.clone(), action.clone()
        proprio[..., [9, 19]] = 0
        action[..., [9, 19]] = 0
        return proprio, action


class ToyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(0.3))
        self.dropout = nn.Dropout(0.9)

    def forward(self, *, domain_id, action_with_noise, proprio, t, vlm_features):
        del domain_id
        # Full 20-D/whole-horizon dependence deliberately tests preserved z.
        global_noise = action_with_noise.mean(dim=(1, 2), keepdim=True)
        return (
            self.theta
            + 0.2 * action_with_noise
            + 0.1 * global_noise
            + self.dropout(vlm_features)[:, None, :]
            + proprio[:, None, :] * 0.01
            + t[:, None, None] * 0.05
        )


class ToyCore(nn.Module):
    num_actions = 3

    def __init__(self):
        super().__init__()
        self.action_space = ToySpace()
        self.transformer = ToyPredictor()

    def forward_vlm(self, input_ids, image_input, image_mask):
        del input_ids, image_input
        return {"vlm_features": image_mask[:, :1].expand(-1, 20).float() * 0.01}


def make_inputs(batch=2):
    return {
        "input_ids": torch.ones(batch, 4, dtype=torch.long),
        "image_input": torch.zeros(batch, 3, 3, 4, 4),
        "image_mask": torch.tensor([[1, 1, 0]]).expand(batch, -1),
        "domain_id": torch.full((batch,), 3, dtype=torch.long),
        "proprio": torch.zeros(batch, 20),
    }


class SamplerTests(unittest.TestCase):
    def test_density_matches_scalar_analytic_formula_all_ten_coordinates(self):
        raw = torch.arange(20, dtype=torch.float32).reshape(1, 2, 10) * 0.03
        mean = torch.full_like(raw, 0.2)
        std = torch.linspace(0.1, 1.0, 10)
        got = log_prob_per_action(raw, mean, std)
        expected = []
        for row in raw[0].tolist():
            total = 0.0
            for x, sigma in zip(row, std.tolist()):
                total += -((x - 0.2) ** 2) / (2 * sigma**2) - math.log(sigma * math.sqrt(2 * math.pi))
            expected.append(total)
        torch.testing.assert_close(got[0, :, 0], torch.tensor(expected), atol=2e-5, rtol=2e-6)
        # The gripper coordinate contributes; env-action first-7 truncation is wrong.
        shifted = raw.clone()
        shifted[..., 9] += 0.7
        self.assertFalse(torch.equal(got, log_prob_per_action(shifted, mean, std)))

    def test_density_mean_gradient_matches_analytic_score(self):
        mean = torch.full((1, 1, 10), 0.2, requires_grad=True)
        raw = torch.linspace(-0.5, 0.5, 10).reshape(1, 1, 10)
        std = torch.linspace(0.1, 1.0, 10)
        log_prob_per_action(raw, mean, std).sum().backward()
        expected = (raw - 0.2) / std.square()
        torch.testing.assert_close(mean.grad, expected, atol=2e-5, rtol=2e-6)

    def test_sample_recompute_ratio_one_with_same_auxiliary_noise(self):
        core, obs = ToyCore(), make_inputs()
        sampler = TerminalCleanGaussianSampler()
        z = torch.linspace(-1, 1, 120).reshape(2, 3, 20)
        eps = torch.linspace(-0.8, 0.9, 40).reshape(2, 2, 10)
        sample = sampler.sample(core, obs, initial_noise=z, terminal_noise=eps, execution_steps=2)
        result = sampler.recompute(core, obs, sample["initial_noise"], sample["raw_actions"])
        torch.testing.assert_close(result["logprobs"], sample["logprobs"], atol=0, rtol=0)
        torch.testing.assert_close(torch.exp(result["logprobs"] - sample["logprobs"]), torch.ones(2, 2, 1))
        self.assertEqual(sample["initial_noise"].shape, (2, 3, 20))
        self.assertEqual(sample["raw_actions"].shape, (2, 2, 10))
        self.assertEqual(sample["logprobs"].dtype, torch.float32)

    def test_clean_recurrence_matches_independent_reference_loop(self):
        core, obs = ToyCore(), make_inputs()
        z = torch.linspace(-1, 1, 120).reshape(2, 3, 20)
        got, _ = clean_action_prediction(core, obs, z, steps=4)
        # Independent closed tensor expression for the toy predictor, not a call
        # to the sampler or transformer; this detects accidental velocity Euler.
        expected = torch.zeros_like(z)
        for i in (4, 3, 2, 1):
            mixed = z * (i / 4) + expected * (1 - i / 4)
            mixed[..., [9, 19]] = 0
            expected = 0.3 + 0.2 * mixed + 0.1 * mixed.mean((1, 2), keepdim=True) + 0.01 + (i / 4) * 0.05
        torch.testing.assert_close(got, expected)
        self.assertTrue(got.requires_grad)
        self.assertFalse(core.training)

    def test_preserves_suffix_and_inactive_auxiliary_coupling(self):
        core, obs = ToyCore(), make_inputs(batch=1)
        z = torch.zeros(1, 3, 20)
        baseline, _ = clean_action_prediction(core, obs, z, steps=2)
        changed = z.clone()
        changed[0, 2, 12] = 10.0  # Unexecuted suffix AND inactive-arm coordinate.
        alternate, _ = clean_action_prediction(core, obs, changed, steps=2)
        self.assertFalse(torch.equal(baseline[:, :1, :10], alternate[:, :1, :10]))
        with self.assertRaises(ValueError):
            clean_action_prediction(core, obs, z[:, :1, :10], steps=2)

    def test_mask_then_sum_is_strict_joint_not_mean(self):
        factors = torch.tensor([[[-2.0], [-3.0], [-5.0]], [[-7.0], [-11.0], [-13.0]]], requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        got = strict_joint_logprob(factors, mask)
        torch.testing.assert_close(got, torch.tensor([[-5.0], [-7.0]]))
        got.sum().backward()
        torch.testing.assert_close(factors.grad, mask[..., None].float())
        torch.testing.assert_close(strict_joint_logprob(factors), torch.tensor([[-10.0], [-31.0]]))

    def test_reward_sign_changes_actor_gradient_and_update(self):
        obs = make_inputs(batch=1)
        core = ToyCore()
        sampler = TerminalCleanGaussianSampler()
        sample = sampler.sample(core, obs, initial_noise=torch.zeros(1, 3, 20), terminal_noise=torch.ones(1, 1, 10), execution_steps=1)
        for advantage in (1.0, -1.0):
            core.zero_grad(set_to_none=True)
            current = sampler.recompute(core, obs, sample["initial_noise"], sample["raw_actions"])
            ratio = torch.exp(strict_joint_logprob(current["logprobs"]) - strict_joint_logprob(sample["logprobs"]))
            loss = -(ratio * advantage).mean()
            loss.backward()
            grad = float(core.transformer.theta.grad)
            self.assertTrue(math.isfinite(grad))
            self.assertNotEqual(grad, 0.0)
            if advantage == 1.0:
                positive = grad
            else:
                self.assertAlmostEqual(grad, -positive, places=4)
        before = core.transformer.theta.detach().clone()
        torch.optim.SGD(core.parameters(), lr=1e-5, weight_decay=0).step()
        self.assertFalse(torch.equal(before, core.transformer.theta.detach()))

    def test_rejects_invalid_noise_history_and_shapes(self):
        for value in (0, -0.1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                TerminalCleanGaussianSampler(pos_std=value)
        with self.assertRaises(ValueError):
            TerminalCleanGaussianSampler(history_mode="native_autoregressive")
        with self.assertRaises(ValueError):
            strict_joint_logprob(torch.zeros(1, 2, 10))
        with self.assertRaises(ValueError):
            strict_joint_logprob(torch.zeros(1, 2, 1), torch.ones(1, 2))
        with self.assertRaises(ValueError):
            log_prob_per_action(torch.zeros(1, 2, 7), torch.zeros(1, 2, 7), torch.ones(10))


if __name__ == "__main__":
    unittest.main()
