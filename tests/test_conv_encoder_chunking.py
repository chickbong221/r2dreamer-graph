import types
import unittest

import torch

from networks import ConvEncoder


class ConvEncoderChunkingTest(unittest.TestCase):
    def test_size100m_mshab_crosses_int32_index_limit(self):
        config = types.SimpleNamespace(
            act="SiLU", depth=48, mults=[2, 3, 4, 4], kernel_size=5, norm=True
        )
        encoder = ConvEncoder(config, input_shape=(112, 112, 6))

        self.assertEqual(encoder._max_rows_without_chunk, 1783)
        self.assertEqual(encoder._rows_per_chunk, 1024)
        self.assertGreater(32 * 64, encoder._max_rows_without_chunk)

    def test_chunked_encoder_matches_single_pass_and_gradients(self):
        torch.manual_seed(0)
        config = types.SimpleNamespace(
            act="SiLU", depth=2, mults=[2, 3], kernel_size=3, norm=True
        )
        encoder = ConvEncoder(config, input_shape=(8, 8, 3))

        obs = torch.randn(2, 3, 8, 8, 3, requires_grad=True)
        expected = encoder(obs)
        expected.square().mean().backward()
        expected_obs_grad = obs.grad.detach().clone()
        expected_param_grads = {
            name: param.grad.detach().clone()
            for name, param in encoder.named_parameters()
            if param.grad is not None
        }

        encoder.zero_grad(set_to_none=True)
        obs.grad = None
        encoder._max_rows_without_chunk = 2
        encoder._rows_per_chunk = 2
        actual = encoder(obs)
        actual.square().mean().backward()

        torch.testing.assert_close(actual, expected.detach())
        torch.testing.assert_close(obs.grad, expected_obs_grad)
        for name, expected_grad in expected_param_grads.items():
            torch.testing.assert_close(
                dict(encoder.named_parameters())[name].grad, expected_grad
            )


if __name__ == "__main__":
    unittest.main()
