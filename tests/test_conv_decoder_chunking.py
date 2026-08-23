import types
import unittest

import torch

from networks import ConvDecoder


class ConvDecoderChunkingTest(unittest.TestCase):
    def test_chunked_decoder_matches_single_pass_and_gradients(self):
        torch.manual_seed(0)
        config = types.SimpleNamespace(
            act="SiLU", depth=2, mults=[2, 3], bspace=2, kernel_size=3, units=8, norm=True
        )
        decoder = ConvDecoder(config, deter=8, flat_stoch=6, shape=(3, 16, 16))

        stoch = torch.randn(2, 3, 6, requires_grad=True)
        deter = torch.randn(2, 3, 8, requires_grad=True)
        expected = decoder(stoch, deter)
        expected.square().mean().backward()
        expected_stoch_grad = stoch.grad.detach().clone()
        expected_deter_grad = deter.grad.detach().clone()
        expected_param_grads = {
            name: param.grad.detach().clone()
            for name, param in decoder.named_parameters()
            if param.grad is not None
        }

        decoder.zero_grad(set_to_none=True)
        stoch.grad = None
        deter.grad = None
        decoder._rows_per_chunk = 2  # Force the chunking path for this small test.
        actual = decoder(stoch, deter)
        actual.square().mean().backward()

        torch.testing.assert_close(actual, expected.detach())
        torch.testing.assert_close(stoch.grad, expected_stoch_grad)
        torch.testing.assert_close(deter.grad, expected_deter_grad)
        for name, expected_grad in expected_param_grads.items():
            torch.testing.assert_close(dict(decoder.named_parameters())[name].grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
