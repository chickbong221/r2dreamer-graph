import unittest
from types import SimpleNamespace

import torch

from rssm import RSSM


def config():
    return SimpleNamespace(
        stoch=2,
        deter=16,
        hidden=8,
        discrete=4,
        img_layers=2,
        obs_layers=1,
        dyn_layers=1,
        blocks=4,
        act="SiLU",
        unimix_ratio=0.01,
        initial="zeros",
        device="cpu",
        sem_layers=1,
    )


class SemanticRSSMTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch = 2
        self.time = 3
        self.embed = torch.randn(self.batch, self.time, 5)
        self.action = torch.randn(self.batch, self.time, 3)
        self.reset = torch.tensor([[True, False, False], [True, False, True]])

    def test_graph_free_interface_is_unchanged(self):
        model = RSSM(config(), embed_size=5, act_dim=3)
        initial = model.initial(self.batch)
        self.assertEqual(len(initial), 2)
        observed = model.observe(self.embed, self.action, initial, self.reset)
        self.assertEqual(len(observed), 3)
        stoch, deter, _ = observed
        self.assertEqual(model.get_feat(stoch, deter).shape[-1], 24)

    def test_semantic_posterior_prior_and_imagination(self):
        model = RSSM(config(), embed_size=5, act_dim=3, semantic=True,
                     graph_token_size=7, graph_dim=6)
        initial = model.initial(self.batch)
        token = torch.randn(self.batch, self.time, 7)
        observed = model.observe(self.embed, self.action, initial, self.reset, token)
        self.assertEqual(len(observed), 4)
        stoch, deter, logit, sem = observed
        # One flat deterministic g, not a categorical state.
        self.assertEqual(sem.shape, (self.batch, self.time, 6))
        self.assertEqual(model.get_feat(stoch, deter, sem).shape[-1], 30)
        _, prior_logit = model.prior(deter, sem)
        self.assertEqual(prior_logit.shape, logit.shape)
        prior_sem = model.semantic_prior_seq(deter)
        self.assertEqual(prior_sem.shape, sem.shape)
        dyn, rep = model.semantic_align_loss(sem, prior_sem)
        self.assertTrue(torch.isfinite(dyn).all() and torch.isfinite(rep).all())
        amp, prior_rms, post_rms = model.semantic_amplitude_loss(sem, prior_sem)
        self.assertEqual(amp.shape, (self.batch, self.time))
        self.assertEqual(prior_rms.shape, amp.shape)
        self.assertTrue(torch.isfinite(amp).all())
        imagined = model.img_step(stoch[:, 0], deter[:, 0], self.action[:, 0], sem[:, 0])
        self.assertEqual(len(imagined), 4)

    def test_rssm_matches_r2dreamer_amp_dtypes(self):
        cfg = config()
        cfg.device = "cuda"
        model = RSSM(cfg, embed_size=5, act_dim=3).cuda()
        stoch, deter = model.initial(self.batch)
        action = self.action[:, 0].cuda()
        embed = self.embed[:, 0].cuda()
        reset = self.reset[:, 0].cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            next_stoch, next_deter, logit = model.obs_step(
                stoch, deter, action, embed, reset
            )
        # OneHotDist and the recurrent bias/norm promote persistent state to
        # float32; GEMM outputs remain under FP16 autocast, as in r2dreamer.
        self.assertEqual(next_stoch.dtype, torch.float32)
        self.assertEqual(next_deter.dtype, torch.float32)
        self.assertEqual(logit.dtype, torch.float16)
        self.assertTrue(torch.isfinite(next_stoch).all())
        self.assertTrue(torch.isfinite(next_deter).all())
        self.assertTrue(torch.isfinite(logit).all())
        self.assertTrue(all(param.dtype == torch.float32 for param in model.parameters()))
        (next_deter.float().square().mean() + logit.float().square().mean()).backward()
        grads = [
            param.grad
            for module in (model._deter_net, model._obs_net)
            for param in module.parameters()
        ]
        self.assertTrue(all(grad is not None and torch.isfinite(grad).all() for grad in grads))


class SemanticAmplitudeLossTest(unittest.TestCase):
    """Scale alignment, and that it stays one-way.

    The directional terms compare RMS-normalised vectors, so they are exactly
    blind to the quantity this one measures -- which is also what makes it
    safe to add: it cannot re-weight anything the direction losses already
    say.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.model = RSSM(config(), embed_size=5, act_dim=3, semantic=True,
                          graph_token_size=7, graph_dim=6)
        self.post = torch.randn(2, 3, 6)
        self.prior = torch.randn(2, 3, 6)

    def test_it_is_zero_exactly_when_the_magnitudes_match(self):
        amp, prior_rms, post_rms = self.model.semantic_amplitude_loss(
            self.post, self.post.clone())
        torch.testing.assert_close(amp, torch.zeros_like(amp), atol=1e-6,
                                   rtol=0)
        torch.testing.assert_close(prior_rms, post_rms)

    def test_direction_alone_is_not_enough(self):
        """A prior at half the posterior's magnitude scores perfectly on the
        directional terms and is exactly what this term is for."""
        half = self.post * 0.5
        dyn, rep = self.model.semantic_align_loss(self.post, half)
        torch.testing.assert_close(dyn, torch.zeros_like(dyn), atol=1e-5,
                                   rtol=0)
        torch.testing.assert_close(rep, torch.zeros_like(rep), atol=1e-5,
                                   rtol=0)
        amp, prior_rms, post_rms = self.model.semantic_amplitude_loss(
            self.post, half)
        # Closed form: halving the magnitude costs (r/2 - r)^2 = r^2/4.
        torch.testing.assert_close(amp, post_rms.square() / 4,
                                   atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(prior_rms * 2, post_rms,
                                   atol=1e-5, rtol=1e-4)
        self.assertTrue((amp > 0).all())

    def test_only_the_prior_receives_a_gradient(self):
        """One-way on purpose: pulling the posterior towards the prior's scale
        is what would let both branches agree by shrinking together."""
        post = self.post.clone().requires_grad_(True)
        prior = self.prior.clone().requires_grad_(True)
        amp, _, _ = self.model.semantic_amplitude_loss(post, prior)
        amp.mean().backward()
        self.assertIsNone(post.grad)
        self.assertIsNotNone(prior.grad)
        self.assertTrue(torch.isfinite(prior.grad).all())
        self.assertTrue(prior.grad.abs().sum() > 0)

    def test_it_is_computed_in_float32_whatever_the_input_dtype(self):
        """A magnitude squared under bfloat16 is the one number here with no
        headroom to spare."""
        reduced = self.post.to(torch.bfloat16)
        amp, prior_rms, post_rms = self.model.semantic_amplitude_loss(
            reduced, reduced.clone())
        for tensor in (amp, prior_rms, post_rms):
            self.assertEqual(tensor.dtype, torch.float32)

    def test_masking_removes_terminal_frames_from_the_average(self):
        """The masked reduction the trainer applies: a frame whose graph token
        was zeroed carries no observation to align against, and must not pull
        the average towards zero either."""
        valid = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        prior = self.post.clone()
        prior[:, -1] *= 8.0  # only the masked frames disagree
        amp, _, _ = self.model.semantic_amplitude_loss(self.post, prior)
        masked = (amp * valid).sum() / valid.sum().clamp_min(1)
        self.assertLess(float(masked), 1e-6)
        self.assertGreater(float(amp.mean()), 1e-3)


if __name__ == "__main__":
    unittest.main()
