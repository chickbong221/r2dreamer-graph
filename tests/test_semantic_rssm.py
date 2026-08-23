import unittest
from types import SimpleNamespace

import torch

from rssm import RSSM, SLOT_META_UID


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
        model = RSSM(config(), embed_size=5, act_dim=3, semantic=True, graph_token_size=7)
        initial = model.initial(self.batch)
        token = torch.randn(self.batch, self.time, 7)
        observed = model.observe(self.embed, self.action, initial, self.reset, token)
        self.assertEqual(len(observed), 5)
        stoch, deter, logit, sem, sem_logit = observed
        self.assertEqual(sem.shape, (self.batch, self.time, 2, 3))
        self.assertEqual(model.get_feat(stoch, deter, sem).shape[-1], 30)
        _, prior_logit = model.prior(deter, sem)
        sem_prior = model.semantic_prior_logits(deter, sem, initial[2], self.reset)
        losses = model.semantic_kl_loss(sem_logit, sem_prior, 1.0)
        self.assertTrue(all(torch.isfinite(value).all() for value in losses))
        self.assertEqual(prior_logit.shape, logit.shape)
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


if __name__ == "__main__":
    unittest.main()
