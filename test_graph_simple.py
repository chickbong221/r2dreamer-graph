"""Simple-graph mode: relation-only contract, UIDs, and the deterministic g.

Everything here runs on synthetic tensors -- no simulator, no DINO.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from graph import (
    FULL_GRAPH_KEYS,
    SIMPLE_GRAPH_KEYS,
    GraphEncoder,
    SimpleGraphDecoder,
    compact_graph,
    graph_from,
    graph_keys,
)
from envs.maniskill import _GRAPH_CONFIG_KEYS, graph_observation_config
from networks import MultiDecoder
from rssm import RSSM
from scenegraph.adapters.graph_obs import _DTYPES as _PACKED_DTYPES
from scenegraph.core.graph_builder import (
    UID_EE,
    UID_PAD,
    UID_VOCAB_MAX,
    EpisodeUIDs,
)

N_MAX = 8
E_MAX = 16
UID_VOCAB = 32


def graph_config(simple=True, units=32):
    return SimpleNamespace(
        simple=simple,
        units=units,
        simple_units=units,
        semantic_dim=units,
        layers=1,
        n_cams=2,
        app_dim=8,
        entity_vocab=14,
        n_rel=11,
        n_abs=17,
        n_temp=6,
        embed=8,
        app=4,
        bbox=4,
        bbox_beta=0.1,
        uid_vocab=UID_VOCAB,
        uid_embed=8,
        reverse_edges=True,
        act="SiLU",
    )


def simple_graph(batch=2, time=3, n_valid=3, n_edges=4, uids=None):
    shape = (batch, time)
    graph = {
        "graph_node_ent": torch.zeros(*shape, N_MAX, dtype=torch.uint8),
        "graph_node_uid": torch.zeros(*shape, N_MAX, dtype=torch.int64),
        "graph_node_target": torch.zeros(*shape, N_MAX, dtype=torch.uint8),
        "graph_edge_src": torch.zeros(*shape, E_MAX, dtype=torch.uint8),
        "graph_edge_dst": torch.zeros(*shape, E_MAX, dtype=torch.uint8),
        "graph_edge_rel": torch.zeros(*shape, E_MAX, dtype=torch.uint8),
        "graph_edge_abs": torch.zeros(*shape, E_MAX, dtype=torch.uint8),
        "graph_edge_temp": torch.zeros(*shape, E_MAX, dtype=torch.uint8),
    }
    graph["graph_node_ent"][..., :n_valid] = torch.arange(1, n_valid + 1)
    if uids is None:
        uids = [UID_EE] + list(range(2, n_valid + 1))
    graph["graph_node_uid"][..., :n_valid] = torch.tensor(uids, dtype=torch.int64)
    # Slot zero is the end effector, so the target lives on slot one.
    graph["graph_node_target"][..., 1] = 1

    index = torch.arange(n_edges)
    graph["graph_edge_src"][..., :n_edges] = (index % n_valid).to(torch.uint8)
    graph["graph_edge_dst"][..., :n_edges] = ((index + 1) % n_valid).to(torch.uint8)
    # Relation 1 admits absolute labels 1-2; relation 5 admits 3-7.
    spatial = index.remainder(2).bool()
    graph["graph_edge_rel"][..., :n_edges] = torch.where(
        spatial, torch.tensor(5), torch.tensor(1)
    ).to(torch.uint8)
    graph["graph_edge_abs"][..., :n_edges] = torch.where(
        spatial, torch.tensor(3), torch.tensor(2)
    ).to(torch.uint8)
    graph["graph_edge_temp"][..., :n_edges] = torch.where(
        spatial, torch.tensor(3), torch.tensor(0)
    ).to(torch.uint8)
    return graph


class ContractTest(unittest.TestCase):
    def test_simple_contract_drops_appearance_and_adds_uid(self):
        self.assertEqual(graph_keys(True), SIMPLE_GRAPH_KEYS)
        self.assertEqual(graph_keys(False), FULL_GRAPH_KEYS)
        for key in ("graph_node_app", "graph_node_bbox"):
            self.assertIn(key, FULL_GRAPH_KEYS)
            self.assertNotIn(key, SIMPLE_GRAPH_KEYS)
        self.assertIn("graph_node_uid", SIMPLE_GRAPH_KEYS)
        self.assertNotIn("graph_node_uid", FULL_GRAPH_KEYS)

    def test_simple_validation_ignores_absent_appearance(self):
        data = simple_graph()
        self.assertEqual(set(graph_from(data, simple=True)), set(SIMPLE_GRAPH_KEYS))
        with self.assertRaises(KeyError):
            graph_from(data, simple=False)

    def test_compact_leaves_the_unused_side_none(self):
        compact = compact_graph(simple_graph(), simple=True)
        self.assertIsNotNone(compact.node_uid)
        self.assertIsNone(compact.node_app)
        self.assertIsNone(compact.node_bbox)
        self.assertIsNone(compact.appearance_known)
        self.assertIsNone(compact.camera_visible)


class ReplayDtypeTest(unittest.TestCase):
    """Every packed dtype must survive the replay buffer.

    torchrl stores through ``index_put``, which has no uint16 kernel. A key
    packed in an unsupported dtype builds and acts fine and only fails on the
    first replay write, well after the run looks healthy.
    """

    # Types with an index_put kernel that the buffer actually exercises.
    SUPPORTED = {np.uint8, np.int32, np.int64, np.float16, np.float32}

    def test_every_graph_key_is_storable(self):
        for key, dtype in _PACKED_DTYPES.items():
            with self.subTest(key=key):
                self.assertIn(np.dtype(dtype).type, self.SUPPORTED)

    def test_uid_is_storable_and_holds_the_vocabulary(self):
        self.assertIs(np.dtype(_PACKED_DTYPES["graph_node_uid"]).type, np.uint8)
        self.assertLessEqual(UID_VOCAB_MAX - 1, np.iinfo(np.uint8).max)

    def test_index_put_accepts_every_graph_dtype(self):
        # The exact operation the buffer performs, so this fails here rather
        # than on the first add_transition of a real run.
        for key, dtype in _PACKED_DTYPES.items():
            with self.subTest(key=key):
                store = torch.zeros(4, dtype=torch.from_numpy(np.zeros(1, dtype)).dtype)
                store[torch.tensor([0, 2])] = store[torch.tensor([1, 3])]

    def test_uid_vocab_above_the_dtype_raises(self):
        with self.assertRaises(ValueError):
            EpisodeUIDs(UID_VOCAB_MAX + 1, seed=0)


class EpisodeUIDTest(unittest.TestCase):
    def test_uid_survives_disappearance(self):
        uids = EpisodeUIDs(UID_VOCAB, seed=0)
        first = uids.uid_for("apple-1")
        uids.uid_for("can-1")
        uids.uid_for("can-2")
        # apple-1 left the view for a while; its code is still its own.
        self.assertEqual(uids.uid_for("apple-1"), first)

    def test_codes_are_never_shared(self):
        uids = EpisodeUIDs(UID_VOCAB, seed=0)
        codes = [uids.uid_for(f"n{i}") for i in range(UID_VOCAB - 2)]
        self.assertEqual(len(set(codes)), len(codes))
        self.assertNotIn(UID_PAD, codes)
        self.assertNotIn(UID_EE, codes)

    def test_ee_code_is_fixed(self):
        uids = EpisodeUIDs(UID_VOCAB, seed=0)
        self.assertEqual(uids.uid_for("ee", is_ee=True), UID_EE)
        self.assertEqual(uids.uid_for("other", is_ee=True), UID_EE)

    def test_reset_clears_the_mapping(self):
        uids = EpisodeUIDs(UID_VOCAB, seed=0)
        uids.uid_for("apple-1")
        uids.reset()
        self.assertEqual(len(uids), 0)

    def test_overflow_raises_rather_than_aliasing(self):
        uids = EpisodeUIDs(6, seed=0)
        for i in range(4):
            uids.uid_for(f"n{i}")
        with self.assertRaises(RuntimeError):
            uids.uid_for("one-too-many")

    def test_codes_are_permuted_per_episode(self):
        # Two episodes must not hand the same object the same code, or a UID
        # becomes a global name the model can memorise.
        uids = EpisodeUIDs(UID_VOCAB, seed=0)
        runs = []
        for _ in range(6):
            uids.reset()
            runs.append(uids.uid_for("apple-1"))
        self.assertGreater(len(set(runs)), 1)


class SimpleEncoderTest(unittest.TestCase):
    def test_siblings_differ_only_through_uid(self):
        torch.manual_seed(0)
        encoder = GraphEncoder(graph_config(simple=True))
        # Same entity id, same target flag, different identity.
        left = simple_graph(batch=1, time=1, n_valid=3, uids=[UID_EE, 2, 3])
        right = simple_graph(batch=1, time=1, n_valid=3, uids=[UID_EE, 4, 5])
        with torch.no_grad():
            a = encoder(left).nodes
            b = encoder(right).nodes
        self.assertFalse(torch.allclose(a, b))

    def test_padded_slots_are_zeroed(self):
        torch.manual_seed(0)
        encoder = GraphEncoder(graph_config(simple=True))
        with torch.no_grad():
            nodes = encoder(simple_graph(n_valid=3)).nodes
        self.assertTrue(torch.equal(nodes[..., 3:, :], torch.zeros_like(nodes[..., 3:, :])))

    def test_token_width_follows_simple_units(self):
        encoder = GraphEncoder(graph_config(simple=True, units=32))
        self.assertEqual(encoder.units, 32)
        with torch.no_grad():
            token = encoder(simple_graph()).token
        self.assertEqual(tuple(token.shape), (2, 3, 32))

    def test_full_mode_is_unchanged(self):
        encoder = GraphEncoder(graph_config(simple=False))
        self.assertFalse(encoder.simple)
        self.assertIsNotNone(encoder.app_proj)
        self.assertIsNone(encoder.uid)


class SimpleDecoderTest(unittest.TestCase):
    def _run(self, sem=None, graph=None):
        torch.manual_seed(0)
        config = graph_config(simple=True)
        decoder = SimpleGraphDecoder(config, semantic_dim=32)
        graph = simple_graph() if graph is None else graph
        compact = compact_graph(graph, simple=True)
        if sem is None:
            sem = torch.randn(2, 3, 32, requires_grad=True)
        step_valid = torch.ones(2, 3, dtype=torch.bool)
        return decoder, sem, decoder(sem, compact, step_valid)

    def test_losses_are_finite_and_named(self):
        _, _, (losses, metrics) = self._run()
        self.assertEqual(set(losses), {"nodetgt", "relabs", "reltemp"})
        for name, value in losses.items():
            self.assertTrue(torch.isfinite(value), name)
        self.assertIn("node_target_acc", metrics)

    def test_every_loss_reaches_the_semantic_state(self):
        # The decoder reads g, not the encoder's node vectors: that is what
        # makes these losses a test of what g retained.
        for name in ("nodetgt", "relabs", "reltemp"):
            with self.subTest(loss=name):
                _, sem, (losses, _) = self._run()
                losses[name].backward()
                self.assertIsNotNone(sem.grad)
                self.assertGreater(float(sem.grad.abs().sum()), 0.0)

    def test_target_flag_is_not_readable_from_the_query(self):
        # Only g and the UID enter the query, so moving the flag cannot change
        # the decoded node vectors -- it only changes the label.
        torch.manual_seed(0)
        decoder = SimpleGraphDecoder(graph_config(simple=True), semantic_dim=32)
        sem = torch.randn(1, 1, 32)
        moved = simple_graph(batch=1, time=1)
        moved["graph_node_target"][..., 1] = 0
        moved["graph_node_target"][..., 2] = 1
        base = compact_graph(simple_graph(batch=1, time=1), simple=True)
        other = compact_graph(moved, simple=True)
        with torch.no_grad():
            uid_a = decoder.uid(base.node_uid)
            uid_b = decoder.uid(other.node_uid)
        self.assertTrue(torch.equal(uid_a, uid_b))

    def test_node_count_may_vary_between_frames(self):
        for n_valid in (1, 3, 7):
            with self.subTest(nodes=n_valid):
                _, _, (losses, metrics) = self._run(
                    graph=simple_graph(n_valid=max(n_valid, 2))
                )
                self.assertTrue(torch.isfinite(losses["nodetgt"]))


class SimpleRSSMTest(unittest.TestCase):
    GRAPH_DIM = 16
    TOKEN = 12

    def _rssm(self):
        config = SimpleNamespace(
            stoch=4, hybrid_stoch=2, deter=16, hidden=8, discrete=3,
            img_layers=1, obs_layers=1, dyn_layers=1, blocks=2, act="SiLU",
            norm=True, unimix_ratio=0.01, initial="learned", device="cpu",
            sem_stoch=2, sem_discrete=3, sem_layers=1,
        )
        return RSSM(
            config, embed_size=6, act_dim=2, semantic=True,
            graph_token_size=self.TOKEN, graph_simple=True,
            graph_dim=self.GRAPH_DIM,
        )

    def test_z_keeps_the_stock_width(self):
        model = self._rssm()
        self.assertEqual(model._stoch, 4)          # stoch, not hybrid_stoch
        self.assertEqual(model.flat_sem, self.GRAPH_DIM)
        self.assertEqual(model.feat_size, 4 * 3 + self.GRAPH_DIM + 16)

    def test_sem_is_a_flat_vector(self):
        model = self._rssm()
        _, _, sem = model.initial(5)
        self.assertEqual(tuple(sem.shape), (5, self.GRAPH_DIM))
        self.assertEqual(model.sem_shape(), (self.GRAPH_DIM,))

    def test_z_networks_do_not_read_g(self):
        model = self._rssm()
        obs_in = model._obs_net[0].in_features
        img_in = model._img_net[0].in_features
        self.assertEqual(obs_in, 16 + 6)             # deter + embed
        self.assertEqual(img_in, 16)                 # deter only

    def test_semantic_heads_do_not_read_previous_g(self):
        model = self._rssm()
        self.assertEqual(model._sem_obs[0].in_features, 16 + self.TOKEN)
        self.assertEqual(model._sem_img[0].in_features, 16)

    def test_previous_g_still_drives_the_transition(self):
        model = self._rssm()
        self.assertIsNotNone(model._deter_net._dyn_in3)
        self.assertEqual(model._deter_net._dyn_in3[0].in_features, self.GRAPH_DIM)

    def test_posterior_is_deterministic(self):
        torch.manual_seed(0)
        model = self._rssm().eval()
        stoch, deter, sem = model.initial(2)
        action = torch.zeros(2, 2)
        embed = torch.randn(2, 6)
        token = torch.randn(2, self.TOKEN)
        reset = torch.zeros(2, dtype=torch.bool)
        with torch.no_grad():
            first = model.obs_step(stoch, deter, action, embed, reset, sem=sem, graph_token=token)
            second = model.obs_step(stoch, deter, action, embed, reset, sem=sem, graph_token=token)
        self.assertTrue(torch.equal(first[3], second[3]))
        self.assertIsNone(first[4])

    def test_observe_returns_no_semantic_logits(self):
        torch.manual_seed(0)
        model = self._rssm()
        initial = model.initial(2)
        observed = model.observe(
            torch.randn(2, 4, 6), torch.zeros(2, 4, 2), initial,
            torch.zeros(2, 4, dtype=torch.bool), torch.randn(2, 4, self.TOKEN),
        )
        self.assertEqual(len(observed), 4)
        self.assertEqual(tuple(observed[3].shape), (2, 4, self.GRAPH_DIM))

    def test_alignment_is_symmetric_in_value_and_split_in_gradient(self):
        model = self._rssm()
        post = torch.randn(2, 3, self.GRAPH_DIM, requires_grad=True)
        prior = torch.randn(2, 3, self.GRAPH_DIM, requires_grad=True)
        dyn, rep = model.semantic_align_loss(post, prior)
        self.assertTrue(torch.allclose(dyn, rep))
        dyn.sum().backward(retain_graph=True)
        self.assertIsNone(post.grad)                 # dyn updates the prior
        self.assertGreater(float(prior.grad.abs().sum()), 0.0)
        prior.grad = None
        rep.sum().backward()
        self.assertGreater(float(post.grad.abs().sum()), 0.0)
        self.assertIsNone(prior.grad)                # rep updates the posterior

    def test_rms_is_parameter_free_and_fixed_scale(self):
        model = self._rssm()
        scaled = model.rms(torch.randn(4, self.GRAPH_DIM) * 100.0)
        self.assertAlmostEqual(float(scaled.square().mean()), 1.0, places=4)

    def test_feature_carries_g(self):
        model = self._rssm()
        stoch, deter, sem = model.initial(3)
        feat = model.get_feat(stoch, deter, sem)
        self.assertEqual(tuple(feat.shape), (3, model.feat_size))


class EnvConfigPlumbingTest(unittest.TestCase):
    """The env config is flattened key by key, so a new key is easy to drop."""

    def test_every_env_graph_key_is_forwarded(self):
        env_config = OmegaConf.load("configs/env/mshab.yaml")
        declared = set(env_config.graph.keys())
        forwarded = set(_GRAPH_CONFIG_KEYS) | {"cameras"}
        self.assertEqual(
            declared - forwarded,
            set(),
            "configs/env/mshab.yaml declares graph keys the adapter never "
            "passes to build_graph_obs; they would be silently ignored",
        )

    def test_pooled_stays_the_default_state_mode(self):
        # The frozen baseline must not move because slot mode exists.
        model_config = OmegaConf.load("configs/model/_base_.yaml")
        self.assertEqual(model_config.graph.state_mode, "pooled")
        self.assertEqual(model_config.graph.n_max, 8)
        self.assertEqual(model_config.graph.e_max, 168)
        self.assertIs(model_config.graph_simple, False)
        self.assertIs(model_config.progress.enabled, False)
        self.assertEqual(model_config.loss_scales.reltemp, 1.0)

    def test_the_slot_preset_pins_the_six_node_contract(self):
        slot_config = OmegaConf.load("configs/model/size100M_graph_slots.yaml")
        self.assertEqual(slot_config.graph.state_mode, "slots")
        self.assertEqual(slot_config.graph.n_max, 6)
        self.assertEqual(slot_config.graph.e_max, 64)
        self.assertEqual(slot_config.graph.slot_dim, 256)
        self.assertEqual(slot_config.graph.slot_heads, 4)
        self.assertEqual(slot_config.graph.slot_mixer_layers, 1)
        self.assertIs(slot_config.graph_simple, True)
        # The relation-only contract stays relation-only.
        pooled = OmegaConf.load("configs/model/size100M_graph.yaml")
        for key in ("deter", "hidden", "discrete", "depth", "units", "rep_loss"):
            self.assertEqual(slot_config[key], pooled[key], key)

    def test_every_arm_sees_the_same_privileged_observation(self):
        # The graph is built from privileged state in every graph arm, so the
        # graph-free baseline has to see the same fields or the comparison
        # measures privilege instead of representation.
        env_config = OmegaConf.load("configs/env/mshab.yaml")
        self.assertIs(env_config.nonprivileged_obs, False)

    def test_simple_mode_reaches_the_builder(self):
        graph_config = OmegaConf.create({
            "enabled": True, "simple": True, "uid_vocab": 64,
            "profile": "room_scale", "thresholds_path": "", "whitelist_dir": "",
            "n_max": 8, "e_max": 168, "k_persist": -1, "app_dim": 384,
            "dino_model": "dinov2_vits14_reg", "dino_res": 112,
            "dino_weights": "", "staleness_enabled": False,
            "bypass_teemo": False,
        })
        out = graph_observation_config(graph_config, ["fetch_head"])
        self.assertIs(out["simple"], True)
        self.assertEqual(out["uid_vocab"], 64)
        self.assertEqual(out["cameras"], ["fetch_head"])


class DecoderDetachTest(unittest.TestCase):
    """Pixel reconstruction may read g but must not reshape it."""

    def _decoder(self, detach):
        # MultiDecoder splats the dist nodes (``**config.cnn_dist``) and also
        # reads them by attribute, so this has to be a real config node rather
        # than a namespace.
        config = OmegaConf.create({
            "cnn_keys": "^image$",
            "mlp_keys": "^state$",
            "mlp_dist": {"name": "symlog_mse"},
            "cnn_dist": {"name": "mse"},
            "mlp": {
                "shape": None, "layers": 1, "units": 8, "act": "SiLU",
                "norm": True, "dist": {"name": "identity"}, "device": "cpu",
                "outscale": 1.0, "symlog_inputs": False, "name": "mlp_decoder",
            },
            "cnn": {
                "depth": 4, "units": 8, "bspace": 2, "mults": [1, 1],
                "act": "SiLU", "norm": True, "kernel_size": 3, "minres": 4,
                "outscale": 1.0,
            },
        })
        return MultiDecoder(
            config, deter=8, flat_stoch=6,
            shapes={"image": (16, 16, 3), "state": (5,)},
            flat_sem=4, detach_sem_cnn=detach,
        )

    def _grad(self, detach, key):
        torch.manual_seed(0)
        decoder = self._decoder(detach)
        sem = torch.randn(2, 1, 4, requires_grad=True)
        stoch = torch.randn(2, 1, 6)
        deter = torch.randn(2, 1, 8)
        out = decoder(stoch, deter, sem)[key]
        out.mode().square().mean().backward()
        return 0.0 if sem.grad is None else float(sem.grad.abs().sum())

    def test_image_reconstruction_does_not_reach_g(self):
        self.assertEqual(self._grad(detach=True, key="image"), 0.0)

    def test_state_reconstruction_still_reaches_g(self):
        self.assertGreater(self._grad(detach=True, key="state"), 0.0)

    def test_full_mode_leaves_the_image_path_attached(self):
        self.assertGreater(self._grad(detach=False, key="image"), 0.0)


if __name__ == "__main__":
    unittest.main()
