"""Direct slot dynamics: observed slots, UID alignment, prior, decoder, progress.

Synthetic tensors only -- no simulator, no omegaconf, no replay buffer. Every
case here is a property the slot mode is supposed to guarantee structurally
rather than learn.
"""

import unittest
from types import SimpleNamespace

import torch

import progress as progress_module
from graph import (
    UID_EE,
    UID_PAD,
    GraphEncoder,
    SlotAligner,
    SlotGraphDecoder,
    SlotObservation,
    SlotReadout,
    compact_graph,
)
from progress import ProgressReward, ProgressScorer
from rssm import (
    SLOT_META_ENT,
    SLOT_META_TARGET,
    SLOT_META_UID,
    RSSM,
)

N_MAX = 6
E_MAX = 16
SLOT_DIM = 16
UID_VOCAB = 32


def graph_config(state_mode="slots", units=32, slot_dim=SLOT_DIM, n_max=N_MAX):
    return SimpleNamespace(
        simple=True,
        state_mode=state_mode,
        units=units,
        simple_units=units,
        semantic_dim=units,
        slot_dim=slot_dim,
        slot_mixer_layers=1,
        slot_heads=2,
        n_max=n_max,
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


def rssm_config():
    return SimpleNamespace(
        stoch=4, hybrid_stoch=2, deter=16, hidden=8, discrete=3,
        img_layers=1, obs_layers=1, dyn_layers=1, blocks=2, act="SiLU",
        norm=True, unimix_ratio=0.01, initial="learned", device="cpu",
        sem_stoch=2, sem_discrete=3, sem_layers=1,
    )


def slot_graph(batch=2, time=3, n_valid=3, n_edges=4, uids=None, ents=None):
    """One relation-only frame batch on the six-slot contract."""
    shape = (batch, time)
    graph = {
        key: torch.zeros(*shape, N_MAX, dtype=torch.uint8)
        for key in ("graph_node_ent", "graph_node_target")
    }
    graph["graph_node_uid"] = torch.zeros(*shape, N_MAX, dtype=torch.int64)
    for key in ("src", "dst", "rel", "abs", "temp"):
        graph[f"graph_edge_{key}"] = torch.zeros(*shape, E_MAX, dtype=torch.uint8)

    if ents is None:
        ents = list(range(1, n_valid + 1))
    graph["graph_node_ent"][..., :n_valid] = torch.tensor(ents, dtype=torch.uint8)
    if uids is None:
        uids = [UID_EE] + list(range(2, n_valid + 1))
    graph["graph_node_uid"][..., :n_valid] = torch.tensor(uids, dtype=torch.int64)
    # Slot zero is the end effector, so the target lives on slot one.
    graph["graph_node_target"][..., 1] = 1

    index = torch.arange(n_edges)
    graph["graph_edge_src"][..., :n_edges] = (index % n_valid).to(torch.uint8)
    graph["graph_edge_dst"][..., :n_edges] = ((index + 1) % n_valid).to(torch.uint8)
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


def observation(batch=1, uids=(UID_EE, 2, 3), ents=None, value=None):
    """A one-step SlotObservation with recognisable slot values."""
    count = len(uids)
    uid = torch.zeros(batch, N_MAX, dtype=torch.long)
    uid[:, :count] = torch.tensor(uids, dtype=torch.long)
    ent = torch.zeros(batch, N_MAX, dtype=torch.long)
    ent[:, :count] = torch.tensor(
        list(ents) if ents is not None else list(range(1, count + 1)), dtype=torch.long
    )
    target = torch.zeros(batch, N_MAX, dtype=torch.long)
    target[:, 1] = 1
    mask = ent.ne(0)
    slots = torch.zeros(batch, N_MAX, SLOT_DIM)
    if value is None:
        # Row i is the constant i+1, so a slot's origin is readable from it.
        value = torch.arange(1, count + 1, dtype=torch.float32)
    slots[:, :count] = value.reshape(1, count, 1)
    return SlotObservation(slots, uid, ent, target, mask)


class ConstantsTest(unittest.TestCase):
    def test_uid_constants_match_the_builder(self):
        from scenegraph.core.graph_builder import UID_EE as BUILDER_EE
        from scenegraph.core.graph_builder import UID_PAD as BUILDER_PAD

        self.assertEqual(UID_EE, BUILDER_EE)
        self.assertEqual(UID_PAD, BUILDER_PAD)


class SlotEncoderTest(unittest.TestCase):
    def test_slot_output_is_six_by_slot_dim(self):
        encoder = GraphEncoder(graph_config())
        with torch.no_grad():
            encoded = encoder(slot_graph(batch=2, time=3))
        self.assertEqual(tuple(encoded.slots.slots.shape), (2, 3, N_MAX, SLOT_DIM))
        for field in ("uid", "ent", "target", "mask"):
            self.assertEqual(
                tuple(getattr(encoded.slots, field).shape), (2, 3, N_MAX)
            )

    def test_no_pooled_token_is_computed(self):
        encoder = GraphEncoder(graph_config())
        self.assertIsNone(encoder.query)
        self.assertIsNone(encoder.key)
        self.assertIsNone(encoder.out)
        with torch.no_grad():
            self.assertIsNone(encoder(slot_graph()).token)
        self.assertFalse(
            any(name.startswith(("key.", "value.", "out.")) or name == "query"
                for name, _ in encoder.named_parameters())
        )

    def test_uid_embedding_is_absent_from_node_encoding(self):
        encoder = GraphEncoder(graph_config())
        self.assertIsNone(encoder.uid)
        self.assertFalse(any("uid" in name for name, _ in encoder.named_parameters()))
        # Same scene, different episode-random identity codes: identical nodes.
        left = slot_graph(batch=1, time=1, uids=[UID_EE, 2, 3])
        right = slot_graph(batch=1, time=1, uids=[UID_EE, 7, 9])
        with torch.no_grad():
            a = encoder(left).slots.slots
            b = encoder(right).slots.slots
        torch.testing.assert_close(a, b)

    def test_padded_slots_are_exactly_zero(self):
        encoder = GraphEncoder(graph_config())
        with torch.no_grad():
            slots = encoder(slot_graph(n_valid=3)).slots.slots
        self.assertTrue(torch.equal(slots[..., 3:, :], torch.zeros_like(slots[..., 3:, :])))

    def test_slot_mode_requires_the_relation_only_contract(self):
        config = graph_config()
        config.simple = False
        with self.assertRaisesRegex(ValueError, "relation-only"):
            GraphEncoder(config)

    def test_pooled_mode_is_structurally_unchanged(self):
        encoder = GraphEncoder(graph_config(state_mode="pooled"))
        self.assertFalse(encoder.slot_mode)
        self.assertIsNotNone(encoder.uid)
        self.assertIsNotNone(encoder.query)
        self.assertEqual(encoder.units, 32)  # simple_units, not slot_dim
        with torch.no_grad():
            encoded = encoder(slot_graph())
        self.assertIsNone(encoded.slots)
        self.assertEqual(tuple(encoded.token.shape), (2, 3, 32))


class SlotAlignerTest(unittest.TestCase):
    def setUp(self):
        self.aligner = SlotAligner(N_MAX)

    def empty(self, batch=1):
        zeros = torch.zeros(batch, N_MAX, dtype=torch.long)
        return zeros, zeros.clone(), zeros.clone()

    def test_end_effector_always_takes_slot_zero(self):
        # The end effector arrives third in the observation and still lands on
        # slot zero, because assignment is by identity and not by position.
        obs = observation(uids=(5, 7, UID_EE))
        align = self.aligner(obs, *self.empty())
        self.assertEqual(int(align.dest[0, 2]), 0)
        self.assertEqual(int(align.uid[0, 0]), UID_EE)
        self.assertTrue(bool(align.mask[0, 0]))

    def test_a_uid_keeps_its_slot_when_the_registry_reorders(self):
        first = self.aligner(observation(uids=(UID_EE, 4, 9)), *self.empty())
        # Same two objects, swapped registry positions.
        second = self.aligner(
            observation(uids=(UID_EE, 9, 4)),
            first.uid,
            first.ent,
            first.target,
        )
        slot_of_4 = int(first.dest[0, 1])
        slot_of_9 = int(first.dest[0, 2])
        self.assertEqual(int(second.dest[0, 2]), slot_of_4)
        self.assertEqual(int(second.dest[0, 1]), slot_of_9)
        self.assertEqual(int(second.uid[0, slot_of_4]), 4)
        self.assertEqual(int(second.uid[0, slot_of_9]), 9)
        self.assertTrue(bool(second.matched[0, slot_of_4]))
        self.assertTrue(bool(second.matched[0, slot_of_9]))

    def test_a_changed_uid_clears_the_slot_it_reuses(self):
        first = self.aligner(observation(uids=(UID_EE, 4)), *self.empty())
        slot_of_4 = int(first.dest[0, 1])
        self.assertEqual(float(first.slots[0, slot_of_4, 0]), 2.0)
        # 4 is gone and 11 is new; with only one object slot occupied and the
        # rest empty, 11 must not inherit 4's latent wherever it lands.
        second = self.aligner(
            observation(uids=(UID_EE, 11), value=torch.tensor([1.0, 5.0])),
            first.uid,
            first.ent,
            first.target,
        )
        slot_of_11 = int(second.dest[0, 1])
        self.assertEqual(int(second.uid[0, slot_of_11]), 11)
        self.assertEqual(float(second.slots[0, slot_of_11, 0]), 5.0)
        self.assertFalse(bool(second.matched[0, slot_of_11]))

    def test_a_full_table_evicts_a_stale_slot_rather_than_dropping_a_node(self):
        # Fill every object slot, then present five entirely new objects.
        first = self.aligner(observation(uids=(UID_EE, 2, 3, 4, 5, 6)), *self.empty())
        self.assertEqual(int(first.mask.sum()), N_MAX)
        second = self.aligner(
            observation(uids=(UID_EE, 12, 13, 14, 15, 16)),
            first.uid,
            first.ent,
            first.target,
        )
        self.assertEqual(int(second.overflow.sum()), 0)
        self.assertEqual(sorted(second.uid[0].tolist()), [UID_EE, 12, 13, 14, 15, 16])
        self.assertEqual(int(second.matched.sum()), 1)  # only the end effector

    def test_zero_uids_never_match_and_never_occupy(self):
        prev_uid = torch.zeros(1, N_MAX, dtype=torch.long)
        prev_ent = torch.zeros(1, N_MAX, dtype=torch.long)
        obs = observation(uids=(UID_EE, 3))
        # Padded rows carry garbage that must not reach a slot.
        obs.uid[:, 4] = 3
        obs.slots[:, 4] = 99.0
        align = self.aligner(obs, prev_uid, prev_ent, prev_ent.clone())
        self.assertEqual(int(align.dest[0, 4]), N_MAX)  # scratch row
        self.assertEqual(int(align.mask.sum()), 2)
        self.assertEqual(float(align.slots.abs().max()), 2.0)

    def test_an_unobserved_slot_keeps_its_identity(self):
        first = self.aligner(observation(uids=(UID_EE, 4, 9)), *self.empty())
        blank = observation(uids=(UID_EE, 4, 9)).keep(torch.zeros(1, dtype=torch.bool))
        second = self.aligner(blank, first.uid, first.ent, first.target)
        torch.testing.assert_close(second.uid, first.uid)
        torch.testing.assert_close(second.ent, first.ent)
        self.assertFalse(bool(second.present.any()))
        self.assertFalse(bool(second.matched.any()))


class SlotRSSMTest(unittest.TestCase):
    def _rssm(self):
        torch.manual_seed(0)
        return RSSM(
            rssm_config(),
            embed_size=6,
            act_dim=2,
            semantic=True,
            graph_slots=True,
            graph_config=graph_config(),
        )

    def _step(self, model, batch=1, reset=False, obs=None, state=None):
        stoch, deter, sem, meta = state or model.initial(batch)
        return model.obs_step(
            stoch,
            deter,
            torch.zeros(batch, 2),
            torch.randn(batch, 6),
            torch.full((batch,), bool(reset)),
            sem=sem,
            slot_meta=meta,
            slot_obs=obs if obs is not None else observation(batch=batch),
        )

    def test_carry_is_slots_plus_identity_metadata(self):
        model = self._rssm()
        self.assertEqual(model.state_keys, ("stoch", "deter", "sem", "slot_meta"))
        stoch, deter, sem, meta = model.initial(4)
        self.assertEqual(tuple(sem.shape), (4, N_MAX, SLOT_DIM))
        self.assertEqual(tuple(meta.shape), (4, N_MAX, 3))
        # No single global semantic vector exists in this mode.
        self.assertIsNone(model._sem_obs)
        self.assertIsNone(model._sem_img)
        self.assertEqual(model.sem_shape(), (N_MAX, SLOT_DIM))

    def test_the_transition_reads_the_slot_matrix_not_a_readout(self):
        model = self._rssm()
        self.assertEqual(
            model._deter_net._dyn_in3[0].in_features, N_MAX * SLOT_DIM + N_MAX
        )
        # The readout is a head input; it must never be reachable from h.
        sem = torch.randn(2, N_MAX, SLOT_DIM)
        meta = torch.zeros(2, N_MAX, 3)
        meta[..., SLOT_META_UID] = 1
        model._deter_net(
            torch.zeros(2, model.flat_stoch // model._discrete, model._discrete),
            torch.zeros(2, model._deter),
            torch.zeros(2, 2),
            model.slot_transition_input(sem, model.slot_mask(meta)),
        ).square().sum().backward()
        self.assertTrue(
            all(p.grad is None for p in model._slot_readout.parameters())
        )
        self.assertIsNone(model._slot_readout.query.grad)

    def test_feature_is_z_h_and_the_pooled_readout(self):
        model = self._rssm()
        stoch, deter, sem, meta = model.initial(3)
        feat = model.get_feat(stoch, deter, sem, meta)
        self.assertEqual(tuple(feat.shape), (3, model.feat_size))
        self.assertEqual(model.feat_size, model.flat_stoch + SLOT_DIM + 16)

    def test_reset_clears_every_slot_and_identity(self):
        model = self._rssm()
        first = self._step(model)
        self.assertGreater(float(first["sem"].abs().sum()), 0.0)
        carried = (
            first["stoch"], first["deter"], first["sem"], first["slot_meta"]
        )
        blank = observation().keep(torch.zeros(1, dtype=torch.bool))
        after = self._step(model, reset=True, obs=blank, state=carried)
        # is_first zeroes the carry, and a blanked frame adds nothing, so the
        # whole slot table has to come out empty.
        self.assertEqual(float(after["sem"].abs().sum()), 0.0)
        self.assertEqual(float(after["slot_meta"].abs().sum()), 0.0)
        self.assertFalse(bool(model.slot_mask(after["slot_meta"]).any()))

    def test_padded_slots_stay_exactly_zero_through_the_prior(self):
        model = self._rssm()
        step = self._step(model, obs=observation(uids=(UID_EE, 3)))
        mask = model.slot_mask(step["slot_meta"])
        self.assertEqual(int(mask.sum()), 2)
        self.assertEqual(float(step["sem"][~mask].abs().max()), 0.0)
        self.assertEqual(float(step["prior_slot"][:, 2:].abs().max()), 0.0)

    def test_padding_cannot_change_a_valid_slot(self):
        model = self._rssm().eval()
        clean = observation(uids=(UID_EE, 3))
        noisy = observation(uids=(UID_EE, 3))
        noisy.slots[:, 3:] = 42.0  # padded rows only
        with torch.no_grad():
            a = self._step(model, obs=clean)
            b = self._step(model, obs=noisy)
        torch.testing.assert_close(a["sem"], b["sem"])
        torch.testing.assert_close(a["slot_meta"], b["slot_meta"])

    def test_posterior_replaces_observed_slots_and_carries_the_prior(self):
        model = self._rssm().eval()
        first = self._step(model)
        carried = (
            first["stoch"], first["deter"], first["sem"], first["slot_meta"]
        )
        # The end effector is still observed; the two objects are not.
        partial = observation(uids=(UID_EE,), value=torch.tensor([7.0]))
        with torch.no_grad():
            second = self._step(model, obs=partial, state=carried)
        present = second["present"]
        self.assertTrue(bool(present[0, 0]))
        self.assertFalse(bool(present[0, 1:].any()))
        # Observed slot: exactly the observation. Carried slot: exactly the prior.
        torch.testing.assert_close(
            second["sem"][0, 0], torch.full((SLOT_DIM,), 7.0)
        )
        torch.testing.assert_close(second["sem"][0, 1], second["prior_slot"][0, 1])

    def test_slot_dynamics_loss_ignores_unmatched_and_padded_slots(self):
        model = self._rssm()
        prior = torch.zeros(1, N_MAX, SLOT_DIM, requires_grad=True)
        post = torch.ones(1, N_MAX, SLOT_DIM)
        valid = torch.zeros(1, N_MAX, dtype=torch.bool)
        valid[0, 1] = True
        loss = model.slot_dynamics_loss(prior, post, valid)
        loss.backward()
        rows = prior.grad.abs().sum(-1)[0]
        self.assertGreater(float(rows[1]), 0.0)
        self.assertEqual(float(rows[[0, 2, 3, 4, 5]].abs().sum()), 0.0)
        # Nothing matched at all is a zero loss, not a division by zero.
        empty = model.slot_dynamics_loss(
            torch.zeros(1, N_MAX, SLOT_DIM), post, torch.zeros(1, N_MAX, dtype=torch.bool)
        )
        self.assertTrue(torch.isfinite(empty))
        self.assertEqual(float(empty), 0.0)

    def test_imagination_moves_slots_and_latches_identity(self):
        model = self._rssm().eval()
        step = self._step(model)
        with torch.no_grad():
            rollout = model.imagine_with_action(
                step["stoch"],
                step["deter"],
                torch.randn(1, 4, 2).clamp(-1, 1),
                step["sem"],
                step["slot_meta"],
            )
        self.assertEqual(tuple(rollout["sem"].shape), (1, 4, N_MAX, SLOT_DIM))
        for i in range(4):
            torch.testing.assert_close(
                rollout["slot_meta"][:, i], step["slot_meta"]
            )
        mask = model.slot_mask(step["slot_meta"])
        self.assertEqual(float(rollout["sem"][:, :, ~mask[0]].abs().max()), 0.0)
        self.assertFalse(
            torch.allclose(rollout["sem"][:, -1], step["sem"], atol=1e-6)
        )

    def test_observe_stacks_the_whole_rollout(self):
        model = self._rssm()
        batch, time = 2, 3
        encoder = GraphEncoder(graph_config())
        with torch.no_grad():
            slot_obs = encoder(slot_graph(batch=batch, time=time)).slots
        observed = model.observe(
            torch.randn(batch, time, 6),
            torch.zeros(batch, time, 2),
            model.initial(batch),
            torch.zeros(batch, time, dtype=torch.bool),
            slot_obs=slot_obs,
        )
        self.assertEqual(tuple(observed["sem"].shape), (batch, time, N_MAX, SLOT_DIM))
        self.assertEqual(tuple(observed["prior_slot"].shape), (batch, time, N_MAX, SLOT_DIM))
        self.assertEqual(tuple(observed["logit"].shape), observed["prior_logit"].shape)
        self.assertEqual(tuple(observed["dest"].shape), (batch, time, N_MAX))


class SlotReadoutTest(unittest.TestCase):
    def test_padding_does_not_reach_the_readout(self):
        torch.manual_seed(0)
        readout = SlotReadout(SLOT_DIM, SLOT_DIM).eval()
        slots = torch.randn(2, N_MAX, SLOT_DIM)
        mask = torch.zeros(2, N_MAX, dtype=torch.bool)
        mask[:, :2] = True
        other = slots.clone()
        other[:, 2:] = 13.0
        with torch.no_grad():
            torch.testing.assert_close(readout(slots, mask), readout(other, mask))

    def test_an_empty_table_reads_out_zero(self):
        readout = SlotReadout(SLOT_DIM, SLOT_DIM).eval()
        with torch.no_grad():
            out = readout(
                torch.randn(2, N_MAX, SLOT_DIM), torch.zeros(2, N_MAX, dtype=torch.bool)
            )
        self.assertEqual(float(out.abs().max()), 0.0)


class SlotDecoderTest(unittest.TestCase):
    def _run(self, requires_grad=True):
        torch.manual_seed(0)
        config = graph_config()
        decoder = SlotGraphDecoder(config)
        graph = slot_graph(batch=2, time=2)
        compact = compact_graph(graph, simple=True)
        post = torch.randn(2, 2, N_MAX, SLOT_DIM, requires_grad=requires_grad)
        prior = torch.randn(2, 2, N_MAX, SLOT_DIM, requires_grad=requires_grad)
        # Identity routing: observation row i is slot i.
        dest = torch.arange(N_MAX).reshape(1, 1, N_MAX).expand(2, 2, N_MAX)
        mask = torch.zeros(2, 2, N_MAX, dtype=torch.bool)
        mask[..., :3] = True
        target = torch.zeros(2, 2, N_MAX, dtype=torch.float32)
        target[..., 1] = 1
        matched = mask.clone()
        step = torch.ones(2, 2, dtype=torch.bool)
        losses, metrics = decoder(
            post, prior, compact, dest, mask, target, matched, step
        )
        return decoder, post, prior, losses, metrics

    def test_losses_are_named_and_finite(self):
        _, _, _, losses, metrics = self._run()
        self.assertEqual(
            set(losses),
            {"nodetgt", "relabs", "reltemp", "prior_relabs", "prior_reltemp"},
        )
        for name, value in losses.items():
            self.assertTrue(torch.isfinite(value), name)
        self.assertIn("node_target_acc", metrics)
        self.assertIn("prior_relabs_acc", metrics)

    def test_the_decoder_never_consumes_uid(self):
        decoder = SlotGraphDecoder(graph_config())
        self.assertFalse(any("uid" in name for name, _ in decoder.named_parameters()))

    def test_posterior_relation_losses_reach_the_observed_slots(self):
        for name in ("nodetgt", "relabs", "reltemp"):
            with self.subTest(loss=name):
                _, post, prior, losses, _ = self._run()
                losses[name].backward()
                self.assertGreater(float(post.grad.abs().sum()), 0.0)
                self.assertIsNone(prior.grad)

    def test_prior_relation_losses_reach_the_predicted_slots_only(self):
        for name in ("prior_relabs", "prior_reltemp"):
            with self.subTest(loss=name):
                _, post, prior, losses, _ = self._run()
                losses[name].backward()
                self.assertGreater(float(prior.grad.abs().sum()), 0.0)
                self.assertIsNone(post.grad)

    def test_unmatched_slots_are_excluded_from_the_prior_losses(self):
        torch.manual_seed(0)
        decoder = SlotGraphDecoder(graph_config())
        compact = compact_graph(slot_graph(batch=1, time=1), simple=True)
        post = torch.randn(1, 1, N_MAX, SLOT_DIM)
        prior = torch.randn(1, 1, N_MAX, SLOT_DIM, requires_grad=True)
        dest = torch.arange(N_MAX).reshape(1, 1, N_MAX)
        mask = torch.zeros(1, 1, N_MAX, dtype=torch.bool)
        mask[..., :3] = True
        target = torch.zeros(1, 1, N_MAX)
        target[..., 1] = 1
        losses, _ = decoder(
            post, prior, compact, dest, mask, target,
            torch.zeros(1, 1, N_MAX, dtype=torch.bool),  # nothing matched
            torch.ones(1, 1, dtype=torch.bool),
        )
        losses["prior_relabs"].backward()
        self.assertEqual(float(losses["prior_relabs"]), 0.0)
        self.assertEqual(float(prior.grad.abs().sum()), 0.0)


class ProgressTest(unittest.TestCase):
    def scorer(self):
        return ProgressScorer(progress_module.PICK_STAGES, 17)

    def probs(self, labels):
        """One-hot relation distributions in the scorer's relation order."""
        scorer = self.scorer()
        out = torch.zeros(1, int(scorer.relations.numel()), 17)
        for index, relation in enumerate(scorer.relations.tolist()):
            out[0, index, labels[relation]] = 1.0
        return out

    FAR = {
        progress_module.REL_PLANAR_DISTANCE: progress_module.ABS_VERY_FAR,
        progress_module.REL_HEIGHT_OFFSET: progress_module.ABS_FAR_BELOW,
        progress_module.REL_CONTACT_COMPAT: progress_module.ABS_UNOBSERVED,
        progress_module.REL_GRASP_COMPAT: progress_module.ABS_UNOBSERVED,
        progress_module.REL_CONTACT: progress_module.ABS_NOT_HOLDS,
        progress_module.REL_GRASP: progress_module.ABS_NOT_HOLDS,
    }
    NEAR = {
        **FAR,
        progress_module.REL_PLANAR_DISTANCE: progress_module.ABS_NEAR,
        progress_module.REL_HEIGHT_OFFSET: progress_module.ABS_LEVEL,
    }
    TOUCHING = {
        **NEAR,
        progress_module.REL_CONTACT_COMPAT: progress_module.ABS_MATCH,
        progress_module.REL_GRASP_COMPAT: progress_module.ABS_MATCH,
        progress_module.REL_CONTACT: progress_module.ABS_HOLDS,
    }
    GRASPED = {**TOUCHING, progress_module.REL_GRASP: progress_module.ABS_HOLDS}

    def test_potential_is_always_in_the_unit_interval(self):
        scorer = self.scorer()
        torch.manual_seed(0)
        random = torch.softmax(torch.randn(64, int(scorer.relations.numel()), 17), -1)
        for hard in (True, False):
            value = scorer.potential(random, hard=hard)
            self.assertGreaterEqual(float(value.min()), 0.0)
            self.assertLessEqual(float(value.max()), 1.0)

    def test_progress_rises_monotonically_and_grasp_scores_highest(self):
        scorer = self.scorer()
        stages = [self.FAR, self.NEAR, self.TOUCHING, self.GRASPED]
        values = [float(scorer.potential(self.probs(s), hard=True)) for s in stages]
        self.assertEqual(values, sorted(values))
        self.assertEqual(values[-1], max(values))
        self.assertAlmostEqual(values[-1], 1.0, places=6)
        self.assertAlmostEqual(values[0], 0.0, places=6)

    def test_soft_and_hard_agree_on_confident_predictions(self):
        scorer = self.scorer()
        for stage in (self.FAR, self.NEAR, self.TOUCHING, self.GRASPED):
            probs = self.probs(stage)
            self.assertAlmostEqual(
                float(scorer.potential(probs, hard=True)),
                float(scorer.potential(probs, hard=False)),
                places=6,
            )

    def test_the_discounted_progress_return_stays_near_one(self):
        horizon = 333
        discount = 1 - 1 / horizon
        reward = ProgressReward(self.scorer(), discount)
        # Solved from the first step and never leaving: the worst case.
        potential = 1.0
        step = (1.0 - discount) * potential
        total = sum(step * discount**t for t in range(20000))
        self.assertLessEqual(total, 1.0 + 1e-6)
        self.assertGreater(total, 0.99)
        self.assertAlmostEqual(reward.discount, discount)

    def test_the_scorer_reads_the_target_slot_not_slot_zero(self):
        slots = torch.zeros(2, N_MAX, SLOT_DIM)
        slots[:, 0] = 1.0  # end effector
        slots[:, 3] = 5.0  # the flagged target
        target = torch.zeros(2, N_MAX)
        target[:, 3] = 1
        mask = torch.zeros(2, N_MAX, dtype=torch.bool)
        mask[:, [0, 3]] = True
        source, chosen, valid = progress_module.slot_pair(slots, target, mask)
        self.assertTrue(bool(valid.all()))
        torch.testing.assert_close(source, torch.ones(2, SLOT_DIM))
        torch.testing.assert_close(chosen, torch.full((2, SLOT_DIM), 5.0))

    def test_a_frame_with_no_target_is_scored_as_no_progress(self):
        torch.manual_seed(0)
        decoder = SlotGraphDecoder(graph_config())
        reward = ProgressReward(self.scorer(), 1 - 1 / 333)
        slots = torch.randn(2, N_MAX, SLOT_DIM)
        mask = torch.ones(2, N_MAX, dtype=torch.bool)
        with torch.no_grad():
            shaped, potential, probs = reward(
                decoder, slots, torch.zeros(2, N_MAX), mask
            )
        self.assertEqual(float(potential.abs().max()), 0.0)
        self.assertEqual(float(shaped.abs().max()), 0.0)
        self.assertEqual(tuple(probs.shape), (2, 6, 17))

    def test_relation_probabilities_are_masked_to_admissible_labels(self):
        torch.manual_seed(0)
        decoder = SlotGraphDecoder(graph_config())
        scorer = self.scorer()
        with torch.no_grad():
            probs = decoder.relation_probs(
                torch.randn(4, SLOT_DIM), torch.randn(4, SLOT_DIM), scorer.relations
            )
        torch.testing.assert_close(probs.sum(-1), torch.ones(4, 6), atol=1e-5, rtol=1e-5)
        legal = decoder.abs_valid.index_select(0, scorer.relations)
        self.assertLess(float(probs[..., ~legal.any(0)].max()), 1e-6)
        # Padding label zero is never admissible for any relation.
        self.assertLess(float(probs[..., 0].max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
