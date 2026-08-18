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
    SCHEMA_SIMPLE_SLOT,
    UID_EE,
    UID_PAD,
    GraphEncoder,
    SlotAligner,
    SlotGraphDecoder,
    SlotObservation,
    SlotReadout,
    compact_graph,
)
from graph import slot_target_label as graph_slot_target_label
from graph import slot_target_logits as graph_slot_target_logits
from progress import ProgressReward, ProgressScorer, target_distribution
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


def graph_config(
    state_mode="slots", units=32, slot_dim=SLOT_DIM, n_max=N_MAX, births=False
):
    return SimpleNamespace(
        simple=True,
        state_mode=state_mode,
        slot_births=births,
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


def pooled_graph(batch=2, time=3, n_valid=3, n_edges=4):
    """The same frames on the pooled contract: boxes instead of a UID."""
    graph = {
        key: value
        for key, value in slot_graph(batch, time, n_valid, n_edges).items()
        if key != "graph_node_uid"
    }
    bbox = torch.zeros(batch, time, N_MAX, 2, 4, dtype=torch.float16)
    boxes = torch.tensor([
        [0.10, 0.40, 0.20, 0.50],
        [0.50, 0.90, 0.10, 0.30],
        [0.00, 0.20, 0.60, 0.80],
    ])
    bbox[..., :min(n_valid, 3), 0, :] = boxes[:min(n_valid, 3)].to(torch.float16)
    graph["graph_node_bbox"] = bbox
    return graph


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

    def test_pooled_mode_keeps_its_own_shape(self):
        # Slot mode must not perturb the pooled arm. Pooled graph-simple has
        # its own contract -- boxes, no UID, masked-mean pooling -- which is
        # asserted in test_graph_simple; what matters here is only that it
        # stays a single token of simple_units width and grows no slot table.
        encoder = GraphEncoder(graph_config(state_mode="pooled"))
        self.assertFalse(encoder.slot_mode)
        self.assertEqual(encoder.units, 32)  # simple_units, not slot_dim
        self.assertIsNone(encoder.uid)
        with torch.no_grad():
            encoded = encoder(pooled_graph())
        self.assertIsNone(encoded.slots)
        self.assertEqual(tuple(encoded.token.shape), (2, 3, 32))


class SlotAlignerTest(unittest.TestCase):
    def setUp(self):
        self.aligner = SlotAligner(N_MAX)

    def empty(self, batch=1):
        zeros = torch.zeros(batch, N_MAX, dtype=torch.long)
        return zeros, zeros.clone(), zeros.clone(), zeros.float()

    def carry(self, align):
        return align.uid, align.ent, align.target, align.alive

    def test_end_effector_always_takes_slot_zero(self):
        # The end effector arrives third in the observation and still lands on
        # slot zero, because assignment is by identity and not by position.
        obs = observation(uids=(5, 7, UID_EE))
        align = self.aligner(obs, *self.empty())
        self.assertEqual(int(align.dest[0, 2]), 0)
        self.assertEqual(int(align.uid[0, 0]), UID_EE)
        self.assertEqual(float(align.alive[0, 0]), 1.0)

    def test_a_uid_keeps_its_slot_when_the_registry_reorders(self):
        first = self.aligner(observation(uids=(UID_EE, 4, 9)), *self.empty())
        # Same two objects, swapped registry positions.
        second = self.aligner(observation(uids=(UID_EE, 9, 4)), *self.carry(first))
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
        # 4 is gone and 11 is new; with the rest of the table empty, 11 takes an
        # inactive slot and must not inherit 4's latent wherever it lands.
        second = self.aligner(
            observation(uids=(UID_EE, 11), value=torch.tensor([1.0, 5.0])),
            *self.carry(first),
        )
        slot_of_11 = int(second.dest[0, 1])
        self.assertEqual(int(second.uid[0, slot_of_11]), 11)
        self.assertEqual(float(second.slots[0, slot_of_11, 0]), 5.0)
        self.assertFalse(bool(second.matched[0, slot_of_11]))
        self.assertTrue(bool(second.born[0, slot_of_11]))

    def test_a_full_table_protects_the_target_and_drops_the_excess(self):
        # Every slot occupied, then a completely different object set arrives.
        # One row cannot be placed, and the retained target must not be the one
        # that loses -- the registry, not the aligner, is meant to prevent this.
        first = self.aligner(observation(uids=(UID_EE, 2, 3, 4, 5, 6)), *self.empty())
        self.assertEqual(int(first.alive.sum()), N_MAX)
        target_slot = int(first.dest[0, 1])
        arriving = observation(uids=(UID_EE, 12, 13, 14, 15, 16))
        arriving.target[:] = 0
        second = self.aligner(arriving, *self.carry(first))
        self.assertEqual(int(second.overflow.sum()), 1)
        self.assertEqual(int(second.uid[0, target_slot]), int(first.uid[0, target_slot]))
        self.assertEqual(int(second.target[0, target_slot]), 1)
        self.assertEqual(int(second.matched.sum()), 1)  # only the end effector
        # Presence is monotone: nothing died, one arrival was refused.
        self.assertEqual(int(second.alive.sum()), N_MAX)

    def test_a_retained_target_slot_is_never_given_to_a_non_target(self):
        first = self.aligner(observation(uids=(UID_EE, 4, 9)), *self.empty())
        target_slot = int(first.dest[0, 1])  # observation row 1 carries the flag
        self.assertEqual(int(first.target[0, target_slot]), 1)

        arriving = observation(uids=(UID_EE, 21))
        arriving.target[:] = 0  # the newcomer is not the target
        second = self.aligner(arriving, *self.carry(first))
        self.assertEqual(int(second.uid[0, target_slot]), 4)
        self.assertEqual(int(second.target[0, target_slot]), 1)
        self.assertNotEqual(int(second.dest[0, 1]), target_slot)
        self.assertFalse(bool(second.replaced[0, target_slot]))

    def test_zero_uids_never_match_and_never_occupy(self):
        obs = observation(uids=(UID_EE, 3))
        # Padded rows carry garbage that must not reach a slot.
        obs.uid[:, 4] = 3
        obs.slots[:, 4] = 99.0
        align = self.aligner(obs, *self.empty())
        self.assertEqual(int(align.dest[0, 4]), N_MAX)  # scratch row
        self.assertEqual(int(align.alive.sum()), 2)
        self.assertEqual(float(align.slots.abs().max()), 2.0)

    def test_an_unobserved_slot_stays_alive_and_keeps_its_identity(self):
        first = self.aligner(observation(uids=(UID_EE, 4, 9)), *self.empty())
        blank = observation(uids=(UID_EE, 4, 9)).keep(torch.zeros(1, dtype=torch.bool))
        second = self.aligner(blank, *self.carry(first))
        torch.testing.assert_close(second.uid, first.uid)
        torch.testing.assert_close(second.ent, first.ent)
        # Absence from a history-off graph is not death.
        torch.testing.assert_close(second.alive, first.alive)
        self.assertFalse(bool(second.present.any()))
        self.assertFalse(bool(second.matched.any()))

    def test_births_match_proposals_by_content_not_by_index(self):
        # Two fresh objects and two inactive proposals. The proposal whose
        # predicted content matches an observation must win it, whichever index
        # it happens to sit at.
        prev = self.empty()
        prior = torch.zeros(1, N_MAX, SLOT_DIM)
        prior[0, 2] = 3.0  # looks like the observation carrying value 3
        prior[0, 1] = 2.0  # looks like the observation carrying value 2
        obs = observation(uids=(UID_EE, 7, 8), value=torch.tensor([1.0, 2.0, 3.0]))
        align = self.aligner(obs, *prev, prior_slot=prior, births=True)
        self.assertEqual(int(align.dest[0, 1]), 1)  # value 2 -> proposal 1
        self.assertEqual(int(align.dest[0, 2]), 2)  # value 3 -> proposal 2
        self.assertEqual(int(align.born.sum()), 2)

    def test_each_graph_is_matched_inside_its_own_row(self):
        # Row 0's cheapest proposal must not be able to claim row 1's node.
        prev = self.empty(batch=2)
        prior = torch.zeros(2, N_MAX, SLOT_DIM)
        prior[0, 1] = 2.0
        prior[1, 3] = 2.0
        obs = observation(batch=2, uids=(UID_EE, 7), value=torch.tensor([1.0, 2.0]))
        align = self.aligner(obs, *prev, prior_slot=prior, births=True)
        self.assertEqual(int(align.dest[0, 1]), 1)
        self.assertEqual(int(align.dest[1, 1]), 3)


class SlotRSSMTest(unittest.TestCase):
    def _rssm(self, births=False):
        torch.manual_seed(0)
        return RSSM(
            rssm_config(),
            embed_size=6,
            act_dim=2,
            semantic=True,
            graph_slots=True,
            graph_config=graph_config(births=births),
        )

    def _step(self, model, batch=1, reset=False, obs=None, state=None):
        stoch, deter, sem, meta, alive = state or model.initial(batch)
        return model.obs_step(
            stoch,
            deter,
            torch.zeros(batch, 2),
            torch.randn(batch, 6),
            torch.full((batch,), bool(reset)),
            sem=sem,
            slot_meta=meta,
            slot_alive=alive,
            slot_obs=obs if obs is not None else observation(batch=batch),
        )

    @staticmethod
    def _carry(step):
        return (
            step["stoch"],
            step["deter"],
            step["sem"],
            step["slot_meta"],
            step["slot_alive"],
        )

    def test_carry_is_slots_presence_and_identity_metadata(self):
        model = self._rssm()
        self.assertEqual(
            model.state_keys, ("stoch", "deter", "sem", "slot_meta", "slot_alive")
        )
        stoch, deter, sem, meta, alive = model.initial(4)
        self.assertEqual(tuple(sem.shape), (4, N_MAX, SLOT_DIM))
        self.assertEqual(tuple(meta.shape), (4, N_MAX, 3))
        self.assertEqual(tuple(alive.shape), (4, N_MAX))
        # Every slot starts inactive, the end effector included.
        self.assertEqual(float(alive.sum()), 0.0)
        # No single global semantic vector exists in this mode.
        self.assertIsNone(model._sem_obs)
        self.assertIsNone(model._sem_img)
        self.assertEqual(model.sem_shape(), (N_MAX, SLOT_DIM))

    def test_the_transition_reads_an_invariant_summary_not_a_readout(self):
        model = self._rssm()
        self.assertEqual(model._deter_net._dyn_in3[0].in_features, 3 * SLOT_DIM + 1)
        # The readout is a head input; it must never be reachable from h.
        sem = torch.randn(2, N_MAX, SLOT_DIM)
        alive = torch.ones(2, N_MAX)
        model._deter_net(
            torch.zeros(2, model.flat_stoch // model._discrete, model._discrete),
            torch.zeros(2, model._deter),
            torch.zeros(2, 2),
            model.slot_transition_input(sem, alive),
        ).square().sum().backward()
        self.assertTrue(
            all(p.grad is None for p in model._slot_readout.parameters())
        )
        self.assertIsNone(model._slot_readout.query.grad)

    def test_object_slot_permutation_leaves_h_unchanged(self):
        model = self._rssm().eval()
        sem = torch.randn(1, N_MAX, SLOT_DIM)
        alive = torch.ones(1, N_MAX)
        order = [0, 3, 1, 4, 2, 5][:N_MAX]
        with torch.no_grad():
            a = model.slot_transition_input(sem, alive)
            b = model.slot_transition_input(sem[:, order], alive[:, order])
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_the_empty_summary_is_finite_and_exactly_zero(self):
        model = self._rssm().eval()
        with torch.no_grad():
            summary = model.slot_transition_input(
                torch.randn(3, N_MAX, SLOT_DIM), torch.zeros(3, N_MAX)
            )
        self.assertTrue(torch.isfinite(summary).all())
        self.assertEqual(float(summary.abs().max()), 0.0)

    def test_the_summary_stays_float32_under_autocast(self):
        # Casting the summary's input is not enough: its projections are Linear
        # layers, so autocast makes their output half precision and the masked
        # maximum's sentinel no longer fits. This is the shape of bug that only
        # appears in a real AMP training step.
        model = self._rssm().eval()
        alive = torch.zeros(2, N_MAX)
        alive[:, :3] = 1.0
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
            summary = model.slot_transition_input(
                torch.randn(2, N_MAX, SLOT_DIM), alive
            )
        self.assertEqual(summary.dtype, torch.float32)
        self.assertTrue(torch.isfinite(summary).all())

    def test_negative_features_do_not_lose_the_masked_max_to_zeros(self):
        model = self._rssm().eval()
        sem = torch.full((1, N_MAX, SLOT_DIM), -5.0)
        alive = torch.zeros(1, N_MAX)
        alive[0, :2] = 1.0
        with torch.no_grad():
            summary = model.slot_transition_input(sem, alive)
            # One live object: its projection is both the mean and the max.
            objects = model._summary.objects(model._summary.norm(sem.float())[:, 1:])
        width = SLOT_DIM
        mean = summary[:, width : 2 * width]
        maximum = summary[:, 2 * width : 3 * width]
        torch.testing.assert_close(mean, maximum, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(maximum, objects[:, 0], atol=1e-5, rtol=1e-5)

    def test_feature_is_z_h_and_the_pooled_readout(self):
        model = self._rssm()
        stoch, deter, sem, _, alive = model.initial(3)
        feat = model.get_feat(stoch, deter, sem, alive)
        self.assertEqual(tuple(feat.shape), (3, model.feat_size))
        self.assertEqual(model.feat_size, model.flat_stoch + SLOT_DIM + 16)

    def test_reset_clears_slots_identity_and_presence(self):
        model = self._rssm()
        first = self._step(model)
        self.assertGreater(float(first["sem"].abs().sum()), 0.0)
        blank = observation().keep(torch.zeros(1, dtype=torch.bool))
        after = self._step(model, reset=True, obs=blank, state=self._carry(first))
        self.assertEqual(float(after["sem"].abs().sum()), 0.0)
        self.assertEqual(float(after["slot_meta"].abs().sum()), 0.0)
        self.assertEqual(float(after["slot_alive"].sum()), 0.0)
        self.assertTrue(bool(after["reset"].all()))

    def test_inactive_slots_hold_exactly_zero_posterior_content(self):
        model = self._rssm()
        step = self._step(model, obs=observation(uids=(UID_EE, 3)))
        alive = model.slot_mask(step["slot_alive"])
        self.assertEqual(int(alive.sum()), 2)
        self.assertEqual(float(step["sem"][~alive].abs().max()), 0.0)

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

    def test_uid_values_do_not_change_the_prior(self):
        # Identical scene, different episode-random identity codes: the prior is
        # computed before alignment and must not see them at all.
        model = self._rssm().eval()
        left = observation(uids=(UID_EE, 4, 9))
        right = observation(uids=(UID_EE, 17, 23))
        with torch.no_grad():
            a = self._step(model, obs=left)
            b = self._step(model, obs=right)
        torch.testing.assert_close(a["prior_slot"], b["prior_slot"])
        torch.testing.assert_close(a["prior_alive_logit"], b["prior_alive_logit"])

    def test_the_current_target_bit_cannot_reach_the_prior(self):
        # Same previous state and action; only which node carries the target
        # flag differs. The prior runs before alignment, so its predictions must
        # be identical even though the posterior may differ.
        model = self._rssm().eval()
        left = observation(uids=(UID_EE, 4, 9))
        right = observation(uids=(UID_EE, 4, 9))
        right.target[:] = 0
        right.target[:, 2] = 1
        with torch.no_grad():
            a = self._step(model, obs=left)
            b = self._step(model, obs=right)
        torch.testing.assert_close(a["prior_slot"], b["prior_slot"])
        torch.testing.assert_close(a["prior_alive_logit"], b["prior_alive_logit"])
        self.assertFalse(torch.equal(a["slot_meta"], b["slot_meta"]))

    def test_posterior_replaces_observed_slots_and_carries_the_prior(self):
        model = self._rssm().eval()
        first = self._step(model)
        # The end effector is still observed; the two objects are not.
        partial = observation(uids=(UID_EE,), value=torch.tensor([7.0]))
        with torch.no_grad():
            second = self._step(model, obs=partial, state=self._carry(first))
        present = second["present"]
        self.assertTrue(bool(present[0, 0]))
        self.assertFalse(bool(present[0, 1:].any()))
        # Observed slot: exactly the observation. Live carried slot: the prior.
        torch.testing.assert_close(
            second["sem"][0, 0], torch.full((SLOT_DIM,), 7.0)
        )
        torch.testing.assert_close(second["sem"][0, 1], second["prior_slot"][0, 1])
        # Presence is monotone, so the unobserved objects are still alive.
        self.assertEqual(float(second["slot_alive"][0, 1]), 1.0)

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

    def test_presence_loss_balances_births_against_inactive_slots(self):
        model = self._rssm()
        logit = torch.zeros(1, N_MAX, requires_grad=True)
        target = torch.zeros(1, N_MAX)
        target[0, 1] = 1.0
        born = torch.zeros(1, N_MAX, dtype=torch.bool)
        born[0, 1] = True
        inactive = torch.zeros(1, N_MAX, dtype=torch.bool)
        inactive[0, 2:] = True
        persistent = torch.zeros(1, N_MAX, dtype=torch.bool)
        loss = model.slot_alive_loss(logit, target, persistent, born, inactive)
        loss.backward()
        # One birth against four inactive slots still carries half the gradient,
        # because the groups are averaged before they are combined.
        birth = float(logit.grad[0, 1].abs())
        dead = float(logit.grad[0, 2:].abs().sum())
        self.assertAlmostEqual(birth, dead, places=5)
        self.assertEqual(float(logit.grad[0, 0]), 0.0)  # not in any group

    def test_imagination_carries_presence_unchanged_without_births(self):
        model = self._rssm(births=False).eval()
        step = self._step(model)
        with torch.no_grad():
            rollout = model.imagine_with_action(
                step["stoch"],
                step["deter"],
                torch.randn(1, 4, 2).clamp(-1, 1),
                step["sem"],
                step["slot_meta"],
                step["slot_alive"],
            )
        self.assertEqual(tuple(rollout["sem"].shape), (1, 4, N_MAX, SLOT_DIM))
        for i in range(4):
            torch.testing.assert_close(rollout["slot_meta"][:, i], step["slot_meta"])
            torch.testing.assert_close(rollout["slot_alive"][:, i], step["slot_alive"])
        self.assertFalse(
            torch.allclose(rollout["sem"][:, -1], step["sem"], atol=1e-6)
        )

    def test_imagined_presence_can_turn_on_with_births_enabled(self):
        model = self._rssm(births=True).eval()
        step = self._step(model, obs=observation(uids=(UID_EE, 3)))
        # Force the presence head to predict "alive" for every slot.
        with torch.no_grad():
            model._slot_prior.alive.bias.fill_(10.0)
            rollout = model.imagine_with_action(
                step["stoch"],
                step["deter"],
                torch.zeros(1, 3, 2),
                step["sem"],
                step["slot_meta"],
                step["slot_alive"],
            )
        started = float(step["slot_alive"].sum())
        ended = float(rollout["slot_alive"][:, -1].sum())
        self.assertGreater(ended, started)
        self.assertEqual(float(rollout["slot_alive"][:, -1, 0]), 1.0)

    def test_the_presence_head_receives_gradient_on_empty_proposals(self):
        model = self._rssm(births=True)
        step = self._step(model)
        step["prior_alive_logit"].square().sum().backward()
        grads = [
            p.grad
            for p in list(model._slot_prior.alive.parameters())
            + [model._slot_prior.birth_query]
        ]
        self.assertTrue(all(g is not None and float(g.abs().sum()) > 0 for g in grads))

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
        for key in ("sem", "prior_slot"):
            self.assertEqual(
                tuple(observed[key].shape), (batch, time, N_MAX, SLOT_DIM)
            )
        for key in ("slot_alive", "prior_alive_logit", "dest"):
            self.assertEqual(tuple(observed[key].shape), (batch, time, N_MAX))
        self.assertEqual(tuple(observed["logit"].shape), observed["prior_logit"].shape)
        self.assertEqual(tuple(observed["reset"].shape), (batch, time))


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
    RELATIONS = torch.tensor([1, 5], dtype=torch.long)

    def _run(self, target_row=1, relations=None):
        torch.manual_seed(0)
        decoder = SlotGraphDecoder(graph_config())
        compact = compact_graph(slot_graph(batch=2, time=2), SCHEMA_SIMPLE_SLOT)
        post = torch.randn(2, 2, N_MAX, SLOT_DIM, requires_grad=True)
        prior = torch.randn(2, 2, N_MAX, SLOT_DIM, requires_grad=True)
        # Identity routing: observation row i is slot i.
        dest = torch.arange(N_MAX).reshape(1, 1, N_MAX).expand(2, 2, N_MAX)
        alive = torch.zeros(2, 2, N_MAX)
        alive[..., :3] = 1.0
        target = torch.zeros(2, 2, N_MAX)
        target[..., target_row] = 1
        step = torch.ones(2, 2, dtype=torch.bool)
        losses, metrics = decoder(
            post,
            prior,
            compact,
            dest,
            alive,
            target,
            step,
            self.RELATIONS if relations is None else relations,
        )
        return decoder, post, prior, losses, metrics

    def test_losses_are_named_and_finite(self):
        _, _, _, losses, metrics = self._run()
        self.assertEqual(
            set(losses),
            {
                "nodetgt",
                "prior_nodetgt",
                "relabs",
                "reltemp",
                "prior_progress_relabs",
            },
        )
        for name, value in losses.items():
            self.assertTrue(torch.isfinite(value), name)
        self.assertIn("node_target_acc", metrics)
        self.assertIn("prior_progress_acc", metrics)
        self.assertGreater(float(metrics["slot_progress_facts"]), 0.0)

    def test_the_decoder_never_consumes_uid(self):
        decoder = SlotGraphDecoder(graph_config())
        self.assertFalse(any("uid" in name for name, _ in decoder.named_parameters()))

    def test_posterior_losses_reach_the_observed_slots_only(self):
        for name in ("nodetgt", "relabs", "reltemp"):
            with self.subTest(loss=name):
                _, post, prior, losses, _ = self._run()
                losses[name].backward()
                self.assertGreater(float(post.grad.abs().sum()), 0.0)
                self.assertIsNone(prior.grad)

    def test_prior_losses_reach_the_predicted_slots_only(self):
        for name in ("prior_nodetgt", "prior_progress_relabs"):
            with self.subTest(loss=name):
                _, post, prior, losses, _ = self._run()
                losses[name].backward()
                self.assertGreater(float(prior.grad.abs().sum()), 0.0)
                self.assertIsNone(post.grad)

    def test_progress_supervision_follows_the_observed_target(self):
        # Slot 1 is the target and slot 0 -> slot 1 carries a fact, so there is
        # something to supervise. Move the flag to a slot with no EE edge and the
        # loss must go to zero rather than borrowing another object's label.
        _, _, _, losses, metrics = self._run(target_row=1)
        self.assertGreater(float(metrics["slot_progress_facts"]), 0.0)
        _, _, prior, other, quiet = self._run(target_row=2)
        self.assertEqual(float(quiet["slot_progress_facts"]), 0.0)
        other["prior_progress_relabs"].backward()
        self.assertEqual(float(prior.grad.abs().sum()), 0.0)

    def test_only_stage_table_relations_are_supervised(self):
        # Relation 5 is in the fixture's EE edge set; a table naming neither
        # relation must leave the progress loss with nothing to say.
        _, _, _, losses, metrics = self._run(
            relations=torch.tensor([7, 8], dtype=torch.long)
        )
        self.assertEqual(float(metrics["slot_progress_facts"]), 0.0)
        self.assertEqual(float(losses["prior_progress_relabs"]), 0.0)

    def test_the_target_objective_is_categorical_with_a_null_class(self):
        decoder = SlotGraphDecoder(graph_config())
        slots = torch.randn(2, N_MAX, SLOT_DIM)
        alive = torch.zeros(2, N_MAX)
        alive[:, :3] = 1.0
        logits = graph_slot_target_logits(decoder.target_logits(slots), alive)
        self.assertEqual(tuple(logits.shape), (2, N_MAX))  # n-1 objects + null
        # No live object leaves all the mass on null.
        empty = graph_slot_target_logits(
            decoder.target_logits(slots), torch.zeros(2, N_MAX)
        )
        self.assertEqual(int(torch.softmax(empty, -1).argmax(-1)[0]), N_MAX - 1)

    def test_the_label_is_latched_and_rejects_two_live_targets(self):
        alive = torch.ones(1, N_MAX)
        flag = torch.zeros(1, N_MAX)
        flag[0, 2] = 1
        label, has_target = graph_slot_target_label(flag, alive)
        self.assertEqual(int(label), 1)  # object slot 2 is class 1
        self.assertTrue(bool(has_target))
        # No live flag is null, not slot zero.
        label, has_target = graph_slot_target_label(torch.zeros(1, N_MAX), alive)
        self.assertEqual(int(label), N_MAX - 1)
        self.assertFalse(bool(has_target))
        # Slot zero can never be the target class.
        ee_only = torch.zeros(1, N_MAX)
        ee_only[0, 0] = 1
        label, has_target = graph_slot_target_label(ee_only, alive)
        self.assertEqual(int(label), N_MAX - 1)
        flag[0, 3] = 1
        with self.assertRaisesRegex(ValueError, "more than one"):
            graph_slot_target_label(flag, alive)


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
    GRASPED = {
        **TOUCHING,
        progress_module.REL_PLANAR_DISTANCE: progress_module.ABS_VERY_NEAR,
        progress_module.REL_GRASP: progress_module.ABS_HOLDS,
    }

    def potential_with(self, relation, label):
        labels = {**self.FAR, relation: label}
        return float(self.scorer().potential(self.probs(labels), hard=True))

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

    def test_planar_distance_rewards_every_improvement_except_the_worst(self):
        labels = (
            progress_module.ABS_VERY_FAR,
            progress_module.ABS_FAR,
            progress_module.ABS_MEDIUM,
            progress_module.ABS_NEAR,
            progress_module.ABS_VERY_NEAR,
        )
        values = [
            self.potential_with(progress_module.REL_PLANAR_DISTANCE, label)
            for label in labels
        ]
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(values[0], 0.0, places=6)
        self.assertAlmostEqual(values[-1], 0.15, places=6)
        self.assertTrue(all(a < b for a, b in zip(values, values[1:])))

    def test_height_and_compatibility_have_graded_credit(self):
        height = [
            self.potential_with(progress_module.REL_HEIGHT_OFFSET, label)
            for label in (
                progress_module.ABS_FAR_BELOW,
                progress_module.ABS_BELOW,
                progress_module.ABS_LEVEL,
                progress_module.ABS_ABOVE,
                progress_module.ABS_FAR_ABOVE,
            )
        ]
        self.assertAlmostEqual(height[0], 0.0, places=6)
        self.assertAlmostEqual(height[1], 0.05, places=6)
        self.assertAlmostEqual(height[2], 0.10, places=6)
        self.assertAlmostEqual(height[3], 0.05, places=6)
        self.assertAlmostEqual(height[4], 0.0, places=6)

        for relation, budget in (
            (progress_module.REL_CONTACT_COMPAT, 0.10),
            (progress_module.REL_GRASP_COMPAT, 0.15),
        ):
            values = [
                self.potential_with(relation, label)
                for label in (
                    progress_module.ABS_UNOBSERVED,
                    progress_module.ABS_POOR_MATCH,
                    progress_module.ABS_PARTIAL_MATCH,
                    progress_module.ABS_MATCH,
                )
            ]
            self.assertTrue(all(a < b for a, b in zip(values, values[1:])))
            self.assertAlmostEqual(values[0], 0.0, places=6)
            self.assertAlmostEqual(values[-1], budget, places=6)

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

    def test_slot_zero_can_never_be_selected_as_the_target(self):
        # The end effector's logit is the largest by far and still cannot win:
        # it is not a class.
        logits = torch.zeros(2, N_MAX)
        logits[:, 0] = 100.0
        alive = torch.ones(2, N_MAX)
        weights, null = target_distribution(logits, alive)
        self.assertEqual(tuple(weights.shape), (2, N_MAX - 1))
        self.assertLess(float(null.max()), 1.0)
        # Mass is spread over the objects, none of it on the end effector.
        torch.testing.assert_close(
            weights.sum(-1) + null, torch.ones(2), atol=1e-5, rtol=1e-5
        )

    def test_an_all_inactive_table_puts_every_target_probability_on_null(self):
        logits = torch.randn(3, N_MAX)
        weights, null = target_distribution(logits, torch.zeros(3, N_MAX))
        self.assertEqual(float(weights.abs().max()), 0.0)
        torch.testing.assert_close(null, torch.ones(3), atol=1e-6, rtol=1e-6)

    def test_relations_are_mixed_after_decoding_not_before(self):
        # Mixing embeddings and then decoding is not the same function as
        # decoding each candidate and mixing the distributions, and only the
        # second one describes objects that exist.
        torch.manual_seed(0)
        decoder = SlotGraphDecoder(graph_config())
        reward = ProgressReward(self.scorer(), 1 - 1 / 333)
        slots = torch.randn(1, N_MAX, SLOT_DIM)
        alive = torch.ones(1, N_MAX)
        logits = torch.zeros(1, N_MAX)  # uniform over the object slots
        with torch.no_grad():
            mixed, _ = reward.relation_probs(decoder, slots, logits, alive)
            weights, _ = target_distribution(logits, alive)
            averaged = (weights[..., None] * slots[..., 1:, :]).sum(-2)
            naive = decoder.relation_probs(
                slots[..., 0, :], averaged, self.scorer().relations
            )
        self.assertFalse(torch.allclose(mixed, naive, atol=1e-4))
        # Still a distribution over admissible labels.
        torch.testing.assert_close(
            mixed.sum(-1), torch.ones_like(mixed.sum(-1)), atol=1e-4, rtol=1e-4
        )

    def test_no_target_and_no_live_slots_score_as_no_progress(self):
        torch.manual_seed(0)
        decoder = SlotGraphDecoder(graph_config())
        reward = ProgressReward(self.scorer(), 1 - 1 / 333)
        slots = torch.randn(2, N_MAX, SLOT_DIM)
        # Every object slot inactive: all target mass lands on null, so the
        # potential is gated to zero and nothing is scored against a phantom.
        alive = torch.zeros(2, N_MAX)
        alive[:, 0] = 1.0
        with torch.no_grad():
            shaped, potential, probs = reward(
                decoder, slots, torch.zeros(2, N_MAX), alive
            )
        self.assertTrue(torch.isfinite(probs).all())
        self.assertEqual(float(potential.abs().max()), 0.0)
        self.assertEqual(float(shaped.abs().max()), 0.0)
        self.assertEqual(tuple(probs.shape), (2, 6, 17))
        # An entirely empty table is finite too, end effector included.
        with torch.no_grad():
            shaped, potential, _ = reward(
                decoder, slots, torch.zeros(2, N_MAX), torch.zeros(2, N_MAX)
            )
        self.assertEqual(float(potential.abs().max()), 0.0)
        self.assertTrue(torch.isfinite(shaped).all())

    def test_progress_follows_a_predicted_new_target(self):
        torch.manual_seed(0)
        decoder = SlotGraphDecoder(graph_config())
        reward = ProgressReward(self.scorer(), 1 - 1 / 333)
        slots = torch.randn(1, N_MAX, SLOT_DIM)
        alive = torch.ones(1, N_MAX)
        # Confidently name slot 2, then slot 3. Different objects, so the
        # relation distributions the potential is built from must differ.
        first = torch.full((1, N_MAX), -10.0)
        first[0, 2] = 10.0
        second = torch.full((1, N_MAX), -10.0)
        second[0, 3] = 10.0
        with torch.no_grad():
            a, _ = reward.relation_probs(decoder, slots, first, alive)
            b, _ = reward.relation_probs(decoder, slots, second, alive)
        self.assertFalse(torch.allclose(a, b, atol=1e-5))

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
