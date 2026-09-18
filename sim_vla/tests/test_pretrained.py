"""Stage 4: the real pretrained SmolVLA checkpoint loads and is wired in.

Skipped, never passed, when lerobot or the weights are unavailable.
"""

from __future__ import annotations

import unittest

from .common import require, require_torch


def load_policy():
    require("lerobot")
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except Exception as exc:                               # noqa: BLE001
        raise unittest.SkipTest(f"SmolVLAPolicy unavailable: {exc}")
    try:
        return SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    except Exception as exc:                               # noqa: BLE001
        raise unittest.SkipTest(f"pretrained weights unavailable: {exc}")


class TestPretrained(unittest.TestCase):
    def test_checkpoint_loads_and_has_parameters(self):
        policy = load_policy()
        total = sum(p.numel() for p in policy.parameters())
        self.assertGreater(total, 1_000_000)

    def test_expert_and_language_modules_resolve(self):
        """The attribute paths this repo searches must match the checkpoint.

        On failure the error names every path tried and prints the real module
        tree, which is what to paste into smolvla_actor.py for a version bump.
        """
        policy = load_policy()
        from sim_vla.models.smolvla_actor import (
            EXPERT_PATHS, LANGUAGE_PATHS, resolve,
        )

        expert, expert_path = resolve(policy, EXPERT_PATHS, "action expert")
        language, language_path = resolve(policy, LANGUAGE_PATHS, "language")
        self.assertIsNotNone(expert)
        self.assertIsNotNone(language)
        print(f"[pretrained] expert={expert_path} language={language_path}")

    def test_actor_wraps_it_and_reports_what_is_trainable(self):
        require_torch()
        policy = load_policy()
        from sim_vla.models.latent_adapter import LatentAdapter
        from sim_vla.models.smolvla_actor import SmolVLAActor

        token_dim = getattr(getattr(policy, "config", None), "hidden_size", 960)
        adapter = LatentAdapter(feature_dim=64, token_dim=int(token_dim))
        actor = SmolVLAActor(policy, adapter, chunk_size=8, action_dim=8)
        report = actor.trainable_report()
        print("[pretrained]", report)
        self.assertIn("adapter", report["trainable_prefixes"])
        self.assertGreater(report["parameters_trainable"], 0)
        self.assertLess(report["parameters_trainable"],
                        report["parameters_total"])


if __name__ == "__main__":
    unittest.main()
