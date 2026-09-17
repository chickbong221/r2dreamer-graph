"""The action mapping is declared with evidence; measurements can only contradict it."""

from __future__ import annotations

import copy
import unittest

import numpy as np

from ..common import load_config, load_yaml, repo_path
from ..preprocessing.audit_dataset import (
    audit_actions,
    audit_gripper,
    declaration_problems,
    measured_contradictions,
    resolve_action_spec,
)

DIMS = list(load_config("dataset")["source"]["dims"])


def recordings(episodes: int = 3, n: int = 150, seed: int = 0):
    """Leader-arm style recordings: joint commands lead the state, the EEF half is the next state exactly."""
    rng = np.random.default_rng(seed)
    tables = {}
    for episode in range(episodes):
        t = np.arange(n, dtype=np.float64)
        state = np.zeros((n, 13))
        for j in range(6):
            state[:, j] = 0.6 * np.sin(0.04 * (j + 1) * t + episode) + 0.1 * j
        closed = (t > 50) & (t < 100)
        state[:, 6] = np.where(closed, 0.62, 1.6)
        state[:, 7:10] = 0.2 * np.stack([np.sin(0.03 * t), np.cos(0.03 * t), 0.5 + 0.1 * np.sin(0.05 * t)], -1)
        state[:, 10:13] = 0.3 * np.stack([np.sin(0.02 * t), np.cos(0.02 * t), np.sin(0.01 * t)], -1)
        action = np.zeros_like(state)
        action[:-1] = state[1:]
        action[-1] = state[-1]
        action[:, :6] += rng.normal(0.0, 0.004, size=(n, 6))
        action[:, 1] += 0.02
        action[:, 6] = np.where(closed, 0.55, 1.62)         # the command squeezes past the fingers
        tables[episode] = {"state": state, "action": action}
    return tables


def declaration(**changes):
    base = {
        "version": "v1", "confirmed": True, "confirmed_by": "tester", "confirmed_on": "2026-09-17",
        "command": {
            "indices": list(range(7)), "names": DIMS[:7], "representation": "absolute_joint_position",
            "units": {name: "rad" for name in DIMS[:7]},
            "evidence": [
                {"kind": "recording_code", "source": "teleop.py", "statement": "records leader joint positions"},
                {"kind": "measurement", "source": "audit", "statement": "tracks the next state"},
            ],
        },
        "derived": {"indices": list(range(7, 13)), "rule": "action[t][i] == state[t+1][i]",
                    "evidence": [{"kind": "measurement", "source": "audit", "statement": "equal to the next state"}]},
        "gripper": {"action_index": 6, "state_index": 6, "units": "rad", "open_value": 1.6, "closed_value": 0.6,
                    "evidence": [{"kind": "hardware", "source": "bench", "statement": "open 1.6, closed 0.6"}]},
        "accepted_contradictions": [],
    }
    out = copy.deepcopy(base)
    for dotted, value in changes.items():
        node = out
        parts = dotted.split("__")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return out


class ActionMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = recordings()
        cls.actions = audit_actions(cls.tables, DIMS)
        cls.gripper = audit_gripper(cls.tables, 6)
        cls.all_actions = np.concatenate([t["action"] for t in cls.tables.values()])

    def spec(self, decl, integrity=()):
        return resolve_action_spec(decl, {"hash": "test"}, DIMS, self.actions, self.gripper, list(integrity),
                                   self.all_actions, len(self.tables), 0.05)

    def test_the_synthetic_recordings_measure_as_intended(self):
        kinds = {r["name"]: r["classification"] for r in self.actions}
        self.assertTrue(all(kinds[name] == "absolute_command" for name in DIMS[:7]), kinds)
        self.assertTrue(all(kinds[name] == "derived_next_state" for name in DIMS[7:]), kinds)

    def test_a_consistent_confirmed_declaration_with_evidence_is_verified(self):
        spec = self.spec(declaration())
        self.assertTrue(spec["verified"], spec["problems"])
        self.assertEqual(spec["command_indices"], list(range(7)))
        self.assertEqual(spec["gripper"]["source"], "declared")
        self.assertEqual(spec["normalization"]["fit_episodes"], "all")
        self.assertEqual(len(spec["normalization"]["low"]), 7)

    def test_measurements_alone_never_verify(self):
        only_numbers = [{"kind": "measurement", "source": "audit", "statement": "tracks the next state"}]
        decl = declaration(command__evidence=only_numbers, gripper__evidence=only_numbers)
        spec = self.spec(decl)
        self.assertFalse(spec["verified"])
        self.assertTrue(any("no evidence other than measurement" in p for p in spec["problems"]))

    def test_an_unconfirmed_declaration_is_not_verified(self):
        for decl in (declaration(confirmed=False), declaration(confirmed_by="")):
            self.assertFalse(self.spec(decl)["verified"])

    def test_unknown_units_and_missing_gripper_values_are_refused(self):
        units = {name: "rad" for name in DIMS[:7]}
        units["gripper"] = "unknown"
        problems = declaration_problems(declaration(command__units=units, gripper__open_value=None), 13)
        self.assertTrue(any("no known unit" in p and "gripper" in p for p in problems))
        self.assertTrue(any("open_value and closed_value" in p for p in problems))

    def test_every_dimension_must_be_declared(self):
        problems = declaration_problems(declaration(derived__indices=list(range(7, 12))), 13)
        self.assertTrue(any("neither command nor derived" in p for p in problems))

    def test_a_contradicting_representation_blocks_until_accepted_with_a_reason(self):
        decl = declaration(command__representation="joint_displacement")
        spec = self.spec(decl)
        self.assertFalse(spec["verified"])
        self.assertTrue(any("command_representation:waist" in p for p in spec["problems"]))
        checks = [c["check"] for c in measured_contradictions(decl, self.actions, self.gripper)]
        accepted = [{"check": check, "reason": "confirmed on the robot"} for check in checks]
        self.assertTrue(self.spec(declaration(command__representation="joint_displacement",
                                              accepted_contradictions=accepted))["verified"])
        unexplained = [{"check": check, "reason": ""} for check in checks]
        self.assertFalse(self.spec(declaration(command__representation="joint_displacement",
                                               accepted_contradictions=unexplained))["verified"])

    def test_a_hindsight_label_declared_as_a_command_is_a_contradiction(self):
        decl = declaration(command__indices=list(range(8)), command__names=DIMS[:8],
                           command__units={name: "rad" for name in DIMS[:8]}, derived__indices=list(range(8, 13)))
        checks = [c["check"] for c in measured_contradictions(decl, self.actions, self.gripper)]
        self.assertIn("command_is_not_derived:eef_x", checks)

    def test_gripper_values_far_from_the_recordings_contradict(self):
        decl = declaration(gripper__open_value=0.6, gripper__closed_value=1.6)
        checks = [c["check"] for c in measured_contradictions(decl, self.actions, self.gripper)]
        self.assertIn("gripper_closing_direction", checks)
        self.assertFalse(self.spec(decl)["verified"])

    def test_data_integrity_problems_still_block(self):
        self.assertFalse(self.spec(declaration(), integrity=["videos do not align with the recorded rows"])["verified"])

    def test_the_shipped_declaration_is_not_yet_verifiable(self):
        shipped = load_yaml(repo_path(load_config("dataset")["action"]["mapping"]))
        problems = declaration_problems(shipped, 13)
        self.assertTrue(any("not confirmed" in p for p in problems))
        self.assertTrue(any("no known unit" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
