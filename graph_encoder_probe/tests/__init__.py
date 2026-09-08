"""Tests for the encoder-sensitivity probe.

Everything here runs without the simulator. Pair construction, verification and
the cache are pure numpy; the model tests need torch but not ManiSkill, and
build their graphs from :mod:`graph_encoder_probe.tests.synthetic` rather than
from a rollout.
"""
