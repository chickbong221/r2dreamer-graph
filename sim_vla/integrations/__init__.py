"""Latent-conditioned SmolVLA for TD-MPC2 and SOLD.

Neither backend is replaced. Each keeps its own world model, its own latent,
its own losses, its own reward convention and its own online algorithm; what
is added is a small adapter from that backend's latent to one SmolVLA
conditioning token, a chunk-imitation stage that trains the adapter and the
action expert against a frozen world model, and a set of narrow hooks that let
the native online algorithm call SmolVLA where it used to call its own
Gaussian policy.

Three stages, per backend::

    demonstrations ---> 1  the backend's own world-model pretraining
                        |     TD-MPC2: encoder + dynamics + reward + Q ensemble
                        |              + the auxiliary Gaussian policy, coupled
                        |     SOLD:    SAVi, then slot dynamics + reward head
                        +-> 2  adapter + SmolVLA action expert, frozen world
                            |  model, chunk-based flow matching
                            +-> 3  the backend's own model-based RL, with
                                   SmolVLA supplying the actions

Layout::

    vendor.py        importing two upstream trees that both claim ``envs``
    params.py        counting what was built, without double counting
    observations.py  recorded cameras into each backend's image contract
    action_space.py  the one boundary between actor units and native units
    chunking.py      the shared imitation contract (causality, masks, chunks)
    latent_actor.py  SmolVLA behind sampling hooks, and no fake log-probability
    checkpoint.py    metadata a set of weights cannot be interpreted without
    tdmpc2/          config, data, policy hooks, three stages, entry point
    sold/            the same, for slot dynamics
    tests/           native-behaviour, integration and gradient tests
"""

from __future__ import annotations

__all__ = ["vendor", "params", "observations", "action_space", "chunking",
           "latent_actor", "checkpoint"]
