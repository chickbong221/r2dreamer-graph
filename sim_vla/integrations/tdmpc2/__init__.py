"""TD-MPC2 with a latent-conditioned SmolVLA policy.

The world model, the MPPI planner, the Q ensemble, the bootstrap structure,
the target updates and the planning settings are upstream's and are unchanged.
What is added:

* an adapter from TD-MPC2's own latent ``z`` to one SmolVLA conditioning token
* a chunk-imitation stage that trains that adapter and the action expert
  against a frozen world model
* five named hooks in ``sim_vla/tdmpc2/tdmpc2.py`` where the agent asks its
  learned policy for an action, so SmolVLA can answer instead

With ``smolvla.enabled: false`` the hooks are inert and the agent is upstream's.
"""

from __future__ import annotations

__all__ = ["config", "data", "agent", "policy", "stages", "online", "run"]
