"""Offline training on the ALOHA kitchen demonstrations.

``locht131/aloha_placing_kitchen_lerobot``: 150 teleoperated episodes of one
right arm putting a banana in a pot and closing the lid, two RGB cameras, 15 Hz.
There is no simulator behind any of it, so everything the ManiSkill path reads
off privileged state -- graph facts, geometry, reward, termination -- is
reconstructed here once, saved, and then read by training as plain arrays.

Every episode trains; a fixed subset of them, the diagnostic episodes, is where
fitting and pipeline behaviour are watched.

Stages, in the order they run:

1. ``preprocessing`` -- download, audit the recorded fields against the
   declared action mapping, Gemini annotation, tracking, Depth Pro geometry,
   and the packed dataset.
2. ``rewards`` -- the dense kitchen reward over the saved geometry and labels.
3. ``training`` -- the world model first, then one latent cache from the frozen
   checkpoint, one-step imagined transitions, and model-assisted IQL, with the
   progress-aware branch as a separate option.
4. ``evaluation`` -- overlays, reward inspection, world-model diagnostics and
   the robot inference wrapper.

The repository's own model and graph modules are imported unchanged; nothing
under this package is imported by them.
"""
