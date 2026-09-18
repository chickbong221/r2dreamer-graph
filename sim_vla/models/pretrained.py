"""One loader for the pretrained SmolVLA checkpoint, with a real pin.

Every caller -- training, evaluation, tests -- goes through :func:`load_policy`,
so a run cannot be trained against one revision and evaluated against another.

"Pinned" here means an immutable commit hash, not a branch name. ``main`` is a
moving target: two runs a week apart can both say they used ``main`` and have
different weights, and nothing in either checkpoint would say so. The requested
revision is therefore resolved to a SHA *before* anything is loaded, that SHA is
what gets loaded, and a resolution failure is an error rather than a silent
fall-through to whatever happens to be cached.

LeRobot's own version is pinned too, in :data:`SUPPORTED_LEROBOT`. The
integration is written against the interface at that version -- ``embed_prefix``
/ ``embed_suffix`` / ``denoise_step`` on ``policy.model``, with the flow
convention and the action padding those imply -- and a different version is
reported rather than assumed compatible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_REPO = "lerobot/smolvla_base"
# The version this integration was written against, by reading its source.
SUPPORTED_LEROBOT = "0.6.1"
_SHA = re.compile(r"^[0-9a-f]{40}$")


class PretrainedError(RuntimeError):
    """A real failure to load. Never converted into a skip."""


@dataclass
class LoadedPolicy:
    """The policy and the facts a run has to record about it."""

    policy: Any
    repo_id: str
    revision: str            # always an immutable commit hash
    requested: str
    lerobot_version: str
    local_dir: Optional[str] = None

    @property
    def model(self) -> Any:
        """``VLAFlowMatching`` -- where the flow methods actually live."""
        model = getattr(self.policy, "model", None)
        if model is None:
            raise PretrainedError(
                "this SmolVLAPolicy has no .model (VLAFlowMatching); the "
                f"integration targets lerobot {SUPPORTED_LEROBOT}")
        return model

    @property
    def config(self) -> Any:
        config = getattr(self.policy, "config", None)
        if config is None:
            raise PretrainedError("this SmolVLAPolicy carries no config")
        return config


def resolve_revision(repo_id: str, revision: str = "main") -> str:
    """Turn a branch or tag into the commit it points at, or fail loudly."""
    if _SHA.match(str(revision or "")):
        return str(revision)
    try:
        from huggingface_hub import HfApi

        sha = HfApi().model_info(repo_id, revision=revision).sha
    except Exception as exc:                               # noqa: BLE001
        raise PretrainedError(
            f"could not resolve {repo_id}@{revision} to a commit: "
            f"{type(exc).__name__}: {exc}. A run pinned to a branch is not "
            "pinned; fix the network or hub credentials, or configure an "
            "explicit 40-character commit hash.") from exc
    if not sha or not _SHA.match(str(sha)):
        raise PretrainedError(
            f"{repo_id}@{revision} resolved to {sha!r}, which is not a commit hash")
    return str(sha)


def lerobot_version() -> str:
    try:
        import lerobot

        return str(getattr(lerobot, "__version__", "unknown"))
    except Exception as exc:                               # noqa: BLE001
        raise PretrainedError(f"lerobot is not importable: {exc}") from exc


def load_policy(repo_id: str = DEFAULT_REPO, revision: str = "main", *,
                local_dir: Optional[str | Path] = None,
                strict_version: bool = False) -> LoadedPolicy:
    """Load the checkpoint at an immutable revision.

    ``local_dir`` downloads a snapshot there first and loads from it, for a
    machine that should not depend on the shared Hugging Face cache. Without
    it the weights come from that cache, which is the default and is stated
    rather than implied.
    """
    version = lerobot_version()
    if strict_version and version != SUPPORTED_LEROBOT:
        raise PretrainedError(
            f"lerobot {version} is installed; this integration was written "
            f"against {SUPPORTED_LEROBOT}. Pin it, or re-read "
            "sim_vla/models/smolvla_actor.py against the installed source.")

    sha = resolve_revision(repo_id, revision)
    source: str | Path = repo_id
    if local_dir is not None:
        from huggingface_hub import snapshot_download

        source = snapshot_download(repo_id=repo_id, revision=sha,
                                   local_dir=str(local_dir))

    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except Exception as exc:                               # noqa: BLE001
        raise PretrainedError(
            f"cannot import SmolVLAPolicy from lerobot {version}: {exc}") from exc

    try:
        policy = (SmolVLAPolicy.from_pretrained(str(source))
                  if local_dir is not None
                  else SmolVLAPolicy.from_pretrained(repo_id, revision=sha))
    except Exception as exc:                               # noqa: BLE001
        raise PretrainedError(
            f"failed to load {repo_id}@{sha}: {type(exc).__name__}: {exc}") from exc

    return LoadedPolicy(
        policy=policy, repo_id=repo_id, revision=sha, requested=str(revision),
        lerobot_version=version,
        local_dir=str(local_dir) if local_dir is not None else None)


def model_facts(loaded: LoadedPolicy) -> Dict[str, Any]:
    """The widths a caller must not assume, read off the loaded model.

    Every one of these has a documented default that the checkpoint is free to
    disagree with, so the adapter width, the chunk length and the action
    padding are taken from here rather than from the configuration file.
    """
    config = loaded.config
    model = loaded.model
    facts: Dict[str, Any] = {
        "chunk_size": int(getattr(config, "chunk_size", 0)),
        "n_action_steps": int(getattr(config, "n_action_steps", 0)),
        "max_state_dim": int(getattr(config, "max_state_dim", 0)),
        "max_action_dim": int(getattr(config, "max_action_dim", 0)),
        "num_steps": int(getattr(config, "num_steps", 0)),
        "tokenizer_max_length": int(getattr(config, "tokenizer_max_length", 0)),
    }
    # Widths are read from the real projection layers, which is the only place
    # they cannot be wrong.
    state_proj = getattr(model, "state_proj", None)
    if state_proj is not None:
        facts["vlm_hidden_size"] = int(state_proj.out_features)
        facts["state_proj_in"] = int(state_proj.in_features)
    action_in = getattr(model, "action_in_proj", None)
    if action_in is not None:
        facts["expert_hidden_size"] = int(action_in.out_features)
        facts["action_in_dim"] = int(action_in.in_features)
    action_out = getattr(model, "action_out_proj", None)
    if action_out is not None:
        facts["action_out_dim"] = int(action_out.out_features)
    missing = [k for k in ("vlm_hidden_size", "expert_hidden_size")
               if k not in facts]
    if missing:
        raise PretrainedError(
            f"could not read {missing} from the loaded model; expected "
            "state_proj / action_in_proj on VLAFlowMatching at lerobot "
            f"{SUPPORTED_LEROBOT}")
    return facts
