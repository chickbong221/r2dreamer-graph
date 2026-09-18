"""One loader for the pretrained SmolVLA checkpoint, with a real pin.

Every caller -- training, evaluation, tests -- goes through :func:`load_policy`,
so a run cannot be trained against one revision and evaluated against another.

"Pinned" here means an immutable commit hash, not a branch name. ``main`` is a
moving target: two runs a week apart can both say they used ``main`` and have
different weights, and nothing in either checkpoint would say so. The requested
revision is therefore resolved to a SHA *before* anything is loaded, that SHA is
what gets loaded, and a resolution failure is an error rather than a silent
fall-through to whatever happens to be cached.

LeRobot's own version is pinned too. :data:`VERIFIED_LEROBOT` lists the
versions whose source was actually read: ``embed_prefix`` / ``embed_suffix`` /
``denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)`` on
``policy.model``, the flow convention ``x_t = t*noise + (1-t)*actions``, and
``make_att_2d_masks`` defined inside ``modeling_smolvla``. Those are identical
across the listed versions; an unlisted one is reported rather than assumed
compatible.

**0.4.4 is the default because 0.6.1 cannot be installed here.** LeRobot 0.6.1
requires Python >= 3.12 and this project pins ``>=3.11,<3.12``; 0.4.4 requires
>= 3.10 and accepts ``torch<2.11``, which the pinned torch 2.8.0 satisfies.
0.3.3 is excluded for the opposite reason: it caps torch below 2.8.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_REPO = "lerobot/smolvla_base"
# Versions whose source was read and whose SmolVLA interface matches what this
# integration calls. Ordered: the first is the one to install here.
VERIFIED_LEROBOT = ("0.4.4", "0.6.1")
SUPPORTED_LEROBOT = VERIFIED_LEROBOT[0]
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


def check_environment() -> None:
    """Name a broken transformers/huggingface_hub pair before it surfaces deep.

    ``transformers`` imports ``is_offline_mode`` from ``huggingface_hub``, and
    the two are released in coherent pairs: transformers 4.x with hub 0.3x, and
    transformers 5.x with hub 1.x. A mixed pair fails on ``import
    transformers`` with an ImportError that names neither package's version,
    which is a bad first thing to see on a new machine.
    """
    try:
        import huggingface_hub
        import transformers
    except ImportError as exc:
        message = str(exc)
        if "is_offline_mode" in message or "huggingface_hub" in message:
            versions = []
            for name in ("transformers", "huggingface_hub"):
                try:
                    from importlib.metadata import version

                    versions.append(f"{name}=={version(name)}")
                except Exception:                          # noqa: BLE001
                    versions.append(f"{name}=?")
            raise PretrainedError(
                f"transformers and huggingface_hub are an incompatible pair "
                f"({', '.join(versions)}): {message}. They ship in matched "
                "pairs -- transformers 4.x with hub 0.3x, transformers 5.x "
                "with hub 1.x. For lerobot "
                f"{SUPPORTED_LEROBOT} install: "
                "'transformers>=4.57.1,<5.0.0' 'huggingface-hub>=0.34.2,<0.36.0'"
            ) from exc
        raise PretrainedError(f"cannot import transformers: {exc}") from exc


def lerobot_version() -> str:
    check_environment()
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
    if strict_version and version not in VERIFIED_LEROBOT:
        raise PretrainedError(
            f"lerobot {version} is installed; the SmolVLA interface was read "
            f"and verified at {list(VERIFIED_LEROBOT)}. Pin one of those, or "
            "re-read sim_vla/models/smolvla_actor.py against the installed "
            "source before trusting it.")

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
