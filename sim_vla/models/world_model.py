"""The world model for both arms, composed from the simulator's own parts.

Nothing here reimplements a recurrent state, an encoder or a graph loss. The
components are the ones ``dreamer.py`` builds -- ``MultiEncoder``, ``RSSM``,
``MultiDecoder``, ``MLPHead``, ``GraphEncoder``, ``SimpleGraphDecoder`` -- wired
together with sim_vla's own training step, so that the simulator code stays
exactly as it is and the two pipelines cannot drift in what a latent means.

One switch decides the arm, and it decides construction, not masking::

    graph_enabled = False -> RSSM(semantic=False), feature = (h, z)
    graph_enabled = True  -> RSSM(semantic=True),  feature = (h, z, g)

A baseline has no graph encoder, no semantic head, no graph decoder and no
graph loss term. It is not a graph model with ``g`` zeroed: there is no ``g``,
``get_feat`` is never handed one, and the feature is narrower. That is why an
arm is chosen before its world model is trained rather than after.

Terminations follow the online environment. ``envs/maniskill.py`` builds it with
``ignore_terminations=True``, so the continuation head is trained against a
signal that is one everywhere inside an episode and bootstraps at the horizon;
the recorded terminal flags stay in the dataset as diagnostics and never reach
this loss. See ``sim_vla/data/dataset.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

import networks
import rssm as rssm_module
from graph import GraphEncoder, SimpleGraphDecoder, compact_graph
from scenegraph.adapters.graph_pack import graph_keys

GRAPH_KEYS = tuple(graph_keys())


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the steps a window actually scores.

    Burn-in and padding are in the arrays and out of the loss; dividing by the
    full length instead would scale every term by how much padding a window
    happened to need.
    """
    mask = mask.to(value.dtype)
    while mask.dim() < value.dim():
        mask = mask[..., None]
    return (value * mask).sum() / mask.sum().clamp(min=1.0)


class WorldModel(nn.Module):
    """Encoder, recurrent dynamics, decoders and heads for one arm."""

    def __init__(self, config, obs_shapes: Mapping[str, Tuple[int, ...]],
                 act_dim: int, *, graph_enabled: bool):
        super().__init__()
        self.config = config
        self.graph_enabled = bool(graph_enabled)
        self.act_dim = int(act_dim)

        shapes = {k: tuple(v) for k, v in obs_shapes.items()}
        # The graph arrays are never encoder inputs: they go to the graph
        # encoder, and a pixel/state encoder that matched them by key would
        # learn the same facts twice.
        model_shapes = {k: v for k, v in shapes.items() if k not in GRAPH_KEYS}
        if self.graph_enabled:
            missing = [k for k in GRAPH_KEYS if k not in shapes]
            if missing:
                raise ValueError(f"graph arm is missing observations: {missing}")

        self.encoder = networks.MultiEncoder(config.encoder, model_shapes)
        self.embed_size = int(self.encoder.out_dim)
        self.image_keys = tuple(self.encoder.cnn_shapes)

        self.graph_encoder = GraphEncoder(config.graph) if self.graph_enabled else None
        graph_token_size = int(self.graph_encoder.units) if self.graph_enabled else 0
        self.graph_dim = int(config.graph.semantic_dim) if self.graph_enabled else 0

        self.rssm = rssm_module.RSSM(
            config.rssm, self.embed_size, self.act_dim,
            semantic=self.graph_enabled,
            graph_token_size=graph_token_size,
            graph_dim=self.graph_dim,
        )
        self.decoder = networks.MultiDecoder(
            config.decoder, int(config.rssm.deter), int(self.rssm.flat_stoch),
            model_shapes, flat_sem=self.graph_dim,
        )
        self.graph_decoder = (
            SimpleGraphDecoder(config.graph, self.graph_dim)
            if self.graph_enabled else None)

        self.reward_head = networks.MLPHead(config.reward, self.feature_dim)
        self.cont_head = networks.MLPHead(config.cont, self.feature_dim)

    # ------------------------------------------------------------------ shape
    @property
    def feature_dim(self) -> int:
        """Width of ``(h, z)`` or ``(h, z, g)``. The arms differ here."""
        return (int(self.config.rssm.deter) + int(self.rssm.flat_stoch)
                + (self.graph_dim if self.graph_enabled else 0))

    def initial(self, batch_size: int):
        return self.rssm.initial(int(batch_size))

    # ---------------------------------------------------------------- encode
    def graph_token(self, batch: Mapping[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Per-step graph embedding, or None for the baseline.

        Returning None rather than zeros is the point: ``RSSM.observe`` refuses
        a semantic rollout without a token, so a misconfigured baseline fails
        instead of training on a constant.
        """
        if not self.graph_enabled:
            return None
        graph = {key: batch[key] for key in GRAPH_KEYS}
        return self.graph_encoder(graph).token

    def observe(self, batch: Mapping[str, torch.Tensor], initial=None
                ) -> Dict[str, Any]:
        """Posterior rollout over a window. No future observation is used."""
        embed = self.encoder(batch)
        action = batch["action"]
        reset = batch["is_first"]
        state = self.initial(action.shape[0]) if initial is None else initial
        post = self.rssm.observe(
            embed, action, state, reset, graph_token=self.graph_token(batch))
        return {"embed": embed, "post": post}

    @staticmethod
    def unpack(post, graph_enabled: bool):
        """``observe`` returns 3 values, or 4 with the graph branch on.

        ``(stoch, deter, post_logit)`` and then ``sem``. There is no prior
        logit in there -- it comes from ``rssm.prior`` -- and ``post[2]`` is
        the posterior logit, not ``sem``. Unpacked in one place because
        indexing it by number at each use is how that gets mixed up.
        """
        stoch, deter, post_logit = post[:3]
        sem = post[3] if graph_enabled else None
        return stoch, deter, post_logit, sem

    def features(self, post) -> torch.Tensor:
        stoch, deter, _logit, sem = self.unpack(post, self.graph_enabled)
        if self.graph_enabled:
            return self.rssm.get_feat(stoch, deter, sem)
        return self.rssm.get_feat(stoch, deter)

    # ------------------------------------------------------------------ loss
    def loss(self, batch: Mapping[str, torch.Tensor], initial=None
             ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        """Total loss, its terms, and the state the caller may want to keep."""
        scales = self.config.loss_scales
        out = self.observe(batch, initial)
        post = out["post"]
        stoch, deter, post_logit, sem = self.unpack(post, self.graph_enabled)
        feat = self.features(post)
        mask = batch["loss_mask"]

        losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, torch.Tensor] = {}

        # MultiDecoder takes stoch first, then deter. Passing them the other
        # way round type-checks and trains on nonsense.
        recon = self.decoder(stoch, deter, sem)
        for key, dist in recon.items():
            if key in batch:
                losses[f"recon_{key}"] = masked_mean(-dist.log_prob(batch[key]), mask)

        losses["reward"] = masked_mean(
            -self.reward_head(feat).log_prob(batch["reward"]), mask)
        # cont is 1 wherever the episode continues. Under the online policy
        # that is everywhere inside a window, and the bootstrap at the end is
        # the value function's job rather than a zero taught here.
        cont_target = (1.0 - batch["is_terminal"].float())
        losses["cont"] = masked_mean(
            -self.cont_head(feat).log_prob(cont_target), mask)

        # The prior is not returned by observe: it is computed from the
        # posterior deter, exactly as dreamer.py:1118 does.
        _prior_stoch, prior_logit = self.rssm.prior(deter, sem)
        dyn, rep = self.rssm.kl_loss(
            post_logit, prior_logit, float(self.config.kl_free))
        losses["dyn"] = masked_mean(dyn, mask)
        losses["rep"] = masked_mean(rep, mask)

        if self.graph_enabled:
            prior_sem = self.rssm.semantic_prior_seq(deter)
            # One call gives both terms. They share a forward value and the
            # loss scales express the asymmetry between them, so computing one
            # of them by hand would be a different objective wearing the same
            # name.
            sem_dyn, sem_rep = self.rssm.semantic_align_loss(sem, prior_sem)
            losses["graphdyn"] = masked_mean(sem_dyn, mask)
            losses["graphrep"] = masked_mean(sem_rep, mask)
            metrics["graph_align_mse"] = losses["graphdyn"].detach()
            # Three return values: the error, and the two RMS gauges that say
            # whether the prior is the right shape at the wrong scale.
            amp_error, prior_rms, post_rms = self.rssm.semantic_amplitude_loss(
                sem, prior_sem)
            losses["graphamp"] = masked_mean(amp_error, mask)
            with torch.no_grad():
                metrics["graph_sem_prior_rms"] = masked_mean(prior_rms, mask)
                metrics["graph_sem_post_rms"] = masked_mean(post_rms, mask)
                metrics["graph_align_cos"] = masked_mean(
                    (self.rssm.rms(sem) * self.rssm.rms(prior_sem)).mean(-1),
                    mask)
            graph = {key: batch[key] for key in GRAPH_KEYS}
            compact = compact_graph(graph)
            flat_sem = sem.reshape(-1, sem.shape[-1])
            graph_losses, graph_metrics = self.graph_decoder(
                flat_sem, compact, mask.reshape(-1))
            losses |= {f"graph_{k}": v for k, v in graph_losses.items()}
            metrics |= {f"graph_{k}": v for k, v in graph_metrics.items()}

        total = sum(
            float(getattr(scales, _scale_name(name), 1.0)) * value
            for name, value in losses.items())
        metrics |= {name: value.detach() for name, value in losses.items()}
        return total, losses, {"post": post, "feat": feat, "metrics": metrics}


def _scale_name(loss_name: str) -> str:
    """Map a loss term to its ``loss_scales`` entry.

    Reconstruction terms are per observation key and share one scale, which is
    how ``dreamer.py`` weights them.
    """
    if loss_name.startswith("recon_"):
        return "image" if loss_name.startswith("recon_image") else "vector"
    if loss_name.startswith("graph_"):
        return loss_name[len("graph_"):]
    return loss_name


def build_world_model(config, obs_shapes, act_dim, *, graph_enabled: bool
                      ) -> WorldModel:
    """Construct an arm's world model and report what it did or did not build."""
    model = WorldModel(config, obs_shapes, act_dim, graph_enabled=graph_enabled)
    kind = "graph (h,z,g)" if graph_enabled else "baseline (h,z)"
    print(f"[world_model] {kind}, feature_dim={model.feature_dim}, "
          f"graph_encoder={'yes' if model.graph_encoder else 'no'}, "
          f"graph_decoder={'yes' if model.graph_decoder else 'no'}", flush=True)
    return model
