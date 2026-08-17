import copy
import math
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

import networks
import rssm
import tools
from graph import (
    GraphDecoder,
    GraphEncoder,
    SimpleGraphDecoder,
    SlotGraphDecoder,
    compact_graph,
    graph_from,
    graph_keys,
    graph_state_mode,
)
from networks import Projector
from progress import ProgressReward, ProgressScorer, load_stages, target_distribution
from optim import LaProp, clip_grad_agc_
from tools import to_f32


def _check_finite_tensors(scope, tensors):
    """Fail with tensor names and locations at the first non-finite boundary."""
    problems = []
    for name, value in tensors:
        if value is None or not (
            torch.is_floating_point(value) or torch.is_complex(value)
        ):
            continue
        finite = torch.isfinite(value)
        if bool(finite.all()):
            continue
        invalid = ~finite
        first = torch.nonzero(invalid, as_tuple=False)[0].tolist()
        bad_rows = []
        if value.ndim:
            row_invalid = invalid.reshape(value.shape[0], -1).any(-1)
            bad_rows = torch.nonzero(row_invalid, as_tuple=False).flatten().tolist()
        finite_values = value[finite]
        finite_peak = (
            float(finite_values.abs().max()) if finite_values.numel() else float("nan")
        )
        problems.append(
            f"{name}: {int(invalid.sum())} invalid value(s), first index {first}, "
            f"bad leading rows={bad_rows[:16]}, finite |max|={finite_peak:.4g}, "
            f"shape={tuple(value.shape)}, dtype={value.dtype}"
        )
    if problems:
        raise FloatingPointError(f"Non-finite {scope}:\n  " + "\n  ".join(problems))


def _check_finite_replay(
    data, initial, *, latent_names=("stoch", "deter", "sem"), log_magnitudes=False
):
    """Fail before model execution if a sampled replay tensor is non-finite."""
    tensors = [(f"data/{key}", value) for key, value in data.items()]
    tensors.extend(
        (f"initial/{latent_names[index]}", value)
        for index, value in enumerate(initial)
    )
    _check_finite_tensors("replay sample before preprocessing/autocast", tensors)
    if not log_magnitudes:
        return
    # Raw magnitudes are diagnostic only. Preprocessing (for example symlog)
    # may change them, and a GEMM can overflow even when every input is below
    # the float16 scalar limit.
    for name, value in tensors:
        if not torch.is_floating_point(value):
            continue
        peak = float(value.abs().max())
        marker = "  <-- exceeds float16 range" if peak > 65504.0 else ""
        print(f"[dreamer] replay |max| {name}={peak:.4g}{marker}", flush=True)


def _mask_terminal_graph(token, is_last):
    """Remove stale graph conditioning on auto-reset terminal observations."""
    valid = ~is_last.bool()
    if valid.ndim == token.ndim and valid.shape[-1] == 1:
        valid = valid.squeeze(-1)
    while valid.ndim < token.ndim:
        valid = valid.unsqueeze(-1)
    return token * valid.to(token.dtype)


def _masked_mean(values, mask):
    mask = mask.float()
    return (values.float() * mask).sum() / mask.sum().clamp_min(1)


def _step_valid(is_last):
    """(B, T) mask of frames that are a real observation."""
    valid = ~is_last.bool()
    if valid.ndim == 3 and valid.shape[-1] == 1:
        valid = valid[..., 0]
    return valid


class Dreamer(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = networks.ReturnEMA(device=self.device)
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)
        self.rep_loss = str(config.rep_loss)
        self.graph_enabled = bool(config.graph.enabled)
        self.graph_only = bool(getattr(config, "graph_only_latent", False))
        self.graph_simple = bool(getattr(config, "graph_simple", False))
        self.graph_slots = (
            self.graph_enabled and graph_state_mode(config.graph) == "slots"
        )
        if self.graph_only and not self.graph_enabled:
            raise ValueError("graph_only_latent=true requires graph.enabled=true")
        if self.graph_simple and not self.graph_enabled:
            raise ValueError("graph_simple=true requires graph.enabled=true")
        if self.graph_slots and not self.graph_simple:
            raise ValueError(
                "graph.state_mode=slots is a relation-only mode; set "
                "model.graph_simple=true as well"
            )
        if self.graph_slots and self.graph_only:
            raise ValueError("graph.state_mode=slots and graph_only_latent conflict")
        if self.graph_simple and self.graph_only:
            raise ValueError(
                "graph_simple and graph_only_latent are mutually exclusive: one "
                "keeps a stock z branch, the other removes z entirely"
            )
        self.graph_keys = graph_keys(self.graph_simple)
        amp_name = str(config.amp_dtype)
        amp_dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        if amp_name not in amp_dtypes:
            raise ValueError(
                f"Unknown model.amp_dtype={amp_name!r}; expected one of {sorted(amp_dtypes)}"
            )
        self._amp_dtype = amp_dtypes[amp_name]
        if self.graph_enabled and self.rep_loss != "dreamer":
            raise ValueError("graph.enabled is a DreamerV3 extension; set model.rep_loss=dreamer")

        # World model components
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        if self.graph_enabled:
            missing = [key for key in self.graph_keys if key not in shapes]
            if missing:
                raise ValueError(
                    f"graph.enabled observation space is missing: {missing}"
                )
        # Every graph key of either contract is excluded from the pixel/state
        # encoder, so a stale key left in the space never reaches the CNN.
        all_graph_keys = set(graph_keys(True)) | set(graph_keys(False))
        model_shapes = {
            key: value for key, value in shapes.items() if key not in all_graph_keys
        }
        encoder_shapes = (
            {key: value for key, value in model_shapes.items() if len(value) != 3}
            if self.graph_only
            else model_shapes
        )
        self.encoder = networks.MultiEncoder(config.encoder, encoder_shapes)
        self.image_keys = tuple(self.encoder.cnn_shapes)
        self.embed_size = self.encoder.out_dim
        self.graph_encoder = GraphEncoder(config.graph) if self.graph_enabled else None
        graph_token_size = (
            0 if self.graph_slots
            else (int(self.graph_encoder.units) if self.graph_enabled else 0)
        )
        self.graph_dim = (
            int(config.graph.semantic_dim)
            if (self.graph_simple and not self.graph_slots)
            else 0
        )
        self.rssm = rssm.RSSM(
            config.rssm,
            self.embed_size,
            self.act_dim,
            semantic=self.graph_enabled,
            graph_token_size=graph_token_size,
            graph_only=self.graph_only,
            graph_simple=self.graph_simple,
            graph_dim=self.graph_dim,
            graph_slots=self.graph_slots,
            graph_config=config.graph if self.graph_enabled else None,
        )
        self.reward = networks.MLPHead(config.reward, self.rssm.feat_size)
        self.cont = networks.MLPHead(config.cont, self.rssm.feat_size)

        config.actor.shape = (act_space.n,) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))
        self.act_discrete = False
        if hasattr(act_space, "multi_discrete"):
            config.actor.dist = config.actor.dist.multi_disc
            self.act_discrete = True
        elif hasattr(act_space, "discrete"):
            config.actor.dist = config.actor.dist.disc
            self.act_discrete = True
        else:
            config.actor.dist = config.actor.dist.cont

        # Actor-critic components
        self.actor = networks.MLPHead(config.actor, self.rssm.feat_size)
        self.value = networks.MLPHead(config.critic, self.rssm.feat_size)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0

        # Progress shaping. A second critic on a second reward, never mixed into
        # the reported score: evaluation still reports MS-HAB reward and success
        # only. Disabled by default so slot dynamics can be measured alone.
        progress_config = getattr(config, "progress", None)
        self.progress_beta = (
            float(progress_config.beta) if progress_config is not None else 0.0
        )
        self.progress_enabled = bool(
            progress_config is not None and progress_config.enabled
        )
        if self.progress_enabled and not self.graph_slots:
            raise ValueError(
                "progress.enabled requires graph.state_mode=slots: the scorer "
                "reads predicted per-slot relations"
            )
        self.progress = None
        self.progress_value = None
        # The stage table is the single source of truth for which relations
        # matter, and prior_progress_relabs needs it whether or not the shaping
        # reward is switched on.
        self.progress_scorer = (
            ProgressScorer(
                load_stages(str(progress_config.stages) if progress_config else ""),
                int(config.graph.n_abs),
            )
            if self.graph_slots
            else None
        )
        if self.progress_enabled:
            scorer = self.progress_scorer
            self.progress = ProgressReward(
                scorer, 1 - 1 / self.horizon, soft=bool(progress_config.soft)
            )
            self.progress_feat_size = (
                self.rssm.flat_stoch
                + self.rssm._deter
                + 2 * self.rssm.slot_dim
                + int(scorer.relations.numel()) * int(config.graph.n_abs)
                + 1  # probability that a target exists at all
            )
            self.progress_value = networks.MLPHead(
                config.critic, self.progress_feat_size
            )
            self._slow_progress = copy.deepcopy(self.progress_value)
            for param in self._slow_progress.parameters():
                param.requires_grad = False
            self.progress_return_ema = networks.ReturnEMA(device=self.device)
        if self.graph_slots:
            self.register_buffer(
                "progress_relations",
                self.progress_scorer.relations.clone(),
                persistent=False,
            )

        self._loss_scales = dict(config.loss_scales)
        self._log_grads = bool(config.log_grads)
        self._replay_input_checked = False
        self._finite_diagnostic_updates = 4
        self._diagnose_finite = False
        self._trace_update = False
        self._trace_update_started = 0.0
        self._trace_stage_started = 0.0

        modules = {
            "rssm": self.rssm,
            "actor": self.actor,
            "value": self.value,
            "reward": self.reward,
            "cont": self.cont,
            "encoder": self.encoder,
        }
        if self.graph_enabled:
            if self.graph_slots:
                self.graph_decoder = SlotGraphDecoder(config.graph)
            elif self.graph_simple:
                self.graph_decoder = SimpleGraphDecoder(config.graph, self.graph_dim)
            else:
                self.graph_decoder = GraphDecoder(config.graph)
            modules.update(
                {"graph_encoder": self.graph_encoder, "graph_decoder": self.graph_decoder}
            )
        if self.progress_enabled:
            modules.update({"progress_value": self.progress_value})

        if self.rep_loss == "dreamer":
            decoder_shapes = {
                key: value for key, value in model_shapes.items() if key != "instruction"
            }
            if self.graph_only:
                decoder_shapes = {
                    key: value for key, value in decoder_shapes.items() if len(value) != 3
                }
            self.decoder = networks.MultiDecoder(
                config.decoder,
                self.rssm._deter,
                self.rssm.flat_stoch,
                decoder_shapes,
                flat_sem=self.rssm.flat_sem,
                # Slot mode passes the attention readout here. Pixels may read
                # it, never reshape it: reconstruction is the highest-bandwidth
                # signal in the model and would turn slots into a second visual
                # latent.
                detach_sem_cnn=self.graph_simple or self.graph_slots,
            )
            recon = self._loss_scales.pop("recon")
            graph_image_recon = self._loss_scales.pop("graph_image_recon")
            # Simple mode detaches g from the CNN, so pixels can no longer
            # distort the semantic state and the downweight that protected it
            # is unnecessary. Keeping stock 1.0 also makes the graph-free arm
            # directly comparable.
            use_graph_image_scale = self.graph_enabled and not (
                self.graph_simple or self.graph_slots
            )
            self._loss_scales.update({
                key: (
                    graph_image_recon
                    if use_graph_image_scale and key in self.decoder.cnn_shapes
                    else recon
                )
                for key in self.decoder.all_keys
            })
            modules.update({"decoder": self.decoder})
        elif self.rep_loss == "r2dreamer" or self.rep_loss == "infonce":
            # add projector for latent to embedding
            self.prj = Projector(self.rssm.feat_size, self.embed_size)
            modules.update({"projector": self.prj})
            self.barlow_lambd = float(config.r2dreamer.lambd)
        elif self.rep_loss == "dreamerpro":
            dpc = config.dreamer_pro
            self.warm_up = int(dpc.warm_up)
            self.num_prototypes = int(dpc.num_prototypes)
            self.proto_dim = int(dpc.proto_dim)
            self.temperature = float(dpc.temperature)
            self.sinkhorn_eps = float(dpc.sinkhorn_eps)
            self.sinkhorn_iters = int(dpc.sinkhorn_iters)
            self.ema_update_every = int(dpc.ema_update_every)
            self.ema_update_fraction = float(dpc.ema_update_fraction)
            self.freeze_prototypes_iters = int(dpc.freeze_prototypes_iters)
            self.aug_max_delta = float(dpc.aug.max_delta)
            self.aug_same_across_time = bool(dpc.aug.same_across_time)
            self.aug_bilinear = bool(dpc.aug.bilinear)

            self._prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.proto_dim))
            self.obs_proj = nn.Linear(self.embed_size, self.proto_dim)
            self.feat_proj = nn.Linear(self.rssm.feat_size, self.proto_dim)
            self._ema_encoder = copy.deepcopy(self.encoder)
            self._ema_obs_proj = copy.deepcopy(self.obs_proj)
            for param in self._ema_encoder.parameters():
                param.requires_grad = False
            for param in self._ema_obs_proj.parameters():
                param.requires_grad = False
            self._ema_updates = 0
            modules.update({
                "prototypes": self._prototypes,
                "obs_proj": self.obs_proj,
                "feat_proj": self.feat_proj,
                "ema_encoder": self._ema_encoder,
                "ema_obs_proj": self._ema_obs_proj,
            })
        # count number of parameters in each module
        for key, module in modules.items():
            if isinstance(module, nn.Parameter):
                print(f"{module.numel():>14,}: {key}")
            else:
                print(f"{sum(p.numel() for p in module.parameters()):>14,}: {key}")
        self._named_params = OrderedDict()
        for name, module in modules.items():
            if isinstance(module, nn.Parameter):
                self._named_params[name] = module
            else:
                for param_name, param in module.named_parameters():
                    self._named_params[f"{name}.{param_name}"] = param
        print(f"Optimizer has: {sum(p.numel() for p in self._named_params.values())} parameters.")

        def _agc(params):
            clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)

        self._agc = _agc
        self._optimizer = LaProp(
            self._named_params.values(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
        # Match the working r2dreamer AMP path: FP16 training uses dynamic loss
        # scaling; full-precision acting does not use the scaler.
        self._scaler = GradScaler(
            enabled=self.device.type == "cuda" and self._amp_dtype == torch.float16,
            init_scale=float(getattr(config, "grad_scale_init", 65536.0)),
        )
        self._skipped_updates = 0
        self._consecutive_skips = 0
        self._scale_settled = False
        self._max_consecutive_skips = 8
        # Below this the scale is no longer protecting anything: fp16's smallest
        # normal is ~6e-5, so a gradient that still overflows here is genuinely
        # out of range rather than merely small.
        self._min_loss_scale = 32.0

        def lr_lambda(step):
            if config.warmup:
                return min(1.0, (step + 1) / config.warmup)
            return 1.0

        self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        self.train()
        self.clone_and_freeze()
        if config.compile and not self.graph_enabled:
            print("Compiling update function with torch.compile...")
            self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")
        elif config.compile and self.graph_enabled:
            print("graph.enabled uses eager real-edge execution; skipping whole-update torch.compile")

    @staticmethod
    def _presence_metrics(logit, born, persistent, inactive):
        """Birth calibration that is cheap enough for the update loop.

        Brier, recall at 0.5, the base rate, and the separation between positive
        and negative probabilities. Ranking metrics (AUROC, AUPRC) and the
        calibration curve need sorting over accumulated samples and belong in a
        periodic dump, not here -- they would cost more than the model.
        """
        with torch.no_grad():
            probability = torch.sigmoid(logit.float())
            positive = born | persistent
            scored = positive | inactive
            target = positive.float()
            weight = scored.float()
            total = weight.sum().clamp_min(1)
            return {
                "presence_brier": (
                    ((probability - target).square() * weight).sum() / total
                ),
                "presence_base_rate": positive.float().sum() / total,
                "presence_birth_recall": _masked_mean(probability.gt(0.5), born),
                "presence_birth_prob": _masked_mean(probability, born),
                "presence_alive_prob": _masked_mean(probability, persistent),
                "presence_dead_prob": _masked_mean(probability, inactive),
            }

    def _note_optimizer_step(self, stepped):
        """Tolerate warm-up back-off; fail on a loss-scale collapse.

        Two different failures need two different instruments, and they are easy
        to confuse. While the scale is still coming down from its initial value,
        consecutive skips and halvings are the same number -- the scale only
        moves when a step is skipped -- so a consecutive-skip limit just
        restates the floor with a tighter bound, and forbids ever reaching a
        scale that fits. The floor is the honest guard there.

        The consecutive limit is for the other failure: a run that settled and
        then started skipping every update. That only means something once the
        scale has proven it can fit at least one step, so it is enforced from
        the first successful update onward.
        """
        if stepped:
            self._consecutive_skips = 0
            self._scale_settled = True
            return
        self._skipped_updates += 1
        self._consecutive_skips += 1
        scale = self._scaler.get_scale()
        collapsed = scale < self._min_loss_scale
        stalled = (
            self._scale_settled
            and self._consecutive_skips >= self._max_consecutive_skips
        )
        if collapsed or stalled:
            reason = (
                f"fell below the {self._min_loss_scale:g} floor"
                if collapsed
                else f"stalled for {self._consecutive_skips} consecutive updates"
            )
            raise FloatingPointError(
                f"float16 loss scale {reason} at {scale:g} "
                f"({self._skipped_updates} skipped updates total). This is past "
                "warm-up back-off: find the loss term that is diverging rather "
                "than lowering the semantic weights to hide it. Set "
                "model.amp_dtype=bfloat16 to remove loss scaling entirely."
            )

    def _update_slow_target(self):
        """Update slow-moving value target network."""
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
                if self.progress_enabled:
                    for v, s in zip(
                        self.progress_value.parameters(),
                        self._slow_progress.parameters(),
                    ):
                        s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    def train(self, mode=True):
        super().train(mode)
        # slow_value should be always eval mode
        self._slow_value.train(False)
        if self.progress_enabled:
            self._slow_progress.train(False)
        return self

    def clone_and_freeze(self):
        # NOTE: "requires_grad" affects whether a parameter is updated
        # not whether gradients flow through its operations
        self._frozen_encoder = copy.deepcopy(self.encoder)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.encoder.named_parameters(), self._frozen_encoder.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        if self.graph_enabled:
            self._frozen_graph_encoder = copy.deepcopy(self.graph_encoder)
            for (name_orig, param_orig), (name_new, param_new) in zip(
                self.graph_encoder.named_parameters(), self._frozen_graph_encoder.named_parameters()
            ):
                assert name_orig == name_new
                param_new.data = param_orig.data
                param_new.requires_grad_(False)

            # Imagination decodes relations from predicted slots, so the graph
            # decoder joins the frozen set: the actor update must not train the
            # world model through the shaping term.
            self._frozen_graph_decoder = copy.deepcopy(self.graph_decoder)
            for (name_orig, param_orig), (name_new, param_new) in zip(
                self.graph_decoder.named_parameters(),
                self._frozen_graph_decoder.named_parameters(),
            ):
                assert name_orig == name_new
                param_new.data = param_orig.data
                param_new.requires_grad_(False)

        if self.progress_enabled:
            self._frozen_progress_value = copy.deepcopy(self.progress_value)
            for (name_orig, param_orig), (name_new, param_new) in zip(
                self.progress_value.named_parameters(),
                self._frozen_progress_value.named_parameters(),
            ):
                assert name_orig == name_new
                param_new.data = param_orig.data
                param_new.requires_grad_(False)

            self._frozen_slow_progress = copy.deepcopy(self._slow_progress)
            for (name_orig, param_orig), (name_new, param_new) in zip(
                self._slow_progress.named_parameters(),
                self._frozen_slow_progress.named_parameters(),
            ):
                assert name_orig == name_new
                param_new.data = param_orig.data
                param_new.requires_grad_(False)

        self._frozen_rssm = copy.deepcopy(self.rssm)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.rssm.named_parameters(), self._frozen_rssm.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_reward = copy.deepcopy(self.reward)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.reward.named_parameters(), self._frozen_reward.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_cont = copy.deepcopy(self.cont)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.cont.named_parameters(), self._frozen_cont.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_actor = copy.deepcopy(self.actor)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.actor.named_parameters(), self._frozen_actor.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_value = copy.deepcopy(self.value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.value.named_parameters(), self._frozen_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_slow_value = copy.deepcopy(self._slow_value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self._slow_value.named_parameters(), self._frozen_slow_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Re-establish shared memory after moving the model to a new device
        self.clone_and_freeze()
        return self

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        """Policy inference step."""
        # obs: dict of (B, *), state: (stoch: (B, S, K), deter: (B, D), prev_action: (B, A))
        torch.compiler.cudagraph_mark_step_begin()
        p_obs = self.preprocess(obs)
        # Keep policy collection in float32, matching the working r2dreamer
        # implementation. AMP is used only by the training update.
        embed = self._frozen_encoder(p_obs)
        encoded = (
            self._frozen_graph_encoder(graph_from(p_obs, self.graph_simple))
            if self.graph_enabled
            else None
        )
        prev_stoch = state.get("stoch")
        prev_deter = state["deter"]
        prev_action = state["prev_action"]
        if self.graph_slots:
            # A terminal frame is an auto-reset artifact, not a scene: blanking
            # it makes the aligner match nothing and every slot fall back to its
            # prior rather than to a graph from the next episode.
            slot_obs = encoded.slots.keep(_step_valid(obs["is_last"]))
            step = self._frozen_rssm.obs_step(
                prev_stoch,
                prev_deter,
                prev_action,
                embed,
                obs["is_first"],
                sem=state.get("sem"),
                slot_meta=state.get("slot_meta"),
                slot_alive=state.get("slot_alive"),
                slot_obs=slot_obs,
            )
            stoch, deter = step["stoch"], step["deter"]
            sem, slot_meta = step["sem"], step["slot_meta"]
            slot_alive = step["slot_alive"]
            feat = self._frozen_rssm.get_feat(stoch, deter, sem, slot_alive)
        else:
            graph_token = encoded.token if encoded is not None else None
            if graph_token is not None:
                graph_token = _mask_terminal_graph(graph_token, obs["is_last"])
            result = self._frozen_rssm.obs_step(
                prev_stoch,
                prev_deter,
                prev_action,
                embed,
                obs["is_first"],
                sem=state.get("sem") if self.graph_enabled else None,
                graph_token=graph_token,
            )
            slot_meta = slot_alive = None
            if self.graph_only:
                deter, sem, _ = result
                stoch = None
            elif self.graph_enabled:
                stoch, deter, _, sem = result[:4]
            else:
                stoch, deter, _ = result
                sem = None
            feat = self._frozen_rssm.get_feat(stoch, deter, sem)
        action_dist = self._frozen_actor(feat)
        action = action_dist.mode if eval else action_dist.rsample()

        action = to_f32(action)
        entries = {"deter": to_f32(deter), "prev_action": action}
        if not self.graph_only:
            entries["stoch"] = to_f32(stoch)
        if self.graph_enabled:
            entries["sem"] = to_f32(sem)
        if self.graph_slots:
            entries["slot_meta"] = to_f32(slot_meta)
            entries["slot_alive"] = to_f32(slot_alive)
        return action, TensorDict(entries, batch_size=state.batch_size)

    @torch.no_grad()
    def get_initial_state(self, B):
        initial = self.rssm.initial(B)
        action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        entries = dict(zip(self.rssm.state_keys, initial))
        entries["prev_action"] = action
        return TensorDict(entries, batch_size=(B,))

    @torch.no_grad()
    def video_pred(self, data, initial):
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        return self._video_pred(p_data, initial)

    def _video_pred(self, data, initial):
        """Video prediction utility."""
        if self.graph_only:
            raise NotImplementedError(
                "graph-only mode has no pixel decoder or open-loop video prediction"
            )
        if self.rep_loss != "dreamer":
            raise NotImplementedError("video_pred requires decoder and is only supported when rep_loss == 'dreamer'.")

        B = min(data["action"].shape[0], 6)
        # (B, T, E)
        embed = self.encoder(data)
        encoded = (
            self.graph_encoder(graph_from(data, self.graph_simple))
            if self.graph_enabled
            else None
        )
        if self.graph_slots:
            slot_obs = encoded.slots.keep(_step_valid(data["is_last"]))
            observed = self.rssm.observe(
                embed[:B, :5],
                data["action"][:B, :5],
                tuple(val[:B] for val in initial),
                data["is_first"][:B, :5],
                slot_obs=slot_obs[:B, :5],
            )
            post_stoch, post_deter = observed["stoch"], observed["deter"]
            post_sem, post_meta = observed["sem"], observed["slot_meta"]
            post_alive = observed["slot_alive"]
            readout = self.rssm.semantic_feature(post_sem, post_alive)
            recon = self.decoder(post_stoch, post_deter, readout)["image"].mode()[:B]
            imagined = self.rssm.imagine_with_action(
                post_stoch[:, -1],
                post_deter[:, -1],
                data["action"][:B, 5:],
                post_sem[:, -1],
                post_meta[:, -1],
                post_alive[:, -1],
            )
            prior_readout = self.rssm.semantic_feature(
                imagined["sem"], imagined["slot_alive"]
            )
            openl = self.decoder(
                imagined["stoch"], imagined["deter"], prior_readout
            )["image"].mode()
            model = torch.cat([recon[:, :5], openl], 1)
            truth = data["image"][:B]
            return torch.cat([truth, model, (model - truth + 1.0) / 2.0], 2)
        graph_token = None
        if self.graph_enabled:
            graph_token = _mask_terminal_graph(encoded.token, data["is_last"])
        observed = self.rssm.observe(
            embed[:B, :5],
            data["action"][:B, :5],
            tuple(val[:B] for val in initial),
            data["is_first"][:B, :5],
            None if graph_token is None else graph_token[:B, :5],
        )
        post_stoch, post_deter = observed[:2]
        post_sem = observed[3] if self.graph_enabled else None
        recon = self.decoder(post_stoch, post_deter, post_sem)["image"].mode()[:B]
        init_stoch, init_deter = post_stoch[:, -1], post_deter[:, -1]
        imagined = self.rssm.imagine_with_action(
            init_stoch,
            init_deter,
            data["action"][:B, 5:],
            None if post_sem is None else post_sem[:, -1],
        )
        prior_stoch, prior_deter = imagined[:2]
        prior_sem = imagined[2] if self.graph_enabled else None
        openl = self.decoder(prior_stoch, prior_deter, prior_sem)["image"].mode()
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:B]
        error = (model - truth + 1.0) / 2.0
        return torch.cat([truth, model, error], 2)

    def _trace_stage_start(self, name):
        """Mark the start of a synchronized first-update diagnostic stage."""
        if not self._trace_update:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._trace_stage_started = time.perf_counter()
        elapsed = self._trace_stage_started - self._trace_update_started
        print(
            f"[dreamer:first-update] START {name} (total={elapsed:.3f}s)",
            flush=True,
        )

    def _trace_stage_done(self, name):
        """Synchronize and mark completion of a first-update diagnostic stage."""
        if not self._trace_update:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        stage = now - self._trace_stage_started
        total = now - self._trace_update_started
        memory = ""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / 2**30
            reserved = torch.cuda.memory_reserved(self.device) / 2**30
            memory = f", allocated={allocated:.2f}GiB, reserved={reserved:.2f}GiB"
        print(
            f"[dreamer:first-update] DONE  {name} "
            f"(stage={stage:.3f}s, total={total:.3f}s{memory})",
            flush=True,
        )

    def update(self, replay_buffer):
        """Sample a batch from replay and perform one optimization step."""
        trace_this_update = not self._replay_input_checked
        if trace_this_update:
            self._trace_update = True
            self._trace_update_started = time.perf_counter()
            print("[dreamer:first-update] BEGIN", flush=True)

        self._trace_stage_start("replay sample")
        data, index, initial = replay_buffer.sample()
        self._trace_stage_done("replay sample")
        first_replay_check = not self._replay_input_checked
        if first_replay_check:
            self._trace_stage_start("replay validation")
        _check_finite_replay(
            data,
            initial,
            latent_names=self.rssm.state_keys,
            log_magnitudes=first_replay_check,
        )
        if first_replay_check:
            self._trace_stage_done("replay validation")
            self._replay_input_checked = True
            print(
                "[dreamer] first replay batch is finite before preprocessing/autocast",
                flush=True,
            )
        torch.compiler.cudagraph_mark_step_begin()
        self._trace_stage_start("preprocess and target update")
        p_data = self.preprocess(data)
        self._update_slow_target()
        if self.rep_loss == "dreamerpro":
            self.ema_update()
        self._trace_stage_done("preprocess and target update")
        metrics = {}
        self._diagnose_finite = self._finite_diagnostic_updates > 0
        with autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            enabled=self.device.type in ("cpu", "cuda"),
        ):
            posterior, mets = self._cal_grad(p_data, initial)
        self._trace_stage_start("optimizer step")
        self._scaler.unscale_(self._optimizer)  # unscale grads in params
        if self._diagnose_finite:
            grads = [(name, param.grad) for name, param in self._named_params.items()]
            if self._scaler.is_enabled():
                # float16 dynamic scaling is *expected* to overflow until the
                # scale settles: GradScaler skips those steps and halves the
                # scale. Raising here would turn normal warm-up into a crash,
                # and the post-step parameter check below still catches real
                # corruption because a skipped step leaves parameters untouched.
                overflow = [
                    name
                    for name, value in grads
                    if value is not None
                    and torch.is_floating_point(value)
                    and not torch.isfinite(value).all()
                ]
                if overflow:
                    print(
                        f"[dreamer] float16 gradient overflow in {len(overflow)}/"
                        f"{len(grads)} tensors at loss scale "
                        f"{self._scaler.get_scale():g}; skipping this step and "
                        "halving the scale",
                        flush=True,
                    )
            else:
                _check_finite_tensors(
                    "gradients before clipping/optimizer step", grads
                )
        if self.rep_loss == "dreamerpro" and self._ema_updates < self.freeze_prototypes_iters:
            self._prototypes.grad.zero_()
        if self._log_grads:
            old_params = [p.data.clone().detach() for p in self._named_params.values()]
            grads = [p.grad for p in self._named_params.values() if p.grad is not None]  # log grads before clipping
            grad_norm = tools.compute_global_norm(grads)
            grad_rms = tools.compute_rms(grads)
            mets["opt/grad_norm"] = grad_norm
            mets["opt/grad_rms"] = grad_rms
        self._agc(self._named_params.values())  # clipping
        scale_before = self._scaler.get_scale()
        self._scaler.step(self._optimizer)  # update params, or skip on non-finite gradients
        self._scaler.update()  # adjust scale
        # A lowered scale is how GradScaler reports that it skipped the step.
        # With the scaler disabled the scale is a constant 1.0, so this reads
        # True and the schedule behaves exactly as before.
        stepped = self._scaler.get_scale() >= scale_before
        self._note_optimizer_step(stepped)
        if self._diagnose_finite:
            _check_finite_tensors(
                "parameters after optimizer step",
                ((name, param) for name, param in self._named_params.items()),
            )
        if stepped:
            # A skipped update must not consume warm-up. The schedule counts
            # applied updates, not attempts.
            self._scheduler.step()  # increment scheduler
        self._optimizer.zero_grad(set_to_none=True)  # reset grads
        self._trace_stage_done("optimizer step")
        mets["opt/lr"] = self._scheduler.get_lr()[0]
        mets["opt/grad_scale"] = self._scaler.get_scale()
        mets["opt/skipped_updates"] = float(self._skipped_updates)
        if self._log_grads:
            updates = [(new - old) for (new, old) in zip(self._named_params.values(), old_params)]
            update_rms = tools.compute_rms(updates)
            params_rms = tools.compute_rms(self._named_params.values())
            mets["opt/param_rms"] = params_rms
            mets["opt/update_rms"] = update_rms
        metrics.update(mets)
        # update latent vectors in replay buffer
        self._trace_stage_start("replay latent writeback")
        replay_buffer.update(
            index,
            **{
                key: value.detach()
                for key, value in zip(self.rssm.state_keys, posterior)
            },
        )
        self._trace_stage_done("replay latent writeback")
        if self._diagnose_finite:
            self._finite_diagnostic_updates -= 1
        if trace_this_update:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - self._trace_update_started
            print(
                f"[dreamer:first-update] COMPLETE (total={elapsed:.3f}s)",
                flush=True,
            )
            self._trace_update = False
        return metrics

    def _cal_grad(self, data, initial):
        """Compute gradients for one batch.

        Notes
        -----
        This function computes:
        1) World model loss (dynamics + representation)
        2) Optional representation loss variants (Dreamer, R2-Dreamer, InfoNCE, DreamerPro)
        3) Imagination rollouts for actor-critic updates
        4) Replay-based value learning
        """
        # data: dict of (B, T, *), initial: (stoch: (B, S, K), deter: (B, D))
        losses = {}
        metrics = {}
        B, T = data.shape

        # === World model: posterior rollout and KL losses ===
        # (B, T, E)
        self._trace_stage_start("encoder")
        embed = self.encoder(data)
        if self._diagnose_finite:
            _check_finite_tensors("encoder output", (("embed", embed),))
        self._trace_stage_done("encoder")
        if self.graph_enabled:
            self._trace_stage_start("graph encoder")
        graph_encoding = (
            self.graph_encoder(graph_from(data, self.graph_simple))
            if self.graph_enabled
            else None
        )
        if self.graph_enabled:
            self._trace_stage_done("graph encoder")
        step_valid = _step_valid(data["is_last"])
        # Slot mode carries the slot table and its identity metadata; every
        # other mode carries a single vector, so these stay None.
        post_meta = post_alive = prior_slot = slot_align = None
        self._trace_stage_start("RSSM posterior rollout")
        if self.graph_slots:
            observed = self.rssm.observe(
                embed,
                data["action"],
                initial,
                data["is_first"],
                slot_obs=graph_encoding.slots.keep(step_valid),
            )
            post_stoch, post_deter = observed["stoch"], observed["deter"]
            post_logit, prior_logit = observed["logit"], observed["prior_logit"]
            post_sem, post_meta = observed["sem"], observed["slot_meta"]
            post_alive = observed["slot_alive"]
            prior_slot, slot_align = observed["prior_slot"], observed
            post_sem_logit = None
        else:
            graph_token = graph_encoding.token if graph_encoding is not None else None
            if graph_token is not None:
                graph_token = _mask_terminal_graph(graph_token, data["is_last"])
            observed = self.rssm.observe(
                embed, data["action"], initial, data["is_first"], graph_token
            )
            prior_logit = None
            if self.graph_only:
                post_deter, post_sem, post_sem_logit = observed
                post_stoch = post_logit = None
            else:
                post_stoch, post_deter, post_logit = observed[:3]
                post_sem = observed[3] if self.graph_enabled else None
                post_sem_logit = (
                    observed[4]
                    if self.graph_enabled and not self.graph_simple
                    else None
                )
        self._trace_stage_done("RSSM posterior rollout")
        # (B, T, S, K)
        if not self.graph_only:
            self._trace_stage_start("RSSM prior and KL")
            if prior_logit is None:
                _, prior_logit = self.rssm.prior(post_deter, post_sem)
            dyn_loss, rep_loss = self.rssm.kl_loss(
                post_logit, prior_logit, self.kl_free
            )
            losses["dyn"] = torch.mean(dyn_loss)
            losses["rep"] = torch.mean(rep_loss)
            self._trace_stage_done("RSSM prior and KL")
        if self.graph_enabled:
            self._trace_stage_start("semantic and graph losses")
            step_float = step_valid.float()
            denominator = step_float.sum().clamp_min(1)
            if self.graph_slots:
                alive = self.rssm.slot_mask(post_alive)
                # A reset transition predicted from nothing, and a capacity
                # replacement predicted a different entity than the one that
                # arrived. Neither has a correspondence to charge a loss against.
                usable = step_valid[..., None] & ~slot_align["reset"][..., None]
                usable = usable & ~slot_align["replaced"]
                persistent = slot_align["matched"] & usable
                born = slot_align["born"] & usable
                # Content regression covers persistence and matched births, and
                # nothing else: an unmatched proposal has no target to regress to.
                losses["slotdyn"] = self.rssm.slot_dynamics_loss(
                    prior_slot, post_sem, persistent | born
                )
                inactive = ~alive & usable
                losses["slotalive"] = self.rssm.slot_alive_loss(
                    slot_align["prior_alive_logit"],
                    post_alive,
                    persistent,
                    born,
                    inactive,
                )
                graph_losses, graph_metrics = self.graph_decoder(
                    post_sem,
                    prior_slot,
                    graph_encoding.compact,
                    slot_align["dest"],
                    post_alive,
                    post_meta[..., rssm.SLOT_META_TARGET],
                    step_valid,
                    self.progress_relations,
                )
                with torch.no_grad():
                    live = alive.float()
                    metrics["slot_occupancy"] = live.sum(-1).mean()
                    metrics["slot_overflow"] = slot_align["overflow"].float().mean()
                    metrics["slot_replacements"] = (
                        slot_align["replaced"].float().sum(-1).mean()
                    )
                    metrics["slot_births"] = born.float().sum(-1).mean()
                    metrics["slot_birth_rate"] = (
                        born.float().sum() / usable.float().sum().clamp_min(1)
                    )
                    metrics["slot_matched_frac"] = (
                        persistent.float().sum() / live.sum().clamp_min(1)
                    )
                    metrics["slot_post_var"] = (
                        post_sem.float().var(-1) * live
                    ).sum() / live.sum().clamp_min(1)
                    metrics["slot_prior_cos"] = _masked_mean(
                        F.cosine_similarity(
                            prior_slot.float(), post_sem.float(), dim=-1, eps=1e-6
                        ),
                        persistent,
                    )
                    metrics["slot_birth_cos"] = _masked_mean(
                        F.cosine_similarity(
                            prior_slot.float(), post_sem.float(), dim=-1, eps=1e-6
                        ),
                        born,
                    )
                    metrics.update(
                        self._presence_metrics(
                            slot_align["prior_alive_logit"], born, persistent, inactive
                        )
                    )
            elif self.graph_simple:
                prior_sem = self.rssm.semantic_prior_seq(post_deter)
                sem_dyn, sem_rep = self.rssm.semantic_align_loss(post_sem, prior_sem)
                # Terminal frames carry a masked graph token, so their
                # posterior is not a real observation to align against.
                losses["graphdyn"] = (sem_dyn * step_float).sum() / denominator
                losses["graphrep"] = (sem_rep * step_float).sum() / denominator
                # Both terms share one forward value; log it once and let the
                # scales express the asymmetry.
                metrics["graph_align_mse"] = losses["graphdyn"].detach()
                with torch.no_grad():
                    # Collapse shows up here first: cosine climbing to one
                    # while both variances fall means the two branches agreed
                    # on a constant rather than on the graph.
                    cos = (
                        self.rssm.rms(post_sem) * self.rssm.rms(prior_sem)
                    ).mean(-1)
                    metrics["graph_align_cos"] = (
                        (cos * step_float).sum() / denominator
                    )
                    metrics["graph_sem_post_var"] = post_sem.float().var(-1).mean()
                    metrics["graph_sem_prior_var"] = prior_sem.float().var(-1).mean()
                graph_losses, graph_metrics = self.graph_decoder(
                    post_sem, graph_encoding.compact, step_valid
                )
            else:
                sem_prior_logit = self.rssm.semantic_prior_logits(
                    post_deter, post_sem, initial[-1], data["is_first"]
                )
                sem_dyn, sem_rep, raw_sem_dyn, raw_sem_rep = self.rssm.semantic_kl_loss(
                    post_sem_logit, sem_prior_logit, self.kl_free
                )
                losses["semdyn"] = (sem_dyn * step_float).mean()
                losses["semrep"] = (sem_rep * step_float).mean()
                metrics["semdyn_raw"] = (raw_sem_dyn * step_float).sum() / denominator
                metrics["semrep_raw"] = (raw_sem_rep * step_float).sum() / denominator
                metrics["sem_entropy"] = (
                    self.rssm.get_sem_dist(post_sem_logit).entropy().mean()
                )
                graph_losses, graph_metrics = self.graph_decoder(
                    graph_encoding, step_valid
                )
            losses.update(graph_losses)
            metrics.update(graph_metrics)
            self._trace_stage_done("semantic and graph losses")
        # === Representation / auxiliary losses ===
        # (B, T, F)
        self._trace_stage_start("representation and reconstruction losses")
        feat = self.rssm.get_feat(post_stoch, post_deter, post_sem, post_alive)
        # The decoder takes the pooled readout in slot mode: reconstruction is a
        # head, and heads never see the slot table directly.
        decoder_sem = (
            self.rssm.semantic_feature(post_sem, post_alive)
            if self.graph_slots
            else post_sem
        )
        if self.rep_loss == "dreamer":
            recon_losses = {
                key: torch.mean(-dist.log_prob(data[key]))
                for key, dist in self.decoder(
                    post_stoch, post_deter, decoder_sem
                ).items()
            }
            losses.update(recon_losses)
        elif self.rep_loss == "r2dreamer":
            # R2-Dreamer: Barlow Twins style redundancy reduction between latent features and encoder embeddings.
            # Flatten batch/time dims for a single cross-correlation matrix.
            # (B, T, F) -> (B*T, F)
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important

            x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)

            c = torch.mm(x1_norm.T, x2_norm) / (B * T)
            invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
            off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
            redundancy_loss = c[off_diag_mask].pow(2).sum()
            losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss
        elif self.rep_loss == "infonce":
            # Contrastive (InfoNCE) objective between projected latent features and encoder embeddings.
            # (B, T, F) -> (B*T, F)
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important
            logits = torch.matmul(x1, x2.T)
            norm_logits = logits - torch.max(logits, 1)[0][:, None]
            labels = torch.arange(norm_logits.shape[0]).long().to(self.device)
            losses["infonce"] = torch.nn.functional.cross_entropy(norm_logits, labels)
        elif self.rep_loss == "dreamerpro":
            # DreamerPro uses augmentation + EMA targets + Sinkhorn assignment.
            with torch.no_grad():
                data_aug = self.augment_data(data)
                initial_aug = (
                    # (B, ...) -> (2B, ...)
                    torch.cat([initial[0], initial[0]], dim=0),
                    torch.cat([initial[1], initial[1]], dim=0),
                )
                ema_proj = self.ema_proj(data_aug)

            embed_aug = self.encoder(data_aug)
            post_stoch_aug, post_deter_aug, _ = self.rssm.observe(
                embed_aug, data_aug["action"], initial_aug, data_aug["is_first"]
            )
            proto_losses = self.proto_loss(post_stoch_aug, post_deter_aug, embed_aug, ema_proj)
            losses.update(proto_losses)
        else:
            raise NotImplementedError
        self._trace_stage_done("representation and reconstruction losses")

        # reward and continue
        self._trace_stage_start("reward and continuation losses")
        losses["rew"] = torch.mean(-self.reward(feat).log_prob(to_f32(data["reward"])))
        cont = 1.0 - to_f32(data["is_terminal"])
        losses["con"] = torch.mean(-self.cont(feat).log_prob(cont))
        # log
        if not self.graph_only:
            metrics["dyn_entropy"] = torch.mean(
                self.rssm.get_dist(prior_logit).entropy()
            )
            metrics["rep_entropy"] = torch.mean(
                self.rssm.get_dist(post_logit).entropy()
            )
        self._trace_stage_done("reward and continuation losses")

        # === Imagination rollout for actor-critic ===
        # (B*T, S, K), (B*T, D)
        if self.graph_only:
            start = (
                post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
                post_sem.reshape(-1, *post_sem.shape[2:]).detach(),
            )
        else:
            start = (
                post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
                post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
            )
        if self.graph_enabled and not self.graph_only:
            start = start + (post_sem.reshape(-1, *post_sem.shape[2:]).detach(),)
        if self.graph_slots:
            start = start + (
                post_meta.reshape(-1, *post_meta.shape[2:]).detach(),
                post_alive.reshape(-1, *post_alive.shape[2:]).detach(),
            )
        # (B, T, ...) -> (B*T, ...)
        self._trace_stage_start("imagination rollout")
        imag_feat, imag_action, imag_extra = self._imagine(
            start, self.imag_horizon + 1
        )
        imag_feat, imag_action = imag_feat.detach(), imag_action.detach()
        self._trace_stage_done("imagination rollout")

        # (B*T, T_imag, 1)
        self._trace_stage_start("imagination objectives")
        imag_reward = self._frozen_reward(imag_feat).mode()
        # (B*T, T_imag, 1)  probability of continuation
        imag_cont = self._frozen_cont(imag_feat).mean
        # (B*T, T_imag, 1)
        imag_value = self._frozen_value(imag_feat).mode()
        imag_slow_value = self._frozen_slow_value(imag_feat).mode()
        disc = 1 - 1 / self.horizon
        # (B*T, T_imag, 1)
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(
            last, term, imag_reward, imag_value, imag_value, disc, self.lamb
        )  # (B*T, T_imag-1, 1)
        ret_offset, ret_scale = self.return_ema(ret)
        # (B*T, T_imag-1, 1)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        if self.progress_enabled:
            # A second return on the shaping reward, with its own critic and its
            # own normalisation, added to the advantage rather than to the
            # reward: the environment return the run reports stays untouched.
            progress_feat = imag_extra["progress_feat"].detach()
            progress_reward = imag_extra["progress_reward"].detach()
            progress_value = self._frozen_progress_value(progress_feat).mode()
            progress_slow = self._frozen_slow_progress(progress_feat).mode()
            progress_ret = self._lambda_return(
                last, term, progress_reward, progress_value, progress_value, disc, self.lamb
            )
            _, progress_scale = self.progress_return_ema(progress_ret)
            progress_adv = (progress_ret - progress_value[:, :-1]) / progress_scale
            adv = adv + self.progress_beta * progress_adv
            progress_dist = self.progress_value(progress_feat)
            progress_padded = torch.cat([progress_ret, 0 * progress_ret[:, -1:]], 1)
            losses["progress_value"] = torch.mean(
                weight[:, :-1].detach()
                * (
                    -progress_dist.log_prob(progress_padded.detach())
                    - progress_dist.log_prob(progress_slow.detach())
                )[:, :-1].unsqueeze(-1)
            )
            metrics["progress_potential"] = torch.mean(imag_extra["progress_potential"])
            metrics["progress_reward"] = torch.mean(progress_reward)
            metrics["progress_ret"] = torch.mean(progress_ret)
            metrics["progress_adv"] = torch.mean(progress_adv)
            metrics["progress_val"] = torch.mean(progress_value)

        policy = self.actor(imag_feat)
        # (B*T, T_imag-1, 1)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))

        imag_value_dist = self.value(imag_feat)
        # (B*T, T_imag, 1)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        losses["value"] = torch.mean(
            weight[:, :-1].detach()
            * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[
                :, :-1
            ].unsqueeze(-1)
        )
        # log
        ret_normed = (ret - ret_offset) / ret_scale
        metrics["ret"] = torch.mean(ret_normed)
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
        metrics["adv"] = torch.mean(adv)
        metrics["adv_std"] = torch.std(adv)
        metrics["con"] = torch.mean(imag_cont)
        metrics["rew"] = torch.mean(imag_reward)
        metrics["val"] = torch.mean(imag_value)
        metrics["tar"] = torch.mean(ret)
        metrics["slowval"] = torch.mean(imag_slow_value)
        metrics["weight"] = torch.mean(weight)
        metrics["action_entropy"] = torch.mean(entropy)
        metrics.update(tools.tensorstats(imag_action, "action"))
        if "imag_alive" in imag_extra:
            alive = imag_extra["imag_alive"]
            metrics["imag_occupancy"] = alive.gt(0.5).float().sum(-1).mean()
            # Nonzero only once births are on: how much occupancy the rollout
            # creates relative to where it started.
            metrics["imag_births"] = (
                (alive[:, -1].gt(0.5).float() - alive[:, 0].gt(0.5).float())
                .clamp_min(0)
                .sum(-1)
                .mean()
            )
        self._trace_stage_done("imagination objectives")

        # === Replay-based value learning (keep gradients through world model) ===
        self._trace_stage_start("replay value objective")
        last, term, reward = (
            to_f32(data["is_last"]),
            to_f32(data["is_terminal"]),
            to_f32(data["reward"]),
        )
        feat = self.rssm.get_feat(post_stoch, post_deter, post_sem, post_alive)
        boot = ret[:, 0].reshape(B, T, 1)
        value = self._frozen_value(feat).mode()
        slow_value = self._frozen_slow_value(feat).mode()
        disc = 1 - 1 / self.horizon
        weight = 1.0 - last
        ret = self._lambda_return(last, term, reward, value, boot, disc, self.lamb)
        ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)

        # Keep this attached to the world model so gradients can flow through
        value_dist = self.value(feat)
        losses["repval"] = torch.mean(
            weight[:, :-1]
            * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value.detach()))[:, :-1].unsqueeze(
                -1
            )
        )
        # log
        metrics.update(tools.tensorstats(ret, "ret_replay"))
        metrics.update(tools.tensorstats(value, "value_replay"))
        metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))
        self._trace_stage_done("replay value objective")

        self._trace_stage_start("backward")
        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()
        self._trace_stage_done("backward")

        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        if self.graph_only:
            posterior = (post_deter, post_sem)
        else:
            posterior = (post_stoch, post_deter)
        # Simple mode still stores g: a sampled chunk's first transition needs
        # g_{t-1} for the transition, exactly as the categorical mode does.
        if self.graph_enabled and not self.graph_only:
            posterior = posterior + (post_sem,)
        if self.graph_slots:
            posterior = posterior + (post_meta, post_alive)
        return posterior, metrics

    def _progress_step(self, stoch, deter, sem, slot_alive):
        """Shaping reward and progress-critic input for one imagined state.

        Everything here is predicted: the slots come from the slot prior, the
        target identity from the frozen decoder's target head, and the relations
        from decoding each candidate slot. No observed label and no latched
        target flag enters an imagined rollout.
        """
        decoder = self._frozen_graph_decoder
        target_logits = decoder.target_logits(sem)
        reward, potential, probs = self.progress(
            decoder, sem, target_logits, slot_alive
        )
        weights, null = target_distribution(target_logits, slot_alive)
        # A soft target embedding is fine as a critic feature -- it is only
        # forbidden as the input to the nonlinear relation decoder.
        objects = sem[..., 1:, :]
        target = (weights[..., None] * objects).sum(-2)
        feat = torch.cat(
            [
                stoch.reshape(*deter.shape[:-1], -1),
                deter,
                sem[..., 0, :],
                target,
                probs.reshape(*probs.shape[:-2], -1).to(deter.dtype),
                (1.0 - null)[..., None].to(deter.dtype),
            ],
            -1,
        )
        return reward.unsqueeze(-1), potential.unsqueeze(-1), feat

    @torch.no_grad()
    def _imagine(self, start, imag_horizon):
        """Roll out the policy in latent space."""
        # (B, S, K), (B, D)
        feats = []
        actions = []
        extra = {}
        slot_meta = slot_alive = None
        if self.graph_only:
            deter, sem = start
            stoch = None
        else:
            stoch, deter = start[:2]
            sem = start[2] if self.graph_enabled else None
            if self.graph_slots:
                slot_meta, slot_alive = start[3], start[4]
        progress_rewards, progress_potentials, progress_feats = [], [], []
        alive_trace = []
        for _ in range(imag_horizon):
            # (B, F)
            feat = self._frozen_rssm.get_feat(stoch, deter, sem, slot_alive)
            # (B, A)
            action = self._frozen_actor(feat).rsample()
            # Append feat and its corresponding sampled action at the same time step.
            feats.append(feat)
            actions.append(action)
            if self.graph_slots:
                alive_trace.append(slot_alive)
            if self.progress_enabled:
                reward, potential, progress_feat = self._progress_step(
                    stoch, deter, sem, slot_alive
                )
                progress_rewards.append(reward)
                progress_potentials.append(potential)
                progress_feats.append(progress_feat)
            if self.graph_slots:
                step = self._frozen_rssm.img_step(
                    stoch, deter, action, sem, slot_meta, slot_alive
                )
                stoch, deter, sem = step["stoch"], step["deter"], step["sem"]
                slot_alive = step["slot_alive"]
                continue
            result = self._frozen_rssm.img_step(stoch, deter, action, sem)
            if self.graph_only:
                deter, sem, _ = result
            elif self.graph_enabled:
                stoch, deter, sem, _ = result
            else:
                stoch, deter = result

        if alive_trace:
            extra["imag_alive"] = torch.stack(alive_trace, dim=1)
        if self.progress_enabled:
            extra.update(
                {
                    "progress_reward": torch.stack(progress_rewards, dim=1),
                    "progress_potential": torch.stack(progress_potentials, dim=1),
                    "progress_feat": torch.stack(progress_feats, dim=1),
                }
            )
        # Stack along sequence dim T_imag.
        # (B, T_imag, F), (B, T_imag, A)
        return torch.stack(feats, dim=1), torch.stack(actions, dim=1), extra

    @torch.no_grad()
    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        """
        lamb=1 means discounted Monte Carlo return.
        lamb=0 means fixed 1-step return.
        """
        assert last.shape == term.shape == reward.shape == value.shape == boot.shape
        live = (1 - to_f32(term))[:, 1:] * disc
        cont = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)

    @torch.no_grad()
    def preprocess(self, data):
        # Shallow-copy the container and replace only normalized image values.
        # This keeps policy inference non-mutating without TensorDict.clone(),
        # whose CUDA foreach path does not support uint16 graph entity IDs.
        data = data.copy()
        for key in self.image_keys:
            if key in data:
                data[key] = to_f32(data[key]) / 255.0
        return data

    @torch.no_grad()
    def augment_data(self, data):
        data_aug = {k: torch.cat([v, v], axis=0) for k, v in data.items()}
        # (B, T, H, W, C) -> (B, T, C, H, W)
        image = data_aug["image"].permute(0, 1, 4, 2, 3)
        data_aug["image"] = self.random_translate(
            image,
            self.aug_max_delta,
            same_across_time=self.aug_same_across_time,
            bilinear=self.aug_bilinear,
        )
        # (B, T, C, H, W) -> (B, T, H, W, C)
        data_aug["image"] = data_aug["image"].permute(0, 1, 3, 4, 2)
        return data_aug

    @torch.no_grad()
    def ema_proj(self, data):
        with torch.no_grad():
            embed = self._ema_encoder(data)
            proj = self._ema_obs_proj(embed)
        return F.normalize(proj, p=2, dim=-1)

    @torch.no_grad()
    def ema_update(self):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)
        self._prototypes.data.copy_(prototypes)
        if self._ema_updates % self.ema_update_every == 0:
            mix = self.ema_update_fraction if self._ema_updates > 0 else 1.0
            for s, d in zip(self.encoder.parameters(), self._ema_encoder.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
            for s, d in zip(self.obs_proj.parameters(), self._ema_obs_proj.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
        self._ema_updates += 1

    def sinkhorn(self, scores):
        """Sinkhorn-Knopp normalization.

        Notes
        -----
        Given a score matrix, we iteratively normalize rows and columns in log
        space so that the resulting assignment matrix is approximately doubly
        stochastic.
        """
        shape = scores.shape
        K = shape[0]
        scores = scores.reshape(-1)
        log_Q = F.log_softmax(scores / self.sinkhorn_eps, dim=0)
        log_Q = log_Q.reshape(K, -1)
        N = log_Q.shape[1]
        for _ in range(self.sinkhorn_iters):
            log_row_sums = torch.logsumexp(log_Q, dim=1, keepdim=True)
            log_Q = log_Q - log_row_sums - math.log(K)
            log_col_sums = torch.logsumexp(log_Q, dim=0, keepdim=True)
            log_Q = log_Q - log_col_sums - math.log(N)
        log_Q = log_Q + math.log(N)
        Q = torch.exp(log_Q)
        return Q.reshape(shape)

    def proto_loss(self, post_stoch, post_deter, embed, ema_proj):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)

        obs_proj = self.obs_proj(embed)
        obs_norm = torch.norm(obs_proj, dim=-1)
        obs_proj = F.normalize(obs_proj, p=2, dim=-1)

        B, T = obs_proj.shape[:2]
        # (B, T, P) -> (B*T, P)
        obs_proj = obs_proj.reshape(B * T, -1)
        obs_scores = torch.matmul(obs_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        obs_scores = obs_scores.reshape(B, T, -1).permute(2, 0, 1)
        obs_scores = obs_scores[:, :, self.warm_up :]
        obs_logits = F.log_softmax(obs_scores / self.temperature, dim=0)
        obs_logits_1, obs_logits_2 = torch.chunk(obs_logits, 2, dim=1)

        # (B, T, P) -> (B*T, P)
        ema_proj = ema_proj.reshape(B * T, -1)
        ema_scores = torch.matmul(ema_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        ema_scores = ema_scores.reshape(B, T, -1).permute(2, 0, 1)
        ema_scores = ema_scores[:, :, self.warm_up :]
        ema_scores_1, ema_scores_2 = torch.chunk(ema_scores, 2, dim=1)

        with torch.no_grad():
            ema_targets_1 = self.sinkhorn(ema_scores_1)
            ema_targets_2 = self.sinkhorn(ema_scores_2)
        ema_targets = torch.cat([ema_targets_1, ema_targets_2], dim=1)

        feat = self.rssm.get_feat(post_stoch, post_deter)
        feat_proj = self.feat_proj(feat)
        feat_norm = torch.norm(feat_proj, dim=-1)
        feat_proj = F.normalize(feat_proj, p=2, dim=-1)

        # (B, T, P) -> (B*T, P)
        feat_proj = feat_proj.reshape(B * T, -1)
        feat_scores = torch.matmul(feat_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        feat_scores = feat_scores.reshape(B, T, -1).permute(2, 0, 1)
        feat_scores = feat_scores[:, :, self.warm_up :]
        feat_logits = F.log_softmax(feat_scores / self.temperature, dim=0)

        swav_loss = -0.5 * torch.mean(torch.sum(ema_targets_2 * obs_logits_1, dim=0)) - 0.5 * torch.mean(
            torch.sum(ema_targets_1 * obs_logits_2, dim=0)
        )
        temp_loss = -torch.mean(torch.sum(ema_targets * feat_logits, dim=0))
        norm_loss = torch.mean(torch.square(obs_norm - 1)) + torch.mean(torch.square(feat_norm - 1))

        return {
            "swav": swav_loss,
            "temp": temp_loss,
            "norm": norm_loss,
        }

    @torch.no_grad()
    def random_translate(self, x, max_delta, same_across_time=False, bilinear=False):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        pad = int(max_delta)

        # Pad
        x_padded = F.pad(x_flat, (pad, pad, pad, pad), "replicate")
        h_padded, w_padded = H + 2 * pad, W + 2 * pad

        # Create base grid
        eps_h = 1.0 / h_padded
        eps_w = 1.0 / w_padded
        arange_h = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, h_padded, device=x.device, dtype=x.dtype)[:H]
        arange_w = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, w_padded, device=x.device, dtype=x.dtype)[:W]
        arange_h = arange_h.unsqueeze(1).repeat(1, W).unsqueeze(2)
        arange_w = arange_w.unsqueeze(0).repeat(H, 1).unsqueeze(2)
        base_grid = torch.cat([arange_w, arange_h], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(B * T, 1, 1, 1)

        # Create shift
        if same_across_time:
            shift = torch.randint(0, 2 * pad + 1, size=(B, 1, 1, 1, 2), device=x.device, dtype=x.dtype)
            shift = shift.repeat(1, T, 1, 1, 1).reshape(B * T, 1, 1, 2)
        else:
            shift = torch.randint(0, 2 * pad + 1, size=(B * T, 1, 1, 2), device=x.device, dtype=x.dtype)

        shift = shift * 2.0 / torch.tensor([w_padded, h_padded], device=x.device, dtype=x.dtype)

        # Apply shift and sample
        grid = base_grid + shift
        mode = "bilinear" if bilinear else "nearest"
        x_translated = F.grid_sample(x_padded, grid, mode=mode, padding_mode="zeros", align_corners=False)

        return x_translated.reshape(B, T, C, H, W)
