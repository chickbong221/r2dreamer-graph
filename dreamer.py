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
    RESERVED_GRAPH_KEYS,
    GraphEncoder,
    SimpleGraphDecoder,
    compact_graph,
    graph_from,
    graph_keys,
)
from networks import Projector
from progress import (
    ProgressReward,
    ProgressScorer,
    TaskScheduleReplayPotential,
    load_stages,
)
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


def _masked_std(values, mask):
    """Population std over the masked entries; zero when fewer than two.

    No ``.item()``: this runs inside the training step, and reading the count
    back to python would sync the device on every update just to log a number.
    """
    mask = mask.float()
    count = mask.sum()
    safe = count.clamp_min(1)
    mean = (values.float() * mask).sum() / safe
    var = (((values.float() - mean) ** 2) * mask).sum() / safe
    return torch.where(count > 1, var.clamp_min(0).sqrt(), torch.zeros_like(var))


def _frame_flag(value):
    """(B, T) view of a per-frame flag stored as either (B, T) or (B, T, 1)."""
    flag = value.bool()
    if flag.ndim == 3 and flag.shape[-1] == 1:
        flag = flag[..., 0]
    return flag


def _step_valid(is_last):
    """(B, T) mask of frames that are a real observation."""
    return ~_frame_flag(is_last)


def _sync_camera_count(graph_config, obs_space) -> None:
    """Size the per-camera layers from the packed graph, not from config.

    How many cameras a ManiSkill task renders is a property of the task and the
    robot it registers, so it is only known once the environment exists -- which
    is after config composition and before this. The builder has already packed
    ``graph_node_bbox`` as ``[n_max, n_cams, 4]``, and that is the shape these
    layers have to match, so read it from there and let config supply only the
    default for suites with no packed graph at all.
    """
    box = obs_space.get("graph_node_bbox", None) if obs_space else None
    shape = getattr(box, "shape", None)
    if not shape or len(shape) != 3:
        return
    observed = int(shape[1])
    if observed != int(graph_config.n_cams):
        print(
            f"[graph] n_cams {int(graph_config.n_cams)} -> {observed}, "
            "taken from the task's rendered cameras",
            flush=True,
        )
        graph_config.n_cams = observed


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
        if self.graph_enabled:
            _sync_camera_count(config.graph, obs_space)
        # graph.enabled is the only graph switch. On means one pooled g beside
        # a stock z, the box-addressed decoder, and the masked-mean readout.
        self.graph_simple = self.graph_enabled
        self.graph_pooled_simple = self.graph_enabled
        self.graph_keys = graph_keys()
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
        # Every key any schema has ever emitted is excluded from the pixel/state
        # encoder, not just the active schema's. A wrapper still exposing
        # graph_node_uid under the pooled schema would otherwise train an
        # identity-conditioned model without anything failing.
        model_shapes = {
            key: value
            for key, value in shapes.items()
            if key not in RESERVED_GRAPH_KEYS
        }
        if self.graph_enabled:
            if "graph_node_uid" in shapes:
                raise ValueError(
                    "pooled graph-simple must not be handed graph_node_uid; the "
                    "environment is emitting the slot contract"
                )
        self.encoder = networks.MultiEncoder(config.encoder, model_shapes)
        self.image_keys = tuple(self.encoder.cnn_shapes)
        self.embed_size = self.encoder.out_dim
        self.graph_encoder = GraphEncoder(config.graph) if self.graph_enabled else None
        graph_token_size = int(self.graph_encoder.units) if self.graph_enabled else 0
        self.graph_dim = int(config.graph.semantic_dim) if self.graph_enabled else 0
        self.rssm = rssm.RSSM(
            config.rssm,
            self.embed_size,
            self.act_dim,
            semantic=self.graph_enabled,
            graph_token_size=graph_token_size,
            graph_dim=self.graph_dim,
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
        # Warm-up on the actor's exposure to progress, in environment steps.
        # Beta is the only thing scheduled: the progress head and
        # `progress_value` train from step 0. The critic reads detached inputs,
        # so its warm-up costs the actor nothing; the head does update the
        # world-model latent, which is deliberate -- it is an auxiliary
        # prediction task like the reward head, and it has its own loss scale
        # if that needs turning down. Beta is the single place progress reaches
        # *behaviour*, so ramping it -- rather than the losses -- is what keeps
        # a cold critic from steering, and an immature world model can
        # hallucinate progress under imagined actions long after the real robot
        # has stopped moving.
        self.progress_beta_start = (
            float(getattr(progress_config, "beta_warmup_start", 0.0))
            if progress_config is not None
            else 0.0
        )
        self.progress_beta_end = (
            float(getattr(progress_config, "beta_warmup_end", 0.0))
            if progress_config is not None
            else 0.0
        )
        if self.progress_beta_end < self.progress_beta_start:
            raise ValueError(
                "progress.beta_warmup_end must not precede "
                "progress.beta_warmup_start"
            )
        # Written by `update`. None means the caller keeps no environment-step
        # count, and the schedule reads as already finished.
        self._env_step = None
        self.progress_enabled = bool(
            progress_config is not None and progress_config.enabled
        )
        if self.progress_enabled and not self.graph_enabled:
            raise ValueError(
                "progress.enabled requires graph.enabled: the potential is "
                "read off the graph state"
            )
        self.progress = None
        self.progress_value = None
        self.progress_head = None
        # Which potential imagination reads. `world_model` is the bounded
        self.progress_mode = str(
            getattr(progress_config, "mode", "ee_target")
            if progress_config is not None
            else "ee_target"
        )
        if self.progress_mode not in ("ee_target", "task_schedule"):
            raise ValueError(
                f"progress.mode={self.progress_mode!r} is not one of "
                "(ee_target, task_schedule)"
            )
        # Built by attach_task_schedule, which is where the task identity and
        # the resolved whitelist directory live. None keeps the replay target
        # on the end-effector stage table.
        self.progress_schedule = None
        self._schedule_n_abs = int(config.graph.n_abs)
        self.progress_schedule_dir = str(
            getattr(progress_config, "schedule_dir", "")
            if progress_config is not None
            else ""
        )

        # The stage table is the single source of truth for which relations
        # matter. Both progress sources need it -- one to supervise predicted
        # labels, the other to turn observed labels into a scalar target.
        self.progress_scorer = (
            ProgressScorer(
                load_stages(),
                int(config.graph.n_abs),
                int(config.graph.n_rel),
            )
            if self.graph_pooled_simple
            else None
        )
        if self.progress_enabled:
            scorer = self.progress_scorer
            self.progress = ProgressReward(
                scorer, 1 - 1 / self.horizon, soft=bool(progress_config.soft)
            )
            relation_width = int(scorer.relations.numel()) * int(config.graph.n_abs)
            if self.graph_pooled_simple:
                # Exactly the latent the policy and the ordinary critic read,
                # and nothing else. The predicted relation block is gone with
                # the relation head: the potential is now a scalar function of
                # this same feature, so appending a second view of it would
                # only give the critic a shortcut the actor does not have.
                self.progress_feat_size = self.rssm.feat_size
            elif self.graph_pooled_simple:
                # imag_feat is already [z, g, h], so the progress critic reads
                # the exact latent the policy and the ordinary critic read, plus
                # the predicted relation block. Nothing else is concatenated: no
                # masks, no counts, no boxes, no observed labels. There is also
                # no target-presence scalar, because the pooled head has no null
                # class to produce one.
                self.progress_feat_size = self.rssm.feat_size + relation_width
            else:
                self.progress_feat_size = (
                    self.rssm.flat_stoch
                    + self.rssm._deter
                    + 2 * self.rssm.slot_dim
                    + relation_width
                    + 1  # probability that a target exists at all
                )
            self.progress_value = networks.MLPHead(
                config.critic, self.progress_feat_size
            )
            self._slow_progress = copy.deepcopy(self.progress_value)
            for param in self._slow_progress.parameters():
                param.requires_grad = False
            self.progress_return_ema = networks.ReturnEMA(
                device=self.device,
                min_scale=float(getattr(progress_config, "return_min_scale", 1.0)),
            )
        if self.progress_scorer is not None:
            self.register_buffer(
                "progress_relations",
                self.progress_scorer.relations.clone(),
                persistent=False,
            )

        self._loss_scales = dict(config.loss_scales)
        # Resolved once. A zero scale means the branch is not computed at all
        # rather than computed and multiplied by zero, which is the only kind of
        # switch this repository needs: losses are keyed by scale, not by an
        # enable flag per head.
        self._progress_model = self.graph_pooled_simple and (
            float(self._loss_scales.get("progress_model", 0.0)) != 0.0
        )
        if self._progress_model:
            # Beside the environment reward head, on the same feature and
            # trained the same way: posterior features from replay, applied to
            # imagined ones. It is a separate head because the two quantities
            # have different supports -- reward is unbounded and twohot,
            # progress is a potential in [0, 1] and a sigmoid.
            self.progress_head = networks.ProgressHead(
                progress_config.head, self.rssm.feat_size
            )
        if self.progress_enabled and not self._progress_model:
            raise ValueError(
                "progress.enabled with loss_scales.progress_model=0 leaves the "
                "progress head untrained; imagination would shape the actor "
                "with an unsupervised readout"
            )
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
            self.graph_decoder = SimpleGraphDecoder(
                config.graph, self.graph_dim, self.progress_scorer.relations
            )
            modules.update(
                {"graph_encoder": self.graph_encoder, "graph_decoder": self.graph_decoder}
            )
        if self.progress_enabled:
            modules.update({"progress_value": self.progress_value})
        if self.progress_head is not None:
            modules.update({"progress_head": self.progress_head})

        if self.rep_loss == "dreamer":
            decoder_shapes = {
                key: value for key, value in model_shapes.items() if key != "instruction"
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
                detach_sem_cnn=self.graph_simple,
            )
            recon = self._loss_scales.pop("recon")
            graph_image_recon = self._loss_scales.pop("graph_image_recon")
            # Simple mode detaches g from the CNN, so pixels can no longer
            # distort the semantic state and the downweight that protected it
            # is unnecessary. Keeping stock 1.0 also makes the graph-free arm
            # directly comparable.
            use_graph_image_scale = self.graph_enabled and not (
                self.graph_simple
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
        elif self.rep_loss == "r2dreamer":
            # add projector for latent to embedding
            self.prj = Projector(self.rssm.feat_size, self.embed_size)
            modules.update({"projector": self.prj})
            self.barlow_lambd = float(config.r2dreamer.lambd)
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
        print(f"[arm] {self.arm_summary()}", flush=True)

    def arm_summary(self) -> str:
        """One line naming what this run actually is.

        Which arm you get is spread over four switches -- `graph.enabled`,
        `progress.enabled` -- and reading
        them back from a config dump means knowing how they combine. Reading
        them back from the object that resolved them does not.
        """
        if not self.graph_enabled:
            graph = "off"
        else:
            graph = "on"
        if not self.progress_enabled:
            progress = "off"
        else:
            progress = str(self.progress_mode)
            progress += f" beta={self.progress_beta:g}"
            if self.progress_schedule is None and self.progress_mode == "task_schedule":
                progress += " (schedule not attached yet)"
        return (f"rep_loss={self.rep_loss} | graph={graph} | progress={progress}")

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

    def attach_task_schedule(self, envs) -> None:
        """Compile this task's phase schedule and use it as the replay target.

        Call before ``.to(device)`` so the compiled buffers travel with the
        module. A no-op unless progress.mode is task_schedule; when it is, a
        missing or unscorable schedule raises here rather than at the first
        gradient step -- a schedule that cannot be scored yields a target of
        zero for the whole run, and nothing downstream would say so.
        """
        # Disabled progress has no target to compile. The baseline arm reaches
        # here: it runs env=maniskill, which sets progress_mode, but turns the
        # graph off -- so there is no whitelist to resolve roles against and
        # nothing that would read the result anyway.
        if not self.progress_enabled or self.progress_mode != "task_schedule":
            return
        import os

        from envs.maniskill import _repo_path, task_schedule_source
        from scenegraph.adapters.graph_vocab import build_entity_vocab
        from scenegraph.core.schedule import compile_from_files

        source = task_schedule_source(envs)
        if source is None:
            raise RuntimeError(
                "progress.mode=task_schedule needs a graph-enabled ManiSkill "
                "env to read the task id and the whitelist directory from"
            )
        env_id, whitelist_dir = source
        # <configs>/subtask_whitelists/<env_id> -> <configs>
        configs = os.path.dirname(os.path.dirname(whitelist_dir))
        schedule = compile_from_files(
            env_id, str(_repo_path(self.progress_schedule_dir)), configs,
            build_entity_vocab(whitelist_dir),
        )
        self.progress_schedule = TaskScheduleReplayPotential(
            schedule, self._schedule_n_abs)
        print(
            f"[progress] {env_id}: {len(schedule.phases)} phases, "
            f"{sum(len(p.clauses) for p in schedule.phases)} clauses, "
            f"{len(schedule.slots)} distinct facts",
            flush=True,
        )

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

        if self.progress_head is not None:
            # Imagination reads the potential through this head, so it joins
            # the frozen set for the same reason the graph decoder does: the
            # actor update must not train the world model through the shaping
            # term.
            self._frozen_progress_head = copy.deepcopy(self.progress_head)
            for (name_orig, param_orig), (name_new, param_new) in zip(
                self.progress_head.named_parameters(),
                self._frozen_progress_head.named_parameters(),
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
            self._frozen_graph_encoder(graph_from(p_obs))
            if self.graph_enabled
            else None
        )
        prev_stoch = state.get("stoch")
        prev_deter = state["deter"]
        prev_action = state["prev_action"]
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
        if self.graph_enabled:
            stoch, deter, _, sem = result[:4]
        else:
            stoch, deter, _ = result
            sem = None
        feat = self._frozen_rssm.get_feat(stoch, deter, sem)
        action_dist = self._frozen_actor(feat)
        action = action_dist.mode if eval else action_dist.rsample()

        action = to_f32(action)
        entries = {"deter": to_f32(deter), "prev_action": action,
                   "stoch": to_f32(stoch)}
        if self.graph_enabled:
            entries["sem"] = to_f32(sem)
        return action, TensorDict(entries, batch_size=state.batch_size)

    @torch.no_grad()
    def get_initial_state(self, B):
        initial = self.rssm.initial(B)
        action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        entries = dict(zip(self.rssm.state_keys, initial))
        entries["prev_action"] = action
        return TensorDict(entries, batch_size=(B,))

    def _pixel_keys(self):
        """Decoded camera keys, in the decoder's own order.

        There is no single ``image``: an env names its cameras, and how many
        there are follows from the task and its robot. Taking the list from the
        decoder keeps truth and reconstruction in the same order, column for
        column, whatever that turns out to be.
        """
        keys = list(getattr(self.decoder, "cnn_shapes", None) or {})
        if not keys:
            raise NotImplementedError(
                "video_pred needs a pixel decoder; this run decodes no images"
            )
        return keys

    @staticmethod
    def _tile_cameras(frames):
        """``[B, T, H, W, C]`` per camera -> one strip along width."""
        return frames[0] if len(frames) == 1 else torch.cat(frames, dim=-2)

    @torch.no_grad()
    def video_pred(self, data, initial):
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        return self._video_pred(p_data, initial)

    def _video_pred(self, data, initial):
        """Video prediction utility."""
        if self.rep_loss != "dreamer":
            raise NotImplementedError("video_pred requires decoder and is only supported when rep_loss == 'dreamer'.")

        B = min(data["action"].shape[0], 6)
        # (B, T, E)
        embed = self.encoder(data)
        encoded = (
            self.graph_encoder(graph_from(data))
            if self.graph_enabled
            else None
        )
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
        keys = self._pixel_keys()
        decoded = self.decoder(post_stoch, post_deter, post_sem)
        recon = self._tile_cameras([decoded[k].mode()[:B] for k in keys])
        init_stoch, init_deter = post_stoch[:, -1], post_deter[:, -1]
        imagined = self.rssm.imagine_with_action(
            init_stoch,
            init_deter,
            data["action"][:B, 5:],
            None if post_sem is None else post_sem[:, -1],
        )
        prior_stoch, prior_deter = imagined[:2]
        prior_sem = imagined[2] if self.graph_enabled else None
        decoded = self.decoder(prior_stoch, prior_deter, prior_sem)
        openl = self._tile_cameras([decoded[k].mode() for k in keys])
        model = torch.cat([recon[:, :5], openl], 1)
        truth = self._tile_cameras([data[k][:B] for k in keys])
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

    def update(self, replay_buffer, step=None):
        """Sample a batch from replay and perform one optimization step.

        `step` is the environment-step count and feeds the progress-beta
        warm-up only; None leaves that schedule at its final value.
        """
        self._env_step = None if step is None else float(step)
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
            self.graph_encoder(graph_from(data))
            if self.graph_enabled
            else None
        )
        if self.graph_enabled:
            self._trace_stage_done("graph encoder")
        step_valid = _step_valid(data["is_last"])
        self._trace_stage_start("RSSM posterior rollout")
        graph_token = graph_encoding.token if graph_encoding is not None else None
        if graph_token is not None:
            graph_token = _mask_terminal_graph(graph_token, data["is_last"])
        observed = self.rssm.observe(
            embed, data["action"], initial, data["is_first"], graph_token
        )
        prior_logit = None
        post_stoch, post_deter, post_logit = observed[:3]
        post_sem = observed[3] if self.graph_enabled else None
        post_sem_logit = None
        self._trace_stage_done("RSSM posterior rollout")
        # (B, T, S, K)
        self._trace_stage_start("RSSM prior and KL")
        _, prior_logit = self.rssm.prior(post_deter, post_sem)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = torch.mean(dyn_loss)
        losses["rep"] = torch.mean(rep_loss)
        self._trace_stage_done("RSSM prior and KL")
        if self.graph_enabled:
            self._trace_stage_start("semantic and graph losses")
            step_float = step_valid.float()
            denominator = step_float.sum().clamp_min(1)
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
                post_sem,
                graph_encoding.compact,
                step_valid,
            )
            losses.update(graph_losses)
            metrics.update(graph_metrics)
            self._trace_stage_done("semantic and graph losses")
        # === Representation / auxiliary losses ===
        # (B, T, F)
        self._trace_stage_start("representation and reconstruction losses")
        feat = self.rssm.get_feat(post_stoch, post_deter, post_sem)
        decoder_sem = post_sem
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
        # === Imagination rollout for actor-critic ===
        # (B*T, S, K), (B*T, D)
        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
        )
        if self.graph_enabled:
            start = start + (post_sem.reshape(-1, *post_sem.shape[2:]).detach(),)
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
            progress_beta = self._progress_beta_at(self._env_step)
            env_adv_abs = torch.mean(adv.abs())
            adv = adv + progress_beta * progress_adv
            progress_dist = self.progress_value(progress_feat)
            progress_padded = torch.cat([progress_ret, 0 * progress_ret[:, -1:]], 1)
            losses["progress_value"] = torch.mean(
                weight[:, :-1].detach()
                * (
                    -progress_dist.log_prob(progress_padded.detach())
                    - progress_dist.log_prob(progress_slow.detach())
                )[:, :-1].unsqueeze(-1)
            )
            potential = imag_extra["progress_potential"]
            progress_adv_abs = torch.mean(progress_adv.abs())
            # --- primary ---------------------------------------------------
            # How much of the actor's advantage the shaping term accounts for:
            # beta * E|A_progress| / E|A_env|. This, not beta alone, is what
            # says whether the ramp landed: roughly 5-20% at the plateau, under
            # 1% means beta is doing nothing, and much over 25% means progress
            # is doing the steering. A wrong value here is a normalisation or
            # beta problem, not a head problem.
            # In float32: early on the environment advantage is near zero,
            # and a float16 autocast would overflow the ratio to inf.
            metrics["progress/influence"] = progress_beta * progress_adv_abs.float() / (
                env_adv_abs.float() + 1e-8
            )
            # Whether the critic has caught up with its own return yet.
            metrics["progress/critic_mae"] = torch.mean(
                (progress_value[:, :-1] - progress_ret).abs()
            )
            # --- raw log ---------------------------------------------------
            # Kept for debugging, not for the dashboard. ``horizon_std`` is the
            # one worth reaching for first: it is the spread within a single
            # rollout, i.e. whether acting changes predicted progress at all.
            # A flat potential produces a zero advantage no matter what beta
            # is, so it says whether beta is even the thing worth changing.
            metrics["progress_potential"] = torch.mean(potential)
            metrics["progress_potential_horizon_std"] = torch.mean(
                torch.std(potential, dim=1)
            )
            metrics["progress_reward"] = torch.mean(progress_reward)
            metrics["progress_ret"] = torch.mean(progress_ret)
            metrics["progress_adv"] = torch.mean(progress_adv)
            metrics["progress_val"] = torch.mean(progress_value)
            metrics["progress_adv_abs"] = progress_adv_abs
            metrics["env_adv_abs"] = env_adv_abs
            metrics["progress_beta"] = progress_beta

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
        self._trace_stage_done("imagination objectives")

        # === Replay-based value learning (keep gradients through world model) ===
        self._trace_stage_start("replay value objective")
        last, term, reward = (
            to_f32(data["is_last"]),
            to_f32(data["is_terminal"]),
            to_f32(data["reward"]),
        )
        feat = self.rssm.get_feat(post_stoch, post_deter, post_sem)
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
        posterior = (post_stoch, post_deter)
        # g is stored too: a sampled chunk's first transition needs g_{t-1}.
        if self.graph_enabled:
            posterior = posterior + (post_sem,)
        return posterior, metrics

    def _progress_beta_at(self, step):
        """Actor weight on the progress advantage at environment step `step`.

        Zero before the window, linear across it, constant after. `None` means
        the caller tracks no environment steps, which reads the same as an
        empty window (`start == end`): beta applies in full from the start.
        """
        start, end = self.progress_beta_start, self.progress_beta_end
        if step is None or end <= start or step >= end:
            return self.progress_beta
        if step <= start:
            return 0.0
        return self.progress_beta * (step - start) / (end - start)

    def _progress_model_loss(self, feat, compact, step_valid):
        """Regress the bounded progress head onto the observed stage ladder.

        The target is read from the packed graph, not predicted: row 0 to row 1
        is the end-effector-to-target block by construction now, and the six
        labels there go through the same scorer the old path applied to
        predicted distributions. One-hot in, so the scalar is exactly the hard
        stage sum.

        Masking, not zeroing. A frame before the target is first observed has
        no ladder to stand on, and a frame missing one of the six facts has an
        incomplete one; scoring either as zero progress would teach the head
        that an unseen target is a failed reach. Huber rather than squared
        error because the target is a weighted step function -- a frame that
        crosses two rungs at once is a real jump, not an outlier to chase.
        """
        graph_count = compact.graph_count
        with torch.no_grad():
            if self.progress_schedule is not None:
                target, valid = self.progress_schedule(
                    compact.node_ent,
                    compact.edge_rel,
                    compact.edge_abs,
                    compact.edge_src_local,
                    compact.edge_dst_local,
                    compact.edge_graph,
                    graph_count,
                )
            else:
                target, valid = self.progress_scorer.replay_potential(
                    compact.edge_rel,
                    compact.edge_abs,
                    compact.edge_src_local,
                    compact.edge_dst_local,
                    compact.edge_graph,
                    graph_count,
                )
            # Terminal frames carry a masked graph token, so whatever their
            # edges say is not an observation of anything.
            valid = valid & step_valid.reshape(graph_count)
            target = target * valid.float()
        phi = self.progress_head(feat).reshape(graph_count)
        error = F.huber_loss(phi, target, reduction="none", delta=0.1)
        denominator = valid.float().sum().clamp_min(1)
        loss = (error * valid.float()).sum() / denominator
        metrics = {
            # The five that matter. `valid_fraction` reads as a persistence
            # bug when it is low; `target_std` says whether behaviour produces
            # any spread to learn at all, and a near-zero one makes every other
            # progress number meaningless.
            "progress/valid_fraction": valid.float().mean(),
            "progress/target_std": _masked_std(target, valid),
            "progress/head_mae": _masked_mean((phi - target).abs(), valid),
            "progress/target_mean": _masked_mean(target, valid),
        }
        return loss, metrics

    def _imagine(self, start, imag_horizon):
        """Roll out the policy in latent space."""
        # (B, S, K), (B, D)
        feats = []
        actions = []
        extra = {}
        stoch, deter = start[:2]
        sem = start[2] if self.graph_enabled else None
        for _ in range(imag_horizon):
            # (B, F)
            feat = self._frozen_rssm.get_feat(stoch, deter, sem)
            # (B, A)
            action = self._frozen_actor(feat).rsample()
            # Append feat and its corresponding sampled action at the same time step.
            feats.append(feat)
            actions.append(action)
            result = self._frozen_rssm.img_step(stoch, deter, action, sem)
            if self.graph_enabled:
                stoch, deter, sem, _ = result
            else:
                stoch, deter = result

        # Stack along sequence dim T_imag. (B, T_imag, F)
        imag_feat = torch.stack(feats, dim=1)
        if self.progress_enabled and self.graph_pooled_simple:
            # One bounded head over the feature the policy just read. Frozen,
            # so imagination pushes nothing back into it.
            potential = self._frozen_progress_head(imag_feat).squeeze(-1)
            reward = (1.0 - self.progress.discount) * potential
            extra.update(
                {
                    "progress_reward": reward.unsqueeze(-1),
                    "progress_potential": potential.unsqueeze(-1),
                    # The critic reads the policy's own feature and nothing
                    # else: the potential is already a function of it.
                    "progress_feat": imag_feat,
                }
            )
        # (B, T_imag, F), (B, T_imag, A)
        return imag_feat, torch.stack(actions, dim=1), extra

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

    def ema_proj(self, data):
        with torch.no_grad():
            embed = self._ema_encoder(data)
            proj = self._ema_obs_proj(embed)
        return F.normalize(proj, p=2, dim=-1)

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
