import torch
from torch import distributions as torchd
from torch import nn

import distributions as dists
from networks import BlockLinear, LambdaLayer
from tools import rpad, weight_init_


class Deter(nn.Module):
    def __init__(
        self, deter, stoch, act_dim, hidden, blocks, dynlayers, act="SiLU", semantic_size=0
    ):
        super().__init__()
        self.blocks = int(blocks)
        self.dynlayers = int(dynlayers)
        act = getattr(torch.nn, act)
        self._dyn_in0 = nn.Sequential(
            nn.Linear(deter, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_in1 = (
            nn.Sequential(
                nn.Linear(stoch, hidden, bias=True),
                nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32),
                act(),
            )
            if stoch
            else None
        )
        self._dyn_in2 = nn.Sequential(
            nn.Linear(act_dim, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_in3 = (
            nn.Sequential(
                nn.Linear(semantic_size, hidden, bias=True),
                nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32),
                act(),
            )
            if semantic_size
            else None
        )
        self._dyn_hid = nn.Sequential()
        input_count = 2 + int(bool(stoch)) + int(bool(semantic_size))
        in_ch = (input_count * hidden + deter // self.blocks) * self.blocks
        for i in range(self.dynlayers):
            self._dyn_hid.add_module(f"dyn_hid_{i}", BlockLinear(in_ch, deter, self.blocks))
            self._dyn_hid.add_module(f"norm_{i}", nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32))
            self._dyn_hid.add_module(f"act_{i}", act())
            in_ch = deter
        self._dyn_gru = BlockLinear(in_ch, 3 * deter, self.blocks)
        self.flat2group = lambda x: x.reshape(*x.shape[:-1], self.blocks, -1)
        self.group2flat = lambda x: x.reshape(*x.shape[:-2], -1)

    def forward(self, stoch, deter, action, sem=None):
        """Deterministic state transition (block-GRU style)."""
        # (B, S, K), (B, D), (B, A)
        B = action.shape[0]

        # Flatten stochastic state and normalize action magnitude.
        # (B, S*K)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        # (B, U)
        x0 = self._dyn_in0(deter)
        x2 = self._dyn_in2(action)

        # Concatenate projected inputs and broadcast over blocks.
        inputs = [x0]
        if self._dyn_in1 is not None:
            if stoch is None:
                raise ValueError("stochastic dynamics require the previous stochastic state")
            inputs.append(self._dyn_in1(stoch.reshape(B, -1)))
        inputs.append(x2)
        if self._dyn_in3 is not None:
            if sem is None:
                raise ValueError("semantic dynamics require the previous semantic state")
            inputs.append(self._dyn_in3(sem.reshape(sem.shape[0], -1)))
        x = torch.cat(inputs, -1)
        # (B, G, 3*U)
        x = x.unsqueeze(-2).expand(-1, self.blocks, -1)

        # Combine per-block deterministic state with per-block inputs.
        # (B, G, D/G + 3*U) -> (B, D + 3*U*G)
        x = self.group2flat(torch.cat([self.flat2group(deter), x], -1))

        # (B, D)
        x = self._dyn_hid(x)
        # (B, 3*D)
        x = self._dyn_gru(x)

        # Split GRU-style gates block-wise.
        # (B, G, 3*D/G)
        gates = torch.chunk(self.flat2group(x), 3, dim=-1)

        # (B, D)
        reset, cand, update = (self.group2flat(x) for x in gates)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        # (B, D)
        return update * cand + (1 - update) * deter


class RSSM(nn.Module):
    def __init__(
        self,
        config,
        embed_size,
        act_dim,
        semantic=False,
        graph_token_size=0,
        graph_only=False,
        graph_simple=False,
        graph_dim=0,
    ):
        super().__init__()
        self.semantic = bool(semantic)
        self.graph_only = bool(graph_only)
        # Simple mode keeps a semantic state but makes it a deterministic
        # vector that no longer gates z. z returns to the stock DreamerV3
        # width and inputs; g survives only in the transition and the feature.
        self.graph_simple = bool(graph_simple)
        if self.graph_only and not self.semantic:
            raise ValueError("graph-only RSSM requires semantic=True")
        if self.graph_simple and not self.semantic:
            raise ValueError("simple-graph RSSM requires semantic=True")
        if self.graph_simple and self.graph_only:
            raise ValueError("graph_simple and graph_only are mutually exclusive")
        if self.graph_only:
            self._stoch = 0
            self._discrete = 0
        else:
            self._stoch = int(
                getattr(config, "hybrid_stoch", config.stoch)
                if self.semantic and not self.graph_simple else config.stoch
            )
            self._discrete = int(config.discrete)
        self._deter = int(config.deter)
        self._hidden = int(config.hidden)
        act = getattr(torch.nn, config.act)
        self._unimix_ratio = float(config.unimix_ratio)
        self._initial = str(config.initial)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self._obs_layers = int(config.obs_layers)
        self._img_layers = int(config.img_layers)
        self._dyn_layers = int(config.dyn_layers)
        self._blocks = int(config.blocks)
        self.flat_stoch = self._stoch * self._discrete
        if self.graph_simple:
            # One flat continuous vector: no categorical factorisation, so
            # _sem_stoch/_sem_discrete stay zero and every reshape keys off
            # graph_simple instead.
            self._sem_stoch = self._sem_discrete = 0
            self._sem_layers = int(config.sem_layers)
            self.flat_sem = int(graph_dim)
            if self.flat_sem <= 0:
                raise ValueError("simple-graph RSSM requires graph_dim > 0")
        elif self.semantic:
            if self.graph_only:
                self._sem_stoch = int(getattr(config, "graph_only_stoch", 32))
                self._sem_discrete = int(
                    getattr(config, "graph_only_discrete", config.discrete)
                )
            else:
                self._sem_stoch = int(config.sem_stoch)
                self._sem_discrete = int(config.sem_discrete)
            self._sem_layers = int(config.sem_layers)
            self.flat_sem = self._sem_stoch * self._sem_discrete
        else:
            self._sem_stoch = self._sem_discrete = self._sem_layers = 0
            self.flat_sem = 0
        self.feat_size = self.flat_stoch + self.flat_sem + self._deter
        self.state_keys = (
            ("deter", "sem")
            if self.graph_only
            else (("stoch", "deter", "sem") if self.semantic else ("stoch", "deter"))
        )
        self._deter_net = Deter(
            self._deter,
            self.flat_stoch,
            act_dim,
            self._hidden,
            blocks=self._blocks,
            dynlayers=self._dyn_layers,
            act=config.act,
            semantic_size=self.flat_sem,
        )

        self._obs_net = None
        self._img_net = None
        # Simple mode's z is stock DreamerV3: q(z | h, o) and p(z | h). g is
        # excluded from both, so graph information reaches z only through the
        # transition, one step later.
        z_sem = 0 if self.graph_simple else self.flat_sem
        if not self.graph_only:
            self._obs_net = nn.Sequential()
            inp_dim = self._deter + embed_size + z_sem
            for i in range(self._obs_layers):
                self._obs_net.add_module(f"obs_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
                self._obs_net.add_module(f"obs_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
                self._obs_net.add_module(f"obs_net_a_{i}", act())
                inp_dim = self._hidden
            self._obs_net.add_module(
                "obs_net_logit",
                nn.Linear(inp_dim, self._stoch * self._discrete, bias=True),
            )
            self._obs_net.add_module(
                "obs_net_lambda",
                LambdaLayer(
                    lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)
                ),
            )

            self._img_net = nn.Sequential()
            inp_dim = self._deter + z_sem
            for i in range(self._img_layers):
                self._img_net.add_module(f"img_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
                self._img_net.add_module(f"img_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
                self._img_net.add_module(f"img_net_a_{i}", act())
                inp_dim = self._hidden
            self._img_net.add_module(
                "img_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete)
            )
            self._img_net.add_module(
                "img_net_lambda",
                LambdaLayer(
                    lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)
                ),
            )
        if self.graph_simple:
            # Q(h, c) and P(h). Neither reads g_{t-1}: it is already in h.
            self._sem_obs = self._deterministic_head(
                self._deter + int(graph_token_size), config.act, "sem_obs"
            )
            self._sem_img = self._deterministic_head(
                self._deter, config.act, "sem_img"
            )
        elif self.semantic:
            self._sem_obs = self._semantic_head(
                self._deter
                + self.flat_sem
                + int(graph_token_size)
                + (int(embed_size) if self.graph_only else 0),
                config.act,
                "sem_obs",
            )
            self._sem_img = self._semantic_head(
                self._deter + self.flat_sem, config.act, "sem_img"
            )
        self.apply(weight_init_)

    def _semantic_head(self, inp_dim, act_name, name):
        act = getattr(torch.nn, act_name)
        net = nn.Sequential()
        for index in range(self._sem_layers):
            net.add_module(f"{name}_{index}", nn.Linear(inp_dim, self._hidden, bias=True))
            net.add_module(
                f"{name}_norm_{index}",
                nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32),
            )
            net.add_module(f"{name}_act_{index}", act())
            inp_dim = self._hidden
        net.add_module(
            f"{name}_logit", nn.Linear(inp_dim, self._sem_stoch * self._sem_discrete)
        )
        net.add_module(
            f"{name}_shape",
            LambdaLayer(
                lambda x: x.reshape(*x.shape[:-1], self._sem_stoch, self._sem_discrete)
            ),
        )
        return net

    def _deterministic_head(self, inp_dim, act_name, name):
        """Flat continuous semantic head. No logits, no sampling."""
        act = getattr(torch.nn, act_name)
        net = nn.Sequential()
        for index in range(self._sem_layers):
            net.add_module(f"{name}_{index}", nn.Linear(inp_dim, self._hidden, bias=True))
            net.add_module(
                f"{name}_norm_{index}",
                nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32),
            )
            net.add_module(f"{name}_act_{index}", act())
            inp_dim = self._hidden
        net.add_module(f"{name}_out", nn.Linear(inp_dim, self.flat_sem))
        return net

    def sem_shape(self):
        """Trailing shape of one semantic state."""
        if self.graph_simple:
            return (self.flat_sem,)
        return (self._sem_stoch, self._sem_discrete)

    def flatten_sem(self, sem):
        """Flatten a semantic state to (..., flat_sem) in either mode."""
        if self.graph_simple:
            return sem
        return sem.reshape(*sem.shape[:-2], self.flat_sem)

    def initial(self, batch_size):
        """Return an initial latent state."""
        # (B, D), (B, S, K)
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        if self.semantic:
            sem = torch.zeros(
                batch_size,
                *self.sem_shape(),
                dtype=torch.float32,
                device=self._device,
            )
            if self.graph_only:
                return deter, sem
        stoch = torch.zeros(
            batch_size,
            self._stoch,
            self._discrete,
            dtype=torch.float32,
            device=self._device,
        )
        if self.semantic:
            return stoch, deter, sem
        return stoch, deter

    def observe(self, embed, action, initial, reset, graph_token=None):
        """Posterior rollout using observations."""
        # (B, T, E), (B, T, A), ((B, S, K), (B, D)) (B, T)
        L = action.shape[1]
        if self.graph_only:
            deter, sem = initial
            if graph_token is None:
                raise ValueError("graph-only RSSM.observe requires graph_token")
            deters, sems, sem_logits = [], [], []
            for i in range(L):
                deter, sem, sem_logit = self.obs_step(
                    None,
                    deter,
                    action[:, i],
                    embed[:, i],
                    reset[:, i],
                    sem=sem,
                    graph_token=graph_token[:, i],
                )
                deters.append(deter)
                sems.append(sem)
                sem_logits.append(sem_logit)
            return (
                torch.stack(deters, dim=1),
                torch.stack(sems, dim=1),
                torch.stack(sem_logits, dim=1),
            )
        if self.semantic:
            stoch, deter, sem = initial
            if graph_token is None:
                raise ValueError("semantic RSSM.observe requires graph_token")
            sems, sem_logits = [], []
        else:
            stoch, deter = initial
            sem = None
        stochs, deters, logits = [], [], []
        for i in range(L):
            # (B, S, K), (B, D), (B, S, K)
            result = self.obs_step(
                stoch,
                deter,
                action[:, i],
                embed[:, i],
                reset[:, i],
                sem=sem,
                graph_token=None if graph_token is None else graph_token[:, i],
            )
            if self.semantic:
                stoch, deter, logit, sem, sem_logit = result
                sems.append(sem)
                sem_logits.append(sem_logit)
            else:
                stoch, deter, logit = result
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        logits = torch.stack(logits, dim=1)
        if self.graph_simple:
            # No posterior logits exist; the prior is computed in one batched
            # call from post_deter instead of stacked per step.
            return stochs, deters, logits, torch.stack(sems, dim=1)
        if self.semantic:
            return (
                stochs,
                deters,
                logits,
                torch.stack(sems, dim=1),
                torch.stack(sem_logits, dim=1),
            )
        return stochs, deters, logits

    def obs_step(self, stoch, deter, prev_action, embed, reset, sem=None, graph_token=None):
        """Single posterior step."""
        # (B, S, K), (B, D), (B, A), (B, E), (B,)
        if not self.graph_only:
            stoch = torch.where(
                rpad(reset, stoch.dim() - int(reset.dim())),
                torch.zeros_like(stoch),
                stoch,
            )
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action
        )
        if self.semantic:
            if sem is None or graph_token is None:
                raise ValueError("semantic obs_step requires sem and graph_token")
            sem = torch.where(rpad(reset, sem.dim() - int(reset.dim())), torch.zeros_like(sem), sem)

        # Deterministic transition then posterior logits conditioned on embed.
        # (B, D)
        deter = self._deter_net(stoch, deter, prev_action, sem)
        if self.graph_simple:
            # Deterministic posterior: g_t = Q(h_t, c_t). No sampling, no
            # logits, and no g_{t-1} -- that already reached h_t.
            sem = self._sem_obs(torch.cat([deter, graph_token], -1))
            sem_logit = None
        elif self.semantic:
            sem_inputs = [deter, sem.reshape(sem.shape[0], -1), graph_token]
            if self.graph_only:
                sem_inputs.append(embed)
            sem_logit = self._sem_obs(torch.cat(sem_inputs, -1))
            sem = self.get_sem_dist(sem_logit).rsample()
        if self.graph_only:
            return deter, sem, sem_logit
        # (B, D + E)
        inputs = [deter]
        if self.semantic and not self.graph_simple:
            inputs.append(sem.reshape(sem.shape[0], -1))
        inputs.append(embed)
        x = torch.cat(inputs, dim=-1)
        # (B, S, K)
        logit = self._obs_net(x)

        # Sample discrete stochastic state via straight-through Gumbel-Softmax.
        # (B, S, K)
        stoch = self.get_dist(logit).rsample()
        if self.semantic:
            return stoch, deter, logit, sem, sem_logit
        return stoch, deter, logit

    def img_step(self, stoch, deter, prev_action, sem=None):
        """Single prior step (no observation)."""

        # (B, D)
        deter = self._deter_net(stoch, deter, prev_action, sem)
        if self.semantic:
            if sem is None:
                raise ValueError("semantic img_step requires sem")
            sem, sem_logit = self.semantic_prior(deter, sem)
        if self.graph_only:
            return deter, sem, sem_logit
        # (B, S, K)
        stoch, _ = self.prior(deter, sem)
        if self.semantic:
            return stoch, deter, sem, sem_logit
        return stoch, deter

    def prior(self, deter, sem=None):
        """Compute prior distribution parameters and sample stoch."""

        if self.graph_only:
            raise RuntimeError("graph-only RSSM has no z prior")

        # (B, S, K)
        inputs = [deter]
        if self.semantic and not self.graph_simple:
            if sem is None:
                raise ValueError("semantic prior requires sem")
            inputs.append(sem.reshape(*sem.shape[:-2], -1))
        logit = self._img_net(torch.cat(inputs, -1))
        stoch = self.get_dist(logit).rsample()
        return stoch, logit

    def imagine_with_action(self, stoch, deter, actions, sem=None):
        """Roll out prior dynamics given a sequence of actions."""
        # (B, S, K), (B, D), (B, T, A)
        L = actions.shape[1]
        stochs, deters, sems = [], [], []
        for i in range(L):
            result = self.img_step(stoch, deter, actions[:, i], sem)
            if self.graph_only:
                deter, sem, _ = result
                sems.append(sem)
            elif self.semantic:
                stoch, deter, sem, _ = result
                sems.append(sem)
            else:
                stoch, deter = result
            if not self.graph_only:
                stochs.append(stoch)
            deters.append(deter)
        # (B, T, S, K), (B, T, D)
        deters = torch.stack(deters, dim=1)
        if self.graph_only:
            return deters, torch.stack(sems, dim=1)
        stochs = torch.stack(stochs, dim=1)
        if self.semantic:
            return stochs, deters, torch.stack(sems, dim=1)
        return stochs, deters

    def get_feat(self, stoch, deter, sem=None):
        """Flatten stoch and concatenate with deter."""
        # (B, S, K), (B, D)
        # (B, S*K)
        parts = []
        if not self.graph_only:
            stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
            parts.append(stoch)
        if self.semantic:
            if sem is None:
                raise ValueError("semantic features require sem")
            parts.append(self.flatten_sem(sem))
        parts.append(deter)
        return torch.cat(parts, -1)

    def get_dist(self, logit):
        if self.graph_only:
            raise RuntimeError("graph-only RSSM has no z distribution")
        return torchd.independent.Independent(dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def get_sem_dist(self, logit):
        if not self.semantic:
            raise RuntimeError("semantic distribution requested from graph-free RSSM")
        return torchd.independent.Independent(
            dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
        )

    def semantic_prior(self, deter, prev_sem):
        if self.graph_simple:
            # Deterministic: g_hat_t = P(h_t). Returned with a None logit so
            # img_step keeps one arity across modes.
            return self._sem_img(deter), None
        logit = self._sem_img(
            torch.cat([deter, prev_sem.reshape(*prev_sem.shape[:-2], self.flat_sem)], -1)
        )
        return self.get_sem_dist(logit).rsample(), logit

    def semantic_prior_seq(self, deter):
        """Batched deterministic prior over a whole posterior rollout."""
        if not self.graph_simple:
            raise RuntimeError("semantic_prior_seq is simple-graph only")
        return self._sem_img(deter)

    @staticmethod
    def rms(x, eps: float = 1e-8):
        """Parameter-free fixed-scale normalisation for semantic alignment.

        Without it both branches can drive the MSE down by shrinking their
        norms instead of agreeing. No learned gain, or the freedom returns.
        """
        x = x.float()
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)

    def semantic_align_loss(self, post_sem, prior_sem):
        """Stop-gradient predictability regularizer, not a KL.

        Both terms take the same value; only their gradient routing differs --
        ``dyn`` updates the prior and the dynamics that produce h, ``rep``
        updates the posterior and, through it, the graph encoder. Nothing here
        resists collapse; the graph and task losses do that.
        """
        post = self.rms(post_sem)
        prior = self.rms(prior_sem)
        dyn = (prior - post.detach()).square().mean(-1)
        rep = (post - prior.detach()).square().mean(-1)
        return dyn, rep

    def semantic_prior_logits(self, deter, post_sem, initial_sem, reset):
        shifted = torch.cat([initial_sem[:, None], post_sem[:, :-1]], 1)
        shifted = torch.where(
            rpad(reset, shifted.dim() - reset.dim()), torch.zeros_like(shifted), shifted
        )
        return self._sem_img(
            torch.cat([deter, shifted.reshape(*shifted.shape[:-2], self.flat_sem)], -1)
        )

    def semantic_kl_loss(self, post_logit, prior_logit, free):
        kld = dists.kl
        raw_rep = kld(post_logit, prior_logit.detach()).sum(-1)
        raw_dyn = kld(post_logit.detach(), prior_logit).sum(-1)
        return (
            torch.clip(raw_dyn, min=free),
            torch.clip(raw_rep, min=free),
            raw_dyn,
            raw_rep,
        )

    def kl_loss(self, post_logit, prior_logit, free):
        kld = dists.kl
        rep_loss = kld(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = kld(post_logit.detach(), prior_logit).sum(-1)
        # Clipped gradients are not backpropagated using torch.clip.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)

        return dyn_loss, rep_loss
