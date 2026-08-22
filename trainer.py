import torch

import tools
from rssm import LATENT_STATE_KEYS


def _graph_builder(envs):
    """Graph builder behind the env stack, or None for non-graph runs."""
    try:
        from envs.maniskill import graph_panel_source
    except Exception:
        return None
    try:
        return graph_panel_source(envs)
    except Exception:
        return None


def _graph_panel(builder, env_idx, height, colormap):
    """The node-link diagram for one env as an RGB uint8 array, or None."""
    if builder is None:
        return None
    graph = getattr(builder, "last_graph_by_env", {}).get(env_idx)
    if graph is None:
        return None
    try:
        from scenegraph.viz.graph_draw import render_graph_array
        return render_graph_array(graph, colormap=colormap, height=height)
    except Exception:
        # Rendering is diagnostic; a broken panel must not end an eval run.
        return None


def _with_panel(frames, panel_fn):
    """Tile ``frames`` left-to-right and append the graph column, if any.

    The panel is sized from the strip it joins, so the diagram scales with
    whatever it is beside -- an encoder-resolution camera strip or a much
    larger render camera -- and the whole thing stays one video rather than
    two that have to be watched side by side.
    """
    if not frames:
        return None
    ref = frames[0]
    panel = panel_fn(int(ref.shape[0])) if panel_fn is not None else None
    if panel is not None:
        import numpy as np

        arr = np.asarray(panel, dtype=np.uint8)
        tile = torch.from_numpy(arr).to(ref.device)
        if ref.dtype.is_floating_point:
            # Cameras arrive normalised; match them or the panel saturates.
            tile = tile.to(ref.dtype) / 255.0
            if float(ref.max()) <= 0.5:
                tile = tile - 0.5
        else:
            tile = tile.to(ref.dtype)
        if tile.shape[0] == ref.shape[0] and tile.shape[-1] == ref.shape[-1]:
            frames.append(tile)
    strip = torch.cat(frames, dim=1)
    # h264 wants even dimensions, and the panel is scaled to whatever height it
    # joins, so its width lands wherever it lands. One black column is cheaper
    # than an eval whose video silently fails to encode.
    if strip.shape[1] % 2:
        strip = torch.cat([strip, torch.zeros_like(strip[:, :1])], dim=1)
    if strip.shape[0] % 2:
        strip = torch.cat([strip, torch.zeros_like(strip[:1])], dim=0)
    return strip


def _render_frame(envs, env_idx, panel_fn=None):
    """One env's human-render camera, with the graph beside it.

    Separate from the observation strip because it is a different camera: the
    encoder reads 112x112 sensors, while this is the task's third-person view
    at whatever `env.eval_render_size` asks for. None when the suite has no
    such camera, and the caller falls back to the observations.
    """
    render = getattr(envs, "render", None)
    if render is None:
        return None
    try:
        frames = render()
    except Exception:
        # Diagnostic only; a renderer that refuses must not end an eval run.
        return None
    if frames is None or len(frames) <= env_idx:
        return None
    return _with_panel([frames[env_idx]], panel_fn)


def _observation_frame(trans, panel_fn=None):
    """First environment's cameras tiled left-to-right as one RGB frame.

    ``panel_fn(height)`` supplies an extra column -- the graph diagram -- sized
    from the camera strip itself, so the two views and the graph read as one
    frame rather than three separate videos.
    """
    if "image" in trans:
        keys = ["image"]
    else:
        # MS-HAB names its fixed view "head", ordinary ManiSkill names it
        # "base". Naming both keeps the fixed view on the left in either suite;
        # anything else falls in sorted after them.
        preferred = ["image_head", "image_base", "image_hand"]
        keys = [key for key in preferred if key in trans]
        keys += sorted(
            key for key in trans.keys()
            if key.startswith("image_") and key not in keys
        )
    frames = []
    for key in keys:
        frame = trans[key]
        while frame.ndim > 3:
            frame = frame[0]
        if frame.ndim != 3 or frame.shape[-1] not in (1, 3):
            continue
        frames.append(frame.detach())
    return _with_panel(frames, panel_fn)


class OnlineTrainer:
    def __init__(self, config, replay_buffer, logger, logdir, train_envs, eval_envs):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.steps = int(config.steps)
        self.pretrain = int(config.pretrain)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.video_pred_log = bool(config.video_pred_log)
        self.video_fps = int(config.video_fps)
        self.params_hist_log = bool(config.params_hist_log)
        self.batch_length = int(config.batch_length)
        batch_steps = int(config.batch_size * config.batch_length)
        self._batch_steps = batch_steps
        # train_ratio is based on data steps rather than environment steps.
        self._updates_needed = tools.Every(batch_steps / config.train_ratio * config.action_repeat)
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(config.update_log_every)
        self._should_eval = tools.Every(self.eval_every)
        self._should_video = tools.Every(config.video_every)
        self._action_repeat = config.action_repeat

    def eval(self, agent, train_step):
        """Run evaluation episodes.

        For CPU-based environments (``ParallelEnv``), stepping is executed on
        CPU and observations are moved to GPU asynchronously.  For GPU-resident
        environments (``IsaacLabVecEnv``), no device transfer is needed —
        ``.to()`` is a no-op when source and target devices match.
        """
        print("Evaluating the policy...")
        envs = self.eval_envs
        agent.eval()
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(envs.env_num, dtype=torch.bool, device=agent.device)
        steps = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        # Gauges have to be reduced the way the training loop reduces them
        # (max over the episode, or a per-frame fraction) or the two are not
        # comparable. Summing a per-frame gauge over a 200-step episode is what
        # turns a 4-entity scene into "730 entities".
        log_sums = {}
        log_maxima = {}
        # cache is only used for video logging / open-loop prediction.
        cache = []
        video_frames = []
        # Graph panel: masks are only built for envs listed here, so recording
        # is scoped to the one env whose video is logged.
        graph_builder = _graph_builder(envs)
        colormap = None
        if graph_builder is not None:
            try:
                from scenegraph.viz.palette import ColorMap
                colormap = ColorMap()      # shared, so colours stay stable
                graph_builder.record_graph_env_indices = {0}
            except Exception:
                graph_builder = None
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while not once_done.all():
            steps += ~done * ~once_done
            # Step environments.  Each env backend handles device placement
            # internally (ParallelEnv converts to CPU, IsaacLabVecEnv keeps
            # on GPU).  The .to() calls below are no-ops when the data is
            # already on agent.device.
            # (B, A), (B,)
            trans, step_done = envs.step(act.detach(), done)
            # dict of (B, 1, *)
            trans = trans.to(agent.device, non_blocking=True)
            panel_fn = (
                (lambda h: _graph_panel(graph_builder, 0, h, colormap))
                if graph_builder is not None else None
            )
            # The render camera when the suite has one, the observation strip
            # otherwise. Not both: two videos of the same episode at different
            # resolutions is worse than one.
            frame = _render_frame(envs, 0, panel_fn)
            if frame is None:
                frame = _observation_frame(trans, panel_fn)
            if frame is not None:
                video_frames.append(frame)
            # (B,)
            done = step_done.to(agent.device)

            # Store transition.
            # We keep the observation and the action that produced it together.
            trans["action"] = act
            if len(cache) < self.batch_length:
                # Each step returns fresh tensors. A shallow container copy is
                # sufficient and supports uint16 graph entity IDs on CUDA.
                cache.append(trans.copy())
            # (B, A)
            act, agent_state = agent.act(trans, agent_state, eval=True)
            returns += trans["reward"][:, 0] * ~once_done
            for key, value in trans.items():
                if key.startswith("log_"):
                    frame = value[:, 0] * ~once_done
                    if key not in log_sums:
                        log_sums[key] = torch.zeros_like(returns)
                        log_maxima[key] = torch.zeros_like(returns)
                    log_sums[key] += frame
                    log_maxima[key] = torch.maximum(log_maxima[key], frame)
            once_done |= done
        if graph_builder is not None:
            graph_builder.record_graph_env_indices = set()
            graph_builder.last_graph_by_env.clear()
            graph_builder.last_masks_by_env.clear()
        # dict of (B, T, *)
        cache = torch.stack(cache, dim=1) if len(cache) else None
        # Its own ``eval/`` namespace rather than ``episode/eval_*``: the two
        # are measured differently -- greedy actions, a separate horizon -- and
        # sharing a prefix puts them on the same dashboard panel as if they
        # were comparable.
        self.logger.scalar("eval/score", returns.mean())
        self.logger.scalar("eval/length", steps.to(torch.float32).mean())
        # Mirrors the training reduction in ``train()``: the target-missing
        # flag becomes the fraction of frames it was set, every other gauge
        # takes the episode max, so a "once" flag reads 1.0 if it ever fired.
        length = steps.to(torch.float32).clamp_min(1)
        for key, value in log_maxima.items():
            if key == "log_graph_target_missing":
                value = log_sums[key] / length
            self.logger.scalar(f"eval/{key[4:]}", value.mean())
        if video_frames:
            video = torch.stack(video_frames, dim=0)
            self.logger.video(
                "eval/video", tools.to_np(video[None]), fps=self.video_fps)
        if self.video_pred_log and cache is not None:
            initial = agent.get_initial_state(1)
            latent_keys = [key for key in LATENT_STATE_KEYS if key in initial]
            self.logger.video(
                "eval/open_loop",
                tools.to_np(
                    agent.video_pred(
                        cache[:1],  # give only first batch
                        tuple(initial[key] for key in latent_keys),
                    )
                ),
            )
        self.logger.write(train_step)
        agent.train()

    def begin(self, agent):
        """Main online training loop.

        For CPU-based environments the loop overlaps CPU stepping and GPU
        model execution via pinned-memory async H2D transfers.  For
        GPU-resident environments (IsaacLab) no transfer is needed —
        ``.to()`` is a no-op when the data is already on the target device.
        """
        envs = self.train_envs
        video_cache = []
        # Keep the same graph panel beside the observation cameras in training
        # videos that evaluation already uses.  Only env 0 is cached because
        # that is the environment selected by ``_observation_frame``.
        graph_builder = _graph_builder(envs)
        colormap = None
        if graph_builder is not None:
            try:
                from scenegraph.viz.palette import ColorMap
                colormap = ColorMap()
                graph_builder.record_graph_env_indices = {0}
            except Exception:
                graph_builder = None
        step = self.replay_buffer.count() * self._action_repeat
        update_count = 0
        policy_fps = tools.FPS()
        train_fps = tools.FPS()
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        lengths = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        episode_log_sums = {}
        episode_log_maxima = {}
        episode_ids = torch.arange(
            envs.env_num, dtype=torch.int32, device=agent.device
        )  # Kept constant so short episodes (< batch_length) remain sampable; RSSM resets via is_first.
        train_metrics = {}
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while step < self.steps:
            # Evaluation
            if self._should_eval(step) and self.eval_episode_num > 0 and self.eval_envs is not None:
                self.eval(agent, step)
            # Save metrics
            if done.any():
                finished = done & lengths.gt(0)
                if finished.any():
                    if len(video_cache) > 0:
                        if self._should_video(step):
                            video = torch.stack(video_cache, axis=0)
                            self.logger.video(
                                "train_video", tools.to_np(video[None]),
                                fps=self.video_fps,
                            )
                        video_cache = []
                    self.logger.scalar("episode/score", returns[finished].mean())
                    self.logger.scalar(
                        "episode/length", lengths[finished].float().mean()
                    )
                    for key, values in episode_log_maxima.items():
                        if key == "log_graph_target_missing":
                            per_env = episode_log_sums[key] / lengths.clamp_min(1)
                        else:
                            per_env = values
                        self.logger.scalar(
                            f"episode/{key[4:]}", per_env[finished].mean()
                        )
                        episode_log_sums[key][finished] = 0
                        episode_log_maxima[key][finished] = 0
                    self.logger.write(step)
                    returns[finished] = 0
                    lengths[finished] = 0
            env_steps = int((~done).sum()) * self._action_repeat
            step += env_steps  # step is based on env side
            policy_fps.step(env_steps)
            lengths += ~done

            # Step environments.  Each env backend handles device placement
            # internally (ParallelEnv converts to CPU, IsaacLabVecEnv keeps
            # on GPU).  The .to() calls below are no-ops when the data is
            # already on agent.device.
            # (B, A), (B,)
            trans, step_done = envs.step(act.detach(), done)
            # dict of (B, 1, *)
            trans = trans.to(agent.device, non_blocking=True)
            # (B,)
            done = step_done.to(agent.device)

            # Policy inference on GPU.
            # "agent_state" is reset by the agent based on the "is_first" flag in trans.
            # (B, A)
            act, agent_state = agent.act(trans, agent_state, eval=False)

            # Store transition.
            # We keep the observation and the action that produced it together.
            # Mask actions after an episode has ended.
            trans["action"] = act * ~done.unsqueeze(-1)
            for key in LATENT_STATE_KEYS:
                if key in agent_state:
                    trans[key] = agent_state[key]
            trans["episode"] = episode_ids  # Don't lift dim
            panel_fn = (
                (lambda h: _graph_panel(graph_builder, 0, h, colormap))
                if graph_builder is not None else None
            )
            frame = _observation_frame(trans, panel_fn)
            if frame is not None:
                video_cache.append(frame)
            self.replay_buffer.add_transition(trans.detach())
            returns += trans["reward"][:, 0]
            active = ~trans["is_first"][:, 0].bool()
            for key, value in trans.items():
                if not key.startswith("log_"):
                    continue
                current = value[:, 0].float() * active
                if key not in episode_log_sums:
                    episode_log_sums[key] = torch.zeros_like(returns)
                    episode_log_maxima[key] = torch.zeros_like(returns)
                episode_log_sums[key] += current
                episode_log_maxima[key] = torch.maximum(
                    episode_log_maxima[key], current
                )
            # Update models after enough data has accumulated
            if step // (envs.env_num * self._action_repeat) > self.batch_length + 1:
                if self._should_pretrain():
                    update_num = self.pretrain
                else:
                    update_num = self._updates_needed(step)
                for _ in range(update_num):
                    # `step` is environment steps; the agent uses it for
                    # the progress-beta warm-up only.
                    _metrics = agent.update(self.replay_buffer, step)
                    train_metrics = _metrics
                    train_fps.step(self._batch_steps)
                update_count += update_num
                # Log training metrics
                if self._should_log(step):
                    for name, value in train_metrics.items():
                        value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                        self.logger.scalar(f"train/{name}", value)
                    self.logger.scalar("train/opt/updates", update_count)
                    if self.video_pred_log:
                        data, _, initial = self.replay_buffer.sample()
                        self.logger.video("open_loop", tools.to_np(agent.video_pred(data, initial)))
                    if self.params_hist_log:
                        for name, param in agent._named_params.items():
                            self.logger.histogram(name, tools.to_np(param))
                    self.logger.scalar("fps/policy", policy_fps.result())
                    self.logger.scalar("fps/train", train_fps.result())
                    for name, value in tools.process_memory_stats().items():
                        self.logger.scalar(name, value)
                    self.logger.write(step)
        if graph_builder is not None:
            graph_builder.record_graph_env_indices = set()
            graph_builder.last_graph_by_env.clear()
            graph_builder.last_masks_by_env.clear()
