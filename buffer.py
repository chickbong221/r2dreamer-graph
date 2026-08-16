import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler

from rssm import LATENT_STATE_KEYS


class Buffer:
    def __init__(self, config):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.num_eps = 0
        self._buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=config.max_size, device=self.storage_device, ndim=2),
            sampler=SliceSampler(
                num_slices=self.batch_size, end_key=None, traj_key="episode", truncated_key=None, strict_length=True
            ),
            prefetch=0,
            batch_size=self.batch_size * (self.batch_length + 1),  # +1 for context
        )

    def add_transition(self, data):
        # This is batched data and lifted for storage.
        # (B, ...) -> (B, 1, ...)
        self._buffer.extend(data.unsqueeze(1))

    def sample(self):
        sample_td, info = self._buffer.sample(return_info=True)
        # The sampler returns a flattened batch of length B*(T+1).
        # (B*(T+1), ...) -> (B, T+1, ...)
        sample_td = sample_td.view(-1, self.batch_length + 1)
        src_dev = sample_td.device
        if src_dev.type == "cpu" and self.device.type == "cuda":
            sample_td = sample_td.pin_memory().to(self.device, non_blocking=True)
        elif src_dev != self.device:
            sample_td = sample_td.to(self.device, non_blocking=True)
        # The initial ones are used only to extract the latent vector
        state_keys = [key for key in LATENT_STATE_KEYS if key in sample_td]
        initial = tuple(sample_td[key][:, 0] for key in state_keys)
        data = sample_td[:, 1:]
        # Action is 1 step back; clone because source and destination overlap.
        data.set_("action", sample_td["action"][:, :-1].clone())
        index = [ind.view(-1, self.batch_length + 1)[:, 1:] for ind in info["index"]]
        return data, index, initial

    def update(self, index, **states):
        # Replay state is float32. Move model outputs back to that storage
        # boundary before indexed assignment (TensorDict requires exact dtype
        # and device matches).
        if "deter" not in states:
            raise KeyError("replay latent update requires deter")
        states = {
            key: value.to(device=self.storage_device, dtype=torch.float32)
            for key, value in states.items()
        }
        # Flatten the data
        index = [ind.reshape(-1) for ind in index]
        # In storage, the length is the first dimension, and the batch (number of environments) is the second dimension.
        n = index[0].shape[0]
        values = {
            key: value.reshape(-1, *value.shape[2:])
            for key, value in states.items()
        }
        self._buffer[index[1], index[0]] = TensorDict(values, batch_size=(n,))

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()
