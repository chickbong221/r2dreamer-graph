"""Read frozen task instructions from the active MS-HAB task plan."""

import numpy as np


def _to_np(value):
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value).reshape(-1)


class InstructionTable:
    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        self.vectors = np.asarray(data["vectors"], np.float32)
        self.keys = [str(key) for key in data["keys"]]
        self.model = str(data["model"])
        if self.vectors.ndim != 2 or len(self.keys) != len(self.vectors):
            raise ValueError(
                f"instruction table {path!r} is malformed: {len(self.keys)} "
                f"keys against vectors {self.vectors.shape}"
            )
        self._rows = {key: index for index, key in enumerate(self.keys)}

    @property
    def dim(self):
        return int(self.vectors.shape[1])

    def row(self, subtask, target):
        key = f"{subtask}/{target}"
        index = self._rows.get(key)
        if index is None:
            raise KeyError(
                f"instruction table has no entry for {key!r}; rebuild it for "
                f"every split used by this run (model={self.model})"
            )
        return self.vectors[index]


class InstructionReader:
    def __init__(self, env, table, num_envs):
        from scenegraph.core.affordance import canonical_affordance_key

        self.table = table
        self.num_envs = int(num_envs)
        self._base = env.unwrapped
        self._canonical = canonical_affordance_key
        self._last = np.zeros((self.num_envs, table.dim), np.float32)
        self._resolved = {}

    def step(self, is_last=None):
        base = self._base
        ptrs = _to_np(getattr(base, "subtask_pointer", None))
        tpis = _to_np(getattr(base, "task_plan_idxs", None))
        bcis = _to_np(getattr(base, "build_config_idxs", None))
        plans = getattr(base, "build_config_idx_to_task_plans", None)
        if ptrs is None or tpis is None or bcis is None or plans is None:
            raise RuntimeError("instruction: MS-HAB task-plan state is unavailable")
        done = (
            np.zeros(self.num_envs, bool)
            if is_last is None
            else np.asarray(is_last, bool).reshape(-1)
        )
        for index in range(self.num_envs):
            if done[index]:
                continue
            triple = (int(bcis[index]), int(tpis[index]), int(ptrs[index]))
            row = self._resolved.get(triple)
            if row is None:
                row = self.table.row(*self._active(plans, triple))
                self._resolved[triple] = row
            self._last[index] = row
        return self._last.copy()

    def _active(self, plans, triple):
        bci, tpi, pointer = triple
        group = plans.get(bci) if hasattr(plans, "get") else plans[bci]
        subtasks = getattr(group[tpi], "subtasks", None) or []
        if not subtasks:
            raise RuntimeError(f"instruction: task plan {bci}/{tpi} has no subtasks")
        subtask = subtasks[min(pointer, len(subtasks) - 1)]
        kind = str(getattr(subtask, "type", ""))
        obj_id = getattr(subtask, "obj_id", None)
        if not obj_id:
            raise RuntimeError(f"instruction: subtask {kind!r} has no object target")
        return kind, "actor:" + str(self._canonical(str(obj_id)))
