import contextlib
import io
import json
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import init as nn_init
from torch.utils.tensorboard import SummaryWriter


class Tee(io.TextIOBase):
    """A text stream that duplicates writes to multiple underlying streams.

    This is used to mirror stdout/stderr to a log file while keeping the
    original console output unchanged.
    """

    def __init__(self, *streams):
        super().__init__()
        # Filter out None and keep a stable order.
        self._streams = [s for s in streams if s is not None]

    def write(self, s):
        # io.TextIOBase requires returning number of characters written.
        # Some streams may return None; we still return len(s).
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        # Preserve tty detection for progress bars etc.
        return any(hasattr(stream, "isatty") and stream.isatty() for stream in self._streams)

    def fileno(self):
        # Some libraries, including Isaac Sim's faulthandler setup, require a
        # real file descriptor from sys.stdout/sys.stderr.
        for stream in self._streams:
            if hasattr(stream, "fileno"):
                try:
                    return stream.fileno()
                except io.UnsupportedOperation:
                    continue
        raise io.UnsupportedOperation("fileno")


class ConsoleLogHandle:
    def __init__(self, stdout, stderr, stdout_tee, stderr_tee, file):
        self._stdout = stdout
        self._stderr = stderr
        self._stdout_tee = stdout_tee
        self._stderr_tee = stderr_tee
        self._file = file
        self._closed = False

    def close(self):
        if self._closed:
            return

        import sys

        if sys.stdout is self._stdout_tee:
            sys.stdout = self._stdout
        if sys.stderr is self._stderr_tee:
            sys.stderr = self._stderr
        self._file.close()
        self._closed = True


def setup_console_log(logdir, filename="console.log"):
    """Mirror stdout/stderr to a file under logdir.

    After calling this, anything written to stdout/stderr (print, tracebacks,
    etc.) will be visible both in the terminal and in the log file.

    Returns
    -------
    file handle
        The opened file handle so that the caller can manage its lifetime.
    """
    import sys

    # Line-buffered text file for timely flushing.
    path = logdir / filename
    f = path.open("a", buffering=1)
    stdout = sys.stdout
    stderr = sys.stderr
    stdout_tee = Tee(stdout, f)
    stderr_tee = Tee(stderr, f)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee
    return ConsoleLogHandle(stdout, stderr, stdout_tee, stderr_tee, f)


def to_np(x):
    return x.detach().cpu().numpy()


def to_f32(x):
    return x.to(dtype=torch.float32)


def to_i32(x):
    return x.to(dtype=torch.int32)


def weight_init_(m, fan_type="in"):
    # RMSNorm: initialize scale to 1.
    if isinstance(m, nn.RMSNorm):
        with torch.no_grad():
            m.weight.fill_(1.0)
        return

    weight = getattr(m, "weight", None)
    if weight is None:
        return

    if weight.numel() == 0:
        return

    # This is a torch private API, but widely used and stable.
    in_num, out_num = nn_init._calculate_fan_in_and_fan_out(weight)

    with torch.no_grad():
        fan = {"avg": (in_num + out_num) / 2, "in": in_num, "out": out_num}[fan_type]
        std = 1.1368 * np.sqrt(1 / fan)
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        # set bias always 0
        bias = getattr(m, "bias", None)
        if bias is not None:
            bias.fill_(0.0)


class CudaBenchmark:
    def __init__(self, comment):
        self._comment = comment

    def __enter__(self):
        self._st = torch.cuda.Event(enable_timing=True)
        self._nd = torch.cuda.Event(enable_timing=True)
        self._st.record()

    def __exit__(self, *args):
        self._nd.record()
        torch.cuda.synchronize()
        print(self._comment, self._st.elapsed_time(self._nd) / 1000)


_WANDB_DIAGNOSTICS = {
    "train/semdyn_raw",
    "train/semrep_raw",
    "train/sem_entropy",
    "train/node_app_cos",
    "train/node_bbox_iou",
    "train/node_vis_acc",
    "train/relabs_acc",
    "train/reltemp_acc",
    "train/node_target_acc",
    "train/node_target_frac",
    "train/graph_real_edges",
}


def wandb_scalars(scalars):
    """Keep only metrics needed to compare learning and graph health."""
    return {
        name: value
        for name, value in scalars
        if name.startswith("episode/")
        or name.startswith("train/loss/")
        or name.startswith("train/opt/")
        or name.startswith("system/process_")
        or name in ("fps/policy", "fps/train")
        or name in _WANDB_DIAGNOSTICS
    }


def _linux_process_rss_bytes(path="/proc/self/status"):
    """Return current resident bytes for this process on Linux."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    # Linux reports VmRSS in KiB.
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass
    return None


def _peak_process_rss_bytes():
    """Return the process lifetime peak RSS using the standard library."""
    try:
        import resource
        import sys

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # ru_maxrss is bytes on macOS and KiB on Linux/BSD.
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return None


def process_memory_stats():
    """Current and peak host RAM owned by the training Python process.

    This intentionally measures process RSS rather than machine-wide memory.
    The replay buffer, environments, DINO model, and Dreamer modules all live
    in this process in the MS-HAB runner, so their resident host allocations
    are included. GPU VRAM is a separate resource and is not counted here.
    """
    gib = float(1024 ** 3)
    current = _linux_process_rss_bytes()
    peak = _peak_process_rss_bytes()
    stats = {}
    if current is not None:
        stats["system/process_ram_gib"] = current / gib
    if peak is not None:
        # Sampling and allocator timing can make a platform-reported peak lag
        # the current read very briefly; a peak must never plot below current.
        peak = max(peak, current or 0)
        stats["system/process_peak_ram_gib"] = peak / gib
    return stats


def prepare_video(value):
    """Convert ``[B,T,H,W,C]`` video into tiled ``[T,C,H,B*W]`` uint8."""
    value = np.asarray(value)
    if value.ndim != 5 or value.shape[-1] not in (1, 3):
        raise ValueError(
            f"video must have shape [B,T,H,W,C], got {value.shape}")
    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(255 * value, 0, 255).astype(np.uint8)
    elif value.dtype != np.uint8:
        value = np.clip(value, 0, 255).astype(np.uint8)
    B, T, H, W, C = value.shape
    tiled = value.transpose(1, 2, 0, 3, 4).reshape(T, H, B * W, C)
    return tiled.transpose(0, 3, 1, 2)


class FPS:
    """Count completed items per wall-clock second between log writes."""

    def __init__(self, clock=time.time):
        self._clock = clock
        self._last_time = clock()
        self._count = 0

    def step(self, amount=1):
        self._count += int(amount)

    def result(self):
        now = self._clock()
        duration = now - self._last_time
        value = self._count / duration if duration > 0 else 0.0
        self._last_time = now
        self._count = 0
        return value


class Logger:
    def __init__(self, logdir, filename="metrics.jsonl", wandb_config=None):
        self._logdir = logdir
        self._filename = filename
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
        self._wandb_run = None
        self._wandb = None
        if wandb_config is not None and bool(wandb_config.enabled):
            try:
                import wandb
            except ImportError as exc:
                raise ImportError(
                    "W&B logging is enabled but wandb is not installed"
                ) from exc
            self._wandb_run = wandb.init(
                project=str(wandb_config.project),
                name=str(wandb_config.name) or None,
                group=str(wandb_config.group) or None,
                entity=str(wandb_config.entity) or None,
                mode=str(wandb_config.mode),
                dir=str(logdir),
            )
            self._wandb = wandb
            self._wandb_run.define_metric("env_step")
            self._wandb_run.define_metric("*", step_metric="env_step")
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self._histograms = {}

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value, fps=16):
        self._videos[name] = (np.array(value), int(fps))

    def histogram(self, name, value):
        self._histograms[name] = np.array(value)

    def write(self, step):
        scalars = list(self._scalars.items())
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        with (self._logdir / self._filename).open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            if "/" not in name:
                self._writer.add_scalar("scalars/" + name, value, step)
            else:
                self._writer.add_scalar(name, value, step)
        for name, value in self._images.items():
            self._writer.add_image(name, value, step)
        wandb_videos = {}
        for name, (value, fps) in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            video = prepare_video(value)
            try:
                self._writer.add_video(name, video[None], step, fps)
            except Exception as exc:
                print(f"Could not encode TensorBoard video {name!r}: {exc}")
            if self._wandb_run is not None:
                try:
                    wandb_videos[f"videos/{name}"] = self._wandb.Video(
                        video, fps=fps, format="mp4")
                except Exception as exc:
                    print(f"Could not encode W&B video {name!r} as mp4: {exc}")
        for name, value in self._histograms.items():
            self._writer.add_histogram(name, value, step)

        if self._wandb_run is not None:
            selected = wandb_scalars(scalars)
            payload = {**selected, **wandb_videos}
            if payload:
                self._wandb_run.log({"env_step": int(step), **payload})

        self._writer.flush()
        self._scalars = {}
        self._images = {}
        self._videos = {}

    def close(self):
        self._writer.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def log_hydra_config(self, config, name="config", step=0, log_hparams=False, hparams_run_name="."):
        """
        Log a Hydra/OmegaConf config to TensorBoard:
          - as YAML text under "{name}/yaml"
          - as flattened hparams to the HParams plugin
        """
        # 1) Log YAML to Text plugin
        yaml_str = None
        try:
            from omegaconf import (
                OmegaConf,  # local import to avoid hard dependency at module import
            )

            yaml_str = OmegaConf.to_yaml(config, resolve=True)
        except ImportError:
            # Fallback to string representation
            yaml_str = str(config)
        self._writer.add_text(f"{name}/yaml", f"```yaml\n{yaml_str}\n```", step)

        # 2) Log flattened hparams to HParams plugin
        flat = {}
        container = None
        try:
            from omegaconf import OmegaConf  # local import again

            container = OmegaConf.to_container(config, resolve=True)
        except Exception:
            container = None

        if log_hparams and container is not None:

            def _flatten(prefix, obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        _flatten(f"{prefix}.{k}" if prefix else k, v)
                elif isinstance(obj, (list, tuple)):
                    flat[prefix] = str(obj)
                elif isinstance(obj, (int, float, bool, str)) or obj is None:
                    flat[prefix] = obj if obj is not None else "null"
                else:
                    flat[prefix] = str(obj)

            _flatten("", container)
            # add_hparams requires a non-empty metrics dict
            with contextlib.suppress(TypeError):
                # Avoid creating a timestamped subdirectory by specifying run_name (PyTorch >= 1.14)
                self._writer.add_hparams(flat, {"_": 0}, run_name=hparams_run_name)


def convert(value, precision=32):
    if isinstance(value, dict):
        return {key: convert(val) for key, val in value.items()}
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


def tensorstats(tensor, prefix):
    return {
        f"{prefix}_mean": torch.mean(tensor),
        f"{prefix}_std": torch.std(tensor),
        f"{prefix}_min": torch.min(tensor),
        f"{prefix}_max": torch.max(tensor),
    }


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def recursively_collect_optim_state_dict(obj, path="", optimizers_state_dicts=None, visited=None):
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()
    # avoid cyclic reference
    if id(obj) in visited:
        return optimizers_state_dicts
    visited.add(id(obj))
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update({k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr})
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(attr, new_path, optimizers_state_dicts, visited)
            )
    return optimizers_state_dicts


def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    for path, state_dict in optimizers_state_dicts.items():
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)
        obj_now.load_state_dict(state_dict)


def build_module_tree(module: nn.Module, module_name: str = "") -> dict:
    """Recursively traverse the given nn.Module and build a dictionary with."""
    # 1) Count direct parameters in this module
    direct_param_count = 0
    param_details = {}
    for pname, p in module.named_parameters(recurse=False):
        nump = p.numel()
        param_details[pname] = nump
        direct_param_count += nump

    # 2) Recursively process child modules
    children_info = {}
    for cname, child in module.named_children():
        children_info[cname] = build_module_tree(child, cname)

    # 3) Calculate total parameter count for this module (including all children)
    total = direct_param_count + sum(child["total"] for child in children_info.values())

    return {
        "name": module_name,
        "params": param_details,
        "children": children_info,
        "total": total,
    }


def print_module_tree(info: dict, parent_path: str = "", indent: int = 0):
    """
    Print the module tree built by build_module_tree() in a hierarchical format:
    "(total_parameter_count) (path_to_module_or_param)"
    The function sorts parameters and submodules in descending order of total size.
    """
    # Construct the current path
    name = info["name"]
    if not parent_path:
        full_path = name  # top level
    else:
        if name:  # submodule name is not empty
            full_path = f"{parent_path}/{name}"
        else:
            full_path = parent_path

    # Print total parameter count for the current module
    line = f"{info['total']:11,d} {full_path}"
    print(" " * indent + line)

    # Create a combined list of param_nodes (parameters) and child_nodes (submodules)
    param_nodes = []
    for param_name, count in info["params"].items():
        param_nodes.append({
            "name": param_name,
            "params": {},
            "children": {},
            "total": count,
        })

    child_nodes = list(info["children"].values())

    # Sort by 'total' in descending order
    combined = param_nodes + child_nodes
    combined.sort(key=lambda x: x["total"], reverse=True)

    # Recursively print all children
    for child_info in combined:
        print_module_tree(child_info, full_path, indent + 2)


def compute_rms(tensors):
    """Compute the root mean square (RMS) of a list of tensors."""
    flattened = torch.cat([t.view(-1) for t in tensors if t is not None])
    if len(flattened) == 0:
        return torch.tensor(0.0)
    return torch.linalg.norm(flattened, ord=2) / (flattened.numel() ** 0.5)


def compute_global_norm(tensors):
    """Compute the global norm (L2 norm) across a list of tensors."""
    flattened = torch.cat([t.view(-1) for t in tensors if t is not None])
    if len(flattened) == 0:
        return torch.tensor(0.0)
    return torch.linalg.norm(flattened, ord=2)


def rpad(x, pad):
    for _ in range(pad):
        x = x.unsqueeze(-1)
    return x


def print_param_stats(model):
    """
    Prints formatted statistical information of the parameter values (not gradients)
    for the trainable parameters (.requires_grad=True) of the specified PyTorch model.

    - mean
    - std  (population standard deviation: std(unbiased=False))
    - L2 norm (param.data.norm())
    - RMS (root mean square: sqrt(mean(tensor^2)))

    The hierarchical name is displayed by replacing '.' with '/' in the default names
    (e.g., converting "layer.weight" to "layer/weight").
    """

    # List to temporarily store the statistics
    stats = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            data = param.data
            mean_val = data.mean().item()
            std_val = data.std(unbiased=False).item()
            l2_val = data.norm().item()
            rms_val = data.pow(2).mean().sqrt().item()

            hierarchical_name = name.replace(".", "/")
            stats.append((hierarchical_name, mean_val, std_val, l2_val, rms_val))

    # Format function to display numbers in scientific notation with 3 significant digits
    def fmt(v):
        return f"{v:.3e}"

    # Column width settings (adjust if necessary)
    col_widths = [60, 15, 15, 15, 15]
    header_format = (
        f"{{:<{col_widths[0]}}}{{:>{col_widths[1]}}}{{:>{col_widths[2]}}}{{:>{col_widths[3]}}}{{:>{col_widths[4]}}}"
    )
    row_format = header_format

    # Print the header
    print(header_format.format("Parameter", "Mean", "Std", "L2 norm", "RMS"))
    print("-" * (sum(col_widths) + 1))

    # Print the main content
    for hname, mean_val, std_val, l2_val, rms_val in stats:
        print(
            row_format.format(
                hname,
                fmt(mean_val),
                fmt(std_val),
                fmt(l2_val),
                fmt(rms_val),
            )
        )
