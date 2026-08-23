"""Static checks for references the cleanup could have left dangling.

None of this needs torch, so it runs anywhere. Each case is one class of
failure that reached the cluster during the switch cleanup and was invisible
here: an import of a deleted name, an attribute on a deleted member, a Hydra
override or preset that no longer exists, a call site left behind by a
signature edit, and an undefined name.
"""

import ast
import pathlib
import re
import subprocess
import sys
import unittest

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
CODE = ["graph.py", "rssm.py", "dreamer.py", "progress.py", "networks.py",
        "trainer.py", "buffer.py", "envs", "scenegraph", "tests", "runs"]


def sources():
    for entry in CODE:
        path = ROOT / entry
        if path.is_file():
            yield path
        else:
            for f in path.rglob("*.py"):
                if "__pycache__" not in str(f) and "logdir" not in str(f):
                    yield f


def tree_of(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def load_group(group, name):
    """Compose one config file with its `defaults: - _base_` parent merged."""
    path = CONFIGS / group / f"{name}.yaml"
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parents = data.pop("defaults", None) or []
    for parent in parents:
        if parent != "_base_":
            continue
        base = yaml.safe_load(
            (CONFIGS / group / "_base_.yaml").read_text(encoding="utf-8"))
        merged = dict(base)
        for key, value in data.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        data = merged
    return data


class ImportsResolveTest(unittest.TestCase):
    """Every name imported from a local module still exists there."""

    def test_no_import_names_a_deleted_symbol(self):
        exported, missing = {}, []
        for path in sources():
            mod = str(path.relative_to(ROOT)).replace("\\", "/")
            mod = mod[:-3].replace("/", ".")
            names = set()
            for node in tree_of(path).body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)):
                    names.add(node.name)
                elif isinstance(node, ast.Assign):
                    names.update(t.id for t in node.targets
                                 if isinstance(t, ast.Name))
                elif isinstance(node, ast.AnnAssign) and isinstance(
                        node.target, ast.Name):
                    names.add(node.target.id)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    names.update(a.asname or a.name.split(".")[0]
                                 for a in node.names)
            exported[mod] = names
        for path in sources():
            for node in ast.walk(tree_of(path)):
                if not isinstance(node, ast.ImportFrom) or node.level:
                    continue
                if node.module not in exported:
                    continue
                for alias in node.names:
                    if alias.name != "*" and alias.name not in exported[node.module]:
                        missing.append(
                            f"{path.relative_to(ROOT)}: from {node.module} "
                            f"import {alias.name}")
        self.assertEqual(missing, [])


class ConfigKeysExistTest(unittest.TestCase):
    """Hydra rejects an override naming a key the config never declares."""

    def _cfg_for(self, model, env):
        root = yaml.safe_load((CONFIGS / "configs.yaml").read_text(encoding="utf-8"))
        cfg = dict(root)
        if model:
            cfg["model"] = load_group("model", model) or {}
        if env:
            cfg["env"] = load_group("env", env) or {}
        return cfg

    @staticmethod
    def _has(tree, dotted):
        for part in dotted.split("."):
            if not isinstance(tree, dict) or part not in tree:
                return False
            tree = tree[part]
        return True

    def _override_lists(self, path):
        """Every list literal that looks like a Hydra override list."""
        for node in ast.walk(tree_of(path)):
            if not isinstance(node, ast.List):
                continue
            items = []
            for el in node.elts:
                if isinstance(el, ast.Constant) and isinstance(el.value, str):
                    items.append(el.value)
                elif isinstance(el, ast.JoinedStr):
                    items.append("".join(v.value for v in el.values
                                         if isinstance(v, ast.Constant)))
            if any("=" in i and "." in i.split("=")[0] for i in items):
                yield items

    def test_every_test_override_names_a_real_key(self):
        stale = []
        for path in (ROOT / "tests").rglob("*.py"):
            if "__pycache__" in str(path):
                continue
            for items in self._override_lists(path):
                model = next((i.split("=", 1)[1] for i in items
                              if i.startswith("model=")), None)
                env = next((i.split("=", 1)[1] for i in items
                            if i.startswith("env=")), None)
                cfg = self._cfg_for(model, env)
                for item in items:
                    key = item.split("=", 1)[0].lstrip("+~")
                    if "." not in key or key.endswith("."):
                        continue
                    if not self._has(cfg, key):
                        stale.append(f"{path.relative_to(ROOT)}: {item}")
        self.assertEqual(stale, [])

    def test_every_sweep_override_names_a_real_key(self):
        """The scripts that go to the cluster, where a stale key costs a job."""
        stale = []
        for script in sorted((ROOT / "runs").rglob("*.sh")):
            text = script.read_text(encoding="utf-8")
            for block in re.findall(r"python train\.py((?:.|\n)*?)\n\n", text):
                args = re.findall(r"([\w.]+)=(\S+)", block)
                chosen = dict(args)
                cfg = self._cfg_for(chosen.get("model"), chosen.get("env"))
                for key, _ in args:
                    if "." not in key:
                        continue
                    if not self._has(cfg, key):
                        stale.append(f"{script.relative_to(ROOT)}: {key}")
        self.assertEqual(stale, [])


class ModelReadsDeclaredKeysTest(unittest.TestCase):
    """`progress_config.stages` outlived the key it reads.

    The model reaches into the config through several names -- `config`,
    `progress_config`, `graph_config` -- so checking only `config.X` missed it.
    """

    SUBCONFIG = {"progress_config": "progress", "graph_config": "graph",
                 "rssm_config": "rssm"}

    def test_every_subconfig_read_is_declared(self):
        base = yaml.safe_load(
            (CONFIGS / "model" / "_base_.yaml").read_text(encoding="utf-8"))
        missing = []
        for name in ("dreamer.py", "graph.py", "rssm.py"):
            path = ROOT / name
            for node in ast.walk(tree_of(path)):
                if not (isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)):
                    continue
                block = self.SUBCONFIG.get(node.value.id)
                if block is None:
                    continue
                if node.attr.startswith("_"):
                    continue
                if node.attr not in base.get(block, {}):
                    missing.append(f"{name}:{node.lineno}: "
                                   f"{node.value.id}.{node.attr} "
                                   f"(not in model.{block})")
        self.assertEqual(sorted(set(missing)), [])


class ConfigDictLiteralKeysTest(unittest.TestCase):
    """`self._loss_scales.pop("graph_image_recon")` after the key was deleted.

    A config dict copied onto an attribute is read by string literal, which
    the attribute check above cannot see.
    """

    DICTS = {"_loss_scales": "loss_scales"}

    def test_every_literal_dict_key_is_declared(self):
        base = yaml.safe_load(
            (CONFIGS / "model" / "_base_.yaml").read_text(encoding="utf-8"))
        missing = []
        for name in ("dreamer.py", "graph.py", "rssm.py"):
            path = ROOT / name
            for node in ast.walk(tree_of(path)):
                owner = key = None
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("pop", "get")
                        and isinstance(node.func.value, ast.Attribute)
                        and node.args
                        and isinstance(node.args[0], ast.Constant)):
                    owner, key = node.func.value.attr, node.args[0].value
                elif (isinstance(node, ast.Subscript)
                      and isinstance(node.value, ast.Attribute)
                      and isinstance(node.slice, ast.Constant)):
                    owner, key = node.value.attr, node.slice.value
                block = self.DICTS.get(owner)
                if block is None or not isinstance(key, str):
                    continue
                if key not in base.get(block, {}):
                    missing.append(f"{name}:{node.lineno}: {owner}[{key!r}] "
                                   f"(not in model.{block})")
        self.assertEqual(sorted(set(missing)), [])


class LossScalesAreProducedTest(unittest.TestCase):
    """A declared scale with no producer means a deleted loss stage.

    The reward, continuation and progress-model losses were dropped with the
    block that computed them. Nothing imported a missing name and nothing read
    a missing key, so every other check here passed.
    """

    # Spread over the decoder's own keys rather than stored under its name.
    SPREAD = {"recon"}

    def test_every_declared_scale_has_a_producer(self):
        base = yaml.safe_load(
            (CONFIGS / "model" / "_base_.yaml").read_text(encoding="utf-8"))
        produced = set()
        for name in ("dreamer.py", "graph.py", "rssm.py"):
            tree = tree_of(ROOT / name)
            for node in ast.walk(tree):
                if (isinstance(node, ast.Subscript)
                        and isinstance(node.ctx, ast.Store)
                        and isinstance(node.value, ast.Name)
                        and node.value.id.endswith("losses")
                        and isinstance(node.slice, ast.Constant)):
                    produced.add(node.slice.value)
            # `proto_loss` returns its losses as a dict instead of storing
            # them, so a `*loss*` function's returned keys count as producers.
            for fn in ast.walk(tree):
                if not (isinstance(fn, ast.FunctionDef) and "loss" in fn.name):
                    continue
                for node in ast.walk(fn):
                    if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                        produced.update(k.value for k in node.value.keys
                                        if isinstance(k, ast.Constant)
                                        and isinstance(k.value, str))
        orphaned = sorted(set(base["loss_scales"]) - produced - self.SPREAD)
        self.assertEqual(orphaned, [])


class WandbAllowlistIsProducedTest(unittest.TestCase):
    """An allowlisted metric nothing emits is a metric silently not logged.

    The list still named the pre-rename `semdyn_raw`, so the collapse
    diagnostics that replaced it never reached wandb.
    """

    @staticmethod
    def _allowlist():
        """Read the set literal without importing tools, which needs torch."""
        for node in ast.walk(tree_of(ROOT / "tools.py")):
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name)
                            and t.id == "_WANDB_DIAGNOSTICS"
                            for t in node.targets)
                    and isinstance(node.value, ast.Set)):
                return {e.value for e in node.value.elts
                        if isinstance(e, ast.Constant)}
        raise AssertionError("_WANDB_DIAGNOSTICS not found in tools.py")

    def test_every_train_diagnostic_is_emitted(self):
        produced = set()
        for name in ("dreamer.py", "graph.py", "rssm.py"):
            for node in ast.walk(tree_of(ROOT / name)):
                if (isinstance(node, ast.Subscript)
                        and isinstance(node.ctx, ast.Store)
                        and isinstance(node.value, ast.Name)
                        and "metrics" in node.value.id
                        and isinstance(node.slice, ast.Constant)):
                    produced.add(node.slice.value)
                elif isinstance(node, ast.Dict):
                    produced.update(k.value for k in node.keys
                                    if isinstance(k, ast.Constant)
                                    and isinstance(k.value, str))
        orphaned = sorted(
            name for name in self._allowlist()
            if name.startswith("train/")
            and name[len("train/"):] not in produced)
        self.assertEqual(orphaned, [])


class NoOrphanedDefinitionsTest(unittest.TestCase):
    """A function nothing calls, which is how the reward loss went missing.

    `_progress_model_loss` survived the cleanup while the block that called it
    did not. Nothing was undefined and no key was missing, so every other check
    here passed.
    """

    IMPLICIT = {"forward", "main", "reset", "step", "render", "close",
                "observe", "act", "update", "seed", "__call__"}
    # Dead before this refactor; not ours to remove here.
    PREEXISTING = {
        "_sample_action", "area", "cartpole_swingup_subtle", "color_list_for",
        "depth_to_color_rgb", "depth_to_gray_rgb", "has_affordance",
        "index_of", "lookup_grasp_components", "pose_to_world_array",
        "print_param_stats", "purge", "read_unwrapped_rgbs",
        "read_unwrapped_sensor", "recursively_load_optim_state_dict",
        "target_unresolved", "unique_seg_ids", "upsert_node",
        "normal_std_fixed",
    }
    # Stranded by the DINO/appearance removal; the adapter is kept for mining.
    PENDING = {"patch_tokens"}

    def test_no_definition_is_unreachable(self):
        scanned = ["train.py", "distributions.py", *CODE]
        paths = []
        for entry in scanned:
            path = ROOT / entry
            if path.is_file():
                paths.append(path)
            elif path.is_dir():
                paths.extend(f for f in path.rglob("*.py")
                             if "__pycache__" not in str(f))
        defined, referenced = {}, set()
        for path in paths:
            tree = tree_of(path)
            in_tests = "tests" in path.parts
            for node in ast.walk(tree):
                if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and not in_tests
                        and not node.name.startswith("__")
                        and node.name not in self.IMPLICIT):
                    defined.setdefault(node.name, str(
                        path.relative_to(ROOT)) + f":{node.lineno}")
                if isinstance(node, ast.Attribute):
                    referenced.add(node.attr)
                elif isinstance(node, ast.Name):
                    referenced.add(node.id)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    referenced.add(node.value)
        orphans = sorted(
            f"{name} ({where})" for name, where in defined.items()
            if name not in referenced
            and name not in self.PREEXISTING and name not in self.PENDING)
        self.assertEqual(orphans, [])


class PresetsExistTest(unittest.TestCase):
    def test_every_selected_group_resolves_to_a_file(self):
        missing = []
        for path in list((ROOT / "tests").rglob("*.py")) + \
                list((ROOT / "runs").rglob("*.sh")):
            if "__pycache__" in str(path):
                continue
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                # A test asserting a preset is *gone* names it on purpose.
                if "assertNot" in line:
                    continue
                for group, name in re.findall(
                        r"[\"' ](model|env)=([A-Za-z0-9_]+)[\"' \\]", line):
                    if not (CONFIGS / group / f"{name}.yaml").exists():
                        missing.append(f"{path.relative_to(ROOT)}: {group}={name}")
        self.assertEqual(missing, [])


class UndefinedNamesTest(unittest.TestCase):
    """pyflakes catches the NameError class a syntax check cannot."""

    def test_pyflakes_reports_no_undefined_names(self):
        try:
            import pyflakes  # noqa: F401
        except ImportError:
            self.skipTest("pyflakes not installed")
        out = subprocess.run(
            [sys.executable, "-m", "pyflakes", *[str(p) for p in sources()]],
            capture_output=True, text=True, cwd=ROOT,
        )
        serious = [line for line in out.stdout.splitlines()
                   if "undefined name" in line]
        self.assertEqual(serious, [])


if __name__ == "__main__":
    unittest.main()
