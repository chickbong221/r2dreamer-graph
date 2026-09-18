"""Report whether this machine can run the pretrained integration, and why not.

The failure that prompted this was::

    ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'

which names neither package's version and happens on ``import transformers``,
before any project code runs. It means the two are a mismatched pair, and the
useful output is the pair.

    python -m sim_vla.doctor
"""

from __future__ import annotations

import sys
from typing import Dict, Optional

from .models.pretrained import SUPPORTED_LEROBOT, VERIFIED_LEROBOT

# The coherent pairs. transformers imports is_offline_mode from huggingface_hub,
# and the two are released together: 4.x with hub 0.3x, 5.x with hub 1.x.
REQUIREMENTS = {
    "0.4.4": {"python": "3.10-3.12", "torch": ">=2.2.1,<2.11.0",
              "transformers": ">=4.57.1,<5.0.0",
              "huggingface-hub": ">=0.34.2,<0.36.0",
              "num2words": ">=0.5.14,<0.6.0",
              "accelerate": ">=1.7.0,<2.0.0"},
    "0.6.1": {"python": ">=3.12", "torch": ">=2.7,<2.12.0",
              "transformers": ">=5.4.0,<5.6.0",
              "huggingface-hub": ">=1.6.0,<2.0.0",
              "num2words": ">=0.5.14,<0.6.0",
              "accelerate": ">=1.14.0,<2.0.0"},
}


REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
# torchrl and tensordict are built against a specific torch release, so an
# upgrade that satisfies lerobot can quietly break the existing Dreamer
# pipeline. They are checked by import, not by version arithmetic.
COUPLED_TO_TORCH = ("torchrl", "tensordict")


def repo_torch_pin() -> Optional[str]:
    """The torch this project pins, read from its own pyproject."""
    try:
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    except Exception:                                      # noqa: BLE001
        return None
    for line in text.splitlines():
        stripped = line.strip().strip('",')
        if stripped.startswith("torch=="):
            return stripped.split("==", 1)[1]
    return None


def import_check(name: str) -> Optional[str]:
    """None if it imports, else the error -- which is the thing worth seeing."""
    try:
        __import__(name)
        return None
    except Exception as exc:                               # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def installed(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:                                      # noqa: BLE001
        return None


def report() -> Dict[str, object]:
    packages = ("lerobot", "transformers", "huggingface-hub", "torch",
                "accelerate", "num2words", "safetensors")
    found = {name: installed(name) for name in packages}
    python = ".".join(str(v) for v in sys.version_info[:3])

    problems = []
    # Absent is a problem, not a clean bill of health: this report exists to
    # say whether the pretrained integration can run here.
    missing = [name for name in ("lerobot", "transformers", "huggingface-hub",
                                 "torch") if not found.get(name)]
    if missing:
        problems.append(f"not installed: {missing}")
    hub, tfm = found.get("huggingface-hub"), found.get("transformers")
    if hub and tfm:
        hub_major = int(hub.split(".")[0])
        tfm_major = int(tfm.split(".")[0])
        # 4.x pairs with hub 0.x; 5.x pairs with hub 1.x. Anything else is the
        # mismatch that produces the is_offline_mode ImportError.
        if (tfm_major >= 5) != (hub_major >= 1):
            problems.append(
                f"transformers {tfm} and huggingface-hub {hub} are a "
                "mismatched pair; this is what raises "
                "\"cannot import name 'is_offline_mode'\"")
    if sys.version_info[:2] >= (3, 12):
        problems.append(
            "note: python >= 3.12, so lerobot 0.6.1 is installable here too")
    elif sys.version_info[:2] < (3, 10):
        problems.append("python < 3.10; no verified lerobot version supports it")

    # Installing lerobot can move torch. That satisfies lerobot and may break
    # the simulator pipeline this repo already runs, which is a different
    # question from whether sim_vla can load a checkpoint.
    pinned = repo_torch_pin()
    torch_version = (found.get("torch") or "").split("+")[0]
    if pinned and torch_version and torch_version != pinned:
        problems.append(
            f"torch {torch_version} is installed but this project pins "
            f"torch=={pinned} (pyproject.toml). The existing Dreamer/ManiSkill "
            "training runs against that pin; sim_vla does not need it moved.")
        for name in COUPLED_TO_TORCH:
            error = import_check(name)
            if error:
                problems.append(
                    f"{name} no longer imports after the torch change: {error}")

    version = found.get("lerobot")
    if version and version not in VERIFIED_LEROBOT:
        problems.append(
            f"lerobot {version} is installed; the interface was verified at "
            f"{list(VERIFIED_LEROBOT)}")
    return {"python": python, "installed": found, "problems": problems}


def main(argv=None) -> int:
    out = report()
    print(f"[doctor] python {out['python']}")
    for name, version in out["installed"].items():
        print(f"[doctor]   {name}: {version or 'NOT INSTALLED'}")
    print(f"\n[doctor] this project targets lerobot {SUPPORTED_LEROBOT}; "
          f"its requirements:")
    for key, value in REQUIREMENTS[SUPPORTED_LEROBOT].items():
        print(f"[doctor]   {key} {value}")
    if out["problems"]:
        print()
        for problem in out["problems"]:
            print(f"[doctor] PROBLEM: {problem}")
        print("\n[doctor] to fix, in this environment:")
        print("  pip install 'lerobot[smolvla]==0.4.4' "
              "'transformers>=4.57.1,<5.0.0' "
              "'huggingface-hub[hf-transfer,cli]>=0.34.2,<0.36.0'")
        pin = repo_torch_pin()
        if pin:
            print("\n[doctor] if torch moved off this project's pin:")
            print(f"  pip install 'torch=={pin}' 'torchrl==0.9.2' "
                  "'tensordict==0.9.1'")
            print("  (lerobot 0.4.4 accepts torch <2.11, so the pin suits both)")
        return 1
    print("\n[doctor] no problems found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
