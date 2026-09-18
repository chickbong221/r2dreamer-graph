"""Run test modules and refuse to call an unverified stage a pass.

``unittest`` exits 0 when every test skipped, which for this suite is exactly
wrong. Worse, a stage that mixes a lightweight test with a required integration
test would exit 0 on the strength of the lightweight one -- so stage 4 could
report green having never loaded the checkpoint.

So a module can be marked **required**: if anything in it skipped, the stage is
INCOMPLETE regardless of what else passed. The pretrained, simulator and
gradient stages are required, because their whole purpose is the integration
they skip.

    passed      every required module ran, and nothing failed
    INCOMPLETE  a required module skipped, or nothing ran at all   (exit 2)
    FAILED      something failed or errored                        (exit 1)

Skip reasons are always printed: "incomplete" is only actionable if it names
the dependency that was missing.
"""

from __future__ import annotations

import argparse
import sys
import unittest
from typing import List, Set

# Modules whose skipping makes their stage unverified, whatever else ran.
REQUIRED = {
    "sim_vla.tests.test_pretrained",     # the real checkpoint
    "sim_vla.tests.test_env",            # the real simulator
    "sim_vla.tests.test_imagination",    # gradient flow through the sampler
    "sim_vla.tests.test_online",         # gradient flow through the return
    "sim_vla.tests.test_world_model",    # both arms actually construct
    "sim_vla.tests.test_adapter",        # gradient into the adapter
    "sim_vla.tests.test_imitation",      # the flow loss actually descends
    "sim_vla.tests.test_checkpoint",     # resume, and the arm refusal
    "sim_vla.tests.test_progress",       # the shaping arithmetic
}


def module_of(test) -> str:
    return type(test).__module__


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one test stage")
    parser.add_argument("modules", nargs="+")
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--require", nargs="*", default=None,
                        help="override which modules count as required")
    args = parser.parse_args(argv)

    required: Set[str] = set(args.require if args.require is not None
                             else [m for m in args.modules if m in REQUIRED])

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in args.modules:
        try:
            suite.addTests(loader.loadTestsFromName(name))
        except Exception as exc:                           # noqa: BLE001
            # A module that will not import is a failure, not a skip: the
            # tests it holds were meant to run.
            print(f"!!! could not load {name}: {exc}")
            return 1

    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)
    ran = result.testsRun
    failed = len(result.failures) + len(result.errors)

    if result.skipped:
        print("\n--- skipped:")
        for test, reason in result.skipped:
            print(f"    {test}: {reason}")

    if failed:
        print(f"\n!!! {failed} failed of {ran}")
        return 1

    skipped_modules = {module_of(test) for test, _ in result.skipped}
    unverified: List[str] = sorted(required & skipped_modules)
    if unverified:
        print(f"\n!!! INCOMPLETE: required module(s) skipped: {unverified}")
        print("    these are the integration this stage exists to verify; "
              "the rest of the stage passing does not cover them")
        return 2
    if ran == 0 or len(result.skipped) == ran:
        print(f"\n!!! INCOMPLETE: {len(result.skipped)}/{ran} skipped, "
              "nothing verified")
        return 2

    verified = ran - len(result.skipped)
    print(f"\n--- {verified}/{ran} verified"
          + (f", {len(result.skipped)} skipped" if result.skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
