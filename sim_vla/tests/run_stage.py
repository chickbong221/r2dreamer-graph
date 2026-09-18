"""Run test modules and report a stage that only skipped as incomplete.

``unittest`` exits 0 when every test skipped, which for this suite is exactly
wrong: a stage that skipped because lerobot is missing, or because ManiSkill is
not installed, or because no dataset was collected, has not been verified and
must not read as passing. Here a stage is:

    passed      at least one test ran and none failed
    INCOMPLETE  nothing ran, everything skipped        (exit 2)
    FAILED      something failed or errored            (exit 1)

The skip reasons are printed, because "incomplete" is only actionable if it
says which dependency was missing.
"""

from __future__ import annotations

import argparse
import sys
import unittest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one test stage")
    parser.add_argument("modules", nargs="+")
    parser.add_argument("--verbosity", type=int, default=1)
    args = parser.parse_args(argv)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in args.modules:
        try:
            suite.addTests(loader.loadTestsFromName(name))
        except Exception as exc:                           # noqa: BLE001
            print(f"!!! could not load {name}: {exc}")
            return 1

    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)
    ran = result.testsRun
    skipped = len(result.skipped)
    failed = len(result.failures) + len(result.errors)

    if result.skipped:
        print("\n--- skipped:")
        for test, reason in result.skipped:
            print(f"    {test}: {reason}")

    if failed:
        print(f"\n!!! {failed} failed of {ran}")
        return 1
    if ran == 0 or skipped == ran:
        print(f"\n!!! INCOMPLETE: {skipped}/{ran} skipped, nothing verified")
        return 2
    print(f"\n--- {ran - skipped}/{ran} verified"
          + (f", {skipped} skipped" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
