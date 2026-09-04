"""Whether a successful episode can pay every rung its final phase weights.

The failure this guards against is quiet. A terminal phase completes on an
exact environment distance while its rungs are labelled from mined quantiles;
when a quantile lands inside that distance, a genuinely successful trajectory
leaves a weighted clause unpaid and the potential stops short of 1.0. Every
frame is readable and every number is in range, so nothing reports it.

Three properties the tool has to hold, each with its own regression here:

* **It fails closed.** "Nothing to check" and "checked and fine" must not
  share an exit status.
* **It enumerates, rather than sampling the endpoints.** Endpoint testing
  quietly assumes every clause is an "or better" ladder, and passes a
  ``medium``-only rung that a successful episode cannot pay.
* **It checks only the pair ``reached`` names.** A clause over any other pair
  is measured on a different scale over a geometry the tolerance does not
  bound.

Bin edges and a tolerance are the inputs, never a hand-written label: a clause
labelled ``very-near`` proves nothing about geometry.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scenegraph.core.spatial_metrics import (
    EE_SITE_HEIGHT_KEY,
    EE_SITE_PLANAR_KEY,
    OBJECT_SITE_HEIGHT_KEY,
    OBJECT_SITE_PLANAR_KEY,
)
from scenegraph.tools.check_terminal_rungs import (
    UNPAYABLE,
    UNVERIFIABLE,
    allowed_interval,
    check,
    clause_findings,
    main,
    reachable_labels,
)

TOL = 0.05
PAIR = ("ee", "rest_site")
KEYS = {"planar-distance": EE_SITE_PLANAR_KEY,
        "height-offset": EE_SITE_HEIGHT_KEY}

# Comfortably wider than the tolerance.
ROOMY = {EE_SITE_PLANAR_KEY: [0.15, 0.30, 0.45, 0.60],
         EE_SITE_HEIGHT_KEY: [-0.45, -0.12, 0.12, 0.45]}
# Tighter: 0.02 planar and a +-0.01 level band both sit inside the ball.
TIGHT = {EE_SITE_PLANAR_KEY: [0.02, 0.05, 0.09, 0.14],
         EE_SITE_HEIGHT_KEY: [-0.09, -0.01, 0.01, 0.09]}

SCHEDULE = Path("scenegraph/configs/schedules/tidy_house/pick.json")
SITES = Path("scenegraph/configs/sites/tidy_house/pick.json")
REST_SITE = "spatial:ee_rest_site"


def _clause(relation, labels, weight=0.1, src="ee", dst="rest_site"):
    return {"relation": relation, "labels": list(labels), "weight": weight,
            "src": src, "dst": dst}


def _run(bins, clauses, tolerance=TOL, keys=None, pair=PAIR):
    return clause_findings(bins, clauses, tolerance, keys or KEYS, pair, "term")


class ReachableLabelTest(unittest.TestCase):
    """Every label the interval can take, not only its ends."""

    EDGES = [0.10, 0.20, 0.30, 0.40]
    NAMES = ["very-near", "near", "medium", "far", "very-far"]

    def test_a_narrow_interval_takes_one_label(self):
        self.assertEqual(reachable_labels(0.0, 0.05, self.EDGES, self.NAMES),
                         ["very-near"])

    def test_a_wide_interval_takes_every_label_between(self):
        self.assertEqual(reachable_labels(0.0, 0.25, self.EDGES, self.NAMES),
                         ["very-near", "near", "medium"])

    def test_an_interval_beyond_the_last_edge_saturates(self):
        self.assertEqual(reachable_labels(0.35, 0.90, self.EDGES, self.NAMES),
                         ["far", "very-far"])

    def test_a_value_equal_to_an_edge_lands_in_the_higher_bin(self):
        """``searchsorted(..., side='right')``. An edge sitting exactly on the
        tolerance is therefore outside the finer bin."""
        self.assertEqual(reachable_labels(0.10, 0.10, self.EDGES, self.NAMES),
                         ["near"])
        self.assertEqual(
            reachable_labels(0.10 - 1e-9, 0.10 - 1e-9, self.EDGES, self.NAMES),
            ["very-near"])


class EndpointAssumptionTest(unittest.TestCase):
    """The two clause shapes endpoint testing would wave through."""

    def test_a_medium_only_planar_rung_is_caught(self):
        """On TIGHT the ball spans very-near, near and medium. An endpoint
        check reads only 0.05, labels it ``medium``, finds it accepted and
        passes -- while the nearer two thirds of the ball are not."""
        findings, checked, _ = _run(TIGHT, [_clause("planar-distance",
                                                    ["medium"])])
        self.assertEqual(checked, 1)
        self.assertEqual({f.kind for f in findings}, {UNPAYABLE})
        self.assertEqual({f.label for f in findings}, {"very-near", "near"})

    def test_a_height_rung_excluding_level_is_caught(self):
        """On TIGHT both endpoints of ``[-tol, +tol]`` land in ``below`` and
        ``above``, which the clause accepts, so an endpoint check passes it --
        while everything between is ``level``, which it does not."""
        findings, checked, _ = _run(
            TIGHT, [_clause("height-offset", ["below", "above"])])
        self.assertEqual(checked, 1)
        self.assertEqual([f.kind for f in findings], [UNPAYABLE])
        self.assertEqual(findings[0].label, "level")

    def test_a_height_rung_spanning_the_interval_passes_the_same_scale(self):
        self.assertEqual(_run(TIGHT, [_clause(
            "height-offset", ["below", "level", "above"])])[0], [])

    def test_a_ladder_rung_spanning_the_whole_interval_passes(self):
        self.assertEqual(_run(TIGHT, [_clause(
            "planar-distance", ["far", "medium", "near", "very-near"])])[0], [])


class CalibrationTest(unittest.TestCase):

    def test_the_finest_rungs_pass_on_a_roomy_scale(self):
        findings, checked, _ = _run(ROOMY, [
            _clause("planar-distance", ["very-near"]),
            _clause("height-offset", ["level"])])
        self.assertEqual(findings, [])
        self.assertEqual(checked, 2)

    def test_the_finest_planar_rung_fails_a_tight_scale(self):
        """Two labels inside the ball the clause does not accept, not one.
        Endpoint testing would have found only the far one."""
        findings, _c, _s = _run(TIGHT, [_clause("planar-distance",
                                                ["very-near"])])
        self.assertEqual({f.kind for f in findings}, {UNPAYABLE})
        # 0.05 sits exactly on the second edge, so it lands in the bin above.
        self.assertEqual({f.label for f in findings}, {"near", "medium"})

    def test_the_level_band_fails_a_tight_scale_on_both_signs(self):
        findings, _c, _s = _run(TIGHT, [_clause("height-offset", ["level"])])
        self.assertEqual({f.label for f in findings}, {"below", "above"})

    def test_planar_and_height_are_judged_independently(self):
        """They are mined from separate statistics, so a wide scale on one
        says nothing about the other."""
        mixed = {EE_SITE_PLANAR_KEY: ROOMY[EE_SITE_PLANAR_KEY],
                 EE_SITE_HEIGHT_KEY: TIGHT[EE_SITE_HEIGHT_KEY]}
        findings, checked, _ = _run(mixed, [
            _clause("planar-distance", ["very-near"]),
            _clause("height-offset", ["level"])])
        self.assertEqual(checked, 2)
        self.assertEqual({f.relation for f in findings}, {"height-offset"})

    def test_the_interval_differs_by_relation(self):
        self.assertEqual(allowed_interval("planar-distance", TOL), (0.0, TOL))
        self.assertEqual(allowed_interval("height-offset", TOL), (-TOL, TOL))
        self.assertIsNone(allowed_interval("grasp", TOL))


class PairRestrictionTest(unittest.TestCase):
    """Only the pair ``reached`` names is bounded by the tolerance."""

    def test_a_weighted_clause_over_another_pair_is_unverifiable(self):
        """It still has to be paid at success, and nothing here can say
        whether it can -- so it is an unanswered question, not a note."""
        findings, checked, _skipped = _run(
            TIGHT, [_clause("planar-distance", ["very-near"], dst="target")])
        self.assertEqual([f.kind for f in findings], [UNVERIFIABLE])
        self.assertEqual(checked, 0)

    def test_the_reason_names_the_pair_it_saw(self):
        findings, _c, _s = _run(
            TIGHT, [_clause("planar-distance", ["very-near"], dst="target")])
        text = str(findings[0])
        self.assertIn("target", text)
        self.assertIn("reached", text)

    def test_a_zero_weight_gate_is_not_checked(self):
        _f, checked, skipped = _run(
            TIGHT, [_clause("planar-distance", ["very-near"], weight=0.0)])
        self.assertEqual(checked, 0)
        self.assertIn("gate", skipped[0])

    def test_a_non_spatial_clause_is_silent(self):
        findings, checked, skipped = _run(TIGHT, [_clause("reached", ["holds"])])
        self.assertEqual((findings, checked, skipped), ([], 0, []))


class FailClosedTest(unittest.TestCase):
    """Nothing-to-check must never read as checked-and-fine."""

    def _schedule(self, phases, roles=None):
        return {"roles": roles or {"rest_site": REST_SITE}, "phases": phases}

    def _sites(self, **over):
        entry = {"site_type": "point", "metric": "euclidean", "subject": "ee"}
        entry.update(over)
        return {REST_SITE: entry}

    def _phase(self, clauses, completion=None):
        return {"name": "term", "weight": 1.0, "clauses": clauses,
                "completion": completion or {
                    "relation": "reached", "src": "ee", "dst": "rest_site"}}

    def test_a_schedule_with_no_reached_phase_is_a_finding(self):
        findings, _notes = check(
            {"bin_edges": ROOMY},
            self._schedule([{"name": "a", "clauses": [],
                             "completion": {"relation": "grasp"}}]),
            self._sites(), TOL)
        self.assertEqual([f.kind for f in findings], [UNVERIFIABLE])

    def test_an_unresolved_site_is_a_finding(self):
        """``reached`` over a pair naming no declared site: the scale that
        labels it cannot be identified, so nothing can be concluded."""
        findings, _notes = check(
            {"bin_edges": ROOMY},
            self._schedule([self._phase([])], roles={"rest_site": "actor:x"}),
            {}, TOL)
        self.assertEqual([f.kind for f in findings], [UNVERIFIABLE])
        self.assertIn("no declared site", str(findings[0]))

    def test_a_surface_site_is_refused_rather_than_guessed(self):
        """It is entered through an oriented tube, not a ball."""
        findings, _notes = check(
            {"bin_edges": ROOMY},
            self._schedule([self._phase([_clause("planar-distance",
                                                 ["very-near"])])]),
            self._sites(site_type="surface"), TOL)
        self.assertEqual([f.kind for f in findings], [UNVERIFIABLE])
        self.assertIn("surface", str(findings[0]))

    def test_a_planar_metric_is_refused(self):
        """A cylinder, not a ball."""
        findings, _notes = check(
            {"bin_edges": ROOMY},
            self._schedule([self._phase([_clause("planar-distance",
                                                 ["very-near"])])]),
            self._sites(metric="planar"), TOL)
        self.assertEqual([f.kind for f in findings], [UNVERIFIABLE])

    def test_a_phase_with_no_checkable_clause_is_a_finding(self):
        findings, _notes = check(
            {"bin_edges": ROOMY},
            self._schedule([self._phase([_clause("grasp", ["holds"])])]),
            self._sites(), TOL)
        self.assertEqual([f.kind for f in findings], [UNVERIFIABLE])
        self.assertIn("nothing about this phase was verified",
                      str(findings[0]))

    def test_a_missing_calibration_is_a_finding_not_a_pass(self):
        """Two facts, both true: the scale is absent, and consequently the
        phase verified nothing."""
        findings, _notes = check(
            {"bin_edges": {}},
            self._schedule([self._phase([_clause("planar-distance",
                                                 ["very-near"])])]),
            self._sites(), TOL)
        self.assertEqual({f.kind for f in findings}, {UNVERIFIABLE})
        text = " ".join(str(f) for f in findings)
        self.assertIn("calibration", text)
        self.assertIn("nothing about this phase was verified", text)

    def test_each_phase_is_judged_on_its_own(self):
        """One phase carrying a verified rung must not vouch for another that
        carried none."""
        good = self._phase([_clause("planar-distance", ["very-near"])])
        good["name"] = "verified"
        bare = self._phase([_clause("grasp", ["holds"])])
        bare["name"] = "bare"
        findings, _notes = check(
            {"bin_edges": ROOMY}, self._schedule([good, bare]),
            self._sites(), TOL)
        self.assertEqual([f.phase for f in findings], ["bare"])

    def test_a_well_formed_phase_passes(self):
        findings, notes = check(
            {"bin_edges": ROOMY},
            self._schedule([self._phase([_clause("planar-distance",
                                                 ["very-near"])])]),
            self._sites(), TOL)
        self.assertEqual(findings, [])
        self.assertIn("1 weighted spatial clause(s) checked", " ".join(notes))


class CommandLineTest(unittest.TestCase):
    """Exit status, because that is what a pipeline reads."""

    def _run(self, bins, schedule, sites, tolerance="0.05"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "asset.json").write_text(json.dumps({"bin_edges": bins}))
            (root / "sched.json").write_text(json.dumps(schedule))
            (root / "sites.json").write_text(json.dumps({"sites": sites}))
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), \
                    contextlib.redirect_stderr(buffer):
                code = main(["--asset", str(root / "asset.json"),
                             "--schedule", str(root / "sched.json"),
                             "--sites", str(root / "sites.json"),
                             "--tolerance", tolerance])
            return code, buffer.getvalue()

    def _schedule(self, clauses):
        return {"roles": {"rest_site": REST_SITE},
                "phases": [{"name": "term", "weight": 1.0, "clauses": clauses,
                            "completion": {"relation": "reached", "src": "ee",
                                           "dst": "rest_site"}}]}

    SITES = {REST_SITE: {"site_type": "point", "metric": "euclidean",
                         "subject": "ee"}}

    def test_a_payable_schedule_exits_zero(self):
        code, out = self._run(
            ROOMY, self._schedule([_clause("planar-distance", ["very-near"])]),
            self.SITES)
        self.assertEqual(code, 0)
        self.assertIn("[ ok ]", out)

    def test_an_unpayable_rung_exits_one(self):
        code, out = self._run(
            TIGHT, self._schedule([_clause("planar-distance", ["very-near"])]),
            self.SITES)
        self.assertEqual(code, 1)
        self.assertIn("unpayable", out)

    def test_an_empty_check_exits_one(self):
        code, out = self._run(
            ROOMY, self._schedule([_clause("grasp", ["holds"])]), self.SITES)
        self.assertEqual(code, 1)
        self.assertIn("unverifiable", out)

    def test_an_unresolved_site_exits_one(self):
        code, _out = self._run(
            ROOMY, self._schedule([_clause("planar-distance", ["very-near"])]),
            {})
        self.assertEqual(code, 1)

    def test_a_non_positive_tolerance_exits_two(self):
        code, _out = self._run(
            ROOMY, self._schedule([_clause("planar-distance", ["very-near"])]),
            self.SITES, tolerance="0")
        self.assertEqual(code, 2)

    def test_an_unreadable_input_exits_two(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), \
                contextlib.redirect_stderr(buffer):
            code = main(["--asset", "no/such.json", "--schedule",
                         "no/such.json", "--tolerance", "0.05"])
        self.assertEqual(code, 2)

    def test_a_weighted_clause_over_another_pair_exits_one(self):
        """It used to be a note the run passed through."""
        code, out = self._run(
            ROOMY,
            self._schedule([_clause("planar-distance", ["very-near"]),
                            _clause("planar-distance", ["very-near"],
                                    dst="target")]),
            self.SITES)
        self.assertEqual(code, 1)
        self.assertIn("unverifiable", out)
        self.assertIn("target", out)

    def test_one_verified_phase_does_not_cover_an_empty_one(self):
        schedule = self._schedule([_clause("planar-distance", ["very-near"])])
        bare = dict(schedule["phases"][0])
        bare["name"] = "bare"
        bare["clauses"] = [_clause("grasp", ["holds"])]
        schedule["phases"] = schedule["phases"] + [bare]
        code, out = self._run(ROOMY, schedule, self.SITES)
        self.assertEqual(code, 1)
        self.assertIn("bare", out)

    def test_the_report_proposes_no_change(self):
        """It states what it found. Which rung to keep is not its call."""
        _code, out = self._run(
            TIGHT, self._schedule([_clause("planar-distance", ["very-near"])]),
            self.SITES)
        self.assertIn("Review required", out)
        self.assertNotIn("Fix by removing", out)


class ShippedScheduleTest(unittest.TestCase):
    """The tidy_house Pick schedule against calibrations of both kinds.

    Its real bins do not exist yet, so this asks the question the final asset
    will have to answer, on stand-ins whose relationship to the tolerance is
    known.
    """

    def _check(self, bin_edges):
        with open(SCHEDULE) as handle:
            schedule = json.load(handle)
        with open(SITES) as handle:
            sites = json.load(handle)["sites"]
        return check({"bin_edges": bin_edges}, schedule, sites, TOL)

    def test_it_finds_the_terminal_phase_and_the_gripper_scale(self):
        _findings, notes = self._check(ROOMY)
        text = " ".join(notes)
        self.assertIn("return_to_rest", text)
        self.assertIn(EE_SITE_PLANAR_KEY, text)

    def test_it_checks_the_site_clauses_and_skips_the_grasp_pair(self):
        """The terminal phase also rewards holding on, over a different pair
        the tolerance does not bound."""
        _findings, notes = self._check(ROOMY)
        text = " ".join(notes)
        self.assertIn("weighted spatial clause(s) checked", text)
        self.assertNotIn("0 weighted spatial clause(s) checked", text)

    def test_it_passes_on_a_roomy_calibration(self):
        findings, _notes = self._check(ROOMY)
        self.assertEqual([str(f) for f in findings], [])

    def test_a_tight_enough_calibration_breaks_a_retained_rung(self):
        """The schedule is not unconditionally safe, and this is the evidence.

        Dropping the finest rungs protects it against a moderately tight
        scale, but on one whose finest planar bin is 0.02 m even the retained
        ``near-or-better`` rung labels a successful return as ``medium``.
        Whether the real scale is anywhere near that tight only the mined
        asset answers, which is why nothing is being redistributed here.
        """
        findings, _notes = self._check(TIGHT)
        self.assertEqual({f.relation for f in findings}, {"planar-distance"})
        self.assertEqual(findings[0].labels, ("near", "very-near"))

    def test_the_height_band_survives_that_same_asset(self):
        """One rung, not the phase: any fix the real bins call for would be
        identifiable rather than wholesale."""
        findings, _notes = self._check(TIGHT)
        self.assertNotIn("height-offset", [f.relation for f in findings])

    def test_the_approach_phase_is_not_examined(self):
        """It completes on grasp, not on a distance, so no implication rule
        applies -- and it does carry the finest rungs."""
        _findings, notes = self._check(TIGHT)
        self.assertNotIn("approach", " ".join(notes))


if __name__ == "__main__":
    unittest.main()
