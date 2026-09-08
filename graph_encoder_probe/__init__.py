"""Does a small edit to a packed graph move the encoder's pooled token?

One question, one loop: the repository's own ``GraphEncoder`` is wired straight
into its ``SimpleGraphDecoder``, trained on reconstruction alone, and probed
with a fixed set of graph pairs that differ in exactly one field. Nothing here
touches the RSSM, imagination or the actor -- the measured vector is the
encoder's pooled readout, before any dynamics see it.

The repository files this imports are used unchanged. Everything the
experiment adds lives under this package.
"""

from __future__ import annotations

__all__ = ["EDIT_GROUPS", "TOKEN_KEY"]

# The four controlled edits. ``assignment`` is the interesting one: both graphs
# carry the same multiset of labels and differ only in which node pair holds
# which, so a pooled readout that merely counts labels cannot separate them.
EDIT_GROUPS = ("absolute", "temporal", "geometry", "assignment")

# What each group is called where a person reads it. The keys stay the internal
# identifiers -- pair names, CSV columns and config all use those -- but a
# legend saying "control" makes a reader work out what was controlled, and one
# saying "no change" does not.
GROUP_LABELS = {
    "absolute": "absolute label changed",
    "temporal": "temporal label changed",
    "geometry": "centroid moved 1-5 cm",
    "assignment": "labels swapped between pairs",
    "control": "no change",
}

# The same thing again for the console, where the line has to stay one line.
GROUP_SHORT = {
    "absolute": "abs",
    "temporal": "temp",
    "geometry": "geom",
    "assignment": "swap",
    "control": "unchanged",
}

# What the probe measures. Named once so a future change to the encoder's
# readout has a single place to be reflected.
TOKEN_KEY = "token"
