"""Tests for the real-robot package.

    python -m unittest discover -s real_robot/tests -t .

Everything but ``test_source`` needs only numpy, PyYAML and the repository's
scenegraph package. ``test_source`` also needs pandas, pyarrow, PyAV and Pillow
and skips, saying so, without them.
"""
