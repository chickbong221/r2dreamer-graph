"""Tests for the real-robot package.

    python -m unittest discover -s real_robot/tests -t .

The numpy-only tests (graph contract, validation with anchor and consistency
checks, the annotation flow against a scripted Gemini, the Gemini client's
failure handling, artifact reuse, packing, reward, episode selection, sequence
windows, the action-mapping audit, geometry, bin grounding, prompt schemas) run
anywhere the repository's scenegraph package imports. Model tests and the
end-to-end run skip themselves without torch and omegaconf.
"""
