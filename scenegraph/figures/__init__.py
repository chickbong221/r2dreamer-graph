"""Paper-figure capture: labelled human-view stills beside their scene graphs.

The eval video already renders both, but composited into one strip at video
resolution. A figure needs them apart and large, from an episode that actually
succeeded. These modules are that path, and nothing in the training stack
imports them.

    scenegraph.tools.render_paper_frames    the CLI that wires them together
"""
