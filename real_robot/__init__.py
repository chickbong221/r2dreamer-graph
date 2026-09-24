"""Scene graphs for the SO-101 LeRobot dataset ``hungho77/so101-multitask``.

Each episode's two camera videos are shown whole to Gemini, which returns the
graph at every frame -- the active target, every fact's labels in the
repository's relation vocabulary, and box keyframes -- and the repository's
own packer turns it into the arrays ``GraphEncoder`` reads, one file per
episode, keyed to the LeRobot frame indices.

    preprocessing.download -> prepare_videos -> annotate_episode -> pack_graphs
    evaluation.render_annotations draws an annotation over its videos.
"""
