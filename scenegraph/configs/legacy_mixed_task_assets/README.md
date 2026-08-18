# Legacy pre-namespacing assets (not loadable)

These are the assets mined on 2026-08-09, before the pipeline was namespaced by
MS-HAB task group. Nothing in the runtime reads this directory; it is kept only
so the numbers behind earlier runs stay inspectable.

They cannot be relabelled into the new layout because they are **a mix of two
tasks**. The collection passed `--task set_table --task prepare_groceries
--task tidy_house` in one run, and the collector then deduplicated by `obj_id`
keeping the first task alphabetically. The result:

* `pick_013_apple.json` — mined from **set_table** rollouts (no
  prepare_groceries checkpoint exists for the apple).
* every other file, `pick_024_bowl.json` included — mined from
  **prepare_groceries** rollouts.
* `pick_all.json` — relation bins taken as the elementwise maximum over that
  mixture, so its scales belong to no single task.

`pick_024_bowl.json` is the clearest symptom: its supporters are
`link:fridge-0/body` and `link:kitchen_counter-0/body`, while set_table's bowl
starts inside a counter drawer. The drawer link is absent, so under set_table
the bowl's actual supporter can never enter the graph.

Re-mine per group instead of reusing these:

    python -m scenegraph.tools.prepare_assets --mshab-task set_table \
        --subtask pick --clean
