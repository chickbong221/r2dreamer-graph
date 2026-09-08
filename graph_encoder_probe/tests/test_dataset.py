"""The cache has to hand back exactly what the packer produced.

The probe measures a difference of about a centimetre in one float32 and one id
in one uint8. A cache that widened a dtype, reordered a shard or silently
concatenated a stale one would change the measurement without changing anything
visible in the report.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS

from ..dataset import FIELD_DTYPES, GraphDataset, GraphFrames, ShardWriter, write_frames
from .synthetic import make_dataset, make_frames

try:
    import torch
except ImportError:                                        # pragma: no cover
    torch = None


def _index_rows(dataset: GraphDataset) -> list[dict]:
    return [
        {key: dataset.index[key][i] for key in dataset.index} for i in range(len(dataset))
    ]


class RoundTrip(unittest.TestCase):
    def setUp(self):
        self.dataset = make_dataset(37, seed=7)

    def _write(self, tmp: str, shard_size: int = 8) -> GraphDataset:
        write_frames(
            tmp,
            (self.dataset.frames.frame(i) for i in range(len(self.dataset))),
            _index_rows(self.dataset),
            self.dataset.meta,
            shard_size=shard_size,
        )
        return GraphDataset.load(tmp)

    def test_packed_tensors_survive_a_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            again = self._write(tmp)
        self.assertEqual(len(again), len(self.dataset))
        for key in GRAPH_KEYS:
            self.assertEqual(again.frames.fields[key].dtype, FIELD_DTYPES[key])
            self.assertTrue(
                np.array_equal(again.frames.fields[key], self.dataset.frames.fields[key]),
                f"{key} changed on the way through disk",
            )
        self.assertEqual(again.frames.fingerprint(), self.dataset.frames.fingerprint())

    def test_frame_order_survives_sharding(self):
        with tempfile.TemporaryDirectory() as tmp:
            again = self._write(tmp, shard_size=5)
        self.assertTrue(np.array_equal(again.index["frame"], self.dataset.index["frame"]))

    def test_provenance_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            again = self._write(tmp)
        for key in ("episode", "seed", "frame", "success"):
            self.assertTrue(np.array_equal(again.index[key], self.dataset.index[key]), key)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_batch_tensors_match_the_stored_arrays(self):
        """The widening of the box table is exact, and nothing else is retyped."""
        batch = self.dataset.frames.torch_batch(np.arange(4))
        for key in GRAPH_KEYS:
            stored = self.dataset.frames.fields[key][:4]
            got = batch[key].numpy()
            self.assertTrue(np.array_equal(got.astype(stored.dtype), stored), key)
            expect = np.float32 if key == "graph_node_bbox" else stored.dtype
            self.assertEqual(got.dtype, expect, key)

    def test_a_stale_shard_cannot_be_concatenated_in(self):
        """Rewriting a cache in place must not inherit the previous run."""
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, shard_size=8)
            again = self._write(tmp, shard_size=8)
            self.assertEqual(len(again), len(self.dataset))

    def test_a_truncated_cache_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp)
            path = os.path.join(tmp, "meta.json")
            with open(path) as handle:
                meta = json.load(handle)
            meta["frames"] = int(meta["frames"]) + 5
            with open(path, "w") as handle:
                json.dump(meta, handle)
            with self.assertRaises(ValueError):
                GraphDataset.load(tmp)

    def test_missing_cache_names_the_stage_to_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                GraphDataset.load(tmp)


class Contract(unittest.TestCase):
    def test_a_wrong_dtype_is_refused_on_the_way_in(self):
        frames = make_frames(2, seed=1)
        frame = frames.frame(0)
        frame["graph_edge_rel"] = frame["graph_edge_rel"].astype(np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            writer = ShardWriter(tmp)
            with self.assertRaises(TypeError):
                writer.add(frame, {"episode": 0, "seed": 0, "frame": 0, "success": True})

    def test_a_missing_key_is_refused(self):
        frames = make_frames(2, seed=1)
        frame = frames.frame(0)
        frame.pop("graph_edge_temp")
        with tempfile.TemporaryDirectory() as tmp:
            writer = ShardWriter(tmp)
            with self.assertRaises(KeyError):
                writer.add(frame, {"episode": 0, "seed": 0, "frame": 0, "success": True})

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_dtypes_mirror_the_online_builder(self):
        """The cache stores what replay stores.

        ``graph_obs`` owns the dtypes the trainer writes into the buffer; this
        table is a copy so a cache can be read without a simulator install, and
        a copy that drifts is a cache measuring different tensors.
        """
        from scenegraph.adapters.graph_obs import _DTYPES

        self.assertEqual(
            {key: np.dtype(value) for key, value in _DTYPES.items()}, FIELD_DTYPES
        )

    def test_keys_mirror_the_packer(self):
        self.assertEqual(tuple(FIELD_DTYPES), tuple(GRAPH_KEYS))


@unittest.skipIf(torch is None, "torch is not installed")
class Residency(unittest.TestCase):
    """Holding the pool on the device is a transport change, not a numeric one."""

    def setUp(self):
        self.frames = make_frames(24, seed=9)
        self.resident = make_frames(24, seed=9).to_device("cpu")

    def test_a_resident_batch_is_identical_to_a_host_batch(self):
        for indices in ([0], [5, 1, 5, 23], list(range(24))):
            host = self.frames.torch_batch(indices)
            gathered = self.resident.torch_batch(indices)
            for key in GRAPH_KEYS:
                self.assertEqual(gathered[key].dtype, host[key].dtype, key)
                self.assertTrue(torch.equal(gathered[key], host[key]), key)

    def test_tensor_indices_gather_the_same_rows(self):
        want = self.frames.torch_batch([3, 8, 2])
        got = self.resident.torch_batch(torch.tensor([3, 8, 2]))
        for key in GRAPH_KEYS:
            self.assertTrue(torch.equal(got[key], want[key]), key)

    def test_a_host_table_still_accepts_tensor_indices(self):
        want = self.frames.torch_batch([1, 4])
        got = self.frames.torch_batch(torch.tensor([1, 4]))
        for key in GRAPH_KEYS:
            self.assertTrue(torch.equal(got[key], want[key]), key)

    def test_derived_tables_do_not_inherit_the_cache(self):
        """``select`` and ``concat`` build different rows; a stale gather would
        silently hand back the wrong graphs."""
        self.assertIsNone(self.resident.select([0, 1]).resident_device)
        self.assertIsNone(self.resident.concat(self.frames).resident_device)

    def test_the_budget_is_measured_not_guessed(self):
        from ..train import resolve_residency

        cpu = torch.device("cpu")
        on, why = resolve_residency("auto", cpu, self.frames, 4.0)
        self.assertFalse(on)
        self.assertIn("cpu", why)
        on, _ = resolve_residency(True, cpu, self.frames, 4.0)
        self.assertTrue(on)
        on, why = resolve_residency("auto", torch.device("cuda"), self.frames, 1e-9)
        self.assertFalse(on)
        self.assertIn("budget", why)
        self.assertGreater(self.frames.device_bytes(), 0)


class Selection(unittest.TestCase):
    def setUp(self):
        self.frames = make_frames(12, seed=4)

    def test_select_and_concat_preserve_rows(self):
        picked = self.frames.select([3, 1, 3])
        self.assertEqual(len(picked), 3)
        self.assertTrue(
            np.array_equal(
                picked.fields["graph_edge_rel"][0], self.frames.fields["graph_edge_rel"][3]
            )
        )
        joined = self.frames.concat(picked)
        self.assertEqual(len(joined), 15)
        self.assertTrue(
            np.array_equal(
                joined.fields["graph_node_ent"][12], self.frames.fields["graph_node_ent"][3]
            )
        )

    def test_fingerprint_follows_content(self):
        other = GraphFrames({key: np.array(v, copy=True) for key, v in self.frames.fields.items()})
        self.assertEqual(other.fingerprint(), self.frames.fingerprint())
        other.fields["graph_node_centroid"][0, 0, 0] += np.float32(0.01)
        self.assertNotEqual(other.fingerprint(), self.frames.fingerprint())

    def test_frame_returns_a_copy(self):
        """Pair construction edits these in place; the table must not follow."""
        original = int(self.frames.fields["graph_edge_abs"][0, 0])
        frame = self.frames.frame(0)
        frame["graph_edge_abs"][0] = original + 1
        self.assertEqual(int(self.frames.fields["graph_edge_abs"][0, 0]), original)


if __name__ == "__main__":
    unittest.main()
