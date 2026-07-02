# Copyright 2025-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the NNX branches of load_state_if_possible."""

import os
import tempfile
import unittest
from unittest import mock

from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common import checkpointing
from maxtext.common import train_state_nnx
import optax
import orbax.checkpoint as ocp


class _Model(nnx.Module):
  """Tiny single-linear NNX model for restore tests."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 1, rngs=rngs)


def _abstract_nnx_state():
  """Build an nnx.State from a TrainStateNNX — same shape that pre_train passes in."""
  model = _Model(rngs=nnx.Rngs(0))
  optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
  return nnx.state(train_state_nnx.TrainStateNNX(model, optimizer))


class TestLoadStateIfPossibleNNX(unittest.TestCase):
  """Cover the NNX branches in load_state_if_possible."""

  def test_emergency_linen_restore_converts_back_to_nnx(self):
    kernel_abstract = jax.ShapeDtypeStruct((2, 1), jnp.float32)
    step_abstract = jax.ShapeDtypeStruct((), jnp.uint32)
    abstract_nnx_pure = {
        "model": {
            "linear": {"kernel": kernel_abstract},
            "dropout": {"rngs": {"params": {"count": step_abstract}}},
        },
        "optimizer": {"step": step_abstract},
    }
    abstract_nnx_state = mock.Mock()
    abstract_nnx_state.to_pure_dict.return_value = abstract_nnx_pure
    restored_linen = {
        "params": {"params": {"linear": {"kernel": jnp.ones((2, 1))}}},
        "step": jnp.asarray(7, dtype=jnp.int32),
    }
    checkpoint_manager = mock.Mock()
    checkpoint_manager.restore.return_value = mock.Mock(state=restored_linen)

    restored = checkpointing._restore_emergency_linen_checkpoint_into_nnx(  # pylint: disable=protected-access
        checkpoint_manager,
        14,
        abstract_nnx_state,
        lambda leaf: ocp.type_handlers.ArrayRestoreArgs(
            global_shape=leaf.shape,
            dtype=leaf.dtype,
        ),
    )

    checkpoint_manager.restore.assert_called_once()
    restore_args = checkpoint_manager.restore.call_args.kwargs["args"].state
    self.assertEqual(
        set(restore_args.item.keys()),
        {"params", "step"},
    )
    self.assertNotIn("model", restore_args.item)
    self.assertNotIn("optimizer", restore_args.item)
    self.assertTrue(restore_args.partial_restore)
    self.assertIn("model", restored)
    self.assertIn("optimizer", restored)
    self.assertNotIn("params", restored)
    self.assertNotIn("opt_state", restored)
    self.assertTrue(bool(jnp.array_equal(restored["model"]["linear"]["kernel"], jnp.ones((2, 1)))))
    self.assertEqual(restored["optimizer"]["step"].dtype, jnp.uint32)
    self.assertEqual(int(restored["optimizer"]["step"]), 7)
    self.assertEqual(
        restored["model"]["dropout"]["rngs"]["params"]["count"].shape,
        (),
    )

  def test_load_parameters_from_path_splits_nnx_state_for_param_view(self):
    """When abstract_unboxed_pre_state is an nnx.State, the function must call
    nnx.split(model, nnx.Param, ...) to get the params and forward them to load_params_from_path."""
    abstract = _abstract_nnx_state()
    sentinel_restored = {"linear": {"kernel": jnp.ones((2, 1)), "bias": jnp.zeros((1,))}}

    with mock.patch.object(checkpointing, "load_params_from_path", return_value=sentinel_restored) as m:
      full, params = checkpointing.load_state_if_possible(
          checkpoint_manager=None,
          data_iterator=None,
          load_parameters_from_path="gs://does-not-exist/params",
          load_full_state_from_path="",
          checkpoint_storage_concurrent_gb=8,
          abstract_unboxed_pre_state=abstract,
      )

    self.assertIsNone(full)
    self.assertIs(params, sentinel_restored)
    m.assert_called_once()
    forwarded_params = m.call_args[0][1]  # second positional arg = abstract_unboxed_params
    # The forwarded params come from nnx.split(..., nnx.Param, ...) — same key shape as the model.
    leaves = jax.tree.leaves(forwarded_params)
    self.assertEqual(len(leaves), 2)  # linear.kernel + linear.bias

  def test_load_parameters_from_path_uses_state_params_for_linen(self):
    """For Linen TrainState, the function must use state.params (not nnx.split)."""
    fake_state = mock.Mock(spec=["params"])
    fake_state.params = {"layer": {"kernel": jnp.ones((2, 2))}}
    sentinel = object()

    with mock.patch.object(checkpointing, "load_params_from_path", return_value=sentinel) as m:
      full, params = checkpointing.load_state_if_possible(
          checkpoint_manager=None,
          data_iterator=None,
          load_parameters_from_path="gs://does-not-exist/params",
          load_full_state_from_path="",
          checkpoint_storage_concurrent_gb=8,
          abstract_unboxed_pre_state=fake_state,
      )

    self.assertIsNone(full)
    self.assertIs(params, sentinel)
    forwarded_params = m.call_args[0][1]
    self.assertIs(forwarded_params, fake_state.params)

  def test_no_paths_returns_none_none(self):
    """Sanity: with no checkpoint manager and no load paths, the function returns (None, None)."""
    full, params = checkpointing.load_state_if_possible(
        checkpoint_manager=None,
        data_iterator=None,
        load_parameters_from_path="",
        load_full_state_from_path="",
        checkpoint_storage_concurrent_gb=8,
        abstract_unboxed_pre_state=_abstract_nnx_state(),
    )
    self.assertIsNone(full)
    self.assertIsNone(params)


class TestLoadParamsIntoNNX(unittest.TestCase):
  """Weight-only load (load_parameters_path) of a Linen-layout checkpoint into NNX."""

  def test_linen_layout_params_restore_into_nnx_state(self):
    """load_params_from_path reshapes an on-disk Linen-layout checkpoint into the NNX params state."""
    model = _Model(rngs=nnx.Rngs(0))
    _, params_abstract, _ = nnx.split(model, nnx.Param, ...)
    weights = {
        "linear": {
            "kernel": jnp.arange(2, dtype=jnp.float32).reshape(2, 1),
            "bias": jnp.array([5.0]),
        }
    }

    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      path = os.path.join(d, "ckpt")
      # On-disk Linen layout: params/params/<weights> plus an unrelated `step`.
      ocp.PyTreeCheckpointer(use_ocdbt=True, use_zarr3=True).save(
          epath.Path(path),
          {"params": {"params": weights}, "step": jnp.array(3)},
          force=True,
      )
      restored = checkpointing.load_params_from_path(path, params_abstract, 8)

    self.assertIsInstance(restored, nnx.State)
    pure = restored.to_pure_dict()
    self.assertTrue(jnp.array_equal(pure["linear"]["kernel"], weights["linear"]["kernel"]))
    self.assertTrue(jnp.array_equal(pure["linear"]["bias"], weights["linear"]["bias"]))


class TestExtractRngState(unittest.TestCase):
  """train_state_nnx.extract_rng_state keeps only the NNX-only rngs/dropout subtrees."""

  def test_keeps_only_rng_and_dropout(self):
    tree = {
        "linear": {"kernel": jnp.ones((2, 1)), "bias": jnp.zeros((1,))},
        "rngs": {"params": {"key": jnp.asarray(1), "count": jnp.asarray(2)}},
        "block": {"dropout": {"count": jnp.asarray(3)}, "kernel": jnp.ones((2, 2))},
    }
    aux = train_state_nnx.extract_rng_state(tree)
    self.assertEqual(set(aux.keys()), {"rngs", "block"})
    self.assertEqual(set(aux["block"].keys()), {"dropout"})  # kernel dropped
    self.assertNotIn("linear", aux)

  def test_empty_when_no_rng_state(self):
    self.assertEqual(train_state_nnx.extract_rng_state({"linear": {"kernel": jnp.ones((2, 1))}}), {})

  def test_extract_is_complement_of_strip(self):
    """Every leaf lands in exactly one of extract / strip — nothing dropped or duplicated."""
    tree = {
        "linear": {"kernel": jnp.ones((2, 1))},
        "rngs": {"params": {"count": jnp.asarray(2)}},
    }
    stripped = train_state_nnx._strip_rng_state(tree)  # pylint: disable=protected-access
    extracted = train_state_nnx.extract_rng_state(tree)
    self.assertEqual(set(stripped.keys()), {"linear"})
    self.assertEqual(set(extracted.keys()), {"rngs"})


class TestDeepMerge(unittest.TestCase):
  """checkpointing._deep_merge overlays leaves onto a base dict."""

  def test_overlay_wins_and_bases_survive(self):
    base = {"model": {"linear": {"kernel": 1}, "rngs": {"count": 0}}}
    overlay = {"model": {"rngs": {"count": 99}}}
    merged = checkpointing._deep_merge(base, overlay)  # pylint: disable=protected-access
    self.assertEqual(merged["model"]["linear"]["kernel"], 1)  # untouched
    self.assertEqual(merged["model"]["rngs"]["count"], 99)  # overlaid

  def test_does_not_mutate_inputs(self):
    base = {"a": {"b": 1}}
    checkpointing._deep_merge(base, {"a": {"c": 2}})  # pylint: disable=protected-access
    self.assertEqual(base, {"a": {"b": 1}})

  def test_non_dict_leaves_prefer_overlay(self):
    """Leaf vs leaf: overlay wins, but a None overlay keeps the base."""
    self.assertEqual(checkpointing._deep_merge(1, 2), 2)  # pylint: disable=protected-access
    self.assertEqual(checkpointing._deep_merge(1, None), 1)  # pylint: disable=protected-access

  def test_adds_keys_absent_from_base(self):
    merged = checkpointing._deep_merge({"a": 1}, {"b": 2})  # pylint: disable=protected-access
    self.assertEqual(merged, {"a": 1, "b": 2})


class TestRngStateAuxPersistence(unittest.TestCase):
  """rngs/dropout persisted as a separate `nnx_aux` item, restored on resume."""

  def _abstract_with_dropout(self):
    abstract = mock.Mock()
    abstract.to_pure_dict.return_value = {
        "model": {
            "linear": {"kernel": jax.ShapeDtypeStruct((2, 1), jnp.float32)},
            "dropout": {"count": jax.ShapeDtypeStruct((), jnp.uint32)},
        },
        "optimizer": {"step": jax.ShapeDtypeStruct((), jnp.uint32)},
    }
    return abstract

  def _write_linen_items(self, step_dir):
    ocp.PyTreeCheckpointer(use_ocdbt=True, use_zarr3=True).save(
        epath.Path(step_dir) / "items",
        {"params": {"params": {"linear": {"kernel": jnp.ones((2, 1))}}}, "step": jnp.asarray(3, jnp.int32)},
        force=True,
    )

  def test_restores_saved_rng_state_instead_of_default(self):
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      step_dir = os.path.join(d, "5")
      self._write_linen_items(step_dir)
      # Persisted rng/dropout state: count=42 (a resumed stream, not the base 0).
      ocp.PyTreeCheckpointer(use_ocdbt=True, use_zarr3=True).save(
          epath.Path(step_dir) / "nnx_aux",
          {"dropout": {"count": jnp.asarray(42, jnp.uint32)}},
          force=True,
      )
      restored = checkpointing._load_linen_checkpoint_into_nnx(  # pylint: disable=protected-access
          os.path.join(step_dir, "items"), self._abstract_with_dropout(), 8, True, True
      )
    self.assertEqual(int(restored["model"]["dropout"]["count"]), 42)
    self.assertTrue(jnp.array_equal(restored["model"]["linear"]["kernel"], jnp.ones((2, 1))))

  def test_falls_back_to_default_when_no_aux_dir(self):
    """A Linen-trained checkpoint (no nnx_aux) still loads; rng/dropout gets the base default."""
    with tempfile.TemporaryDirectory() as d:  # pylint: disable=consider-using-with
      step_dir = os.path.join(d, "5")
      self._write_linen_items(step_dir)  # no nnx_aux written
      restored = checkpointing._load_linen_checkpoint_into_nnx(  # pylint: disable=protected-access
          os.path.join(step_dir, "items"), self._abstract_with_dropout(), 8, True, True
      )
    self.assertEqual(int(restored["model"]["dropout"]["count"]), 0)  # _default_for_sds

  def test_save_checkpoint_adds_nnx_aux_item_when_present(self):
    manager = mock.Mock()
    manager.save.return_value = True
    config = mock.Mock(
        enable_checkpointing=False,
        dataset_type="tfds",
        lora=None,
        checkpoint_storage_target_data_file_size_bytes=1,
    )
    aux = {"dropout": {"count": jnp.asarray(7, jnp.uint32)}}
    checkpointing.save_checkpoint(manager, 5, {"params": {}}, config, None, False, aux)
    composite = manager.save.call_args.kwargs["args"]
    self.assertIn("nnx_aux", composite.keys())

  def test_save_checkpoint_omits_nnx_aux_when_empty(self):
    manager = mock.Mock()
    manager.save.return_value = True
    config = mock.Mock(
        enable_checkpointing=False,
        dataset_type="tfds",
        lora=None,
        checkpoint_storage_target_data_file_size_bytes=1,
    )
    checkpointing.save_checkpoint(manager, 5, {"params": {}}, config, None, False, {})
    composite = manager.save.call_args.kwargs["args"]
    self.assertNotIn("nnx_aux", composite.keys())


if __name__ == "__main__":
  unittest.main()
