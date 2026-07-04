# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tensor handling utility functions for checkpoint conversion."""

from functools import partial
from typing import Any, Callable, List
import jax
import jax.numpy as np
import numpy as onp


def apply_hook_fns(weight, target_shape, hook_fns):
  """Apply hook functions, essential for to_maxtext and to_huggingface"""
  # If hook is unsepecified, use identity
  if hook_fns is None:
    return weight
  if not isinstance(hook_fns, list):
    hook_fns = [hook_fns]
  # Apply a list of hooks, be careful of order
  for hook_fn in hook_fns:
    weight = hook_fn(weight, target_shape)
  return weight


def _binary_chunked_stack(tensors: List[np.ndarray], axis: int) -> np.ndarray:
  """Stacks JAX arrays along axis by binary division to limit memory usage from JAX compiler."""
  if not tensors:
    raise ValueError("Cannot stack empty list of tensors.")
  if len(tensors) == 1:
    return np.expand_dims(tensors[0], axis=axis)
  if len(tensors) == 2:
    return np.stack(tensors, axis=axis)

  mid = len(tensors) // 2
  left = _binary_chunked_stack(tensors[:mid], axis=axis)
  right = _binary_chunked_stack(tensors[mid:], axis=axis)
  return np.concatenate([left, right], axis=axis)


def get_safe_local_array(array):
  """Safely extracts a SingleDevice sharded tensor into a purely local CPU Numpy Array 
  (returning Zeros on nodes that do not own the payload) to enable math processing."""
  is_np = not hasattr(array, "addressable_shards")
  host_id = jax.process_index()

  if not is_np and len(array.addressable_shards) > 0:
    is_owner = True
  elif not is_np and hasattr(array, "sharding") and hasattr(array.sharding, "device_set"):
    owner_device = list(array.sharding.device_set)[0]
    is_owner = (owner_device.process_index == host_id)
  else:
    is_owner = is_np

  if is_owner:
    return onp.asarray(array), is_owner
  return onp.zeros(array.shape, dtype=array.dtype), is_owner


def reshard_to_target(array, sharding, hook_fns=None, target_shape=None, is_owner=None):
  """Reshards a local CPU array cross-host explicitly to the target sharding."""
  if hasattr(array, "sharding") and isinstance(array.sharding, jax.sharding.NamedSharding):
    # For fully formed NamedSharding inputs, we process the hooks normally here.
    if hook_fns is not None and target_shape is not None:
      array = apply_hook_fns(array, target_shape, hook_fns)
    _reshard = jax.jit(lambda x: x, out_shardings=sharding)
    return _reshard(array)

  # Secure purely CPU memory
  if is_owner is None:
    local_arr, is_owner = get_safe_local_array(array)
  else:
    local_arr = onp.asarray(array)

  if hook_fns is not None and target_shape is not None:
    local_arr = apply_hook_fns(local_arr, target_shape, hook_fns)
    local_arr = onp.asarray(local_arr)
  
  import math
  bytes_per_slice = math.prod(local_arr.shape[1:]) * local_arr.itemsize if len(local_arr.shape) > 1 else local_arr.itemsize
  max_bytes_per_chunk = 128 * 1024 * 1024  # 128 MB max buffer per transmission

  if bytes_per_slice >= max_bytes_per_chunk:
    chunk_size = 1
  else:
    chunk_size = max(1, max_bytes_per_chunk // bytes_per_slice)
    
  total_len = local_arr.shape[0]

  # Broadcast over network sequentially in mathematically stable chunks 
  # to rigorously prevent Host OOM Resource Exhausted Limits on the FSDP Grid.
  if total_len <= chunk_size or total_len == 0:
    global_replicated = jax.experimental.multihost_utils.broadcast_one_to_all(
        local_arr, is_source=is_owner
    )
    cpu_replicated = onp.asarray(global_replicated)
    del global_replicated
  else:
    replicated_chunks = []
    for i in range(0, total_len, chunk_size):
      chunk_arr_cpu = local_arr[i : i + chunk_size]
      global_replicated_chunk = jax.experimental.multihost_utils.broadcast_one_to_all(
          chunk_arr_cpu, is_source=is_owner
      )
      replicated_chunks.append(onp.asarray(global_replicated_chunk))
      del global_replicated_chunk

    cpu_replicated = onp.concatenate(replicated_chunks, axis=0)
    del replicated_chunks

  del local_arr

  # jax.make_array_from_callback directly pulls the memory slices mathematically matched 
  # by the FSDP Sharding layout and instantiates them purely into device memory! 
  res = jax.make_array_from_callback(
      cpu_replicated.shape,
      sharding,
      lambda index: cpu_replicated[index]
  )
  return res


def _build_multi_axis_stacked_tensor(
    hf_source_keys: List[List[str]],
    tensor_getter_fn: Callable[[str], np.ndarray],
    hook_fns: Any,
    target_leaf: Any,
    config,
) -> np.ndarray:
  """Builds a MaxText tensor by stacking HF weights along two axes directly in place on device."""
  if hasattr(target_leaf, "sharding"):
    target_shape = target_leaf.shape
    target_sharding = target_leaf.sharding
    target_dtype = target_leaf.dtype
  else:
    target_shape = target_leaf
    target_sharding = None
    target_dtype = target_leaf.dtype if hasattr(target_leaf, "dtype") else np.float32

  all_expert_tensors = []
  for layer_keys_for_expert in hf_source_keys:
    layer_tensors_for_expert = []
    for hf_key_single in layer_keys_for_expert:
      # Retrieve array natively on TPU (SingleDeviceSharding on Host 0)
      layer_tensors_for_expert.append(tensor_getter_fn(hf_key_single))

    expert_tensor = _binary_chunked_stack(layer_tensors_for_expert, axis=0)
    all_expert_tensors.append(expert_tensor)

  stacked_array = _binary_chunked_stack(all_expert_tensors, axis=0).astype(target_dtype)
  
  if target_sharding is not None:
    stacked_array = reshard_to_target(stacked_array, target_sharding, hook_fns=hook_fns, target_shape=target_shape)
  return stacked_array


def _build_single_axis_stacked_tensor(
    hf_source_keys: List[str],
    tensor_getter_fn: Callable[[str], np.ndarray],
    hook_fns: Any,
    target_leaf: Any,
    config,
) -> np.ndarray:
  """Builds a MaxText tensor by stacking HF weights along a single axis directly in place on device."""
  if hasattr(target_leaf, "sharding"):
    target_shape = target_leaf.shape
    target_sharding = target_leaf.sharding
    target_dtype = target_leaf.dtype
  else:
    target_shape = target_leaf
    target_sharding = None
    target_dtype = target_leaf.dtype if hasattr(target_leaf, "dtype") else np.float32

  axis_to_stack = config.param_scan_axis if config.scan_layers else 0

  tensors_to_stack = []
  for hf_key_single in hf_source_keys:
    tensors_to_stack.append(tensor_getter_fn(hf_key_single))

  stacked_array = _binary_chunked_stack(tensors_to_stack, axis=axis_to_stack).astype(target_dtype)
  
  if target_sharding is not None:
    stacked_array = reshard_to_target(stacked_array, target_sharding, hook_fns=hook_fns, target_shape=target_shape)
  return stacked_array


def get_hf_loading_function(hf_source_keys_or_key, tensor_getter, hook_fn, mt_target_leaf, config):
  """Determine the loading function for HF keys."""
  if not isinstance(hf_source_keys_or_key, list):
    # Case 1: Single hf key (str)
    def _loader(getter, key, leaf, hook):
      if hasattr(leaf, "sharding"):
        return reshard_to_target(getter(key), leaf.sharding, hook_fns=hook, target_shape=leaf.shape)
      else:
        # Fallback Local Path
        local_arr, _ = get_safe_local_array(getter(key))
        return apply_hook_fns(local_arr, leaf.shape, hook)

    return partial(
        _loader,
        tensor_getter,
        hf_source_keys_or_key,
        mt_target_leaf,
        hook_fn,
    )
  # Stacked mapping
  elif not isinstance(hf_source_keys_or_key[0], list):
    # Case 2 or 3: Single-Axis Stacked hf keys (un-nested list)
    return partial(
        _build_single_axis_stacked_tensor,
        hf_source_keys_or_key,
        tensor_getter,
        hook_fn,
        mt_target_leaf,
        config,
    )
  else:
    # isinstance(hf_source_keys_or_key[0], list)
    # Case 4: Multi-Axis Stacked hf keys (nested list)
    return partial(
        _build_multi_axis_stacked_tensor,
        hf_source_keys_or_key,
        tensor_getter,
        hook_fn,
        mt_target_leaf,
        config,
    )
