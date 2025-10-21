/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuco/detail/bitwise_compare.cuh>
#include <cuco/detail/utils.cuh>

#include <cub/block/block_reduce.cuh>
#include <cuda/std/atomic>

#include <cooperative_groups.h>

namespace cuco::detail {
namespace cg = cooperative_groups;

CUCO_SUPPRESS_KERNEL_WARNINGS
/**
 * @brief Initializes each slot in the flat `slots` storage to contain `k` and `v`.
 *
 * Each space in `slots` that can hold a key value pair is initialized to a
 * `pair_atomic_type` containing the key `k` and the value `v`.
 *
 * @tparam atomic_key_type Type of the `Key` atomic container
 * @tparam atomic_mapped_type Type of the `Value` atomic container
 * @tparam Key key type
 * @tparam Value value type
 * @tparam pair_atomic_type key/value pair type
 *
 * @param slots Pointer to flat storage for the map's key/value pairs
 * @param k Key to which all keys in `slots` are initialized
 * @param v Value to which all values in `slots` are initialized
 * @param size Size of the storage pointed to by `slots`
 */
template <std::size_t block_size,
          typename atomic_key_type,
          typename atomic_mapped_type,
          typename Key,
          typename Value,
          typename pair_atomic_type>
CUCO_KERNEL void initialize(pair_atomic_type* const slots, Key k, Value v, int64_t size)
{
  int64_t const loop_stride = gridDim.x * block_size;
  int64_t idx               = block_size * blockIdx.x + threadIdx.x;
  while (idx < size) {
    new (&slots[idx].first) atomic_key_type{k};
    new (&slots[idx].second) atomic_mapped_type{v};
    idx += loop_stride;
  }
}

/**
 * @brief Inserts all key/value pairs in the range `[first, last)`.
 *
 * If multiple keys in `[first, last)` compare equal, it is unspecified which
 * element is inserted.
 *
 * @tparam block_size
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `value_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 *
 * @param first Beginning of the sequence of key/value pairs
 * @param n Number of the key/value pairs to insert
 * @param num_successes The number of successfully inserted key/value pairs
 * @param view Mutable device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function used to compare two keys for equality
 */
template <std::size_t block_size,
          typename InputIt,
          typename viewT,
          typename Hash,
          typename KeyEqual>
CUCO_KERNEL void insert(
  InputIt first, int64_t n, unsigned long long* num_successes, viewT view, Hash hash, KeyEqual key_equal)
{
  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_successes = 0;

  int64_t const loop_stride = gridDim.x * block_size;
  int64_t idx               = block_size * blockIdx.x + threadIdx.x;

  while (idx < n) {
    typename viewT::value_type const insert_pair{*(first + idx)};
    if (view.insert(insert_pair, hash, key_equal)) { thread_num_successes++; }
    idx += loop_stride;
  }

  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
  std::size_t block_num_successes = BlockReduce(temp_storage).Sum(thread_num_successes);
  if (threadIdx.x == 0) { atomicAdd(num_successes, block_num_successes); }
}

/**
 * @brief Inserts all key/value pairs in the range `[first, last)`.
 *
 * If multiple keys in `[first, last)` compare equal, it is unspecified which
 * element is inserted. Uses the CUDA Cooperative Groups API to leverage groups
 * of multiple threads to perform each key/value insertion. This provides a
 * significant boost in throughput compared to the non Cooperative Group
 * `insert` at moderate to high load factors.
 *
 * @tparam block_size
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform
 * inserts
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `value_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 *
 * @param first Beginning of the sequence of key/value pairs
 * @param n Number of the key/value pairs to insert
 * @param num_successes The number of successfully inserted key/value pairs
 * @param view Mutable device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function used to compare two keys for equality
 */
template <std::size_t block_size,
          uint32_t tile_size,
          typename InputIt,
          typename viewT,
          typename Hash,
          typename KeyEqual>
CUCO_KERNEL void insert(
  InputIt first, int64_t n, unsigned long long* num_successes, viewT view, Hash hash, KeyEqual key_equal)
{
  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_successes = 0;

  auto tile = cg::tiled_partition<tile_size, cg::thread_block>(cg::this_thread_block());
  int64_t const loop_stride = gridDim.x * block_size / tile_size;
  int64_t idx               = (block_size * blockIdx.x + threadIdx.x) / tile_size;

  while (idx < n) {
    // force conversion to value_type
    typename viewT::value_type const insert_pair{*(first + idx)};
    if (view.insert(tile, insert_pair, hash, key_equal) && tile.thread_rank() == 0) {
      thread_num_successes++;
    }
    idx += loop_stride;
  }

  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
  std::size_t block_num_successes = BlockReduce(temp_storage).Sum(thread_num_successes);
  if (threadIdx.x == 0) { atomicAdd(num_successes, block_num_successes); }
}

/**
 * @brief Erases the key/value pairs corresponding to all keys in the range `[first, last)`.
 *
 * If the key `*(first + i)` exists in the map, its slot is erased and made available for future
 * insertions.
 * Else, no effect.
 *
 * @tparam block_size The size of the thread block
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 *
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param num_successes The number of successfully erased key/value pairs
 * @param view Device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template <std::size_t block_size,
          typename InputIt,
          typename viewT,
          typename Hash,
          typename KeyEqual>
CUCO_KERNEL void erase(
  InputIt first, int64_t n, unsigned long long* num_successes, viewT view, Hash hash, KeyEqual key_equal)
{
  using BlockReduce = cub::BlockReduce<std::size_t, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_successes = 0;

  const int64_t loop_stride = gridDim.x * block_size;
  int64_t idx               = block_size * blockIdx.x + threadIdx.x;

  while (idx < n) {
    if (view.erase(*(first + idx), hash, key_equal)) { thread_num_successes++; }
    idx += loop_stride;
  }

  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
  std::size_t block_num_successes = BlockReduce(temp_storage).Sum(thread_num_successes);
  if (threadIdx.x == 0) {
    atomicAdd(num_successes, block_num_successes);
  }
}

/**
 * @brief Erases the key/value pairs corresponding to all keys in the range `[first, last)`.
 *
 * If the key `*(first + i)` exists in the map, its slot is erased and made available for future
 * insertions.
 * Else, no effect.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform erase
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 *
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param num_successes The number of successfully erased key/value pairs
 * @param view Device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template <std::size_t block_size,
          uint32_t tile_size,
          typename InputIt,
          typename viewT,
          typename Hash,
          typename KeyEqual>
CUCO_KERNEL void erase(
  InputIt first, int64_t n, unsigned long long* num_successes, viewT view, Hash hash, KeyEqual key_equal)
{
  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_successes = 0;

  auto tile = cg::tiled_partition<tile_size, cg::thread_block>(cg::this_thread_block());
  int64_t const loop_stride = gridDim.x * block_size / tile_size;
  int64_t idx               = (block_size * blockIdx.x + threadIdx.x) / tile_size;

  while (idx < n) {
    if (view.erase(tile, *(first + idx), hash, key_equal) and tile.thread_rank() == 0) {
      thread_num_successes++;
    }
    idx += loop_stride;
  }

  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
  std::size_t block_num_successes = BlockReduce(temp_storage).Sum(thread_num_successes);
  if (threadIdx.x == 0) {
    atomicAdd(num_successes, block_num_successes);
  }
}

/**
 * @brief Finds the values corresponding to all keys in the range `[first, last)`.
 *
 * If the key `*(first + i)` exists in the map, copies its associated value to `(output_begin + i)`.
 * Else, copies the empty value sentinel.
 * @tparam block_size The size of the thread block
 * @tparam Value The type of the mapped value for the map
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 *
 * @param first Beginning of the sequence of keys
 * @param n Number of keys to query
 * @param output_begin Beginning of the sequence of values retrieved for each key
 * @param view Device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template <std::size_t block_size,
          typename Value,
          typename InputIt,
          typename OutputIt,
          typename viewT,
          typename Hash,
          typename KeyEqual>
CUCO_KERNEL void find(
  InputIt first, int64_t n, OutputIt output_begin, viewT view, Hash hash, KeyEqual key_equal)
{
  int64_t const loop_stride = gridDim.x * block_size;
  int64_t idx               = block_size * blockIdx.x + threadIdx.x;
  __shared__ Value writeBuffer[block_size];

  while (idx < n) {
    auto key   = *(first + idx);
    auto found = view.find(key, hash, key_equal);

    /*
     * The ld.relaxed.gpu instruction used in view.find causes L1 to
     * flush more frequently, causing increased sector stores from L2 to global memory.
     * By writing results to shared memory and then synchronizing before writing back
     * to global, we no longer rely on L1, preventing the increase in sector stores from
     * L2 to global and improving performance.
     */
    writeBuffer[threadIdx.x] = found == view.end()
                                 ? view.empty_value_sentinel()
                                 : found->second.load(cuda::std::memory_order_relaxed);
    __syncthreads();
    *(output_begin + idx) = writeBuffer[threadIdx.x];
    idx += loop_stride;
  }
}
 
template <std::size_t block_size,
          typename Value,
          typename InputIt,
          typename OutputIt,
          typename MaskOutIt,
          typename viewT,
          typename Hash,
          typename KeyEqual>
CUCO_KERNEL void find_and_mask(
  InputIt first, int64_t n, OutputIt output_begin, MaskOutIt mask_begin, viewT view, Hash hash, KeyEqual key_equal)
{
  int64_t const loop_stride = gridDim.x * block_size;
  int64_t idx               = block_size * blockIdx.x + threadIdx.x;
  __shared__ Value writeBuffer[block_size];
  __shared__ bool maskBuffer[block_size];
 
  while (idx < n) {
    auto key   = *(first + idx);
    auto found = view.find(key, hash, key_equal);
 
    /*
     * The ld.relaxed.gpu instruction used in view.find causes L1 to
     * flush more frequently, causing increased sector stores from L2 to global memory.
     * By writing results to shared memory and then synchronizing before writing back
     * to global, we no longer rely on L1, preventing the increase in sector stores from
     * L2 to global and improving performance.
     */
    writeBuffer[threadIdx.x] = found == view.end()
                                 ? view.empty_value_sentinel()
                                 : found->second.load(cuda::std::memory_order_relaxed);
    maskBuffer[threadIdx.x] = found == view.end();
    __syncthreads();
    *(output_begin + idx) = writeBuffer[threadIdx.x];
    *(mask_begin + idx) = maskBuffer[threadIdx.x];
    idx += loop_stride;
  }
}
 
template <std::size_t block_size,
          uint32_t tile_size,
          typename Value,
          typename InputIt,
          typename OutputIt,
          typename MaskOutIt,
          typename viewT,
          typename Hash,
          typename KeyEqual>
CUCO_KERNEL void find_and_mask(
  InputIt first, int64_t n, OutputIt output_begin, MaskOutIt mask_begin, viewT view, Hash hash, KeyEqual key_equal)
{
  auto tile = cg::tiled_partition<tile_size, cg::thread_block>(cg::this_thread_block());
  int64_t const loop_stride = gridDim.x * block_size / tile_size;
  int64_t idx               = (block_size * blockIdx.x + threadIdx.x) / tile_size;
#pragma nv_diagnostic push
#pragma nv_diag_suppress static_var_with_dynamic_init
  // Get rid of a false-positive build warning with ARM
  __shared__ Value writeBuffer[block_size / tile_size];
  __shared__ bool maskBuffer[block_size / tile_size];
#pragma nv_diagnostic pop
 
  while (idx < n) {
    auto key   = *(first + idx);
    auto found = view.find(tile, key, hash, key_equal);
 
    /*
     * The ld.relaxed.gpu instruction used in view.find causes L1 to
     * flush more frequently, causing increased sector stores from L2 to global memory.
     * By writing results to shared memory and then synchronizing before writing back
     * to global, we no longer rely on L1, preventing the increase in sector stores from
     * L2 to global and improving performance.
     */
    if (tile.thread_rank() == 0) {
      writeBuffer[threadIdx.x / tile_size] =
        found == view.end() ? view.empty_value_sentinel()
                            : found->second.load(cuda::std::memory_order_relaxed);
      maskBuffer[threadIdx.x / tile_size] = found == view.end();
    }
    __syncthreads();
    if (tile.thread_rank() == 0) { 
      *(output_begin + idx) = writeBuffer[threadIdx.x / tile_size];
      *(mask_begin + idx) = maskBuffer[threadIdx.x / tile_size];
    }
    idx += loop_stride;
  }
}

/**
 * @brief Finds the values corresponding to all keys in the range `[first, last)`.
 *
 * If the key `*(first + i)` exists in the map, copies its associated value to `(output_begin + i)`.
 * Else, copies the empty value sentinel. Uses the CUDA Cooperative Groups API to leverage groups
 * of multiple threads to find each key. This provides a significant boost in throughput compared
 * to the non Cooperative Group `find` at moderate to high load factors.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform
 * inserts
 * @tparam Value The type of the mapped value for the map
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 *
 * @param first Beginning of the sequence of keys
 * @param n Number of keys to query
 * @param output_begin Beginning of the sequence of values retrieved for each key
 * @param view Device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template <std::size_t block_size,
          uint32_t tile_size,
          typename Value,
          typename InputIt,
          typename OutputIt,
          typename viewT,
          typename Hash,
          typename KeyEqual>
CUCO_KERNEL void find(
  InputIt first, int64_t n, OutputIt output_begin, viewT view, Hash hash, KeyEqual key_equal)
{
  auto tile = cg::tiled_partition<tile_size, cg::thread_block>(cg::this_thread_block());
  int64_t const loop_stride = gridDim.x * block_size / tile_size;
  int64_t idx               = (block_size * blockIdx.x + threadIdx.x) / tile_size;
#pragma nv_diagnostic push
#pragma nv_diag_suppress static_var_with_dynamic_init
  // Get rid of a false-positive build warning with ARM
  __shared__ Value writeBuffer[block_size / tile_size];
#pragma nv_diagnostic pop

  while (idx < n) {
    auto key   = *(first + idx);
    auto found = view.find(tile, key, hash, key_equal);

    /*
     * The ld.relaxed.gpu instruction used in view.find causes L1 to
     * flush more frequently, causing increased sector stores from L2 to global memory.
     * By writing results to shared memory and then synchronizing before writing back
     * to global, we no longer rely on L1, preventing the increase in sector stores from
     * L2 to global and improving performance.
     */
    if (tile.thread_rank() == 0) {
      writeBuffer[threadIdx.x / tile_size] =
        found == view.end() ? view.empty_value_sentinel()
                            : found->second.load(cuda::std::memory_order_relaxed);
    }
    __syncthreads();
    if (tile.thread_rank() == 0) { *(output_begin + idx) = writeBuffer[threadIdx.x / tile_size]; }
    idx += loop_stride;
  }
}

/**
 * @brief Indicates whether the keys in the range `[first, last)` are contained in the map.
 *
 * Writes a `bool` to `(output + i)` indicating if the key `*(first + i)` exists in the map.
 *
 * @tparam block_size The size of the thread block
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 *
 * @param first Beginning of the sequence of keys
 * @param n Number of keys to query
 * @param output_begin Beginning of the sequence of booleans for the presence of each key
 * @param view Device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template <std::size_t block_size,
          typename InputIt,
          typename OutputIt,
          typename viewT,
          typename Hash,
          typename KeyEqual>
CUCO_KERNEL void contains(
  InputIt first, int64_t n, OutputIt output_begin, viewT view, Hash hash, KeyEqual key_equal)
{
  int64_t const loop_stride = gridDim.x * block_size;
  int64_t idx               = block_size * blockIdx.x + threadIdx.x;
  __shared__ bool writeBuffer[block_size];

  while (idx < n) {
    auto key = *(first + idx);

    /*
     * The ld.relaxed.gpu instruction used in view.find causes L1 to
     * flush more frequently, causing increased sector stores from L2 to global memory.
     * By writing results to shared memory and then synchronizing before writing back
     * to global, we no longer rely on L1, preventing the increase in sector stores from
     * L2 to global and improving performance.
     */
    writeBuffer[threadIdx.x] = view.contains(key, hash, key_equal);
    __syncthreads();
    *(output_begin + idx) = writeBuffer[threadIdx.x];
    idx += loop_stride;
  }
}

/**
 * @brief Indicates whether the keys in the range `[first, last)` are contained in the map.
 *
 * Writes a `bool` to `(output + i)` indicating if the key `*(first + i)` exists in the map.
 * Uses the CUDA Cooperative Groups API to leverage groups of multiple threads to perform the
 * contains operation for each key. This provides a significant boost in throughput compared
 * to the non Cooperative Group `contains` at moderate to high load factors.
 *
 * @tparam block_size The size of the thread block
 * @tparam tile_size The number of threads in the Cooperative Groups used to perform
 * inserts
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the map's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to the map's `mapped_type`
 * @tparam viewT Type of device view allowing access of hash map storage
 * @tparam Hash Unary callable type
 * @tparam KeyEqual Binary callable type
 *
 * @param first Beginning of the sequence of keys
 * @param n Number of keys to query
 * @param output_begin Beginning of the sequence of booleans for the presence of each key
 * @param view Device view used to access the hash map's slot storage
 * @param hash The unary function to apply to hash each key
 * @param key_equal The binary function to compare two keys for equality
 */
template <std::size_t block_size,
          uint32_t tile_size,
          typename InputIt,
          typename OutputIt,
          typename viewT,
          typename Hash,
          typename KeyEqual>
CUCO_KERNEL void contains(
  InputIt first, int64_t n, OutputIt output_begin, viewT view, Hash hash, KeyEqual key_equal)
{
  auto tile = cg::tiled_partition<tile_size, cg::thread_block>(cg::this_thread_block());
  int64_t const loop_stride = gridDim.x * block_size / tile_size;
  int64_t idx               = (block_size * blockIdx.x + threadIdx.x) / tile_size;
  __shared__ bool writeBuffer[block_size / tile_size];

  while (idx < n) {
    auto key   = *(first + idx);
    auto found = view.contains(tile, key, hash, key_equal);

    /*
     * The ld.relaxed.gpu instruction used in view.find causes L1 to
     * flush more frequently, causing increased sector stores from L2 to global memory.
     * By writing results to shared memory and then synchronizing before writing back
     * to global, we no longer rely on L1, preventing the increase in sector stores from
     * L2 to global and improving performance.
     */
    if (tile.thread_rank() == 0) { writeBuffer[threadIdx.x / tile_size] = found; }
    __syncthreads();
    if (tile.thread_rank() == 0) { *(output_begin + idx) = writeBuffer[threadIdx.x / tile_size]; }
    idx += loop_stride;
  }
}

template <std::size_t block_size,
          typename pair_atomic_type,
          typename viewT,
          typename Hash,
          typename KeyEqual>
CUCO_KERNEL void rehash(
  pair_atomic_type* first, int64_t n, viewT view, Hash hash, KeyEqual key_equal)
{
  int64_t const loop_stride = gridDim.x * block_size;
  int64_t idx               = block_size * blockIdx.x + threadIdx.x;

  while (idx < n) {
    auto current_slot         = first + idx;
    auto const existing_key   = current_slot->first.load(cuda::std::memory_order_relaxed);
    auto const existing_value = current_slot->second.load(cuda::std::memory_order_relaxed);

    auto const is_filled =
      not(cuco::detail::bitwise_compare(view.empty_key_sentinel(), existing_key) or
          cuco::detail::bitwise_compare(view.erased_key_sentinel(), existing_key));
    if (is_filled) {
      typename viewT::value_type const insert_pair{existing_key, existing_value};
      view.insert(insert_pair, hash, key_equal);
    }
    idx += loop_stride;
  }
}

}  // namespace cuco::detail