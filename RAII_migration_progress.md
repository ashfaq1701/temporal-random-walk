# RAII Migration Progress

**Branch:** `claude/temporal-random-walk-raii-vIfAH`
**Base:** `master` at `1b8dc0f`
**Status:** In progress — 4 commits landed, store-level migration (B.2–B.6) pending.

## Motivation

The codebase manages CUDA/CPU memory manually via `new`/`delete`, `cudaMalloc`/`cudaFree`, bare `T*` fields, and `owns_data` flags. An audit flagged:

- **A real double-free bug** in `WalksWithEdgeFeatures` (destructor + compiler-synthesized copy ctor).
- Many classes that violated Rule-of-0/3/5 via raw pointers + user-declared destructors.
- `owns_data` flags scattered across 8+ types to manually simulate ownership semantics.
- No `std::unique_ptr` / `std::shared_ptr` usage anywhere, even in pure-host proxy classes.
- CUDA scratch allocations (`cudaMalloc` / `cudaFree` pairs) in kernel launch sites with no exception-safety wrapper.

The goal is to move toward actual RAII: resource lifetime tied to object lifetime, no manual cleanup chains, no ownership flags.

## Key discoveries

### The Stores are secretly dual-role

`EdgeDataStore`, `NodeEdgeIndexStore`, `TemporalGraphStore`, and `TemporalRandomWalkStore` each play **two roles at different times**:

1. **Owner** on the host (when `owns_data == true`, the destructor frees all pointer fields).
2. **Device view** when `owns_data == false` and the struct has been shallow-copied with its pointer fields reassigned to device addresses, then `cudaMemcpy`'d to device for kernel consumption.

This works because every field is a raw pointer, keeping the struct trivially copyable. The `owns_data` flag is a manual implementation of "this instance doesn't actually own these pointers."

### RAII breaks the dual role

Replacing raw pointer fields with `CudaBuffer<T>` (or `std::unique_ptr`) makes the struct **non-copyable**, which breaks two things simultaneously:

1. `StoreType temp = *store;` inside `to_device_ptr` stops compiling.
2. `cudaMemcpy(device, &store, sizeof(StoreType))` becomes UB: memmove on a non-trivially-copyable type is undefined per `[basic.types]`.

### The fix cascades across the whole store hierarchy

Splitting ownership from device-view-layout means:

- Owning `Store` has `CudaBuffer<T>` fields; lives on host; destroys its own resources.
- Non-owning `View` POD struct with raw `T*` fields; is what kernels see; trivially copyable.
- `to_device_ptr(Store*)` builds a `View` on the host, `cudaMalloc`s device memory for it, `cudaMemcpy`s the view bytes.
- Device function signatures flip from `const StoreType*` to `const ViewType*`.

This cascades: `edge_data::to_device_ptr` returns `EdgeDataView*`, so `TemporalGraphStore::to_device_ptr` must build a `TemporalGraphView` (not a store shallow-copy), which means `TemporalRandomWalkStore::to_device_ptr` must build a `TemporalRandomWalkView`, etc. **One store migration drags in the whole chain.**

## Completed work

### `7568f09` — CudaBuffer + WalksWithEdgeFeatures

- Added `temporal_random_walk/src/common/cuda_buffer.cuh` — a move-only RAII wrapper with `data()`, `release()`, `reset()`, `size()`, `operator bool`. Uses `cudaFree` when `use_gpu`, `free()` otherwise.
- `WalksWithEdgeFeatures::walk_edge_features` migrated from `float*` + manual destructor to `CudaBuffer<float>`. Struct is now explicitly non-copyable and noexcept-movable.
- Consumers updated: pybind11 `_temporal_random_walk.cu` uses `release()` to transfer ownership to NumPy; `test_utils.h` and `test_temporal_random_walk_edge_features.cpp` use `.data()`.
- Closes the double-free hole.

### `a6b70c8` — NodeFeaturesStore

- Raw `float* node_features` + `node_features_size` + `owns_data` + hand-rolled destructor → single `CudaBuffer<float>` member.
- `ensure_size` allocates a fresh `CudaBuffer`, copies old values via `copy_memory`, zero-fills the extension, move-assigns in. Semantically equivalent to the previous `resize_memory` path but exception-safe by construction.
- First store with full RAII. No dual-role concern here (`NodeFeaturesStore` is never `to_device_ptr`'d).

### `ef13323` — Proxy-level `unique_ptr`

- `NodeFeatures::node_features` → `std::unique_ptr<NodeFeaturesStore>`.
- `TemporalRandomWalk::temporal_random_walk` → `std::unique_ptr<TemporalRandomWalkStore>`.
- `TemporalRandomWalk::node_features` → `std::unique_ptr<NodeFeatures>`.
- 4 `new`/`delete` pairs gone, 2 proxy destructors gone.
- All call sites that forwarded these members to free-function APIs now pass `.get()`.

### `41e26c6` — Device-facing view POD structs

- Added `EdgeDataView`, `NodeEdgeIndexView`, `TemporalGraphView`, `TemporalRandomWalkView` as pure POD aggregates.
- Views carry only the subset of fields device code reads (see survey below).
- Default-constructible + trivially copyable (`static_assert`-verified), so they're ready to be `cudaMemcpy`'d to device memory by future `to_device_ptr` routines.
- **No function signature uses them yet** — this commit is pure type foundation.

## Remaining work

The remaining store migration (B.2 through B.6) is a single logical unit because the `cudaMemcpy` boundary has to flip from `Store*` to `View*` across the whole hierarchy at once. Below is the intended breakdown once we have a CUDA-capable dev environment to compile-verify each step:

### B.2 — `EdgeDataStore` → `CudaBuffer<T>` + View signatures

- Replace 11 raw pointer fields with `CudaBuffer<T>` members.
- Drop `owns_data` flag and user-declared destructor.
- All `HOST DEVICE` / `DEVICE` edge_data:: functions take `const EdgeDataView*` (not `const EdgeDataStore*`).
- `edge_data::to_device_ptr(const EdgeDataStore*)` returns `EdgeDataView*` (device pointer). Host-side helper builds an `EdgeDataView` from the store (extracts `.data()` / `.size()` into raw pointer + size fields) and `cudaMemcpy`s that to device memory.
- Add `edge_data::make_view(const EdgeDataStore*)` helper so host call sites can cheaply construct a view on demand.
- Host call sites in `temporal_graph.cu`, `proxies/EdgeData.cu`, `edge_selectors.cuh` (host path), and tests either pass the locally-constructed view or adapt to `.data()`-based access.
- Touches: `edge_data.{cuh,cu}` (~1.1k lines), `temporal_graph.cu`, `proxies/EdgeData.cu`, `edge_selectors.cuh`, `test_edge_data.cpp`, `test_temporal_graph.cpp`.

### B.3 — `NodeEdgeIndexStore` → `CudaBuffer<T>` + View signatures

Same shape as B.2. 11 pointer fields. All are device-accessed (no host-only fields). Touches: `node_edge_index.{cuh,cu}` (~1.9k lines), `temporal_graph.cu`, `proxies/NodeEdgeIndex.cu`, `edge_selectors.cuh`, tests.

### B.4 — `TemporalGraphStore` → `unique_ptr` + `TemporalGraphView`

- `TemporalGraphStore::edge_data`, `node_edge_index` → `std::unique_ptr`.
- `temporal_graph::to_device_ptr` builds a `TemporalGraphView` (not a `TemporalGraphStore` shallow copy); the view's `edge_data` / `node_edge_index` point at the device views returned by the nested `to_device_ptr` calls.
- Kernels that take `const TemporalGraphStore*` flip to `const TemporalGraphView*`.
- Drop `owns_data`.
- Touches: `temporal_graph.{cuh,cu}`, `edge_selectors.cuh`, kernel headers, `proxies/TemporalGraph.cu`, tests.

### B.5 — `TemporalRandomWalkStore` → `unique_ptr` + `TemporalRandomWalkView`

- `temporal_graph` → `std::unique_ptr<TemporalGraphStore>`.
- `cuda_device_prop` → `std::unique_ptr<cudaDeviceProp>`.
- `last_batch_unique_sources/targets` → `CudaBuffer<int>` (host-only CPU buffers).
- `temporal_random_walk::to_device_ptr` builds a `TemporalRandomWalkView`.
- Kernels flip to `const TemporalRandomWalkView*`.
- Drop `owns_data`.

### B.6 — Sweep remaining `owns_data` flags

After B.2–B.5, there should be no live reader of `owns_data` on any of the above stores. This commit deletes the flag from each struct and any test/proxy that still references it. Also deletes `NodeEdgeIndex::clear()` and the `node_edge_index::clear` free function (both were manual-cleanup APIs that RAII makes redundant).

### Independent cleanup tracks (orderable any time)

- **CUDA scratch-allocation RAII.** The `cudaMalloc` / `cudaFree` pairs around kernel launches leak on any early return. Sites: `RandomPicker.cu` (4 pickers × scratch buffers), `proxies/TemporalGraph.cu:91-102`, `proxies/EdgeData.cu:82-92`, `proxies/NodeEdgeIndex.cu:68-81`. Each pair wraps neatly in `CudaBuffer<T>`. Fully independent of B.2–B.6; can land at any time.
- **`DataBlock<T>` Rule-of-5.** `DataBlock<T>` in `data/structs.cuh:68-102` has a destructor that frees via `cudaFree` / `free`, but no copy/move constructors. Any accidental copy double-frees. Identical hazard to the one fixed in `WalksWithEdgeFeatures`. Should get the same treatment (make it non-copyable, add move, or migrate to `CudaBuffer<T>`).
- **Aliased proxies (`TemporalGraph`, `EdgeData`, `NodeEdgeIndex`).** These proxies have a `T* store` + `bool owns_*` two-state pattern (owning construction vs. non-owning view construction). After B.2–B.6 lands, these can be split into an owning proxy (holds `unique_ptr`) and a non-owning view proxy (holds raw pointer / reference). Or collapsed to `unique_ptr` + a separate view type.

## Constraints and gotchas

- **No local CUDA toolchain during this work.** Commits have been verified by building small extracted translation units with plain `g++ -std=c++17` where feasible. Anything touching `nvcc`-specific semantics (kernels, `thrust`, `cuda::std::`) has not been compile-verified. CI or a local CUDA dev machine is required to land B.2+ confidently.
- **`cudaMemcpy` over `CudaBuffer`-containing structs is UB.** This is the architectural reason for the View/Store split. Do not regress by memcpy'ing a `Store` directly.
- **Device code must never invoke `CudaBuffer` accessors** unless those accessors are marked `HOST DEVICE`. Currently they're `HOST` only, which is correct as long as device code goes through `View` (not `Store`). If a future change exposes `CudaBuffer` fields to device code, the accessors will need the `HOST DEVICE` marker.
- **Field parity between Store and View.** When adding a new pointer field to a Store that kernels will touch, add the corresponding `T*` + `size_t` pair to the View and populate it in `to_device_ptr`. Missing fields won't produce a compile error — they'll silently be `nullptr` on the device.

## Reference: device-accessed field survey

From an audit of the stores and kernel code:

| Store | Device-accessed fields | Host-only fields |
|---|---|---|
| `EdgeDataStore` | `sources`, `targets`, `timestamps`, `unique_timestamps`, `timestamp_group_offsets`, `active_node_ids`, `forward_cumulative_weights_exponential`, `backward_cumulative_weights_exponential`, `max_node_id` | `edge_features`, `feature_dim`, `node_adj_offsets`, `node_adj_neighbors`, `enable_*` flags, `use_gpu` |
| `NodeEdgeIndexStore` | all 11 pointer arrays (every field is device-accessed) | none |
| `TemporalGraphStore` | `edge_data` (→ `EdgeDataView*`), `node_edge_index` (→ `NodeEdgeIndexView*`), `is_directed`, `inv_p`, `inv_q`, `latest_timestamp` | `max_time_capacity`, `timescale_bound`, `enable_*` flags, `use_gpu`, `node2vec_p/q` |
| `TemporalRandomWalkStore` | `temporal_graph` (→ `TemporalGraphView*`), `is_directed`, `walk_padding_value`, `global_seed`, `shuffle_walk_order`, `node2vec_p/q` | `cuda_device_prop`, `last_batch_unique_*`, `max_time_capacity`, `timescale_bound`, `enable_*` flags |

The view structs in `41e26c6` already encode these splits.
