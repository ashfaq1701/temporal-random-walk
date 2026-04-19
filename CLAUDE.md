# Claude guide — temporal-random-walk

C++17/CUDA library for generating temporal random walks over dynamic graphs, with a pybind11 Python binding.

## Layout

```
temporal_random_walk/
  src/
    common/         shared utilities — CudaBuffer, memory helpers, CUDA error macros, macros (HOST/DEVICE)
    data/           POD types (Edge, SizeRange, DataBlock<T>, WalkSet, WalksWithEdgeFeatures)
    stores/         owning data structures (EdgeDataStore, NodeEdgeIndexStore, TemporalGraphStore, NodeFeaturesStore)
    core/           TemporalRandomWalkStore + walk-generation kernels + CPU fallback
    proxies/        host-only wrapper classes exposed through the public API and pybind
    utils/          random, OpenMP helpers
  py_interface/     pybind11 module definition
  test/             gtest-based unit tests
  test_run/         standalone benchmark executables
build_scripts/
  build_12_6, build_12_8      Dockerfiles pinning CUDA toolkit versions
py_tests/           Python-level integration tests
```

## Build + test workflow

**Run this full sequence before considering any change done.** All commands from the repo root.

CUDA toolkit, pybind11, OpenMP, TBB, and GTest must be installed. The Dockerfiles in `build_scripts/build_12_8/` are the canonical reference for the exact toolchain.

**1. Configure** (once per clean checkout, or after touching CMake files):

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
```

Add `-DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake` if GTest comes from vcpkg. Set `-DHAS_CUDA=OFF` only when deliberately testing the CPU-only fallback (e.g. under ASAN on a GPU-less box).

**2. Build**:

```
cmake --build build -j
```

Re-run this after every edit. CMake rebuilds only touched translation units.

**3. Run the unit tests under compute-sanitizer**:

```
compute-sanitizer --tool memcheck ./build/temporal_random_walk/test/test_temporal_random_walk
```

**Success criteria — both must hold:**

- `compute-sanitizer` footer reports `ERROR SUMMARY: 0 errors`. No leaks, no invalid accesses, no race conditions.
- All gtest cases pass. Last lines show `[  PASSED  ] N tests.` with no `[  FAILED  ]` entries.

If either check fails, investigate the root cause before committing. Do not commit a leak or a test regression to paper over later.

### Other targets

- `test_run_temporal_random_walk_2` and siblings — end-to-end benchmark binaries; they land in `build/bin/` per their CMakeLists override. For performance measurement, not correctness. Not part of the standard workflow.
- pybind module — built by `setup.py bdist_wheel` / `python -m build`, not by the bare `cmake` target. Run `pytest py_tests/` after `pip install -e .` when touching the Python surface.

## Architectural patterns

**Store / Proxy split.** Low-level data lives in "Store" types (plain `struct`s in `src/stores/` and `src/core/`). User-facing classes in `src/proxies/` wrap them with stable public APIs and are what pybind binds. Keep kernels and `edge_data::`/`node_edge_index::` free-function helpers out of proxies.

**Host / device dual-path.** Most operations have `_std` (CPU, OpenMP + TBB) and `_cuda` (GPU, Thrust + custom kernels) implementations. The proxy chooses based on `use_gpu`. Both paths should produce identical output for the same seed.

**Owner / view split (in progress).** Stores that are `cudaMemcpy`'d to device memory for kernels used to play both roles at once, gated by an `owns_data` flag. Ongoing RAII refactor splits them into owning `Store` types (host, with `CudaBuffer<T>` fields) and plain-POD `*View` types (what kernels actually see). See `RAII_migration_progress.md`. `EdgeDataView`, `NodeEdgeIndexView`, `TemporalGraphView`, `TemporalRandomWalkView` are defined but not yet wired into function signatures.

**Memory primitives.** Prefer `CudaBuffer<T>` (`src/common/cuda_buffer.cuh`) for any new owning buffer — move-only, allocator aware via `use_gpu`. Legacy code still uses the `allocate_memory` / `clear_memory` / `resize_memory` / `append_memory` / `copy_memory` free functions in `src/common/memory.cuh`; they're fine for now but are not exception-safe. `DataBlock<T>` is a partial RAII wrapper for rvalue returns — it has a destructor but no copy/move ctor and is a latent double-free hazard (flagged in the RAII doc).

**Macros.** `HOST` / `DEVICE` macros in `src/common/macros.cuh` expand to `__host__` / `__device__` under `HAS_CUDA`, empty otherwise. `CUDA_CHECK_AND_CLEAR` / `CUDA_KERNEL_CHECK` in `src/common/error_handlers.cuh` are the canonical error checks; they `std::exit` on failure rather than throwing, so exception-safety concerns are limited in practice.

## Conventions visible in the code

- 4-space indent, braces on same line, `HOST`/`DEVICE` before return type.
- Fields that point to GPU memory mirror the `bool use_gpu` on their enclosing struct — respect it when freeing.
- `#pragma omp parallel for` for CPU parallelism; `thrust::` execution policy constant `DEVICE_EXECUTION_POLICY` for GPU.
- Free-function namespaces (`edge_data::`, `node_edge_index::`, `temporal_graph::`, `temporal_random_walk::`) hold the operations on each store; keep the store structs themselves data-only.
- No exceptions as control flow — `std::exit` is the standard response to CUDA errors (via the `CUDA_CHECK_AND_CLEAR` macro).

## Active work

**Branch `claude/temporal-random-walk-raii-vIfAH`** holds an in-progress RAII refactor. `RAII_migration_progress.md` at the repo root tracks what's done, what's left, and the constraints discovered along the way. Pick up there before making store-level changes — there are non-obvious cross-store dependencies (the `cudaMemcpy` boundary for the store hierarchy has to flip from `Store*` to `View*` atomically).

## When verifying changes

- Always run the full build + test + memcheck workflow above. "It compiles" is not sufficient.
- Type errors in `nvcc`-only code are invisible to plain `g++` — do not rely on a stripped-down CPU compile as a shortcut.
- For Python-facing changes, also run `py_tests/` with `pytest`.
- Changes to `CudaBuffer<T>`, `DataBlock<T>`, `WalkSet`, or any `Store`'s destructor are high-blast-radius — the memcheck run is non-negotiable for these.
