"""
temporal_random_walk — high-performance temporal random walk sampling
with optional GPU acceleration.
"""
from __future__ import annotations

import ctypes
import glob
import os
import site


def _preload_nvidia_cuda_libs() -> None:
    """Pre-load nvidia-*-cu12 CUDA libraries via RTLD_GLOBAL.

    nvidia-*-cu12 pip packages install .so files under
    ``<site-packages>/nvidia/<component>/lib/``. Loading each one with
    ``RTLD_GLOBAL`` makes its symbols visible to subsequent ``dlopen()``
    calls, so the extension module's dynamic-link dependencies
    (libcudart.so.12, libcurand.so.10) resolve to the pip-provided
    copies rather than failing with "cannot open shared object file".

    Silently tolerates missing libs — CPU-only builds and unusual
    install layouts still work.
    """
    site_roots = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        site_roots.append(user_site)

    # Deduplicate while preserving order — some environments list the
    # same path twice.
    seen: set[str] = set()
    unique_roots = [r for r in site_roots if not (r in seen or seen.add(r))]

    # Patterns matched against nvidia/<component>/lib/. Add a pattern
    # here if we start linking against another CUDA library.
    patterns = ("libcudart.so.12*", "libcurand.so.10*")

    for root in unique_roots:
        for pattern in patterns:
            for so_path in glob.glob(
                os.path.join(root, "nvidia", "*", "lib", pattern)
            ):
                try:
                    ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    # Already loaded, ABI-incompatible, or permission
                    # denied — fall through to the extension's own
                    # load attempt, which will give a clearer error
                    # if the lib is genuinely missing.
                    pass


_preload_nvidia_cuda_libs()

from _temporal_random_walk import *  # noqa: E402, F401, F403
