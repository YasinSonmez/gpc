import os
import sys
from pathlib import Path

# Fix for cuda_nvcc.__file__ being None in namespace packages
try:
    import nvidia.cuda_nvcc
    if nvidia.cuda_nvcc.__file__ is None:
        nvidia.cuda_nvcc.__file__ = nvidia.cuda_nvcc.__path__[0] + "/__init__.py"
        sys.modules['nvidia.cuda_nvcc'].__file__ = nvidia.cuda_nvcc.__file__
except (ImportError, AttributeError):
    pass  # CUDA not available or not needed

# Set XLA/JAX flags for better performance and memory management.
# Use setdefault / append so callers can override and compose flags.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_triton_gemm_any=true" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_triton_gemm_any=true").strip()

import jax
import mujoco  # noqa: F401 (imported for side effects)

jax.config.update("jax_default_matmul_precision", "highest")
_cache_dir = Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache" / "jax")))
_cache_dir.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_cache_dir))
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update(
    # "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
# )
