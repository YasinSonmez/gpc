import os
import sys

# Fix for cuda_nvcc.__file__ being None in namespace packages
try:
    import nvidia.cuda_nvcc
    if nvidia.cuda_nvcc.__file__ is None:
        nvidia.cuda_nvcc.__file__ = nvidia.cuda_nvcc.__path__[0] + "/__init__.py"
        sys.modules['nvidia.cuda_nvcc'].__file__ = nvidia.cuda_nvcc.__file__
except (ImportError, AttributeError):
    pass  # CUDA not available or not needed

# Set XLA/JAX flags for better performance and memory management
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Don't preallocate all GPU memory
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true "

import jax
import mujoco  # noqa: F401 (imported for side effects)

jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update(
    # "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
# )
