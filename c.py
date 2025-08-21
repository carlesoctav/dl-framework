import jax
import jax.numpy as jnp
from jax import P
from functools import partial
import os
import subprocess
import sys

def set_XLA_flags_gpu():
    flags = os.environ.get("XLA_FLAGS", "")
    flags += (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=false "
        "--xla_gpu_enable_async_collectives=true "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_highest_priority_async_stream=true "
    )
    os.environ["XLA_FLAGS"] = flags


def simulate_CPU_devices(device_count: int = 8):
    flags = os.environ.get("XLA_FLAGS", "")
    flags += f" --xla_force_host_platform_device_count={device_count}"
    os.environ["XLA_FLAGS"] = flags
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import ml_collections
    except ImportError:
        install_package("ml_collections")


def install_package(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])





simulate_CPU_devices()


mesh = jax.make_mesh((4, 2), ('x', 'y'))

a = jnp.arange( 8 * 16.).reshape(8, 16)
b = jnp.arange(16 *  4.).reshape(16, 4)

@partial(jax.shard_map, mesh=mesh, in_specs=(P('x', 'y'), P('y', None)),
         out_specs=P('x', None))
def matmul_basic(a_block, b_block):
  # a_block: f32[2, 8]
  # b_block: f32[8, 4]
  jax.debug.visualize_array_sharding(a_block)
  jax.debug.visualize_array_sharding(b_block)
  c_partialsum = jnp.dot(a_block, b_block)
  c_block = jax.lax.psum(c_partialsum, 'y')
  # c_block: f32[2, 4]
  return c_block

c = matmul_basic(a, b)   # c: f32[8, 4]
