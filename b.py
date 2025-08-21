from functools import partial
from flax import linen as nn
import jax
from jax.experimental.mesh_utils import create_device_mesh
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


ki = nn.linear.default_kernel_init



simulate_CPU_devices()


key = jax.random.key(0)
x = jax.random.normal(key, (8, 8))

module = nn.Dense(
    80,
    kernel_init=nn.with_partitioning(ki, ( None, 'data'))
)

params = module.init(key, x)

devices = create_device_mesh((8,), jax.devices())
mesh = jax.sharding.Mesh(devices, axis_names = ("data",))

pspec = nn.get_partition_spec(params)
print(f"DEBUGPRINT[77]: b.py:56: pspec={pspec}")


@partial(
    jax.shard_map,
    in_specs = (pspec, jax.P("data", None)),
    out_specs = jax.P("data", None),
    mesh = mesh
)
def train(param, x):
    jax.debug.visualize_array_sharding(param["params"]["kernel"].value)
    pass


train(params, x)
