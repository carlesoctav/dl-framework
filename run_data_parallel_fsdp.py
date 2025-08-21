#!/usr/bin/env python3
"""Run a minimal Data-Parallel (DP) and Fully-Sharded Data-Parallel (FSDP)
example from the notebook `data_parallel_fsdp.ipynb`.

Default: simulates CPU devices (8) so you can run without GPUs.

Usage:
  python run_data_parallel_fsdp.py            # simulate CPU (default)
  python run_data_parallel_fsdp.py --no-simulate-cpu   # use real devices (GPUs) if available
  python run_data_parallel_fsdp.py --device-count 4    # change simulated device count

Dependencies: jax, flax, optax. The script will not install packages for you.
"""

import os
import argparse
import pprint
import jax


def simulate_cpu_devices(device_count: int = 8) -> None:
    """Set XLA environment variables to simulate a host with multiple devices."""
    flags = os.environ.get("XLA_FLAGS", "")
    flags += f" --xla_force_host_platform_device_count={device_count}"
    os.environ["XLA_FLAGS"] = flags
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    parser = argparse.ArgumentParser(description="Run DP and FSDP demo")
    parser.add_argument("--no-simulate-cpu", action="store_true", help="Do not simulate CPU devices; use real devices if available")
    parser.add_argument("--device-count", type=int, default=8, help="Number of simulated CPU devices (when simulating)")
    parser.add_argument("--steps", type=int, default=3, help="Number of training steps to run")
    parser.add_argument("--use-scan", action="store_true", help="Use `lax.scan` based gradient accumulation (faster compile)")
    args = parser.parse_args()

    if not args.no_simulate_cpu:
        print(f"Simulating CPU devices: {args.device_count}")
        simulate_cpu_devices(args.device_count)

    # Delay importing JAX/Flax until after environment flags set
    import functools
    from typing import Any, Callable, Dict, Sequence, Tuple

    import jax
    import jax.numpy as jnp
    import numpy as np
    import flax.linen as nn
    from flax.struct import dataclass as flax_dataclass
    from flax.training import train_state
    import optax
    from jax import lax
    from jax.experimental.shard_map import shard_map
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    try:
        from ml_collections import ConfigDict
    except Exception:
        class ConfigDict(dict):
            def __init__(self, d=None):
                super().__init__(d or {})
                for k, v in list(self.items()):
                    if isinstance(v, dict):
                        self[k] = ConfigDict(v)

            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(name)

            def __setattr__(self, name, value):
                self[name] = value

            def __repr__(self):
                return f"ConfigDict({dict(self)})"

    PyTree = Any
    Metrics = Dict[str, Tuple[jax.Array, ...]]

    @flax_dataclass
    class Batch:
        inputs: jax.Array
        labels: jax.Array

    class TrainState(train_state.TrainState):
        rng: jax.Array

    def print_metrics(metrics: Metrics, title: str | None = None) -> None:
        metrics = jax.device_get(metrics)
        lines = [f"{k}: {v[0] / v[1]:.6f}" for k, v in metrics.items()]
        if title:
            title = f" {title} "
            max_len = max(len(title), max(map(len, lines)))
            lines = [title.center(max_len, "=")] + lines
        print("\n".join(lines))

    def get_num_params(state: TrainState) -> int:
        return sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(state.params))

    # Gradient accumulation implementations (loop and scan versions)
    def accumulate_gradients_loop(
        state: TrainState,
        batch: Batch,
        rng: jax.random.PRNGKey,
        num_minibatches: int,
        loss_fn: Callable,
    ) -> Tuple[PyTree, Metrics]:
        batch_size = batch.inputs.shape[0]
        minibatch_size = batch_size // num_minibatches
        rngs = jax.random.split(rng, num_minibatches)
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        grads = None
        metrics = None
        for minibatch_idx in range(num_minibatches):
            with jax.named_scope(f"minibatch_{minibatch_idx}"):
                start = minibatch_idx * minibatch_size
                end = start + minibatch_size
                minibatch = jax.tree.map(lambda x: x[start:end], batch)
                (_, step_metrics), step_grads = grad_fn(
                    state.params, state.apply_fn, minibatch, rngs[minibatch_idx]
                )
                if grads is None:
                    grads = step_grads
                    metrics = step_metrics
                else:
                    grads = jax.tree.map(jnp.add, grads, step_grads)
                    metrics = jax.tree.map(jnp.add, metrics, step_metrics)
        grads = jax.tree.map(lambda g: g / num_minibatches, grads)
        return grads, metrics

    def accumulate_gradients_scan(
        state: TrainState,
        batch: Batch,
        rng: jax.random.PRNGKey,
        num_minibatches: int,
        loss_fn: Callable,
    ) -> Tuple[PyTree, Metrics]:
        batch_size = batch.inputs.shape[0]
        minibatch_size = batch_size // num_minibatches
        rngs = jax.random.split(rng, num_minibatches)
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        def _minibatch_step(minibatch_idx: jax.Array | int) -> Tuple[PyTree, Metrics]:
            minibatch = jax.tree.map(
                lambda x: jax.lax.dynamic_slice_in_dim(
                    x, start_index=minibatch_idx * minibatch_size, slice_size=minibatch_size, axis=0
                ),
                batch,
            )
            (_, step_metrics), step_grads = grad_fn(
                state.params, state.apply_fn, minibatch, rngs[minibatch_idx]
            )
            return step_grads, step_metrics

        def _scan_step(carry: Tuple[PyTree, Metrics], minibatch_idx: jax.Array | int) -> Tuple[Tuple[PyTree, Metrics], None]:
            step_grads, step_metrics = _minibatch_step(minibatch_idx)
            carry = jax.tree.map(jnp.add, carry, (step_grads, step_metrics))
            return carry, None

        grads_shapes, metrics_shape = jax.eval_shape(_minibatch_step, 0)
        grads = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shapes)
        metrics = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), metrics_shape)
        (grads, metrics), _ = jax.lax.scan(
            _scan_step, init=(grads, metrics), xs=jnp.arange(num_minibatches), length=num_minibatches
        )
        grads = jax.tree.map(lambda g: g / num_minibatches, grads)
        return grads, metrics

    def accumulate_gradients(
        state: TrainState,
        batch: Batch,
        rng: jax.random.PRNGKey,
        num_minibatches: int,
        loss_fn: Callable,
        use_scan: bool = False,
    ) -> Tuple[PyTree, Metrics]:
        if use_scan:
            return accumulate_gradients_scan(
                state=state, batch=batch, rng=rng, num_minibatches=num_minibatches, loss_fn=loss_fn
            )
        else:
            return accumulate_gradients_loop(
                state=state, batch=batch, rng=rng, num_minibatches=num_minibatches, loss_fn=loss_fn
            )

    # fold RNG across a mesh axis so each device gets a different dropout key
    def fold_rng_over_axis(rng: jax.random.PRNGKey, axis_name: str) -> jax.random.PRNGKey:
        axis_index = jax.lax.axis_index(axis_name)
        return jax.random.fold_in(rng, axis_index)

    # ---- Model and config setup ----
    data_config = ConfigDict(dict(batch_size=8, num_classes=10, input_size=784))
    model_config = ConfigDict(
        dict(
            hidden_size=512,
            dropout_rate=0.1,
            dtype=jnp.float32,
            num_classes=data_config.num_classes,
            data_axis_name="data",
            min_weight_size=2 ** 4,
        )
    )
    optimizer_config = ConfigDict(dict(learning_rate=1e-3, num_minibatches=1))
    config = ConfigDict(
        dict(model=model_config, optimizer=optimizer_config, data=data_config, data_axis_name=model_config.data_axis_name, seed=42)
    )

    class DPClassifier(nn.Module):
        config: ConfigDict

        @nn.compact
        def __call__(self, x: jax.Array, train: bool) -> jax.Array:
            x = nn.Dense(features=self.config.hidden_size, dtype=self.config.dtype, name="input_dense")(x)
            x = nn.silu(x)
            x = nn.Dropout(rate=self.config.dropout_rate, deterministic=not train)(x)
            x = nn.Dense(features=self.config.num_classes, dtype=self.config.dtype, name="output_dense")(x)
            x = x.astype(jnp.float32)
            return x

    optimizer = optax.adamw(learning_rate=config.optimizer.learning_rate)

    model_dp = DPClassifier(config=config.model)

    rng = jax.random.PRNGKey(config.seed)
    model_init_rng, data_inputs_rng, data_labels_rng = jax.random.split(rng, 3)
    batch = Batch(
        inputs=jax.random.normal(data_inputs_rng, (config.data.batch_size, config.data.input_size)),
        labels=jax.random.randint(data_labels_rng, (config.data.batch_size,), 0, config.data.num_classes),
    )

    def init_dp(rng: jax.random.PRNGKey, x: jax.Array, model: nn.Module) -> TrainState:
        init_rng, rng = jax.random.split(rng)
        variables = model.init({"params": init_rng}, x, train=False)
        params = variables.pop("params")
        state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer, rng=rng)
        return state

    device_array = np.array(jax.devices())
    mesh = Mesh(device_array, (config.data_axis_name,))

    init_dp_fn = jax.jit(
        shard_map(
            functools.partial(init_dp, model=model_dp),
            mesh,
            in_specs=(P(), P(config.data_axis_name)),
            out_specs=P(),
            check_rep=False,
        )
    )

    state_dp = init_dp_fn(model_init_rng, batch.inputs)
    print("DP Parameters")
    pprint.pprint(jax.tree.map(lambda x: (getattr(x, "shape", None), getattr(x, "sharding", None)), state_dp.params))

    # loss and train step for DP
    def loss_fn(params: PyTree, apply_fn: Any, batch: Batch, rng: jax.Array):
        dropout_rng = fold_rng_over_axis(rng, config.data_axis_name)
        logits = apply_fn({"params": params}, batch.inputs, train=True, rngs={"dropout": dropout_rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch.labels)
        correct_pred = jnp.equal(jnp.argmax(logits, axis=-1), batch.labels)
        batch_size = batch.inputs.shape[0]
        step_metrics = {"loss": (loss.sum(), batch_size), "accuracy": (correct_pred.sum(), batch_size)}
        loss = loss.mean()
        return loss, step_metrics

    def train_step_dp(state: TrainState, metrics: Metrics | None, batch: Batch) -> Tuple[TrainState, Metrics]:
        rng, step_rng = jax.random.split(state.rng)
        grads, step_metrics = accumulate_gradients(
            state,
            batch,
            step_rng,
            config.optimizer.num_minibatches,
            loss_fn=loss_fn,
            use_scan=args.use_scan,
        )
        with jax.named_scope("sync_gradients"):
            grads = jax.tree.map(lambda g: jax.lax.pmean(g, axis_name=config.data_axis_name), grads)
        new_state = state.apply_gradients(grads=grads, rng=rng)
        with jax.named_scope("sync_metrics"):
            step_metrics = jax.tree.map(lambda x: jax.lax.psum(x, axis_name=config.data_axis_name), step_metrics)
        if metrics is None:
            metrics = step_metrics
        else:
            metrics = jax.tree.map(jnp.add, metrics, step_metrics)
        return new_state, metrics

    train_step_dp_fn = jax.jit(
        shard_map(
            train_step_dp,
            mesh,
            in_specs=(P(), P(), P(config.data_axis_name)),
            out_specs=(P(), P()),
            check_rep=False,
        ),
        donate_argnames=("state", "metrics"),
    )

    _, metric_shapes = jax.eval_shape(train_step_dp_fn, state_dp, None, batch)
    metrics_dp = jax.tree.map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes)

    print("Running DP training loop...")
    for _ in range(args.steps):
        state_dp, metrics_dp = train_step_dp_fn(state_dp, metrics_dp, batch)
    final_metrics_dp = jax.tree.map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes)
    state_dp, final_metrics_dp = train_step_dp_fn(state_dp, final_metrics_dp, batch)
    print_metrics(final_metrics_dp, "DP - Final metrics")

    # ---- FSDP: parameter sharding ----
    Parameter = jax.Array | nn.Partitioned

    @jax.named_scope("shard_params")
    def shard_params(params: PyTree, axis_name: str, min_weight_size: int = 2 ** 18) -> PyTree:
        axis_idx = jax.lax.axis_index(axis_name)
        axis_size = jax.lax.psum(1, axis_name)

        def _split(x: Parameter) -> Parameter:
            if isinstance(x, nn.Partitioned):
                value, names = x.value, x.names
            else:
                value = x
                names = (None,) * value.ndim
            if axis_name in names:
                return x
            elif value.size <= min_weight_size:
                return x
            else:
                shape = value.shape
                idx = np.argsort(shape)[::-1]
                for i in idx:
                    if shape[i] % axis_size == 0 and names[i] is None:
                        split_size = shape[i] // axis_size
                        p_sharded = nn.Partitioned(
                            value=lax.dynamic_slice_in_dim(value, axis_idx * split_size, split_size, axis=i),
                            names=names[:i] + (axis_name,) + names[i + 1 :],
                        )
                        return p_sharded
                return x

        return jax.tree.map(
            _split,
            params,
            is_leaf=lambda x: isinstance(x, nn.Partitioned),
        )

    def gather_array_with_mean_grads(x: jax.Array, axis: int, axis_name: str):
        axis_size = jax.lax.psum(1, axis_name)

        @jax.custom_gradient
        def f(x):
                def grad_fn(g):
                    return jax.lax.psum_scatter(g, axis_name, scatter_dimension=axis, tiled=True) / axis_size

                return jax.lax.all_gather(x, axis_name, axis=axis, tiled=True), grad_fn

        return f(x)

    @jax.named_scope("gather_params")
    def gather_params(params: PyTree, axis_name: str) -> PyTree:
        def _gather(p: Parameter) -> Parameter:
            if isinstance(p, nn.Partitioned) and axis_name in p.names:
                param_shard = p.names
                shard_axis = param_shard.index(axis_name)
                value = gather_array_with_mean_grads(p.value, axis=shard_axis, axis_name=axis_name)
                param_shard = param_shard[:shard_axis] + (None,) + param_shard[shard_axis + 1 :]
                if any([name is not None for name in param_shard]):
                    return nn.Partitioned(value, param_shard)
                else:
                    return value
            else:
                return p

        return jax.tree.map(_gather, params, is_leaf=lambda x: isinstance(x, nn.Partitioned))

    def shard_module_params(target: nn.Module | Callable, axis_name: str, min_weight_size: int = 2 ** 18) -> nn.Module | Callable:
        return nn.map_variables(
            target,
            trans_in_fn=functools.partial(gather_params, axis_name=axis_name),
            trans_out_fn=functools.partial(shard_params, axis_name=axis_name, min_weight_size=min_weight_size),
            mapped_collections="params",
            mutable=True,
        )

    class FSDPClassifier(nn.Module):
        config: ConfigDict

        @nn.compact
        def __call__(self, x: jax.Array, train: bool) -> jax.Array:
            sharded_dense = shard_module_params(nn.Dense, axis_name=self.config.data_axis_name, min_weight_size=self.config.min_weight_size)
            x = sharded_dense(features=self.config.hidden_size, dtype=self.config.dtype, name="input_dense")(x)
            x = nn.silu(x)
            x = nn.Dropout(rate=self.config.dropout_rate, deterministic=not train)(x)
            x = sharded_dense(features=self.config.num_classes, dtype=self.config.dtype, name="output_dense")(x)
            x = x.astype(jnp.float32)
            return x

    # create FSDP model and initialize shapes/specs
    model_fsdp = FSDPClassifier(config=config.model)
    # first init with unknown output spec to get shapes
    init_fsdp_fn = shard_map(
        functools.partial(init_dp, model=model_fsdp),
        mesh,
        in_specs=(P(), P(config.data_axis_name)),
        out_specs=P(),
        check_rep=False,
    )
    state_fsdp_shapes = jax.eval_shape(init_fsdp_fn, model_init_rng, batch.inputs)
    state_fsdp_specs = nn.get_partition_spec(state_fsdp_shapes)

    # re-jit with correct output specs
    init_fsdp_fn = jax.jit(
        shard_map(
            functools.partial(init_dp, model=model_fsdp),
            mesh,
            in_specs=(P(), P(config.data_axis_name)),
            out_specs=state_fsdp_specs,
            check_rep=False,
        )
    )
    state_fsdp = init_fsdp_fn(model_init_rng, batch.inputs)

    print("FSDP Parameters")
    pprint.pprint(jax.tree.map(lambda x: getattr(x, "shape", None), jax.device_get(state_fsdp.params)))

    # sync gradients function for partitioned params
    def sync_gradients(grads: PyTree, axis_names: Sequence[str]) -> PyTree:
        def sync_grad(g: Parameter) -> Parameter:
            if isinstance(g, nn.Partitioned):
                replication_axis_names = [name for name in axis_names if name not in jax.tree_util.tree_leaves(g.names)]
                if len(replication_axis_names) == 0:
                    return g
                else:
                    return g.replace(value=jax.lax.pmean(g.value, axis_name=replication_axis_names))
            else:
                return jax.lax.pmean(g, axis_name=axis_names)

        return jax.tree.map(sync_grad, grads, is_leaf=lambda x: isinstance(x, nn.Partitioned))

    def train_step_fsdp(state: TrainState, metrics: Metrics | None, batch: Batch) -> Tuple[TrainState, Metrics]:
        rng, step_rng = jax.random.split(state.rng)
        grads, step_metrics = accumulate_gradients(
            state,
            batch,
            step_rng,
            config.optimizer.num_minibatches,
            loss_fn=loss_fn,
            use_scan=args.use_scan,
        )
        with jax.named_scope("sync_gradients"):
            grads = sync_gradients(grads, (config.data_axis_name,))
        new_state = state.apply_gradients(grads=grads, rng=rng)
        with jax.named_scope("sync_metrics"):
            step_metrics = jax.tree.map(lambda x: jax.lax.psum(x, axis_name=config.data_axis_name), step_metrics)
        if metrics is None:
            metrics = step_metrics
        else:
            metrics = jax.tree.map(jnp.add, metrics, step_metrics)
        return new_state, metrics

    train_step_fsdp_fn = jax.jit(
        shard_map(
            train_step_fsdp,
            mesh,
            in_specs=(state_fsdp_specs, P(), P(config.data_axis_name)),
            out_specs=(state_fsdp_specs, P()),
            check_rep=False,
        ),
        donate_argnames=("state", "metrics"),
    )

    _, metric_shapes = jax.eval_shape(train_step_fsdp_fn, state_fsdp, None, batch)
    metrics_fsdp = jax.tree.map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes)

    print("Running FSDP training loop...")
    jax.profiler.start_trace("traces/")
    for i in range(args.steps):
         with jax.profiler.StepTraceAnnotation("train_step", step_num=i + 1):
            state_fsdp, metrics_fsdp = train_step_fsdp_fn(state_fsdp, metrics_fsdp, batch)
    jax.profiler.stop_trace()
    final_metrics_fsdp = jax.tree.map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes)
    state_fsdp, final_metrics_fsdp = train_step_fsdp_fn(state_fsdp, final_metrics_fsdp, batch)
    print_metrics(final_metrics_fsdp, "FSDP - Final metrics")

    # Compare DP and FSDP metrics
    metrics_dp = jax.device_get(metrics_dp)
    metrics_fsdp = jax.device_get(metrics_fsdp)
    for key in metrics_dp.keys():
        val_dp = metrics_dp[key][0] / metrics_dp[key][1]
        val_fsdp = metrics_fsdp[key][0] / metrics_fsdp[key][1]
        print(f"Metrics DP Avg {key}: {val_dp:.4f}")
        print(f"Metrics FSDP Avg {key}: {val_fsdp:.4f}")
        try:
            np.testing.assert_allclose(val_dp, val_fsdp, atol=1e-2)
        except AssertionError as e:
            print(f"Warning: metric {key} differs between DP and FSDP: {e}")

    # Compare parameters and optimizer state
    params_dp = jax.device_get({"params": state_dp.params, "opt_state": state_dp.opt_state})
    params_fsdp = jax.device_get({"params": state_fsdp.params, "opt_state": state_fsdp.opt_state})
    params_fsdp = jax.tree.map(lambda x: x.value if isinstance(x, nn.Partitioned) else x, params_fsdp, is_leaf=lambda x: isinstance(x, nn.Partitioned))
    try:
        jax.tree.map(lambda x, y: np.testing.assert_allclose(x, y, atol=1e-4), params_dp, params_fsdp)
        print("Parameters match between DP and FSDP")
    except AssertionError as e:
        print("Parameters differ between DP and FSDP:", e)


if __name__ == "__main__":
    main()
