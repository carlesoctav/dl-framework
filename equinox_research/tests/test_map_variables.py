import equinox as eqx
import jax
import jax.numpy as jnp


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))

    def __call__(self, x):
        return self.weight @ x + self.bias


def test_map_variables_pure():
    key = jax.random.key(0)
    lin = Linear(3, 3, key)
    orig = lin.weight

    def zero_out(mapped):
        # mapped keeps structure; weight selected, bias selected (arrays)
        return jax.tree_util.tree_map(lambda leaf: jnp.zeros_like(leaf) if leaf is not None else None, mapped)

    ZLinear = eqx.map_variables(Linear, where=eqx.is_array, map_in_fn=zero_out, mutate=False)
    zlin = ZLinear(3, 3, key)
    out = zlin(jnp.ones((3,)))
    # Parameters should remain unchanged (pure)
    assert jnp.allclose(zlin.weight, orig)
    assert zlin is not lin  # new instance; sanity
    assert out.shape == (3,)


def test_map_variables_mutate():
    key = jax.random.key(1)
    lin = Linear(2, 2, key)
    def scale_up(mapped):
        return jax.tree_util.tree_map(lambda leaf: leaf * 2 if leaf is not None else None, mapped)

    MLinear = eqx.map_variables(Linear, map_in_fn=scale_up, map_out_fn=scale_up, mutate=True, allow_traced_mutation=True)
    mlin = MLinear(2, 2, key)
    before = mlin.weight.copy()
    _ = mlin(jnp.ones((2,)))
    # After call weights should have been scaled twice (in then out) => *4
    assert jnp.allclose(mlin.weight, before * 4)


def test_map_variables_methods():
    class TwoCalls(eqx.Module):
        w: jax.Array
        def __init__(self, key):
            self.w = jax.random.normal(key, (3,))
        def first(self, x):
            return self.w + x
        def second(self, x):
            return self.w * x
    key = jax.random.key(2)
    def negate(mapped):
        return jax.tree_util.tree_map(lambda leaf: -leaf if leaf is not None else None, mapped)
    MT = eqx.map_variables(TwoCalls, map_in_fn=negate, methods=("first", "second"))
    inst = MT(key)
    x = jnp.ones((3,))
    out1 = inst.first(x)
    out2 = inst.second(x)
    # Original w should be untouched
    assert jnp.allclose(out1, -inst.w + x)
    assert jnp.allclose(out2, -inst.w * x)
