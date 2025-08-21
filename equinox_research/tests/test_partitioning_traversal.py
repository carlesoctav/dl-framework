import equinox as eqx
import jax.random as jrandom


class TwoLayer(eqx.Module):
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear

    def __init__(self, key):
        k1, k2 = jrandom.split(key)
        self.l1 = eqx.nn.Linear(8, 8, key=k1)
        self.l2 = eqx.nn.Linear(8, 8, key=k2)

    def __call__(self, x):
        return self.l2(self.l1(x))


def test_wrap_leaves_fsdp():
    key = jrandom.PRNGKey(0)
    model = TwoLayer(key)
    eqx.wrap_leaves(model, lambda m: isinstance(m, eqx.nn.Linear), [lambda m: eqx.fsdp_wrap(m, axis="data")])
    specs = eqx.get_partition_specs(model)
    assert specs["l1"]["weight"][0] == "data"
    assert specs["l2"]["weight"][0] == "data"
