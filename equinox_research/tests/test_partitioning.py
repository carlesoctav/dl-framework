import equinox as eqx
import jax
import jax.random as jrandom


def test_fsdp_linear_specs():
    key = jrandom.PRNGKey(0)
    lin = eqx.nn.Linear(8, 16, key=key)
    lin = eqx.fsdp_wrap(lin, axis="data")
    specs = eqx.get_partition_specs(lin)
    w_spec = specs["weight"]
    b_spec = specs["bias"]
    assert w_spec[0] == "data" and all(dim is None for dim in w_spec[1:])
    assert b_spec[0] == "data"


def test_tp_fsdp_order_invariance():
    key = jrandom.PRNGKey(0)
    lin1 = eqx.nn.Linear(8, 16, key=key)
    lin1 = eqx.tp_column_wrap(lin1, axis="tp")
    lin1 = eqx.fsdp_wrap(lin1, axis="fsdp")

    lin2 = eqx.nn.Linear(8, 16, key=key)
    lin2 = eqx.fsdp_wrap(lin2, axis="fsdp")
    lin2 = eqx.tp_column_wrap(lin2, axis="tp")

    specs1 = eqx.get_partition_specs(lin1)
    specs2 = eqx.get_partition_specs(lin2)

    assert specs1["weight"] == specs2["weight"]
    # Expect combined axes on dim 0 as tuple sorted lexicographically (fsdp,tp) -> ("fsdp","tp")
    w_spec = specs1["weight"]
    assert isinstance(w_spec[0], tuple)
    assert set(w_spec[0]) == {"fsdp", "tp"}


def test_no_transform_defaults():
    key = jrandom.PRNGKey(0)
    lin = eqx.nn.Linear(4, 4, key=key)
    specs = eqx.get_partition_specs(lin)
    w_spec = specs["weight"]
    assert all(dim is None for dim in w_spec)


def test_gather_exec_outside_parallel():
    key = jrandom.PRNGKey(0)
    lin = eqx.nn.Linear(4, 4, key=key)
    lin = eqx.fsdp_wrap(lin, axis="data")
    x = jax.random.normal(key, (4,))
    # Should execute without failing gather (falls back silently)
    _ = lin(x)
