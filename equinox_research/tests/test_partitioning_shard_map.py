import equinox as eqx
import jax.random as jrandom


def test_get_shard_map_specs_linear():
    key = jrandom.PRNGKey(0)
    lin = eqx.nn.Linear(8, 16, key=key)
    lin = eqx.tp_column_wrap(lin, axis="tp")
    specs = eqx.get_shard_map_specs(lin)
    # params present
    assert "weight" in specs["params"]
    # output spec uses partitioning of weight dim0
    w_spec = specs["params"]["weight"]
    out_spec = specs["out_specs"]
    if w_spec[0] is None:
        assert out_spec is None
    else:
        assert out_spec[0] == w_spec[0]


def test_get_shard_map_specs_order_invariance():
    key = jrandom.PRNGKey(0)
    lin1 = eqx.fsdp_wrap(eqx.tp_column_wrap(eqx.nn.Linear(8, 16, key=key), axis="tp"), axis="fsdp")
    lin2 = eqx.tp_column_wrap(eqx.fsdp_wrap(eqx.nn.Linear(8, 16, key=key), axis="fsdp"), axis="tp")
    specs1 = eqx.get_shard_map_specs(lin1)
    specs2 = eqx.get_shard_map_specs(lin2)
    assert specs1["params"]["weight"] == specs2["params"]["weight"]
    assert specs1["out_specs"] == specs2["out_specs"]
