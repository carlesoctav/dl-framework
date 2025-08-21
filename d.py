from jax import tree_util as jtu
import jax
import equinox as eq
from equinox import nn


class Linear(eq.Module):
    linear1: jax.Array
    linear2: jax.Array
    use_bias: bool = eq.field(static = True)



key = jax.random.key(0)
linear = Linear(
    linear1 = jax.random.normal(key, (10,10)),
    linear2 = jax.random.normal(key, (100, 100)),
    use_bias = False
)


flat, tree_def = jtu.tree_flatten_with_path(linear)
print(f"DEBUGPRINT[87]: d.py:22: flat={type(flat)}")
print(f"DEBUGPRINT[88]: d.py:22: tree_def={type(tree_def)}")
print(f"DEBUGPRINT[81]: d.py:19: flat={len( flat )}")
print(f"DEBUGPRINT[80]: d.py:19: flat={flat}")
print(f"DEBUGPRINT[82]: d.py:20: tree_def={tree_def}")

array, static = eq.partition(linear, eq.is_array, is_leaf = lambda x: isinstance(x, jax.Array))
print(f"DEBUGPRINT[85]: d.py:27: array={type(array)}")
print(f"DEBUGPRINT[86]: d.py:27: static={type(static)}")
print(f"DEBUGPRINT[83]: d.py:25: array={array}")
print(f"DEBUGPRINT[84]: d.py:25: static={static}")

