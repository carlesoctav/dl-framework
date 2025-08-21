from flax import nnx
from flax.nnx import Rngs, initializers 
from flax.training.train_state import TrainState

new_init = nnx.with_partitioning(initializers.normal(), sharding = (None, "data"))


linear = nnx.Linear(10, 10, kernel_init = new_init, rngs = Rngs(0))
print(f"DEBUGPRINT[57]: a.py:7: linear={linear}")
pspec = nnx.get_partition_spec(nnx.state(linear))
print(f"DEBUGPRINT[58]: a.py:9: pspec={pspec}")
