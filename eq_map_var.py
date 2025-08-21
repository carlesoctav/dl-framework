import functools as ft
import types
from collections.abc import Callable
from typing import Any

import equinox as eq


Pytree = Any

def map_variables(
    cls: type[eq.Module] | eq.Module,
    map_in_fn, 
    map_out_fn = None,
    map_name: str = "",
    *,
    where = eq.is_array,
    mutate = False,
    methods = ("__call__",)
)-> type[eq.Module]:
    if not isinstance(cls, type) or not issubclass(cls, eq.Module):
        raise TypeError("First argument must be a subclass of Module")
    if mutate and map_out_fn is None:
        map_out_fn = lambda x: x

    def _wrap_method(name, orig_fn):
        @ft.wraps(orig_fn)
        def wrapped(self, *args, **kwargs):
            to_map, others = eq.partition(self, where) 
            mapped = map_in_fn(to_map) 
            mapped_self = eq.combine(mapped, others)
            output = orig(mapped_self, *args, **kwargs)
            if mutate:
                mapped_after = map_out_fn(mapped) 
                return output, eq.combine(mapped_after, others) 
            return output
        return wrapped

    dct = {}
    for m in methods:
        if not hasattr(cls, m):
            raise ValueError(f"{cls.__name__} doesn't have {m} method")
        orig = getattr(cls, m)

        if isinstance(orig, types.FunctionType | types.MethodType):
            func = orig.__func__ if isinstance(orig, types.MethodType) else orig
        else:
            raise TypeError(f"{m} is not types.MethodType or types.FunctionType")

        dct[m] = _wrap_method(m, func)


    New = type(f"{cls.__name__}{map_name}", (cls,), dct)
    New.__doc__ = (cls.__doc__ or "") + f"\n\n[Wrapped for {map_name} by map_variables]"
    return New

