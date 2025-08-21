import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, Optional

import jax
import jax.tree_util as jtu
from jaxtyping import PyTree

from ._filters import partition, combine, is_array, AxisSpec
from ._module import Module


def _identity(x):
    return x


def map_variables(
    module_cls: type[Module],
    where: AxisSpec = is_array,
    map_in_fn: Callable[[PyTree], PyTree] | None = None,
    map_out_fn: Callable[[PyTree], PyTree] | None = None,
    *,
    mutate: bool = False,
    methods: Optional[Sequence[str]] = None,
    allow_traced_mutation: bool = False,
) -> type[Module]:
    """Returns a new Module subclass that maps over (a view of) its variables.

    This is a lightweight analogue of ``flax.linen.map_variables`` for Equinox.

    The returned subclass applies ``map_in_fn`` to a PyTree containing only the
    selected leaves (as specified by ``where``) just before executing each wrapped
    method, and (optionally) ``map_out_fn`` afterwards to persist changes.

    Behaviour:
    - The *input* PyTree passed to ``map_in_fn`` has the same structure as the
      original module, except that unselected leaves are ``None``.
    - ``map_in_fn`` must return a PyTree with identical structure (``None`` must
      remain ``None`` in the unselected positions). The transformed leaves are
      used for the forward call.
    - If ``mutate=False`` (default) then the underlying module parameters are
      left untouched after the call (pure view / masking semantics).
    - If ``mutate=True`` then after the forward pass we call ``map_out_fn`` (or
      the identity if it is ``None``) on the (post-forward) mapped PyTree, and
      write the resulting leaves back in-place onto the *original* instance using
      ``object.__setattr__``. This is an impure side effect and is disallowed
      under JAX transformations unless ``allow_traced_mutation=True``.

    Arguments:
      module_cls: The ``equinox.Module`` subclass to transform. (Pass the *class*,
        not an instance.)
      where: A filter spec (bool or callable) identifying which leaves to expose
        to the mapping functions. Defaults to ``equinox.is_array``.
      map_in_fn: Function applied to the mapped PyTree before the method runs.
        Defaults to identity.
      map_out_fn: Function applied (only if ``mutate=True``) after the method to
        obtain the new values to persist. Defaults to identity.
      mutate: If ``True`` perform in-place persistence of the (mapped) variables
        after each wrapped method call.
      methods: Iterable of method names to wrap. If ``None`` then ``__call__`` is
        wrapped (if present). Each method is wrapped independently but shares the
        same mapping functions.
      allow_traced_mutation: Permit ``mutate=True`` usage under a JAX trace
        (e.g. inside ``jit``). This is unsafe; mutated values may be silently
        ignored across JIT boundaries. Only set this if you know what you are
        doing.

    Returns:
      A new subclass of ``module_cls`` with wrapped methods.
    """

    if not issubclass(module_cls, Module):  # type: ignore[arg-type]
        raise TypeError("`map_variables` expects a subclass of `equinox.Module`.")

    if map_in_fn is None:
        map_in_fn = _identity
    if map_out_fn is None:
        map_out_fn = _identity

    if methods is None:
        if hasattr(module_cls, "__call__"):
            methods = ["__call__"]
        else:
            raise ValueError(
                "`methods` was not provided and the module has no `__call__` method."
            )
    if isinstance(methods, str):  # type: ignore[unreachable]
        methods = [methods]  # pragma: no cover (defensive; Sequence[str] covers list)

    # Validate all named methods exist on the base class.
    for m in methods:
        if not hasattr(module_cls, m):
            raise AttributeError(f"Method '{m}' not found on {module_cls}.")

    def _wrap_method(method_name: str):
        original = getattr(module_cls, method_name)

        def wrapped(self: Module, *args, **kwargs):
            # Extract mapped and unmapped parts.
            mapped, remainder = partition(self, where)
            mapped_in = map_in_fn(mapped)

            # Structural checks.
            if jtu.tree_structure(mapped_in) != jtu.tree_structure(mapped):
                raise ValueError(
                    "`map_in_fn` must return a PyTree with the same structure as its input."
                )

            # Disallow introducing new leaves where previously None.
            introduced = False
            for before, after in zip(
                jtu.tree_leaves(mapped), jtu.tree_leaves(mapped_in)
            ):
                if before is None and after is not None:
                    introduced = True
                    break
            if introduced:
                raise ValueError(
                    "`map_in_fn` produced values for leaves that were not selected by `where`."
                )

            updated_self = combine(mapped_in, remainder)

            # Call underlying original method (unbound function call).
            out = original(updated_self, *args, **kwargs)

            if mutate:
                # Partition again in case forward created new structure (unlikely but safe).
                mapped_after, remainder_after = partition(updated_self, where)
                mapped_out = map_out_fn(mapped_after)
                if jtu.tree_structure(mapped_out) != jtu.tree_structure(mapped_after):
                    raise ValueError(
                        "`map_out_fn` must return a PyTree with the same structure as its input."
                    )
                for before, after in zip(
                    jtu.tree_leaves(mapped_after), jtu.tree_leaves(mapped_out)
                ):
                    if before is None and after is not None:
                        raise ValueError(
                            "`map_out_fn` produced values for leaves not selected by `where`."
                        )
                # JAX trace safety check.
                if not allow_traced_mutation:
                    if any(
                        isinstance(leaf, jax.core.Tracer)
                        for leaf in jtu.tree_leaves(mapped_out)
                        if leaf is not None
                    ):
                        raise RuntimeError(
                            "In-place mutation under a JAX trace is disallowed. Set "
                            "`allow_traced_mutation=True` to override (unsafe)."
                        )
                final_self = combine(mapped_out, remainder_after)
                # Persist dynamic field values back onto the *original* instance.
                for f in dataclasses.fields(type(self)):  # type: ignore[arg-type]
                    if f.name in getattr(self, f.name, {}):  # pragma: no cover (defensive)
                        pass
                    value = getattr(final_self, f.name)
                    object.__setattr__(self, f.name, value)

            return out

        wrapped.__name__ = method_name
        wrapped.__qualname__ = method_name
        wrapped.__doc__ = (
            f"Wrapped `{method_name}` with variable mapping via `equinox.map_variables`."
        )
        return wrapped

    namespace: dict[str, Any] = {m: _wrap_method(m) for m in methods}
    namespace["__map_variables_config__"] = dict(
        where=where,
        map_in_fn=map_in_fn,
        map_out_fn=map_out_fn,
        mutate=mutate,
        methods=tuple(methods),
        allow_traced_mutation=allow_traced_mutation,
        base_class=module_cls,
    )

    new_name = f"MapVariables{module_cls.__name__}"
    subclass = type(new_name, (module_cls,), namespace)
    return subclass  # type: ignore[return-value]
