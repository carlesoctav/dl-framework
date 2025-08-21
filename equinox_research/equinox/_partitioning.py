from __future__ import annotations

"""Experimental partitioning/transform utilities.

This provides a lightweight metadata system for annotating Equinox ``Module``
instances (initially ``nn.Linear``) with sharding transforms such as FSDP and
Tensor Parallel (column) splits. The goals are:

- Attach declarative per-parameter sharding rules (which dimension maps to which
  mesh axis name) in a composable way.
- Provide a single ``get_partition_specs`` helper that derives JAX
  ``PartitionSpec`` objects for all parameters, in a way that is *order
  independent* with respect to applying multiple transforms (e.g. FSDP then TP
  vs TP then FSDP).
- (Prototype) Optionally materialise gathered parameter views inside the forward
  pass via ``map_variables`` without permanently updating stored parameters.

Runtime collective semantics here are deliberately minimal: only a basic
all-gather-on-dimension-0 is implemented (sufficient for the FSDP-style
"gather params before forward" pattern). Tensor Parallel currently only adds
spec metadata (no specialised runtime behaviour yet).

This is a research scaffold and intentionally narrow in scope. It is not part of
Equinox's public API and may change or be removed without notice.

Example
-------
>>> import equinox as eqx, jax.random as jr
>>> key = jr.PRNGKey(0)
>>> lin = eqx.nn.Linear(8, 16, key=key)
>>> lin = eqx.fsdp_wrap(lin, axis="data")
>>> lin = eqx.tp_column_wrap(lin, axis="tp")
>>> specs = eqx.get_partition_specs(lin)
>>> specs["weight"][0]  # combined axes on dim 0 (order independent)
('data', 'tp')
"""

from dataclasses import dataclass, fields
from typing import Dict, List, Sequence

import jax
from jax.sharding import PartitionSpec

from ._filters import is_array
from ._map import map_variables
from ._module import Module


# ---------------------------------------------------------------------------
# Core metadata types
# ---------------------------------------------------------------------------


@dataclass
class ParamRule:
    """Rule describing how a single parameter (attribute) is sharded.

    Attributes:
      attr: Name of the attribute on the Module (e.g. "weight", "bias").
      dim_axes: Mapping ``param_dim_index -> mesh_axis_name``. Only one axis per
        dim per transform (composition is handled later by accumulation).
      gather: If ``True`` attempt to all-gather the (dim 0) sharded value inside
        the forward pass before usage (non-persistent view).
    """

    attr: str
    dim_axes: Dict[int, str]
    gather: bool = False


@dataclass
class PartitionTransform:
    """A named collection of ``ParamRule`` objects representing one transform.

    Example: an FSDP transform that shards ``weight`` and ``bias`` dim 0 over a
    mesh axis ``fsdp`` and gathers them before the forward pass.
    """

    name: str
    param_rules: Sequence[ParamRule]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_param_spec(shape: Sequence[int], transforms: Sequence[PartitionTransform], attr: str) -> PartitionSpec:
    """Accumulate axis names for each tensor dimension across all transforms.

    Order independence is achieved by sorting the (de-duplicated) axis names per
    dimension lexicographically.
    """
    per_dim_axes: List[List[str]] = [[] for _ in shape]
    for tr in transforms:
        for rule in tr.param_rules:
            if rule.attr == attr:
                for dim, axis in rule.dim_axes.items():
                    if dim < 0 or dim >= len(per_dim_axes):  # defensive
                        continue
                    per_dim_axes[dim].append(axis)
    parts = []
    for axes in per_dim_axes:
        if not axes:
            parts.append(None)
        else:
            # Deduplicate then sort for canonical ordering.
            axes_unique = sorted(dict.fromkeys(axes))  # dict preserves order pre 3.7 spec
            if len(axes_unique) == 1:
                parts.append(axes_unique[0])
            else:
                parts.append(tuple(axes_unique))
    return PartitionSpec(*parts)


def _build_map_in_fn(transforms: Sequence[PartitionTransform]):
    """Create a ``map_in_fn`` applying any gathering rules.

    Currently only supports gathering along dim 0 (common for FSDP-style fully
    sharded parameters) and silently falls back (no-op) if executed outside a
    context providing the named axis (e.g. outside pmap or shard_map environment).
    """

    def map_in_fn(mapped_module):  # mapped_module is a shallow copy with selected leaves
        for tr in transforms:
            for rule in tr.param_rules:
                if not rule.gather:
                    continue
                # Only support dim 0 gather at the moment.
                if 0 not in rule.dim_axes:
                    continue
                axis_name = rule.dim_axes[0]
                if hasattr(mapped_module, rule.attr):
                    value = getattr(mapped_module, rule.attr)
                    if value is None:
                        continue
                    try:
                        gathered = jax.lax.all_gather(value, axis_name=axis_name)
                        # Shape: (axis_size, *value.shape); flatten first two dims.
                        new_shape = (gathered.shape[0] * gathered.shape[1],) + gathered.shape[2:]
                        value = gathered.reshape(new_shape)
                        object.__setattr__(mapped_module, rule.attr, value)
                    except Exception:
                        # Outside of a parallel context; skip (acts as local shard view).
                        pass
        return mapped_module

    return map_in_fn


# ---------------------------------------------------------------------------
# Public transformation application API
# ---------------------------------------------------------------------------


def apply_partition_transform(module: Module, transform: PartitionTransform) -> Module:
    """Applies (adds) a partition transform to a Module instance in-place.

    This re-wraps the instance's class via ``map_variables`` each time to ensure
    the composed ``map_in_fn`` reflects all accumulated transforms.
    """

    if not isinstance(module, Module):  # type: ignore[arg-type]
        raise TypeError("apply_partition_transform expects an Equinox Module instance")

    existing: List[PartitionTransform] = list(getattr(module, "__partition_transforms__", []))
    existing.append(transform)

    base_cls = getattr(module, "__original_base_class__", type(module))
    # Build new wrapper class with composed map_in_fn.
    map_in_fn = _build_map_in_fn(existing)
    Wrapped = map_variables(base_cls, where=is_array, map_in_fn=map_in_fn, mutate=False)

    # Rebind instance to new subclass and attach metadata.
    module.__class__ = Wrapped  # type: ignore[attr-defined]
    module.__partition_transforms__ = existing  # type: ignore[attr-defined]
    module.__original_base_class__ = base_cls  # type: ignore[attr-defined]
    return module


# Convenience creators for common transforms (initial prototypes)

def fsdp_wrap(module: Module, axis: str = "fsdp") -> Module:
    """Shard Linear parameters over ``axis`` along dim 0; gather before forward.

    Currently assumes target module has attributes ``weight`` and optional ``bias``.
    """

    rules = [ParamRule("weight", {0: axis}, gather=True)]
    if hasattr(module, "bias"):
        rules.append(ParamRule("bias", {0: axis}, gather=True))
    tr = PartitionTransform(name=f"fsdp[{axis}]", param_rules=rules)
    return apply_partition_transform(module, tr)


def tp_column_wrap(module: Module, axis: str = "tp") -> Module:
    """Column (output-dimension) tensor parallelism metadata.

    For now this only annotates sharding (dim 0) and does *not* change runtime
    computation semantics (i.e. it still gathers parameters if another transform
    requests it). Set ``gather=False`` so weights remain sharded views unless an
    FSDP transform is also applied.
    """

    rules = [ParamRule("weight", {0: axis}, gather=False)]
    if hasattr(module, "bias"):
        rules.append(ParamRule("bias", {0: axis}, gather=False))
    tr = PartitionTransform(name=f"tp_col[{axis}]", param_rules=rules)
    return apply_partition_transform(module, tr)


# ---------------------------------------------------------------------------
# Spec extraction
# ---------------------------------------------------------------------------


def get_partition_specs(module: Module):
    """Return a nested dict mapping attribute names to ``PartitionSpec``.

    Only array leaves are included. For arrays without any transform-applied
    sharding rules the spec will have ``None`` for every dimension (replicated).
    Non-array / non-Module attributes are omitted for brevity.
    """

    def recurse(obj):
        if isinstance(obj, Module):  # type: ignore[arg-type]
            transforms = getattr(obj, "__partition_transforms__", [])
            result = {}
            base_cls = getattr(obj, "__original_base_class__", type(obj))
            for f in fields(base_cls):  # dataclass fields
                name = f.name
                val = getattr(obj, name)
                if isinstance(val, Module):  # nested module
                    result[name] = recurse(val)
                elif is_array(val):
                    spec = _compute_param_spec(val.shape, transforms, name)
                    result[name] = spec
            return result
        else:
            return {}

    return recurse(module)


# ---------------------------------------------------------------------------
# Traversal utility
# ---------------------------------------------------------------------------


def wrap_leaves(module: Module, predicate, wrappers):
    """Traverse a Module tree and apply wrapper functions to leaves.

    Arguments:
      module: Root module instance (modified in-place).
      predicate: Callable ``(leaf: Module) -> bool``; leaves passing it are wrapped.
      wrappers: Sequence of callables each ``(leaf: Module) -> Module`` applied in order.
    Returns the root module (for chaining).
    """
    if isinstance(module, Module):  # type: ignore[arg-type]
        for name in dir(module):
            # Restrict to dataclass fields only
            if name.startswith("_"):
                continue
            if not hasattr(module, name):
                continue
            try:
                getattr(type(module), name)
            except Exception:
                pass
            val = getattr(module, name)
            if isinstance(val, Module):  # type: ignore[arg-type]
                if predicate(val):
                    for w in wrappers:
                        val = w(val)
                    object.__setattr__(module, name, val)
                else:
                    wrap_leaves(val, predicate, wrappers)
    return module


# ---------------------------------------------------------------------------
# shard_map spec derivation (prototype)
# ---------------------------------------------------------------------------


def get_shard_map_specs(module: Module):
    """Return a dict with parameter specs plus default in/out specs for shard_map.

    Heuristics:
      - Input activation for Linear: if weight dim0 has composite or any axis, we
        assume output activation is sharded the same way along its feature dim.
      - Input vector for Linear assumed replicated (None) unless we see a rule on
        weight dim1 in the future (not yet implemented).

    Returns dict with keys: 'params', 'in_specs', 'out_specs'.
    This is a shallow prototype; multi-layer inference just aggregates params and
    sets in/out to None unless a single Linear root.
    """
    param_specs = get_partition_specs(module)
    in_specs = None
    out_specs = None
    # Simple case: module is a (wrapped) Linear
    base_cls = getattr(module, "__original_base_class__", type(module))
    if base_cls.__name__ == "Linear":
        w_spec = param_specs.get("weight")
        if w_spec is not None:
            # Output spec mirrors weight dim0 partitioning for its single feature dim.
            if w_spec[0] is None:
                out_specs = None
            else:
                out_specs = PartitionSpec(w_spec[0]) if not isinstance(w_spec[0], tuple) else PartitionSpec(w_spec[0])
        in_specs = None  # Currently assume replicated input feature vector
    return {"params": param_specs, "in_specs": in_specs, "out_specs": out_specs}

__all__ = [
    "ParamRule",
    "PartitionTransform",
    "apply_partition_transform",
    "fsdp_wrap",
    "tp_column_wrap",
    "get_partition_specs",
    "wrap_leaves",
    "get_shard_map_specs",
]
