# Note: libigl SLIM Python Binding API (nanobind)

**Date:** 2026-05-16
**Status:** Resolved

---

## Symptom

Running `compare_flattening_methods.py` raised:

```
TypeError: __init__(): incompatible function arguments. The following argument types are supported:
    1. __init__(self, arg0: numpy.ndarray[dtype=float64, shape=(*, *), order='F'],
                      arg1: numpy.ndarray[dtype=int32,  shape=(*, *), order='F'], /) -> None
Invoked with types: igl.pyigl_core.SLIMData
```

at the line `data = igl.SLIMData()`.

## Root cause

The installed libigl Python binding is **nanobind**-based (recent libigl) and
its SLIM API differs from the older pybind11 tutorial examples that the
original `flatten_slim` was modelled on:

1. There is no separate `SLIMData` construction step in normal use —
   `igl.slim_precompute(...)` **returns** a `SLIMData` directly.
2. The energy is selected through the `igl.MappingEnergyType` enum
   (member `SYMMETRIC_DIRICHLET`), not a top-level
   `igl.SLIM_ENERGY_TYPE_SYMMETRIC_DIRICHLET` constant.
3. `slim_precompute` takes `(V, F, V_init, slim_energy, b, bc, soft_p)` —
   the `SLIMData` is a return value, not an in-out argument.
4. `slim_solve(data, iter_num)` returns the optimised `V_o`; its column
   count matches `V_init` (so a 2-column `V_init` gives a 2-column UV
   array).

The old `igl.SLIMData()` no-argument pattern matches older tutorial
snippets but not the installed binding, which insists on `SLIMData(V, F)`
when the class **is** constructed directly — though that direct
construction is unnecessary because `slim_precompute` does it internally.

## Resolution

`code/scripts/compare_flattening_methods.py::flatten_slim()` was rewritten
to use the nanobind API:

```python
data = igl.slim_precompute(
    V,
    F,
    uv_init,
    igl.MappingEnergyType.SYMMETRIC_DIRICHLET,
    b,
    bc,
    1e5,           # soft_p
)
for _ in range(n_iter):
    uv = igl.slim_solve(data, 1)
```

Verified by:
- `python -c "import igl; help(igl.slim_precompute)"` — confirmed signature.
- Synthetic 17-vertex disk mesh: SLIM produces a finite (17, 2) UV array
  with the centre vertex pinned at (0, 0) and boundary[0] at (1, 0).

## How to inspect the installed API (for future drift)

```python
import igl
help(igl.slim_precompute)
help(igl.slim_solve)
print(list(igl.MappingEnergyType.__members__))
```

## Related files

- `code/scripts/compare_flattening_methods.py` — `flatten_slim()` function (fixed)
- `docs/development/plans/active/compare-flattening-methods.md` — plan
- libigl tutorial #709: https://libigl.github.io/tutorial/#scalable-locally-injective-maps
