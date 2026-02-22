# Dev Note: CuPy import order with preprocessing packages

**Problem class:** Importing CuPy after the `preprocessing` package tree causes
a `TypeError: Alias 'bool8' was removed in NumPy 2.0` crash inside CuPy's
Cython initialiser.

| Field | Value |
|-------|-------|
| Resolved in | Entry-point scripts and `neural_kinect_scene_viewer.py` |
| Affected versions | CuPy ≤ 14.0.0 + NumPy ≥ 2.0 + Open3D / pyk4a C-extensions |
| Platform | Windows / WSL2 |
| Related bug | [bug-cupy-bool8-import-order.md](bug-cupy-bool8-import-order.md) |

---

## 1. Symptom

```
TypeError: Alias 'bool8' was removed in NumPy 2.0.
  Use a name without a digit at the end.
```

Raised inside `cupy/_core/_dtype.pyx` (`_init_dtype_dict`) when `import cupy`
runs after any module from the `preprocessing` package tree has been imported.
Works fine in a fresh Python session with no preprocessing imports present.

---

## 2. Investigation path

See [bug-cupy-bool8-import-order.md](bug-cupy-bool8-import-order.md) for the full diagnostic sequence.
The key finding was that importing CuPy *before* the preprocessing packages —
and then importing those packages — works reliably.  Once CuPy is cached in
`sys.modules`, subsequent `import cupy` calls in downstream modules return the
cached object and skip the Cython init entirely.

---

## 3. Root Cause

The `preprocessing` packages transitively load compiled C-extensions (Open3D,
pyk4a) at DLL-load time.  These extensions **mutate NumPy's internal dtype
registry** — specifically the aliases that CuPy's Cython init
(`_init_dtype_dict`) references.  If CuPy's Cython init runs after that
mutation it encounters dtype names that no longer exist and raises `TypeError`.

Importing CuPy first (while NumPy's state is pristine) completes `_init_dtype_dict`
successfully.  The module is then cached; the dtype mutation that happens later
(when preprocessing packages load) no longer affects it.

---

## 4. Architecture Constraints

- The mutation is an **undocumented side-effect** of loading Open3D/pyk4a.
  There is no API to suppress or reverse it.
- The constraint applies to **any** entry-point or module that both uses CuPy
  and imports from `preprocessing.*`, regardless of whether it directly touches
  the affected dtype names.
- Lazy importing CuPy (e.g., inside a function body) does **not** help unless
  the function is guaranteed to be called before any preprocessing import.

---

## 5. Fix Applied

### Entry-point scripts

Add an early, guarded `import cupy` at the top of the file, **before** any
`preprocessing.*` import.  The `try/except` allows the script to run on
machines without a GPU or CuPy installation.

```python
try:
    import cupy  # noqa: F401  — must precede preprocessing imports
except Exception:
    pass

from preprocessing.some_module import SomeClass  # safe after cupy init
```

### Modules that use CuPy internally

Place the `import cupy` line in the module's third-party import section,
**above** any `from preprocessing...` import:

```python
_CUPY_AVAILABLE: bool = False
try:
    import cupy as _cp
    _cp.array([1.0])          # confirm GPU round-trip succeeds
    _CUPY_AVAILABLE = True
    del _cp
except Exception as _exc:
    print(f"CuPy not available ({type(_exc).__name__}: {_exc})")
    del _exc
```

---

## 6. Reusable Pattern

Use this checklist for any new script or module that requires both CuPy and
the `preprocessing` package tree.

- [ ] **Import CuPy (or attempt it) before any `preprocessing.*` import** —
  even if CuPy is only used deep inside the module.
- [ ] **Wrap the import in `try/except Exception`** so the script degrades
  gracefully on CPU-only machines.
- [ ] **For entry-point scripts:** place the guarded `import cupy` block near
  the top, as the first substantive statement after stdlib imports.
- [ ] **For library modules:** place the guarded `import cupy` in the
  third-party imports section, above internal `preprocessing.*` imports.
- [ ] **Do not lazy-import CuPy** inside a function or class method unless you
  can guarantee that function is called before any preprocessing import.
- [ ] Add a `# noqa: F401` comment if the early import is unused at the
  module level (to silence linter warnings about unused imports).

---

## 7. References

| Document | Location |
|----------|----------|
| Bug report (investigation + fix history) | [bug-cupy-bool8-import-order.md](bug-cupy-bool8-import-order.md) |
| Entry-point example | `code/scripts/view_merged_neural_kinect.py` |
| Module-level example | `code/src/preprocessing/common/gui/neural_kinect_scene_viewer.py` |
