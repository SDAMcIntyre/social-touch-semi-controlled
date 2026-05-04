# Bug: RF Explorer contact-point parsing hangs on large CSVs

**Date:** 2026-05-04  
**Status:** Fixed (two rounds)  
**Observed:** 15+ minutes hang before the "contact points" print for a 619,868-row CSV  

- Round 1 (2026-05-04): row-wise regex replaces double-join approach — eliminated
  the giant-string bottleneck but introduced 1.86M `np.fromstring` + `np.vstack`
  overhead (still 10+ min at scale).
- Round 2 (2026-05-04): collect raw float strings in a flat list, single
  `np.array(...).reshape(-1, 3)` at the end — eliminates per-row numpy allocation.

---

## Symptom

When `load_explorer_data()` processes a large series-augmented CSV (619k rows,
nearly all with valid contact points), the terminal freezes after:

```
[RF Explorer] [session_stem]: computing (cache miss)...
[RF Explorer] [session_stem]   CSV read: Xs  rows=619868
```

The "contact points" line never appears within a practical timeframe.

---

## Root cause

The contact-parsing block in `rf_explorer_data.py` (around line 240) performs
three serial Python-level string operations on the entire column at once:

```python
combined = " ".join(cp_raw)                              # ~619k strings → ~40–80 MB string
match_strings = re.findall(r'\[([^\[\]]+)\]', combined)  # regex on the giant string
all_pts = np.fromstring(" ".join(match_strings), ...)    # join millions of matches → parse
```

With ~1.86M contact points (619k rows × ~3 pts/row average) this produces:
- A ~50–80 MB `combined` string
- ~1.86M regex matches
- A second `" ".join` on 1.86M short strings before `np.fromstring`

All three steps are O(N) but with large Python-level memory allocation overhead.
The regex engine and the double-join are the wall; NumPy never gets involved until
everything is already in one string.

---

## Scale analysis

Designed-for session size: ~1,800 rows. The 619k-row CSV is ~340× larger.

Original projected pipeline time (pre-fix, hypothesis):

| Step | Original estimate | Post-fix estimate | Notes |
|---|---|---|---|
| Contact parsing | 30–60 min | ~10–30 s | Flat list + single reshape eliminates per-row numpy allocation |
| Rotation + mesh load | < 10 s | < 10 s | Unchanged |
| KDTree query (1.86M queries) | 30–90 min | ~1–5 s | Original estimate was wrong — `cKDTree.query` is C-compiled, k=1 on ~10k verts is fast |
| Cache write (savez_compressed) | 5–15 min | ~5–60 s | Depends on disk; compressed .npz of filtered arrays |
| **Total** | **1–3 hours** | **~30 s – 2 min** | Dominated by CSV read + contact parsing |

The original KDTree estimate (30–90 min) was a hypothesis, not a measurement.
`cKDTree` is SciPy's C-compiled KD-tree; 1.86M nearest-neighbour queries
against ~10k vertices completes in seconds.

---

## Fix applied

### Round 1 — row-wise regex (insufficient)

Replaced the double-join + single-regex with per-row `re.findall` + `np.fromstring`:

```python
row_pts = []
for s in cp_raw:
    for m in _bracket_re.findall(s):
        row_pts.append(np.fromstring(m, sep=' ', dtype=np.float64))
all_pts = np.vstack(row_pts)
```

This eliminated the giant-string bottleneck but created 1.86M tiny numpy arrays
and called `np.vstack` on the full list — still 10+ minutes at 619k-row scale
due to per-call allocation overhead and vstack's internal iteration.

### Round 2 — flat list + single allocation (final fix)

Collect raw float strings in a plain Python list, single numpy allocation at the end:

```python
raw_floats: list[str] = []
for s in cp_raw:
    for m in _bracket_re.findall(s):
        raw_floats.extend(m.split())
all_pts = np.array(raw_floats, dtype=np.float64).reshape(-1, 3)
```

No numpy objects created inside the loop — `str.split()` and `list.extend` are
cheap Python ops. The single `np.array(...)` call at the end handles all
type conversion and memory allocation in one shot.

Progress prints added at ~20% intervals for both the bracket-counting and
contact-parsing loops.

---

## Affected file

`code/src/analysis/receptive_field_mapping/rf_explorer_data.py` — `load_explorer_data()`
