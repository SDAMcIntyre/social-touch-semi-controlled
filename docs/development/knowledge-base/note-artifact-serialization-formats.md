# Dev Note: Choosing a serialization format for a new pipeline artifact

**Problem class:** Every plan that adds a per-vertex or otherwise ragged artifact
faces the same three-way choice — `.npz`, parquet, or extra CSV columns — and
re-derives the same trade-off from scratch. This note records the decision the
repo has actually made, and when to make a different one.

| Field | Value |
|-------|-------|
| Scope | Any new on-disk artifact produced by a pipeline task |
| Incumbent format | `numpy.savez_compressed` (`.npz`) |
| Also in use since 2026-08-12 | parquet, via `pyarrow` |

---

## 1. The repo now has both

Until August 2026 the answer was always `.npz`. `population-rf-vertex-data-export.md`
chose it explicitly and rejected HDF5 as "new dependency, overkill";
`bug-rf-explorer-contact-parsing-scale.md` and
`note-rf-camera-settings-connections.md` both use it.

`contact-depth-field-sidecar.md` introduced **`pyarrow`** as a hard dependency
(`requirements.txt`, and the `pip:` block of `environment.yml`) for
`<video_stem>_contact_depth_field.parquet`. That is the first parquet file in
the repository. `pyproject.toml`'s dependency list is a deliberately minimal
subset — it does not name `open3d` either — and was left alone.

## 2. Which to pick

**Reach for `.npz` when** the artifact is a handful of rectangular arrays
consumed inside this repo. It has no dependency cost and matches everything
already on disk.

**Reach for parquet when at least two of these hold:**

- The consumer is the **separate analytics repo**, not this one. A numpy
  container makes every consumer re-derive the layout; a parquet schema does not.
- The data is **long-form / ragged** — one row per (frame, vertex), where a
  frame's row count varies. `.npz` needs padding or an offsets array; parquet
  needs neither.
- The file must be **self-describing**. Parquet carries schema-level key–value
  metadata (`schema.with_metadata`), which is the correct home for provenance:
  `schema_version`, `coordinate_space`, `units`, `sign_convention`,
  `source_recording`, `produced_by`. Constant-valued *columns* would also work
  and dictionary-encode to near-nothing, but they imply the value could vary per
  row.
- The consumer needs **partial reads** or column filtering without loading the
  whole file.

**Never reach for extra CSV columns** for ragged data. That is what produced the
`contact_points` `[[x y z] ...]` string blob, whose parsing took 15+ minutes on
619k rows and needed two rounds of optimisation
(`bug-rf-explorer-contact-parsing-scale.md`). It is also quantised to `%.1f` by
`serialize_contact_points`, so it cannot carry full precision.

## 3. Two things to get right whichever you pick

**Version the schema from day one.** `population-rf-vertex-data-export.md`
shipped without a `schema_version` and had its file renamed and its schema
widened by a follow-on plan **five days later**. The reader should *raise* on an
unknown version rather than decode on a guess.

**Isolate the format behind a writer/reader pair** in a module that knows
nothing about sessions, configs, DAGs or Prefect — see
`preprocessing/motion_analysis/tactile_quantification/io/contact_depth_field_io.py`.
That module imports its DTO under `TYPE_CHECKING` only, so pure serialisation
never drags a geometry SDK into a caller. If the dependency is later judged
unacceptable, two function bodies change and nothing else does.

## 4. Precision is a schema decision, not a default

`signed_depth_mm` is stored at **float64, not float32**, because the artifact's
strongest correctness check is that `max(|signed_depth_mm|)` recovered per frame
equals the CSV's `contact_depth` *bit-identically*. float32 would silently
degrade that to "approximately" — an uncheckable invariant. The positional
`x/y/z` stay float32 (0.1 µm at millimetre scale). Cost: roughly 8 MB → 12 MB
per recording.

The corollary for tests: any `pd.read_csv` in an exact-equality assertion needs
`float_precision="round_trip"`. The default parser perturbs ~9% of the values in
a real recording by one ULP.

## 5. Idempotency — list the new artifact as an output

A new artifact added to an **existing** task hits the first-run trap: every
already-processed session has a current primary output and no sidecar, so
`should_process_task` would skip forever and the artifact would never appear.

`should_process_task` already accepts `PathInput = Union[Path, List[Path]]`, so
the fix is `output_paths=[primary, sidecar]` — any missing output forces the
rerun, and staleness compares the newest input against the **oldest** output, so
a sidecar written later cannot mask a stale primary. Pair it with
`clean_task_outputs([primary, sidecar])` so a failed run leaves neither.

Be aware this triggers a full reprocessing sweep of every existing session. That
is the intended cost, not an accident — flag it before merging.

---

## Related

- `docs/development/plans/active/contact-depth-field-sidecar.md` — the parquet decision in full
- `docs/development/plans/completed/population-rf-vertex-data-export.md` — the `.npz` precedent
- `docs/development/plans/completed/fix-contact-points-csv-corruption.md` — serialization history
- `docs/development/knowledge-base/bug-rf-explorer-contact-parsing-scale.md` — why columnar
- `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md` — mm throughout
