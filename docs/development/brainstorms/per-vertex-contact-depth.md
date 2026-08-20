# Brainstorm: Per-vertex contact depth

**Started:** 2026-08-12   **Last matured:** 2026-08-12   **Status:** Handed off

> Handed off to [`docs/development/plans/pending/per-vertex-contact-depth-poc.md`](../plans/pending/per-vertex-contact-depth-poc.md)
> — a standalone proof of concept scoped to ONE recording with an interactive 3D viewer. The
> Parquet sidecar and postprocessing-chain plumbing described below remain future work.

## Real goal (north star)

Not "a better depth number". The aim is a **spatial weighting kernel for attributing IFF to
contact vertices**. Today every vertex in a contact patch contributes equally to the single IFF
value recorded for that frame, which is physically wrong: for a spherical fingertip the stress
is concentrated at the patch centre. A per-vertex depth field is the most unitary quantity from
which such a weighting can be built — and from which area, max-depth, depth-at-RF and any future
stress model can all be re-derived.

Secondary, downstream of the same field: ask "what was the indentation *at the recorded
afferent's receptive field*?" rather than "what was the deepest indentation anywhere on the arm?"

## Where it stands

Direction chosen: **per-contact-point depth** (option 2 of 3 — see Alternatives). The per-vertex
signed distances already exist in
`code/src/preprocessing/motion_analysis/tactile_quantification/model/objects_interaction_processor.py`
at line ~170 as `tri_distances[inside_mask]`, and are destroyed by a single `np.max` at line 171.
The computation is essentially free; the entire cost of this feature is **plumbing and storage**.

Decided: this payload will **not** go in the existing per-frame CSV.

**Settled design (2026-08-12):**

- **Fork B — self-describing table.** Depth travels *with* its coordinates, not as a slot-keyed
  sidecar. Rejects the cheap slot-keyed variant because `deduplicate_xy_points` reorders/merges
  contact points and would silently misalign depths with no crash — the exact failure class the
  fail-fast rule exists to prevent.
- **Parquet, long/tidy form**, one file per recording: `(frame_index, x, y, z, depth)`, joined to
  the existing per-frame CSV on `frame_index`. Ragged frame sizes need no padding or offsets.
  float32 gives 0.1 µm precision at mm scale. Beats HDF5 (needs CSR offsets, random access not
  needed), NetCDF/xarray (wants rectangular data; padding would need a fill sentinel, awkward
  under no-silent-fallbacks), and NPZ+offsets (fastest to write, worst to maintain).
- **Patch definition unchanged** — keep the existing "all 3 triangle vertices penetrate" mask
  rather than the softer per-vertex `d < 0` criterion, so `contact_area` stays consistent.
- **Store raw signed depth**, not `abs` and not a pre-applied weighting kernel (see Hertz thread).

**Two consequences that fall out of matching the existing patch definition:**

1. The patch vertex set is `np.unique(active_tris)` — which is *already* `unique_contact_indices`,
   the array that populates `contact_points`. So the field is simply
   `distances_np[unique_contact_indices]`: index-aligned with the existing coordinates by
   construction. No new mask, no new geometry, no ambiguity.
2. **Exact regression invariant.** `tri_distances[inside_mask]` is `distances_np[active_tris]`,
   i.e. the same values as the unique set with duplicates from shared triangles; `max` is
   invariant to duplication. Therefore `max(|depth_field|) == existing contact_depth` *exactly*.
   The PoC must assert bit-identical agreement across a whole session.

**Hard requirement derived from the IFF goal:** the weight on vertex *i* is
`w(depth_i) × RF_sensitivity(position_i)` — two fields multiplying. The depth field must therefore
be expressible in the RF-centred frame, so it must survive the full postprocessing chain
(ICP → dedup → projection → PCA → RF-centring). It cannot be a preprocessing-only debug artefact.

## Alternatives on the table

- **(1) Distribution statistics only** — replace the scalar with `depth_max/mean/p95/std`. Free,
  no schema change, but discards all spatial structure. Rejected as the goal: cannot weight
  vertices. Retained as a cheap by-product.
- **(2) Per-contact-point depth, index-aligned with `contact_points`** — CHOSEN. Same computation,
  defers the stable-index problem, fits existing postprocessing plumbing.
- **(3) Depth as a field over forearm-of-reference vertex ids** — persist the global vertex indices
  that `project_contacts_onto_forearm.py:93-104` already computes and discards. Gives a stable
  session-wide index space for cross-trial and cross-neuron spatial maps. Parked as the natural
  successor to (2), not the PoC.

## Threads explored

- **Contact area is not a vertex count** (user's initial premise corrected): it is the summed
  surface area in mm² of forearm triangles whose three vertices all penetrate. KEPT as context.
- **Rigid-body interpenetration ≠ skin indentation.** Real skin deforms and distributes load
  beyond the geometric intersection; the penetration field is sharply truncated at the patch
  boundary in a way real skin never is. Open — decides whether the deliverable is the raw field or
  a contact-mechanics-smoothed field.
- **Hertz vs. geometric overlap profile.** For a sphere of radius R indented by δ, the geometric
  overlap profile is parabolic, `g(r) = δ - r²/2R`, reaching zero at `r = √(2Rδ)`; the Hertzian
  pressure profile is `p(r) = p₀√(1 - r²/a²)` with contact radius `a = √(Rδ)`. These are different
  shapes AND different supports (overlap radius overestimates Hertz contact radius by ~√2, hence
  contact *area* by ~2×). Consequence: **store raw depth, do not bake in a weighting kernel** — the
  depth→weight transform is a scientific choice to be applied post-hoc and revised.
- **`pressure = depth / area`** (analysis repo) is dimensionally 1/mm, not pressure. Likely
  obsoleted or redefined by this field. Parked — lives outside this repo.

## Open questions

Carried into planning — none of these block starting.

- **Dedup reduction rule**: when two contact points collapse to one XY location with different
  depths — max, mean, or keep both? Scientific choice, not technical. Needs a decision before
  `deduplicate_xy_points` can be extended.
- **Scale**: how many frames-with-contact per session, and what is the forearm mesh vertex
  density (or total vertex count of a typical forearm mesh)? At 20k contact frames × 300 points
  Parquet lands around 30–60 MB/session; at 50k × 3000 it approaches 1–2 GB and the chunking
  strategy needs revisiting. Not blocking for a PoC.
- **Does `contact_points` eventually get retired?** The new Parquet table is strictly more
  capable than a stringified `[[x y z] ...]` blob rounded to 0.1 mm in a CSV cell. PoC should
  dual-write and leave the CSV column untouched; note the deprecation path but do not take it now.
- **Rigid-body vs. deformable skin** (from Threads): whether a later iteration applies
  contact-mechanics smoothing so the field is not sharply truncated at the patch rim. Explicitly
  out of scope for the PoC — raw field only.
- **`pressure = depth / area`** in the analysis repo: likely obsoleted or redefined per-vertex.
  Out of this repo's scope; flag to the downstream repo when the field lands.

## Session log

- 2026-08-12 — Located the computation; corrected the "area = vertex count" premise. Established
  the real goal: spatial weighting of vertices for IFF attribution, not a better scalar. Chose
  per-contact-point depth (option 2). Ruled out storing it in the existing CSV. Surfaced the
  Hertz-vs-overlap profile mismatch, which argues for storing raw depth and deferring the
  weighting kernel. Format selection left open.
- 2026-08-12 — Settled the design: Fork B (depth travels with its coordinates), Parquet long-form
  sidecar, existing patch definition retained, signed raw depth. Derived the hard requirement that
  the field must survive the full postprocessing chain (RF-centred frame needed for the
  `w(depth) × RF_sensitivity(position)` product), and the exact regression invariant
  `max(|depth_field|) == contact_depth`. Status → Matured; ready for a plan document.
