# Pipeline Capabilities for the Paper

What the analysis pipeline produces that is paper-ready.

---

## Spatial Response Field

**Boundary method (novel):** Gaussian-smoothed population heatmap → Laplacian
zero-crossing → negative-Laplacian flood-fill from peak → marching-squares
contour → polygon metrics. Computed on SLIM (Symmetric-Dirichlet) UV
parameterization — a quasi-isometric (angle-and-area preserving) 2D
flattening of the 3D forearm mesh. **No published precedent** for either
inflection-contour boundary extraction or this UV mapping approach in tactile
afferent literature.

### Metrics produced

| Metric | Unit | Source task |
|--------|------|-------------|
| Response field area (3D surface) | mm² | `spatial_extract_boundaries` |
| Response field perimeter (3D) | mm | `spatial_extract_boundaries` |
| Circularity (4π·area/perimeter²) | 0–1 | `spatial_extract_boundaries` |
| PCA major axis | mm | `spatial_extract_boundaries` |
| PCA minor axis | mm | `spatial_extract_boundaries` |
| PCA aspect ratio | unitless | derived (major/minor) |
| PCA orientation | degrees | `spatial_extract_boundaries` |
| Centroid location (UV + 3D) | mm | `spatial_extract_boundaries` |
| Hotspot / peak location (UV + 3D) | mm | `spatial_extract_boundaries` |
| Mean IFF on boundary contour | Hz | `spatial_extract_boundaries` |
| Centroid shift proximal vs. distal | mm | `spatial_compare_rf_centers` |
| Hotspot shift proximal vs. distal | mm | `spatial_compare_rf_centers` |
| Cross-session boundary overlays | PNG | `spatial_compare_boundaries` |
| Session × gesture metric heatmaps | PNG | `spatial_compare_boundaries` |

All computed per gesture subset: all, tap, stroke, stroke_proximal, stroke_distal.

### Output files

- `spatial_compare_boundaries/iff_mean/session_rf_boundary_summary.csv`
- `spatial_compare_rf_centers/iff_mean/rf_center_proximal_distal_summary.csv`
- `spatial_extract_boundaries/{session_id}/{session_id}_population_response_fields.npz`

---

## Stimulus-Response Tuning

### Tuning curves (`stimulus_iff_tuning_curves`)

- Binned sliding-window tuning: feature bins → mean IFF ± STD + touch count
- Raw-dots + polynomial fit: individual touches + R², CI bands
- Configured features: `contact_area_mean`, `pressure_mean`,
  `hand_velocity_amplitude_mean`, `contact_depth_mean`
- Response metrics: mean IFF, max IFF, spike count per touch
- Per-session CSVs + cross-session overlay PNGs (by session, by neuron type)

### Instruction tuning (`stimulus_iff_instruction_tuning`)

- Designed metadata categories: `speed_metadata`, `force_metadata`,
  `contact_area_metadata`
- Bar charts of IFF per instruction level, per session
- Cross-session overlays with jittered dots

### Radar / spider plots (`stimulus_render_radar`)

- Multi-feature gesture profiles: median + IQR per feature per gesture
- Session-normalized and globally-normalized variants
- Enabled radar features: contact_area, contact_depth, hand_velocity_amplitude
  (pressure and mechanics-of-solids features are extracted elsewhere but are
  NOT in the enabled radar config group)

### Cross-session distributions (`stimulus_compare_sessions`)

- Box-strip / violin / bar-error per session per feature per gesture
- Pooled feature summary CSV across all sessions

---

## Cross-Domain (Spatial × Stimulus)

### 2D Feature grids (`cross_map_feature_grid`) — NOT FEASIBLE

- Configured velocity×depth grid = 10,000 cells → per session ≈ 295/10,000 filled → ~97% empty (per session, ≤1 touch/cell)
- Configured velocity×pressure grid = 1,100 cells → per session ≈ 295/1,100 filled → ~73% empty (per session, ≤1 touch/cell)
- Root cause: median 295 touches/session; need 5–10 per cell minimum
- **Verdict: not feasible for this dataset**

### 1D Velocity grid (`cross_render_sessions`) — WORKS

- 20 velocity bins (step = 25 mm/s), already enabled and functional
- Produces session × velocity × gesture heatmaps
- **Recommended for the paper**

### Alternative: coarse 2D for largest sessions only

- ST16-05 (1,983 touches) and ST18-01 (1,836 touches) could support a
  very coarse 5×5 = 25-cell grid with ~40–80 touches/cell
- Would be supplementary, not main analysis
