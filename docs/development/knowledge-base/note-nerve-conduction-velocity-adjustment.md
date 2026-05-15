# Dev Note: Full nerve preprocessing pipeline (deleted + re-implemented)

**Problem class:** Understanding the origin and generation of the intermediate
nerve data directories (`1_csv_files/`, `2_block-order/`, `3_cond-velocity-adj/`),
which the merging pipeline depends on but whose producing scripts were deleted
in a pipeline refactoring. All four deleted scripts have since been
re-implemented in Python as a unified "Nerve [Auto]" pipeline.

| Field | Value |
|-------|-------|
| Scope | Full nerve preprocessing pipeline (deleted + re-implemented) |
| Data paths | `2_processed/nerve/0_matlab_files/` → `1_csv_files/` → `2_block-order/` → `3_cond-velocity-adj/` |
| Sessions | ST13, ST14, ST15, ST16, ST18 (semi-controlled, June 2022) |

---

## 1. Full pipeline chain

The nerve data flows through four sequential directories before it reaches the
merging pipeline:

```
0_matlab_files/         Raw MATLAB recordings (.mat per unit per session)
0_files_no-TTL/         Baseline nerve recordings without TTL sync (reference data,
                        not processed by the pipeline — used for offline comparison)
1_csv_files/            Per-block CSVs extracted from .mat files, organised in
                        {date}_{UnitName}/ session subfolders
2_block-order/          CSVs renamed to kinect block-order convention:
                        {session_id}_semicontrolled_block-order{NN}_nerve.csv
3_cond-velocity-adj/    CSVs with spike timing shifted for conduction delay
                        (this is the direct input to the merging pipeline)
```

The `3_cond-velocity-adj/` directory holds CSV files where the temporal signal
has been shifted to compensate for **neural conduction delay** — the time it
takes an action potential to travel from the mechanoreceptor end-organ to the
recording electrode.

The shift is computed per unit as:

```
lag_sec = electrode_endorgan_distance_m / conduction_velocity_m_s
```

Columns adjusted: `Nervespike1`, `Freq`.

A summary report (`conduction_velocity_lag_report.csv`) is generated alongside
the adjusted files.

## 2. Full deleted pipeline chain

All four scripts that produced these directories were **deleted** in commit
`c047112` (2025-08-26) during a major refactoring that reorganised the pipeline
from `source/` into `code/scripts/`.

### Deleted scripts

| Script | Purpose | Deleted in |
|--------|---------|------------|
| `source/2_primary_processing/standard_naming_and_formatting/2.0.4_processed_nerve_mat_correct_date.m` | Correct date metadata in .mat files (ST14-01, ST14-02: −1 day; ST16-02: +2 days) | `c047112` |
| `source/2_primary_processing/standard_naming_and_formatting/2.0.5_processed_nerve_mat2csv.m` | Extract per-block tables from .mat structs → CSVs in `1_csv_files/` | `c047112` |
| `source/2_primary_processing/standard_naming_and_formatting/2.0.6_processed_nerve_standardize_filename_with_TTLdata.py` | Rename CSVs from mat-stem naming to block-order convention → `2_block-order/` | `c047112` |
| `source/3_preprocessing/3.1_preprocess_nerve_conduction_velocity.py` | Shift spike timing by conduction delay → `3_cond-velocity-adj/` | `c047112` |

### Commit history of the NCV script (most evolved)

| Commit | Date | Description |
|--------|------|-------------|
| `9571352` | 2024-07-09 | Created (folder renumbering) |
| `784d461` | 2024-07-16 | Added visualisation of shifting during processing |
| `3a05db6` | 2024-07-31 | Update |
| `f2d698d` | 2025-04-14 | Config variables change |
| `c047112` | 2025-08-26 | **Deleted** in pipeline refactoring |

### Recovery commands

Retrieve any deleted script from the commit immediately before the refactoring:

```bash
git show c047112~1:source/2_primary_processing/2_standard_naming_and_formatting/2.0.4_processed_nerve_mat_correct_date.m
git show c047112~1:source/2_primary_processing/2_standard_naming_and_formatting/2.0.5_processed_nerve_mat2csv.m
git show c047112~1:source/2_primary_processing/2_standard_naming_and_formatting/2.0.6_processed_nerve_standardize_filename_with_TTLdata.py
git show c047112~1:source/3_preprocessing/3.1_preprocess_nerve_conduction_velocity.py
```

### Script logic summaries

**2.0.4 — Date correction (MATLAB, in-memory):**
Applied per-session date offsets to the `YYYYMMDD` field before writing CSVs.
Known corrections: ST14-01: −1 day; ST14-02: −1 day; ST16-01: −1 day (present
in original script, not confirmed in data); ST16-02: +2 days.

**2.0.5 — Mat-to-CSV extraction (MATLAB):**
1. Load `.mat` file via `load()`.
2. Extract top-level struct fields: `Exp`, `Unit`, `Zoom`, `UnitName`,
   `UnitNumber`, `Stimulus`.
3. Skip if `Stimulus != "Semi_contr"`.
4. For each block `b` in `S.FullPeriod_D.ContD`: extract table `D`, write CSV
   `{matname}_block{b}_table.csv` into `{date}_{UnitName}/`.
5. Write `{matname}_metadata.txt`.

**2.0.6 — Block-order rename (Python):**
1. Load quality-check xlsx (`semicontrolled_data-collection_quality-check.xlsx`).
2. For each `*_table.csv` in `1_csv_files/{session_id}/`: extract `zoom_id` and
   `block_id` from filename, look up `block_order` via `(zoom_id, block_id)` key.
3. Copy CSV to `2_block-order/{session_id}/{session_id}_semicontrolled_block-order{NN}_nerve.csv`.

**3.1 — Conduction velocity adjustment (Python):**
1. Read nerve CSV files from `2_block-order/`.
2. Load metadata CSV (`semicontrol_unit-name_to_unit-type.csv`) with per-unit
   conduction velocity (m/s) and electrode-endorgan distance (cm).
3. Compute `lag_sec = distance / velocity` per unit.
4. Shift `Nervespike1` and `Freq` columns by the computed lag.
5. Write adjusted CSVs to `3_cond-velocity-adj/{session_id}/`.
6. Write `conduction_velocity_lag_report.csv`.

## 3. Pipeline diagram

```
0_matlab_files/           (.mat per unit)
    │  [MATLAB 2.0.4 → date correction (in-memory)]
    │  [MATLAB 2.0.5 → table extraction]
    ▼
1_csv_files/              ({date}_{UnitName}/{mat_stem}_block{N}_table.csv)
    │  [Python 2.0.6 → block-order rename using quality-check xlsx]
    ▼
2_block-order/            ({session_id}_semicontrolled_block-order{NN}_nerve.csv)
    │  [Python 3.1 → conduction velocity adjustment]
    ▼
3_cond-velocity-adj/      (same naming, spike timing shifted)
    │  [merge_neural_and_kinect_data.py → TTL cross-correlation]
    ▼
3_merged/                 (neural + kinect data aligned)
```

`0_files_no-TTL/` sits alongside `0_matlab_files/` and holds baseline nerve
recordings acquired without TTL synchronisation. It is reference data only and
is not processed by the pipeline.

## 4. Current state

All four deleted scripts were **re-implemented** in May 2026 as a unified
"Nerve [Auto]" Prefect pipeline covering the full chain from raw `.mat` files
to conduction-velocity-adjusted CSVs.

### Re-implemented stages

| Stage | Core function | Module |
|-------|--------------|--------|
| Mat → CSV (date correction + extraction) | `convert_nerve_mat_to_csv()` | `code/scripts/_3_preprocessing/_9_nerve_data_extraction/` |
| CSV → block-order rename | `rename_nerve_to_block_order()` | `code/scripts/_3_preprocessing/_9_nerve_data_extraction/` |
| Block-order → conduction velocity adj | `adjust_nerve_conduction_velocity()` | `code/scripts/_3_preprocessing/_8_nerve_velocity_adjustment/` |

### Shared pipeline artefacts

| Artefact | Path |
|----------|------|
| Pipeline script | `code/scripts/preprocess_pipeline_nerve_auto.py` |
| DAG config | `configs/preprocess_pipeline_nerve_auto_dag.yaml` |
| Unit tests (mat2csv + rename) | `code/tests/test_nerve_data_extraction.py` |
| Unit tests (NCV) | `code/tests/test_nerve_conduction_velocity.py` |
| GUI launcher entry | **Preprocess → Nerve [Auto]** |

### Key data references

**xlsx lookup** (used by `rename_nerve_to_block_order`):
- File: `semicontrolled_data-collection_quality-check.xlsx` (relative to `project_data_root`)
- Key columns: `Zoom` (int), `Zoom Block ID` (int), `Block order` (int)

**Metadata CSV** (used by `adjust_nerve_conduction_velocity`):
- File: `1_primary/nerve/semicontrol_unit-name_to_unit-type.csv` (relative to `project_data_root`)
- Required columns: `Unit_name`, `conduction_velocity (m/s)`, `electrode_endorgan_distance (cm)`

**Date corrections** (hardcoded in `convert_nerve_mat_to_csv`):
- ST14-01: −1 day
- ST14-02: −1 day
- ST16-01: −1 day (present in original script, not confirmed in data)
- ST16-02: +2 days

**Input path** for NCV step: `2_processed/nerve/2_block-order/{session_id}/` — derived from
`config.nerve_processed_dir` via:
```python
block_order_dir = config.nerve_processed_dir.parent.parent / "2_block-order" / config.session_id
```

**Output path** for NCV step: `2_processed/nerve/3_cond-velocity-adj/{session_id}/`
(same as `config.nerve_processed_dir`)

The downstream merging script
(`code/scripts/_4_merging/merge_neural_and_kinect_data.py`) performs
additional temporal alignment (TTL cross-correlation) on top of the
conduction-velocity-adjusted data.

## 5. Implications

- The full pipeline from raw `.mat` files to `3_cond-velocity-adj/` can now be
  **regenerated** by running the Nerve [Auto] pipeline from the GUI or directly:
  ```bash
  python code/scripts/preprocess_pipeline_nerve_auto.py --dag-config configs/preprocess_pipeline_nerve_auto_dag.yaml
  ```
- All three tasks are wired sequentially in a single pipeline run:
  `convert_mat_to_csv` → `rename_to_block_order` → `adjust_conduction_velocity`.
- Users can **disable upstream tasks** if intermediate artefacts already exist
  (e.g. disable `convert_mat_to_csv` and `rename_to_block_order` to run only
  the conduction velocity adjustment on pre-existing `2_block-order/` files).
- Set `force_processing: true` per-task independently in the DAG config to
  regenerate outputs that already exist.
- The pipeline is **idempotent**: re-running with `force_processing: false`
  skips sessions whose outputs are already up-to-date.
