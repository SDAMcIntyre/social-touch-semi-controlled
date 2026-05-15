# Dev Note: Nerve conduction velocity adjustment (`3_cond-velocity-adj`)

**Problem class:** Understanding the origin and generation of the
`3_cond-velocity-adj` nerve data directory, which the merging pipeline
consumes but no current code produces.

| Field | Value |
|-------|-------|
| Scope | Nerve preprocessing (deleted), merging pipeline |
| Data path | `2_processed/nerve/3_cond-velocity-adj/{session_id}/` |
| Sessions | ST13, ST14, ST15, ST16, ST18 (semi-controlled, June 2022) |

---

## 1. What `3_cond-velocity-adj` contains

Each session directory holds CSV files with nerve spike data where the
temporal signal has been shifted to compensate for **neural conduction delay**
-- the time it takes an action potential to travel from the mechanoreceptor
end-organ to the recording electrode.

The shift is computed per unit as:

```
lag_sec = electrode_endorgan_distance_m / conduction_velocity_m_s
```

Columns adjusted: `Nervespike1`, `Freq`.

A summary report (`conduction_velocity_lag_report.csv`) was also generated
alongside the adjusted files.

## 2. Original implementation (deleted)

The script that produced these outputs was:

```
source/3_preprocessing/3.1_preprocess_nerve_conduction_velocity.py
```

**Deleted** in commit `c047112` (2025-08-26) during a major refactoring that
reorganised the pipeline from `source/` into `code/scripts/`.

### Commit history of the script

| Commit | Date | Description |
|--------|------|-------------|
| `9571352` | 2024-07-09 | Created (folder renumbering) |
| `784d461` | 2024-07-16 | Added visualisation of shifting during processing |
| `3a05db6` | 2024-07-31 | Update |
| `f2d698d` | 2025-04-14 | Config variables change |
| `c047112` | 2025-08-26 | **Deleted** in pipeline refactoring |

### How to retrieve the full source

```bash
git show c047112~1:source/3_preprocessing/3.1_preprocess_nerve_conduction_velocity.py
```

### Script logic (summary)

1. Read nerve CSV files from the `2_block-order` directory.
2. Load metadata CSV (`semicontrol_unit-name_to_unit-type.csv`) containing
   per-unit conduction velocity (m/s) and electrode-endorgan distance (cm).
3. Compute lag per unit using `distance / velocity`.
4. Shift `Nervespike1` and `Freq` columns by the computed lag.
5. Write adjusted CSVs to `3_cond-velocity-adj/{session_id}/`.
6. Write `conduction_velocity_lag_report.csv`.

## 3. Related upstream MATLAB scripts (also deleted)

These handled earlier stages of nerve data conversion from `.mat` to CSV:

| File | Purpose | Last relevant commit |
|------|---------|---------------------|
| `source/2_primary_processing/standard_naming_and_formatting/0_processed_nerve_mat2csv.m` | Convert `.mat` nerve files to CSV | `c9a2458` (2024-04-11) |
| `source/2_primary_processing/standard_naming_and_formatting/2.0.4_processed_nerve_mat_correct_date.m` | Correct date info in `.mat` files | `c047112` (2025-08-26) |

## 4. Current state

The conduction velocity adjustment was **re-implemented** in May 2026 as a
standalone Prefect pipeline:

| Artefact | Path |
|----------|------|
| Core function | `code/scripts/_3_preprocessing/_8_nerve_velocity_adjustment/adjust_nerve_conduction_velocity.py` |
| Pipeline script | `code/scripts/preprocess_pipeline_nerve_auto.py` |
| DAG config | `configs/preprocess_pipeline_nerve_auto_dag.yaml` |
| Unit tests | `code/tests/test_nerve_conduction_velocity.py` |

The pipeline is registered in the GUI launcher under **Preprocess → Nerve [Auto]**.

**Metadata CSV** (per-unit conduction velocity and electrode-endorgan distance):
`1_primary/nerve/semicontrol_unit-name_to_unit-type.csv` (relative to `project_data_root`)

**Input path:** `2_processed/nerve/2_block-order/{session_id}/` — derived from
`config.nerve_processed_dir` via:
```python
block_order_dir = config.nerve_processed_dir.parent.parent / "2_block-order" / config.session_id
```

**Output path:** `2_processed/nerve/3_cond-velocity-adj/{session_id}/` (same as
`config.nerve_processed_dir`)

The downstream merging script
(`code/scripts/_4_merging/merge_neural_and_kinect_data.py`) performs
additional temporal alignment (TTL cross-correlation) on top of the
conduction-velocity-adjusted data.

## 5. Implications

- The `3_cond-velocity-adj` outputs can now be **regenerated** by running the
  Nerve [Auto] pipeline from the GUI or directly:
  ```bash
  python code/scripts/preprocess_pipeline_nerve_auto.py --dag-config configs/preprocess_pipeline_nerve_auto_dag.yaml
  ```
- Set `force_processing: true` in the DAG config to regenerate outputs that
  already exist.
- The pipeline is **idempotent**: re-running with `force_processing: false`
  skips sessions whose outputs are already up-to-date.
- The metadata CSV (`semicontrol_unit-name_to_unit-type.csv`) must contain
  `Unit_name`, `conduction_velocity (m/s)`, and `electrode_endorgan_distance (cm)`
  columns for every unit encountered in the input nerve CSV filenames.
