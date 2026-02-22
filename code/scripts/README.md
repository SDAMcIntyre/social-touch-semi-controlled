# code/scripts/

Entry-point scripts for the social-touch Kinect processing pipeline.

## Top-level workflow scripts

Each script corresponds to a pipeline stage and reads its task graph from a
DAG config in `configs/`:

| Script | Purpose |
|--------|---------|
| `primary_workflow_kinect_auto.py` | Stage 2: automated primary video processing (MKV → RGB/depth) |
| `preprocess_workflow_kinect_auto.py` | Stage 3: automated preprocessing (LED, stickers, forearm, hand tracking) |
| `preprocess_workflow_kinect_manual.py` | Stage 3: same as above, triggered manually per session |
| `preprocess_workflow_kinect_visualisation.py` | Stage 3: visualisation pass for preprocessing outputs |
| `preprocess_pipeline_extract_forearm_manual.py` | Stage 3 (standalone): forearm extraction with interactive parameter tuning |
| `preprocess_handmesh_find_scale_factor.py` | Stage 3 (standalone): compute hand mesh scale factor |
| `postprocess_workflow_kinect_auto.py` | Stage 5: postprocessing (receptive fields, XYZ references) |
| `merging_pipeline_neuron_to_kinect_auto.py` | Stage 4: merge neural and Kinect data |
| `analysis_workflow.py` | Analysis: unified summary generation across sessions |
| `view_merged_neural_kinect.py` | Viewer: 3D visualisation of merged neural + Kinect recordings |

## Numbered subdirectory convention

Helper and component scripts are organised into stage-numbered subdirectories
that mirror the pipeline stages:

```
_1_acquisition/              Stage 1  — data acquisition (experiment runners)
_2_primary_processing/       Stage 2  — MKV stream extraction, config preparation
_3_preprocessing/            Stage 3  — tracking and analysis substages:
    _1_sticker_tracking/         Sticker position tracking
    _2_hand_tracking/            Hand mesh assignment (HaMeR)
    _3_forearm_extraction/       Forearm point-cloud extraction
    _4_somatosensory_quantification/
    _5_led_tracking/             LED/TTL signal tracking
    _6_metadata_matching/        Session metadata matching
    _7_unification/              Data unification across modalities
_4_merging/                  Stage 4  — neural + Kinect merge
_5_postprocessing/           Stage 5  — receptive field and XYZ reference computation
```

## `__misc/`

Ad-hoc development and test scripts — not part of the main pipeline. This
directory is gitignored and its contents vary between workstations.

## Dependencies

LED ROI analysis requires `ffprobe` (FFmpeg) on `PATH`. Without it, the LED
tracking subprocess will fail with an error.
