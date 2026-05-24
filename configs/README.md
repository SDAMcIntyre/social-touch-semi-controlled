# configs/

YAML configuration files for the social-touch Kinect processing pipeline.

## Directory structure

```
configs/
├── *_dag.yaml              # Root-level pipeline DAGs (tracked)
├── _dag_templates/         # Template DAG files for generating new configs
├── forearm_configs/        # Per-session forearm extraction parameters
└── kinect_configs/         # Per-recording-block Kinect acquisition configs
```

## Root-level DAG files (tracked)

Each root-level `*_dag.yaml` file defines a task graph for one workflow: which
processing steps are enabled, their options, and their `depends_on` dependencies.
They map directly to the top-level workflow scripts in `code/scripts/`:

| Config file | Workflow script |
|-------------|----------------|
| `primary_workflow_kinect_auto_dag.yaml` | `primary_workflow_kinect_auto.py` |
| `preprocess_workflow_kinect_auto_dag.yaml` | `preprocess_workflow_kinect_auto.py` |
| `preprocess_workflow_kinect_manual_dag.yaml` | `preprocess_workflow_kinect_manual.py` |
| `preprocess_workflow_kinect_visualisation_dag.yaml` | `preprocess_workflow_kinect_visualisation.py` |
| `postprocess_workflow_kinect_auto_dag.yaml` | `postprocess_workflow_kinect_auto.py` |
| `merging_pipeline_neuron_to_kinect_auto_dag.yaml` | `merging_pipeline_neuron_to_kinect_auto.py` |
| `analyse_workflow_processing_dag.yaml` | `analysis_workflow_processing.py` |
| `analyse_workflow_viewers_dag.yaml` | `analysis_workflow_viewers.py` |

## `_dag_templates/`

Parameterised templates used to generate session-specific DAG configs
programmatically. Not intended to be run directly.

## `forearm_configs/`

One YAML per recording session containing forearm extraction parameters (data
paths, ROI settings). Files are named `session_<date>_<id>.yaml`.

## `kinect_configs/`

Per-recording-block Kinect acquisition configs, named
`kinect_config_<date>_<id>_..._block-order<N>.yaml`. Files are grouped into
subdirectories by session subset (e.g. `valid_configs_ST13-01/`, `all_configs/`,
`tmp_specific/`) to make it easy to point a workflow at a specific batch.

## `.gitignore` rules

Only root-level YAML files are tracked. YAML files inside subdirectories are
gitignored because they contain machine-specific absolute paths and are
generated or hand-crafted per researcher workstation:

```gitignore
configs/**/*.yaml   # ignore per-session/block configs in subdirectories
!configs/*.yaml     # except root-level DAG files
```

This means `forearm_configs/*.yaml` and `kinect_configs/**/*.yaml` are **not**
under version control. They must be created locally before running a workflow.

## Config loading

Pipeline DAG files are loaded by `utils.DagConfigHandler`
(`code/src/utils/pipeline/`). Kinect block configs are loaded by
`primary_processing.KinectConfigFileHandler`.
