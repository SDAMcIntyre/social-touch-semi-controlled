"""Unit tests for DagConfigModel round-trip YAML fidelity."""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from utils.pipeline.dag_config_model import DagConfigModel

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
DAG_FILES = sorted(CONFIGS_DIR.glob("*_dag.yaml"))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _copy_to_tmp(src: Path, tmp_path: Path) -> Path:
    dst = tmp_path / src.name
    shutil.copy2(src, dst)
    return dst


# ------------------------------------------------------------------
# Round-trip fidelity
# ------------------------------------------------------------------


@pytest.mark.parametrize("dag_file", DAG_FILES, ids=lambda p: p.stem)
def test_roundtrip_preserves_comments(dag_file: Path, tmp_path: Path) -> None:
    """Load → save → verify comment lines are preserved."""
    tmp = _copy_to_tmp(dag_file, tmp_path)
    original_text = tmp.read_text()

    model = DagConfigModel(tmp)
    model.save()

    saved_text = tmp.read_text()
    original_comments = [
        ln.strip() for ln in original_text.splitlines() if ln.strip().startswith("#")
    ]
    saved_comments = [
        ln.strip() for ln in saved_text.splitlines() if ln.strip().startswith("#")
    ]
    assert original_comments == saved_comments


@pytest.mark.parametrize("dag_file", DAG_FILES, ids=lambda p: p.stem)
def test_roundtrip_toggle_persists(dag_file: Path, tmp_path: Path) -> None:
    """Load → toggle a task → save → reload → toggle persisted."""
    tmp = _copy_to_tmp(dag_file, tmp_path)
    model = DagConfigModel(tmp)

    tasks = model.get_task_names()
    if not tasks:
        pytest.skip("no tasks in config")

    first = tasks[0]
    original_state = model.is_task_enabled(first)
    model.set_task_enabled(first, not original_state)
    model.save()

    reloaded = DagConfigModel(tmp)
    assert reloaded.is_task_enabled(first) is (not original_state)


# ------------------------------------------------------------------
# Config type detection
# ------------------------------------------------------------------


def test_config_type_kinect() -> None:
    cfg = CONFIGS_DIR / "preprocess_workflow_kinect_auto_dag.yaml"
    if not cfg.exists():
        pytest.skip("config not found")
    model = DagConfigModel(cfg)
    assert model.get_config_type() == "kinect_configs"


def test_config_type_forearm() -> None:
    cfg = CONFIGS_DIR / "preprocess_pipeline_extract_forearm_manual_dag.yaml"
    if not cfg.exists():
        pytest.skip("config not found")
    model = DagConfigModel(cfg)
    assert model.get_config_type() == "forearm_configs"


def test_get_config_entries_single_dir(tmp_path: Path) -> None:
    cfg = CONFIGS_DIR / "preprocess_workflow_kinect_manual_dag.yaml"
    if not cfg.exists():
        pytest.skip("config not found")
    model = DagConfigModel(cfg)
    entries = model.get_config_entries()
    assert len(entries) == 1
    assert isinstance(entries[0], str)


def test_get_config_entries_multi(tmp_path: Path) -> None:
    cfg = CONFIGS_DIR / "analyse_workflow_dag.yaml"
    if not cfg.exists():
        pytest.skip("config not found")
    model = DagConfigModel(cfg)
    entries = model.get_config_entries()
    assert len(entries) > 1


# ------------------------------------------------------------------
# set_config_entries round-trip
# ------------------------------------------------------------------


def test_set_config_entries_single_roundtrip(tmp_path: Path) -> None:
    cfg = CONFIGS_DIR / "preprocess_workflow_kinect_manual_dag.yaml"
    if not cfg.exists():
        pytest.skip("config not found")

    tmp = _copy_to_tmp(cfg, tmp_path)
    model = DagConfigModel(tmp)

    model.set_config_entries(["valid_configs_ST14-01_ST14-02"])
    model.save()

    reloaded = DagConfigModel(tmp)
    assert reloaded.get_config_entries() == ["valid_configs_ST14-01_ST14-02"]


def test_set_config_entries_multi_flow_style(tmp_path: Path) -> None:
    cfg = CONFIGS_DIR / "preprocess_workflow_kinect_manual_dag.yaml"
    if not cfg.exists():
        pytest.skip("config not found")

    tmp = _copy_to_tmp(cfg, tmp_path)
    model = DagConfigModel(tmp)

    model.set_config_entries(["valid_configs_ST13-01", "valid_configs_ST13-02"])
    model.save()

    reloaded = DagConfigModel(tmp)
    assert reloaded.get_config_entries() == ["valid_configs_ST13-01", "valid_configs_ST13-02"]
    # Flow-style list must be on a single line: [item1, item2]
    saved_text = tmp.read_text()
    assert "[valid_configs_ST13-01, valid_configs_ST13-02]" in saved_text


# ------------------------------------------------------------------
# Task ordering
# ------------------------------------------------------------------


@pytest.mark.parametrize("dag_file", DAG_FILES, ids=lambda p: p.stem)
def test_task_names_order_matches_yaml(dag_file: Path) -> None:
    """get_task_names() preserves YAML-defined key order."""
    model = DagConfigModel(dag_file)
    names = model.get_task_names()

    # Also verify order by reading raw YAML keys
    from ruamel.yaml import YAML

    yaml = YAML()
    with open(dag_file) as f:
        raw = yaml.load(f)
    expected = list((raw.get("tasks") or {}).keys())
    assert names == expected


# ------------------------------------------------------------------
# Task options
# ------------------------------------------------------------------


def test_set_task_option_roundtrip(tmp_path: Path) -> None:
    cfg = CONFIGS_DIR / "preprocess_workflow_kinect_auto_dag.yaml"
    if not cfg.exists():
        pytest.skip("config not found")

    tmp = _copy_to_tmp(cfg, tmp_path)
    model = DagConfigModel(tmp)

    tasks = model.get_task_names()
    # Find a task with force_processing option
    target = None
    for t in tasks:
        if "force_processing" in model.get_task_options(t):
            target = t
            break
    if target is None:
        pytest.skip("no task with force_processing")

    original = model.get_task_option(target, "force_processing")
    model.set_task_option(target, "force_processing", not original)
    model.save()

    reloaded = DagConfigModel(tmp)
    assert reloaded.get_task_option(target, "force_processing") is (not original)


# ------------------------------------------------------------------
# Synthetic YAML (no filesystem dependency)
# ------------------------------------------------------------------


def test_model_with_synthetic_yaml(tmp_path: Path) -> None:
    """Verify model works on a minimal synthetic DAG file."""
    content = textwrap.dedent("""\
        # Test DAG
        parameters:
          kinect_configs: "test_dir"

        tasks:
          # Stage 1
          task_a:
            enabled: true
            options:
              force_processing: false
            depends_on: []

          task_b:
            enabled: false
            options:
              force_processing: true
              monitor: false
            depends_on: [task_a]
    """)
    cfg = tmp_path / "test_dag.yaml"
    cfg.write_text(content)

    model = DagConfigModel(cfg)
    assert model.get_config_type() == "kinect_configs"
    assert model.get_config_entries() == ["test_dir"]
    assert model.get_task_names() == ["task_a", "task_b"]
    assert model.is_task_enabled("task_a") is True
    assert model.is_task_enabled("task_b") is False
    assert model.get_task_dependencies("task_b") == ["task_a"]
    assert model.get_task_option("task_b", "monitor") is False

    # Modify and round-trip
    model.set_task_enabled("task_b", True)
    model.set_config_entries(["new_dir"])
    model.save()

    reloaded = DagConfigModel(cfg)
    assert reloaded.is_task_enabled("task_b") is True
    assert reloaded.get_config_entries() == ["new_dir"]

    # Comments preserved
    saved = cfg.read_text()
    assert "# Test DAG" in saved
    assert "# Stage 1" in saved


def test_model_no_parameters_section(tmp_path: Path) -> None:
    """DAG with no parameters section returns 'kinect_configs' type and empty entries."""
    content = textwrap.dedent("""\
        tasks:
          task_x:
            enabled: true
            depends_on: []
    """)
    cfg = tmp_path / "no_params_dag.yaml"
    cfg.write_text(content)

    model = DagConfigModel(cfg)
    assert model.get_config_type() == "kinect_configs"
    assert model.get_config_entries() == []


def test_dirty_flag(tmp_path: Path) -> None:
    content = textwrap.dedent("""\
        parameters:
          kinect_configs: "test_dir"
        tasks:
          t1:
            enabled: true
            depends_on: []
    """)
    cfg = tmp_path / "dirty_dag.yaml"
    cfg.write_text(content)

    model = DagConfigModel(cfg)
    assert model.dirty is False

    model.set_task_enabled("t1", False)
    assert model.dirty is True

    model.save()
    assert model.dirty is False

    model.set_config_entries(["other_dir"])
    assert model.dirty is True

    model.reload()
    assert model.dirty is False


# ------------------------------------------------------------------
# Grid group spec — get / set round-trip
# ------------------------------------------------------------------

_GRID_GROUP_YAML = textwrap.dedent("""\
    parameters:
      kinect_configs: "test_dir"

    tasks:
      cross_map_feature_grid:
        enabled: true
        options:
          force_processing: false
          grid_groups:
            velocity_pressure_2d:
              enabled: true
              neuron_mode: iff
              per_gesture_type: true
              vertex_threshold_ratio: 0.25
              compute_baseline: true
              features:
                hand_velocity_amplitude_mean_during_iff: {min: 0, max: 500, step: 5, span: 10}
                pressure_mean_during_iff: {min: 0.002, max: 0.22, step: 0.02, span: 0.04}
        depends_on: []
""")


def _grid_group_cfg(tmp_path: Path) -> Path:
    cfg = tmp_path / "grid_group_dag.yaml"
    cfg.write_text(_GRID_GROUP_YAML)
    return cfg


def test_get_grid_group_spec_returns_plain_dict(tmp_path: Path) -> None:
    model = DagConfigModel(_grid_group_cfg(tmp_path))
    spec = model.get_grid_group_spec(
        "cross_map_feature_grid", "grid_groups", "velocity_pressure_2d"
    )
    assert isinstance(spec, dict)
    assert spec["enabled"] is True
    assert spec["neuron_mode"] == "iff"
    assert spec["per_gesture_type"] is True
    assert spec["vertex_threshold_ratio"] == 0.25
    assert spec["compute_baseline"] is True
    feats = spec["features"]
    assert "hand_velocity_amplitude_mean_during_iff" in feats
    assert feats["hand_velocity_amplitude_mean_during_iff"]["min"] == 0
    assert feats["hand_velocity_amplitude_mean_during_iff"]["max"] == 500


def test_get_grid_group_spec_missing_task_raises(tmp_path: Path) -> None:
    model = DagConfigModel(_grid_group_cfg(tmp_path))
    with pytest.raises(KeyError, match="no_such_task"):
        model.get_grid_group_spec("no_such_task", "grid_groups", "velocity_pressure_2d")


def test_get_grid_group_spec_missing_opt_key_raises(tmp_path: Path) -> None:
    model = DagConfigModel(_grid_group_cfg(tmp_path))
    with pytest.raises(KeyError, match="no_such_key"):
        model.get_grid_group_spec(
            "cross_map_feature_grid", "no_such_key", "velocity_pressure_2d"
        )


def test_get_grid_group_spec_missing_name_raises(tmp_path: Path) -> None:
    model = DagConfigModel(_grid_group_cfg(tmp_path))
    with pytest.raises(KeyError, match="no_such_group"):
        model.get_grid_group_spec(
            "cross_map_feature_grid", "grid_groups", "no_such_group"
        )


def test_set_grid_group_spec_roundtrip(tmp_path: Path) -> None:
    """set → save → reload → get produces identical scalar values."""
    cfg = _grid_group_cfg(tmp_path)
    model = DagConfigModel(cfg)
    original = model.get_grid_group_spec(
        "cross_map_feature_grid", "grid_groups", "velocity_pressure_2d"
    )

    model.set_grid_group_spec(
        "cross_map_feature_grid", "grid_groups", "velocity_pressure_2d", original
    )
    assert model.dirty is True
    model.save()

    reloaded = DagConfigModel(cfg)
    result = reloaded.get_grid_group_spec(
        "cross_map_feature_grid", "grid_groups", "velocity_pressure_2d"
    )
    assert result["neuron_mode"] == original["neuron_mode"]
    assert result["per_gesture_type"] == original["per_gesture_type"]
    assert result["vertex_threshold_ratio"] == original["vertex_threshold_ratio"]
    assert result["compute_baseline"] == original["compute_baseline"]
    assert set(result["features"].keys()) == set(original["features"].keys())
    for feat, bounds in original["features"].items():
        for k, v in bounds.items():
            assert result["features"][feat][k] == v


def test_set_grid_group_spec_features_are_flow_style(tmp_path: Path) -> None:
    """After set_grid_group_spec, each feature bounds dict uses flow style.

    Flow style means the bounds are rendered with curly braces on the same
    line as the feature key (ruamel.yaml may wrap long lines, but the opening
    brace always appears on the key's line).
    """
    cfg = _grid_group_cfg(tmp_path)
    model = DagConfigModel(cfg)
    spec = model.get_grid_group_spec(
        "cross_map_feature_grid", "grid_groups", "velocity_pressure_2d"
    )
    model.set_grid_group_spec(
        "cross_map_feature_grid", "grid_groups", "velocity_pressure_2d", spec
    )
    model.save()

    saved_text = cfg.read_text()
    for line in saved_text.splitlines():
        if "hand_velocity_amplitude_mean_during_iff:" in line:
            assert "{" in line, f"Expected flow-style opening brace on: {line!r}"
        if "pressure_mean_during_iff:" in line:
            assert "{" in line, f"Expected flow-style opening brace on: {line!r}"


def test_set_grid_group_spec_new_group(tmp_path: Path) -> None:
    """set_grid_group_spec can add a brand-new group name."""
    cfg = _grid_group_cfg(tmp_path)
    model = DagConfigModel(cfg)
    new_spec = {
        "enabled": False,
        "neuron_mode": "sa1",
        "per_gesture_type": False,
        "vertex_threshold_ratio": 0.1,
        "compute_baseline": False,
        "features": {
            "pressure_mean_during_iff": {"min": 0.0, "max": 1.0, "step": 0.05, "span": 0.1},
        },
    }
    model.set_grid_group_spec(
        "cross_map_feature_grid", "grid_groups", "new_group", new_spec
    )
    model.save()

    reloaded = DagConfigModel(cfg)
    result = reloaded.get_grid_group_spec(
        "cross_map_feature_grid", "grid_groups", "new_group"
    )
    assert result["neuron_mode"] == "sa1"
    assert result["features"]["pressure_mean_during_iff"]["max"] == 1.0

    original_still_present = reloaded.get_grid_group_spec(
        "cross_map_feature_grid", "grid_groups", "velocity_pressure_2d"
    )
    assert original_still_present["neuron_mode"] == "iff"


def test_set_grid_group_spec_against_real_config(tmp_path: Path) -> None:
    """Round-trip get → set on the real analyse_workflow_processing_dag.yaml."""
    real_cfg = CONFIGS_DIR / "analyse_workflow_processing_dag.yaml"
    if not real_cfg.exists():
        pytest.skip("analyse_workflow_processing_dag.yaml not found")
    tmp = _copy_to_tmp(real_cfg, tmp_path)
    model = DagConfigModel(tmp)

    task_name = "cross_map_feature_grid"
    if task_name not in model.get_task_names():
        pytest.skip(f"task '{task_name}' not in config")

    groups = model.get_task_option(task_name, "grid_groups") or {}
    if not groups:
        pytest.skip("no grid_groups in real config")

    group_name = next(iter(groups))
    spec = model.get_grid_group_spec(task_name, "grid_groups", group_name)
    model.set_grid_group_spec(task_name, "grid_groups", group_name, spec)
    model.save()

    reloaded = DagConfigModel(tmp)
    result = reloaded.get_grid_group_spec(task_name, "grid_groups", group_name)
    assert result["neuron_mode"] == spec["neuron_mode"]
    assert set(result["features"].keys()) == set(spec["features"].keys())
