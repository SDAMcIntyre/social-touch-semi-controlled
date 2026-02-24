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
# Kinect directory mode detection
# ------------------------------------------------------------------


def test_singular_kinect_dir_mode() -> None:
    cfg = CONFIGS_DIR / "preprocess_workflow_kinect_auto_dag.yaml"
    if not cfg.exists():
        pytest.skip("config not found")
    model = DagConfigModel(cfg)
    assert model.get_kinect_dir_mode() == "single"


def test_plural_kinect_dir_mode() -> None:
    cfg = CONFIGS_DIR / "analyse_workflow_dag.yaml"
    if not cfg.exists():
        pytest.skip("config not found")
    model = DagConfigModel(cfg)
    assert model.get_kinect_dir_mode() == "multi"


def test_get_kinect_directories_singular(tmp_path: Path) -> None:
    cfg = CONFIGS_DIR / "preprocess_workflow_kinect_auto_dag.yaml"
    if not cfg.exists():
        pytest.skip("config not found")
    model = DagConfigModel(cfg)
    dirs = model.get_kinect_directories()
    assert len(dirs) == 1
    assert isinstance(dirs[0], str)


def test_get_kinect_directories_plural(tmp_path: Path) -> None:
    cfg = CONFIGS_DIR / "analyse_workflow_dag.yaml"
    if not cfg.exists():
        pytest.skip("config not found")
    model = DagConfigModel(cfg)
    dirs = model.get_kinect_directories()
    assert len(dirs) > 1


# ------------------------------------------------------------------
# Exclude files
# ------------------------------------------------------------------


def test_set_exclude_files_roundtrip(tmp_path: Path) -> None:
    cfg = CONFIGS_DIR / "preprocess_workflow_kinect_auto_dag.yaml"
    if not cfg.exists():
        pytest.skip("config not found")

    tmp = _copy_to_tmp(cfg, tmp_path)
    model = DagConfigModel(tmp)

    model.set_exclude_files(["a.yaml", "b.yaml"])
    model.save()

    reloaded = DagConfigModel(tmp)
    assert reloaded.get_exclude_files() == ["a.yaml", "b.yaml"]


def test_set_exclude_files_empty_removes_key(tmp_path: Path) -> None:
    cfg = CONFIGS_DIR / "preprocess_workflow_kinect_auto_dag.yaml"
    if not cfg.exists():
        pytest.skip("config not found")

    tmp = _copy_to_tmp(cfg, tmp_path)
    model = DagConfigModel(tmp)

    # Add then remove
    model.set_exclude_files(["x.yaml"])
    model.save()
    model.set_exclude_files([])
    model.save()

    reloaded = DagConfigModel(tmp)
    assert reloaded.get_exclude_files() == []
    # Key should be absent from the YAML
    assert "exclude_files" not in tmp.read_text()


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
          kinect_configs_directory: "kinect_configs/test_dir"

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
    assert model.get_kinect_dir_mode() == "single"
    assert model.get_kinect_directories() == ["kinect_configs/test_dir"]
    assert model.get_task_names() == ["task_a", "task_b"]
    assert model.is_task_enabled("task_a") is True
    assert model.is_task_enabled("task_b") is False
    assert model.get_task_dependencies("task_b") == ["task_a"]
    assert model.get_task_option("task_b", "monitor") is False

    # Modify and round-trip
    model.set_task_enabled("task_b", True)
    model.set_kinect_directories(["kinect_configs/new_dir"])
    model.save()

    reloaded = DagConfigModel(cfg)
    assert reloaded.is_task_enabled("task_b") is True
    assert reloaded.get_kinect_directories() == ["kinect_configs/new_dir"]

    # Comments preserved
    saved = cfg.read_text()
    assert "# Test DAG" in saved
    assert "# Stage 1" in saved


def test_model_no_parameters_section(tmp_path: Path) -> None:
    """DAG with no parameters section returns 'none' mode and empty lists."""
    content = textwrap.dedent("""\
        tasks:
          task_x:
            enabled: true
            depends_on: []
    """)
    cfg = tmp_path / "no_params_dag.yaml"
    cfg.write_text(content)

    model = DagConfigModel(cfg)
    assert model.get_kinect_dir_mode() == "none"
    assert model.get_kinect_directories() == []
    assert model.get_exclude_files() == []


def test_dirty_flag(tmp_path: Path) -> None:
    content = textwrap.dedent("""\
        parameters:
          kinect_configs_directory: "test"
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

    model.set_exclude_files(["x.yaml"])
    assert model.dirty is True

    model.reload()
    assert model.dirty is False
