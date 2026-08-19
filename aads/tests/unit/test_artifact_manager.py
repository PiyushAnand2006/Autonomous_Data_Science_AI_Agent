"""Tests for aads.agents.artifact_manager — run directory and artifact registration."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from aads.agents.artifact_manager import ArtifactManager, _SUBDIRS, _VIZ_SUBDIRS
from aads.core.exceptions import ArtifactError
from aads.core.schemas import ArtifactType


@pytest.fixture
def artifact_mgr(tmp_path) -> ArtifactManager:
    """ArtifactManager rooted in a temporary directory."""
    return ArtifactManager(storage_root=tmp_path)


@pytest.fixture
def sample_file(tmp_path) -> Path:
    """Create a small CSV file to use as raw data."""
    f = tmp_path / "source_data.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(f, index=False)
    return f


class TestInitializeRun:

    def test_creates_all_subdirs(self, artifact_mgr):
        run_dir = artifact_mgr.initialize_run("test_run_001")
        assert run_dir.exists()
        for key, subdir in _SUBDIRS.items():
            assert (run_dir / subdir).is_dir(), f"Missing subdir: {subdir}"

    def test_creates_viz_subdirs(self, artifact_mgr):
        run_dir = artifact_mgr.initialize_run("test_run_002")
        viz_root = run_dir / _SUBDIRS["visualizations"]
        for sub in _VIZ_SUBDIRS:
            assert (viz_root / sub).is_dir(), f"Missing viz subdir: {sub}"

    def test_duplicate_run_id_raises(self, artifact_mgr):
        artifact_mgr.initialize_run("dup_run")
        mgr2 = ArtifactManager(storage_root=artifact_mgr.storage_root)
        with pytest.raises(ArtifactError, match="already exists"):
            mgr2.initialize_run("dup_run")

    def test_returns_absolute_path(self, artifact_mgr):
        run_dir = artifact_mgr.initialize_run("abs_test")
        assert run_dir.is_absolute()


class TestGetPath:

    def test_valid_key(self, artifact_mgr):
        artifact_mgr.initialize_run("path_test")
        path = artifact_mgr.get_path("raw_data")
        assert path.name == "01_Raw_Data"
        assert path.is_dir()

    def test_invalid_key_raises(self, artifact_mgr):
        artifact_mgr.initialize_run("path_test2")
        with pytest.raises(ArtifactError, match="Unknown artifact key"):
            artifact_mgr.get_path("nonexistent_key")

    def test_not_initialized_raises(self):
        mgr = ArtifactManager()
        with pytest.raises(ArtifactError, match="not initialized"):
            mgr.get_path("raw_data")


class TestCopyRawData:

    def test_copies_file(self, artifact_mgr, sample_file):
        artifact_mgr.initialize_run("copy_test")
        dest = artifact_mgr.copy_raw_data(sample_file)
        assert dest.exists()
        assert dest.name == sample_file.name
        # Verify content matches
        original = pd.read_csv(sample_file)
        copied = pd.read_csv(dest)
        pd.testing.assert_frame_equal(original, copied)

    def test_registers_artifact(self, artifact_mgr, sample_file):
        artifact_mgr.initialize_run("reg_test")
        artifact_mgr.copy_raw_data(sample_file)
        assert len(artifact_mgr.artifacts) == 1
        assert artifact_mgr.artifacts[0].artifact_type == ArtifactType.RAW_DATA

    def test_nonexistent_source_raises(self, artifact_mgr):
        artifact_mgr.initialize_run("no_src_test")
        with pytest.raises(ArtifactError, match="not found"):
            artifact_mgr.copy_raw_data("/nonexistent/file.csv")


class TestRegisterArtifact:

    def test_register_and_retrieve(self, artifact_mgr, tmp_path):
        run_dir = artifact_mgr.initialize_run("register_test")
        # Create a fake artifact file
        model_path = run_dir / _SUBDIRS["models"] / "model.pkl"
        model_path.write_text("fake model")

        record = artifact_mgr.register_artifact(
            artifact_type=ArtifactType.MODEL,
            path=model_path,
            description="Test model",
        )
        assert record.artifact_type == ArtifactType.MODEL
        assert "model.pkl" in record.path

    def test_get_artifacts_by_type(self, artifact_mgr, sample_file):
        artifact_mgr.initialize_run("filter_test")
        artifact_mgr.copy_raw_data(sample_file)

        raw = artifact_mgr.get_artifacts_by_type(ArtifactType.RAW_DATA)
        assert len(raw) == 1

        models = artifact_mgr.get_artifacts_by_type(ArtifactType.MODEL)
        assert len(models) == 0
