"""
Tests for AADS Charting and Visualization Tools.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aads.tools.visualization.charts import (
    plot_categorical,
    plot_correlations,
    plot_distributions,
    plot_outliers,
    plot_target_relationships,
)


@pytest.fixture
def sample_viz_df() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "num1": np.random.normal(10, 2, 50),
        "num2": np.random.exponential(5, 50),
        "cat1": np.random.choice(["Alpha", "Beta", "Gamma"], 50),
        "target": np.random.choice([0, 1], 50),
    })


class TestChartingTools:
    """Verify chart generation functions create valid image files."""

    def test_plot_distributions(self, tmp_path, sample_viz_df):
        out_dir = tmp_path / "distributions"
        files = plot_distributions(sample_viz_df, ["num1", "num2"], out_dir)

        assert len(files) == 2
        for f in files:
            p = Path(f)
            assert p.exists()
            assert p.stat().st_size > 0

    def test_plot_categorical(self, tmp_path, sample_viz_df):
        out_dir = tmp_path / "categorical"
        files = plot_categorical(sample_viz_df, ["cat1"], out_dir)

        assert len(files) == 1
        p = Path(files[0])
        assert p.exists()
        assert p.stat().st_size > 0

    def test_plot_correlations(self, tmp_path, sample_viz_df):
        out_dir = tmp_path / "correlations"
        files = plot_correlations(sample_viz_df, ["num1", "num2", "target"], out_dir)

        assert len(files) == 1
        p = Path(files[0])
        assert p.exists()
        assert p.stat().st_size > 0

    def test_plot_outliers(self, tmp_path, sample_viz_df):
        out_dir = tmp_path / "outliers"
        files = plot_outliers(sample_viz_df, ["num1", "num2"], out_dir)

        assert len(files) == 2
        for f in files:
            p = Path(f)
            assert p.exists()
            assert p.stat().st_size > 0

    def test_plot_target_relationships(self, tmp_path, sample_viz_df):
        out_dir = tmp_path / "target_rel"
        files = plot_target_relationships(sample_viz_df, "target", ["num1", "cat1"], out_dir)

        assert len(files) >= 1
        for f in files:
            p = Path(f)
            assert p.exists()
            assert p.stat().st_size > 0
