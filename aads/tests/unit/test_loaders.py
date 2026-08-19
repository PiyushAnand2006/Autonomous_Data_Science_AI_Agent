"""Tests for aads.tools.loaders — CSV, Excel, Parquet loaders and registry."""

from pathlib import Path

import pandas as pd
import pytest

from aads.core.exceptions import DataLoadError
from aads.tools.loaders.csv_loader import CSVLoader
from aads.tools.loaders.excel_loader import ExcelLoader
from aads.tools.loaders.parquet_loader import ParquetLoader
from aads.tools.loaders.registry import LoaderRegistry


# ---------------------------------------------------------------------------
# Fixtures — create small test files
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path) -> Path:
    """Create a small CSV test file."""
    csv_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [85.5, 92.0, 78.3],
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_tsv(tmp_path) -> Path:
    """Create a small TSV test file."""
    tsv_path = tmp_path / "test_data.tsv"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_csv(tsv_path, index=False, sep="\t")
    return tsv_path


@pytest.fixture
def sample_excel(tmp_path) -> Path:
    """Create a small Excel test file."""
    xlsx_path = tmp_path / "test_data.xlsx"
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "value": [10.0, 20.0, 30.0],
    })
    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    return xlsx_path


@pytest.fixture
def sample_parquet(tmp_path) -> Path:
    """Create a small Parquet test file."""
    pq_path = tmp_path / "test_data.parquet"
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "category": ["A", "B", "C"],
    })
    df.to_parquet(pq_path, index=False, engine="pyarrow")
    return pq_path


# ---------------------------------------------------------------------------
# CSV Loader
# ---------------------------------------------------------------------------

class TestCSVLoader:

    def test_supported_extensions(self):
        loader = CSVLoader()
        assert ".csv" in loader.supported_extensions()
        assert ".tsv" in loader.supported_extensions()

    def test_supports_csv(self, sample_csv):
        loader = CSVLoader()
        assert loader.supports(sample_csv)

    def test_does_not_support_xlsx(self, sample_excel):
        loader = CSVLoader()
        assert not loader.supports(sample_excel)

    def test_load_csv(self, sample_csv):
        loader = CSVLoader()
        df = loader.load(sample_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "name" in df.columns

    def test_load_tsv(self, sample_tsv):
        loader = CSVLoader()
        df = loader.load(sample_tsv)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    def test_load_nonexistent_raises(self):
        loader = CSVLoader()
        with pytest.raises(DataLoadError, match="not found"):
            loader.load("/nonexistent/path.csv")

    def test_validate_wrong_extension(self, sample_excel):
        loader = CSVLoader()
        with pytest.raises(DataLoadError, match="Unsupported extension"):
            loader.validate(sample_excel)


# ---------------------------------------------------------------------------
# Excel Loader
# ---------------------------------------------------------------------------

class TestExcelLoader:

    def test_supported_extensions(self):
        loader = ExcelLoader()
        assert ".xlsx" in loader.supported_extensions()
        assert ".xls" in loader.supported_extensions()

    def test_load_xlsx(self, sample_excel):
        loader = ExcelLoader()
        df = loader.load(sample_excel)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "value" in df.columns


# ---------------------------------------------------------------------------
# Parquet Loader
# ---------------------------------------------------------------------------

class TestParquetLoader:

    def test_supported_extensions(self):
        loader = ParquetLoader()
        assert ".parquet" in loader.supported_extensions()

    def test_load_parquet(self, sample_parquet):
        loader = ParquetLoader()
        df = loader.load(sample_parquet)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "category" in df.columns


# ---------------------------------------------------------------------------
# Loader Registry
# ---------------------------------------------------------------------------

class TestLoaderRegistry:

    def test_auto_registers_builtins(self):
        registry = LoaderRegistry()
        exts = registry.supported_extensions
        assert ".csv" in exts
        assert ".xlsx" in exts
        assert ".parquet" in exts

    def test_load_csv_via_registry(self, sample_csv):
        registry = LoaderRegistry()
        df = registry.load(sample_csv)
        assert len(df) == 3

    def test_load_excel_via_registry(self, sample_excel):
        registry = LoaderRegistry()
        df = registry.load(sample_excel)
        assert len(df) == 3

    def test_load_parquet_via_registry(self, sample_parquet):
        registry = LoaderRegistry()
        df = registry.load(sample_parquet)
        assert len(df) == 3

    def test_unsupported_extension_raises(self, tmp_path):
        registry = LoaderRegistry()
        fake_file = tmp_path / "data.xyz"
        fake_file.write_text("hello")
        with pytest.raises(DataLoadError, match="No loader registered"):
            registry.load(fake_file)

    def test_get_loader_returns_correct_type(self):
        registry = LoaderRegistry()
        loader = registry.get_loader("data.csv")
        assert isinstance(loader, CSVLoader)

        loader = registry.get_loader("data.xlsx")
        assert isinstance(loader, ExcelLoader)

        loader = registry.get_loader("data.parquet")
        assert isinstance(loader, ParquetLoader)
