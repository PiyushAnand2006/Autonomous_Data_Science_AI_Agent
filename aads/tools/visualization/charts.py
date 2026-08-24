"""
AADS Charting Tools — automated exploratory visualization generation.

Generates standard distributions, categorical frequency charts, correlation heatmaps,
outlier boxplots, and feature-target relationship charts using Matplotlib with headless rendering.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import matplotlib
# Use headless non-interactive backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aads.core.logging import get_logger

logger = get_logger(__name__)


def _clean_filename(name: str) -> str:
    """Sanitize column name for safe filenames."""
    return re.sub(r"[^\w\-]", "_", str(name)).strip("_")


def plot_distributions(
    df: pd.DataFrame,
    numeric_cols: list[str],
    output_dir: Path,
    max_cols: int = 12,
) -> list[str]:
    """Generate distribution histograms for numeric columns.

    Args:
        df: DataFrame.
        numeric_cols: List of numeric column names.
        output_dir: Output directory path.
        max_cols: Maximum number of distribution plots to produce.

    Returns:
        List of generated file paths as strings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    # Dynamic RAM Auto-Scaling: 20k rows on low-RAM containers, up to 100k rows on local RAM
    try:
        from aads.tools.ml.trainer import get_memory_budget_samples
        sample_limit = 20000 if get_memory_budget_samples() <= 25000 else 100000
    except Exception:
        sample_limit = 20000

    plot_df = df.sample(n=min(len(df), sample_limit), random_state=42) if len(df) > sample_limit else df

    cols_to_plot = numeric_cols[:max_cols]
    for col in cols_to_plot:
        if col not in plot_df.columns:
            continue
        series = plot_df[col].dropna()
        if len(series) == 0:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            ax.hist(series, bins=30, color="#3b82f6", edgecolor="white", alpha=0.85, density=True)
            # Add mean & median vertical lines
            mean_val = series.mean()
            median_val = series.median()
            ax.axvline(mean_val, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
            ax.axvline(median_val, color="#10b981", linestyle="-", linewidth=1.5, label=f"Median: {median_val:.2f}")
            ax.set_title(f"Distribution of {col}", fontsize=12, fontweight="bold")
            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(axis="y", linestyle=":", alpha=0.6)
            plt.tight_layout()

            fname = f"dist_{_clean_filename(col)}.png"
            dest = output_dir / fname
            fig.savefig(dest, dpi=150)
            generated.append(str(dest))
        except Exception as e:
            logger.warning("plot_distribution_failed", col=col, error=str(e))
        finally:
            plt.close(fig)

    return generated


def plot_categorical(
    df: pd.DataFrame,
    cat_cols: list[str],
    output_dir: Path,
    max_cols: int = 8,
    top_n: int = 10,
) -> list[str]:
    """Generate frequency bar charts for categorical columns.

    Args:
        df: DataFrame.
        cat_cols: List of categorical column names.
        output_dir: Output directory path.
        max_cols: Maximum number of categorical plots to produce.
        top_n: Top N categories to display.

    Returns:
        List of generated file paths as strings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    plot_df = df.sample(n=min(len(df), 20000), random_state=42) if len(df) > 20000 else df

    cols_to_plot = cat_cols[:max_cols]
    for col in cols_to_plot:
        if col not in plot_df.columns:
            continue
        series = plot_df[col].dropna()
        if len(series) == 0:
            continue

        counts = series.value_counts().head(top_n)

        fig, ax = plt.subplots(figsize=(7, 4))
        try:
            bars = ax.barh([str(k) for k in counts.index][::-1], counts.values[::-1], color="#8b5cf6", alpha=0.85)
            ax.set_title(f"Top {top_n} Categories in {col}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Count")
            ax.grid(axis="x", linestyle=":", alpha=0.6)
            plt.tight_layout()

            fname = f"cat_{_clean_filename(col)}.png"
            dest = output_dir / fname
            fig.savefig(dest, dpi=150)
            generated.append(str(dest))
        except Exception as e:
            logger.warning("plot_categorical_failed", col=col, error=str(e))
        finally:
            plt.close(fig)

    return generated


def plot_correlations(
    df: pd.DataFrame,
    numeric_cols: list[str],
    output_dir: Path,
) -> list[str]:
    """Generate a correlation heatmap for numeric features.

    Args:
        df: DataFrame.
        numeric_cols: List of numeric column names.
        output_dir: Output directory path.

    Returns:
        List containing the generated correlation heatmap path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df.sample(n=min(len(df), 20000), random_state=42) if len(df) > 20000 else df
    valid_cols = [c for c in numeric_cols if c in plot_df.columns]
    if len(valid_cols) < 2:
        return []

    # Limit to top 20 numeric columns to avoid unreadable massive heatmaps
    subset_cols = valid_cols[:20]
    corr_df = plot_df[subset_cols].corr(method="pearson").fillna(0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(subset_cols) * 0.6), max(6, len(subset_cols) * 0.5)))
    try:
        cax = ax.matshow(corr_df, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(cax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(subset_cols)))
        ax.set_yticks(range(len(subset_cols)))
        ax.set_xticklabels(subset_cols, rotation=45, ha="left", fontsize=9)
        ax.set_yticklabels(subset_cols, fontsize=9)

        # Annotate cells if <= 12 cols
        if len(subset_cols) <= 12:
            for i in range(len(subset_cols)):
                for j in range(len(subset_cols)):
                    val = corr_df.iloc[i, j]
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

        ax.set_title("Pearson Correlation Heatmap", fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()

        dest = output_dir / "correlation_matrix.png"
        fig.savefig(dest, dpi=150)
        return [str(dest)]
    except Exception as e:
        logger.warning("plot_correlations_failed", error=str(e))
        return []
    finally:
        plt.close(fig)


def plot_outliers(
    df: pd.DataFrame,
    numeric_cols: list[str],
    output_dir: Path,
    max_cols: int = 12,
) -> list[str]:
    """Generate outlier box plots for numeric columns.

    Args:
        df: DataFrame.
        numeric_cols: List of numeric column names.
        output_dir: Output directory path.
        max_cols: Max boxplots to create.

    Returns:
        List of generated file paths as strings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    plot_df = df.sample(n=min(len(df), 20000), random_state=42) if len(df) > 20000 else df
    cols_to_plot = numeric_cols[:max_cols]
    for col in cols_to_plot:
        if col not in plot_df.columns:
            continue
        series = plot_df[col].dropna()
        if len(series) < 5:
            continue

        fig, ax = plt.subplots(figsize=(6, 3.5))
        try:
            # Use orientation='horizontal' (or vert=False fallback)
            try:
                ax.boxplot(
                    series,
                    orientation="horizontal",
                    patch_artist=True,
                    boxprops=dict(facecolor="#60a5fa", color="#1d4ed8"),
                    medianprops=dict(color="#dc2626", linewidth=2),
                    flierprops=dict(marker="o", markerfacecolor="#ef4444", markersize=4, alpha=0.6),
                )
            except TypeError:
                ax.boxplot(
                    series,
                    vert=False,
                    patch_artist=True,
                    boxprops=dict(facecolor="#60a5fa", color="#1d4ed8"),
                    medianprops=dict(color="#dc2626", linewidth=2),
                    flierprops=dict(marker="o", markerfacecolor="#ef4444", markersize=4, alpha=0.6),
                )
            ax.set_title(f"Outlier Box Plot: {col}", fontsize=11, fontweight="bold")
            ax.set_xlabel(col)
            ax.grid(axis="x", linestyle=":", alpha=0.6)
            plt.tight_layout()

            fname = f"box_{_clean_filename(col)}.png"
            dest = output_dir / fname
            fig.savefig(dest, dpi=150)
            generated.append(str(dest))
        except Exception as e:
            logger.warning("plot_outlier_failed", col=col, error=str(e))
        finally:
            plt.close(fig)

    return generated


def plot_target_relationships(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    output_dir: Path,
    max_features: int = 6,
) -> list[str]:
    """Generate target vs feature relationship charts.

    Args:
        df: DataFrame.
        target_col: Target column name.
        feature_cols: Features to plot against the target.
        output_dir: Output directory.
        max_features: Maximum relationships to plot.

    Returns:
        List of generated file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if target_col not in df.columns:
        return []

    target_series = df[target_col].dropna()
    is_target_num = pd.api.types.is_numeric_dtype(target_series) and not pd.api.types.is_bool_dtype(target_series)
    is_target_cat = not is_target_num or target_series.nunique() <= 5

    generated: list[str] = []
    features_to_plot = [f for f in feature_cols if f != target_col][:max_features]

    for feat in features_to_plot:
        if feat not in df.columns:
            continue
        clean_df = df[[feat, target_col]].dropna()
        if len(clean_df) < 5:
            continue

        is_feat_num = pd.api.types.is_numeric_dtype(clean_df[feat]) and not pd.api.types.is_bool_dtype(clean_df[feat])

        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            # Case 1: Numeric Feature vs Categorical Target (Boxplot per category)
            if is_target_cat and is_feat_num:
                categories = clean_df[target_col].unique()
                data_by_cat = [clean_df[clean_df[target_col] == c][feat].values for c in categories]
                ax.boxplot(data_by_cat, patch_artist=True)
                ax.set_xticks(range(1, len(categories) + 1))
                ax.set_xticklabels([str(c) for c in categories], rotation=30, ha="right", fontsize=9)
                ax.set_title(f"{feat} by {target_col}", fontsize=11, fontweight="bold")
                ax.set_ylabel(feat)
                ax.set_xlabel(target_col)

            # Case 2: Numeric Feature vs Continuous Numeric Target (Scatter plot)
            elif not is_target_cat and is_feat_num:
                ax.scatter(clean_df[feat], clean_df[target_col], alpha=0.5, color="#2563eb", edgecolors="none")
                # Add simple trendline
                if len(clean_df) > 2:
                    z = np.polyfit(clean_df[feat], clean_df[target_col], 1)
                    p = np.poly1d(z)
                    sorted_x = np.sort(clean_df[feat])
                    ax.plot(sorted_x, p(sorted_x), "r--", alpha=0.8, linewidth=1.5)
                ax.set_title(f"{target_col} vs {feat}", fontsize=11, fontweight="bold")
                ax.set_xlabel(feat)
                ax.set_ylabel(target_col)

            # Case 3: Categorical Feature vs Target
            else:
                top_cats = clean_df[feat].value_counts().head(8).index
                filtered = clean_df[clean_df[feat].isin(top_cats)]
                if not is_target_cat:  # Categorical feature vs Continuous target
                    data_by_cat = [filtered[filtered[feat] == c][target_col].values for c in top_cats]
                    ax.boxplot(data_by_cat, patch_artist=True)
                    ax.set_xticks(range(1, len(top_cats) + 1))
                    ax.set_xticklabels([str(c) for c in top_cats], rotation=30, ha="right", fontsize=9)
                    ax.set_title(f"{target_col} by {feat}", fontsize=11, fontweight="bold")
                    ax.set_ylabel(target_col)
                    ax.set_xlabel(feat)
                else:  # Categorical vs Categorical (Crosstab count bar chart)
                    ct = pd.crosstab(filtered[feat], filtered[target_col])
                    ct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", alpha=0.85)
                    ax.set_title(f"{target_col} distribution by {feat}", fontsize=11, fontweight="bold")
                    ax.set_xlabel(feat)
                    ax.set_ylabel("Count")

            ax.grid(True, linestyle=":", alpha=0.5)
            plt.tight_layout()

            fname = f"target_vs_{_clean_filename(feat)}.png"
            dest = output_dir / fname
            fig.savefig(dest, dpi=150)
            generated.append(str(dest))
        except Exception as e:
            logger.warning("plot_target_relationship_failed", feature=feat, error=str(e))
        finally:
            plt.close(fig)

    return generated
