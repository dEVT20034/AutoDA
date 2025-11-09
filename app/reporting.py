"""Reporting utilities for assembling and exporting AutoDA analytics summaries."""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as plotly_io


@dataclass
class ReportInputs:
    """Lightweight container describing the analytics journey."""

    project_name: str
    target: str | None
    dataset: dict[str, Any]
    profiling: dict[str, Any] | None
    cleaning: dict[str, Any] | None
    feature_engineering: dict[str, Any] | None
    feature_selection: dict[str, Any] | None
    split: dict[str, Any] | None
    training: dict[str, Any] | None
    data_quality: dict[str, Any] | None
    visualizations: list[dict[str, Any]] | None = None


def _chunked(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    cache = list(items)
    for idx in range(0, len(cache), size):
        yield cache[idx : idx + size]


def _format_kv(metrics: dict[str, Any] | None) -> list[str]:
    if not metrics:
        return []
    lines: list[str] = []
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, int):
            rendered = f"{value:,}"
        elif isinstance(value, float):
            rendered = f"{value:.3f}"
        elif isinstance(value, list):
            rendered = ", ".join(map(str, value[:8]))
            if len(value) > 8:
                rendered += " â€¦"
        elif isinstance(value, dict):
            rendered = ", ".join(f"{k}: {v}" for k, v in value.items())
        else:
            rendered = str(value)
        lines.append(f"{key.replace('_', ' ').title()}: {rendered}")
    return lines


def _wrap_lines(text: str, width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        if current and current_len + len(word) + 1 > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + (1 if current_len else 0)
    if current:
        lines.append(" ".join(current))
    return lines


def _percent_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().rstrip("%")
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        rendered = f"{value:.3f}".rstrip("0").rstrip(".")
        return rendered or "0"
    return str(value)


def build_report_summary(inputs: ReportInputs) -> dict[str, Any]:
    """Create a structured summary of the analytics pipeline."""

    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    dataset = inputs.dataset or {}
    data_quality = inputs.data_quality or {}
    profiling = inputs.profiling or {}

    raw_rows = dataset.get("rows_raw")
    cleaned_rows = dataset.get("rows_cleaned")
    target = inputs.target or "Not configured"

    summary_bits: list[str] = []
    if raw_rows is not None and cleaned_rows is not None:
        summary_bits.append(
            f"Processed {cleaned_rows:,} rows (from {raw_rows:,} raw records)."
        )
    elif raw_rows:
        summary_bits.append(f"Ingested dataset with {raw_rows:,} rows.")

    if inputs.training and inputs.training.get("winner"):
        winner = inputs.training["winner"]
        primary = winner.get("metric_primary", {})
        summary_bits.append(
            f"Selected {winner.get('name', 'Best model')} with {primary.get('label', 'score')} {primary.get('value', 'n/a')} as the champion model."
        )

    if data_quality.get("raw") is not None and data_quality.get("cleaned") is not None:
        summary_bits.append(
            f"Data Quality Score improved from {data_quality['raw']} to {data_quality['cleaned']}."
        )

    executive_summary = " ".join(summary_bits) or (
        f"AutoDA analysed project {inputs.project_name} on {generated_at}."
    )

    sections: list[dict[str, Any]] = []

    dataset_section = [
        f"Dataset: {dataset.get('name', 'Uploaded file')}",
        f"Rows Ã— Columns: {dataset.get('rows_cleaned', dataset.get('rows_raw', 'n/a'))} Ã— {dataset.get('columns', 'n/a')}",
        f"Modalities: {', '.join(dataset.get('modalities', [])) or 'Tabular'}",
    ]
    if data_quality.get("raw") is not None:
        dataset_section.append(f"Data Quality (raw): {data_quality['raw']}")
    if data_quality.get("cleaned") is not None:
        dataset_section.append(f"Data Quality (cleaned): {data_quality['cleaned']}")
    sections.append({"title": "Dataset overview", "bullets": dataset_section})

    project_details = [
        f"Project name: {inputs.project_name}",
        f"Target column: {target}",
        f"Dataset: {dataset.get('name', 'Uploaded file')}",
        f"Generated at: {generated_at}",
        f"Modalities: {', '.join(dataset.get('modalities', [])) or 'Tabular'}",
    ]
    if dataset.get("columns"):
        project_details.append(f"Columns: {dataset['columns']}")
    if dataset.get("rows_raw"):
        project_details.append(f"Rows (raw): {dataset['rows_raw']:,}")
    if dataset.get("rows_cleaned"):
        project_details.append(f"Rows (cleaned): {dataset['rows_cleaned']:,}")

    if profiling:
        prof_bullets: list[str] = []
        numeric_stats = profiling.get("numeric_stats", [])[:3]
        categorical_stats = profiling.get("categorical_stats", [])[:3]
        missing = profiling.get("missing", {})
        if numeric_stats:
            prof_bullets.append(
                "Numeric highlights: "
                + "; ".join(f"{item['label']} ({item['summary']})" for item in numeric_stats)
            )
        if categorical_stats:
            prof_bullets.append(
                "Categorical highlights: "
                + "; ".join(
                    f"{item['label']} ({item['summary']})" for item in categorical_stats
                )
            )
        if missing:
            ordered = sorted(
                missing.items(),
                key=lambda pair: _percent_value(pair[1]) or 0,
                reverse=True,
            )[:3]
            prof_bullets.append(
                "Missing values: " + ", ".join(f"{col} {pct}" for col, pct in ordered)
            )
        prof_bullets.append(f"Duplicate rows: {profiling.get('duplicates', 0):,}")
        prof_bullets.append(f"Estimated outliers: {profiling.get('outliers', 0):,}")
        sections.append({"title": "Profiling insights", "bullets": prof_bullets})

    if inputs.cleaning:
        cleaning_metrics = _format_kv(inputs.cleaning.get("metrics"))
        sections.append(
            {
                "title": "Cleaning summary",
                "bullets": cleaning_metrics
                + [f"Quality score: {inputs.cleaning.get('quality_score') or 'n/a'}"],
            }
        )

    if inputs.feature_engineering:
        engineered = inputs.feature_engineering
        bullets = [
            f"New columns created: {engineered.get('total_new', engineered.get('features_created', 'n/a'))}",
        ]
        new_cols = engineered.get("new_columns") or engineered.get("new_features") or []
        if new_cols:
            bullets.append(
                "Highlights: "
                + ", ".join(new_cols[:6])
                + (" â€¦" if len(new_cols) > 6 else "")
            )
        sections.append({"title": "Feature engineering", "bullets": bullets})

    if inputs.feature_selection:
        selected = inputs.feature_selection
        bullets = [
            f"Method: {selected.get('method', selected.get('strategy', 'n/a'))}",
            f"Selected features: {selected.get('count', len(selected.get('selected_features', [])))}",
        ]
        feature_list = selected.get("selected") or selected.get("selected_features") or []
        if feature_list:
            bullets.append(
                "Top picks: "
                + ", ".join(feature_list[:6])
                + (" â€¦" if len(feature_list) > 6 else "")
            )
        sections.append({"title": "Feature selection", "bullets": bullets})

    if inputs.split and inputs.split.get("splits"):
        split_bullets = [f"Strategy: {inputs.split.get('strategy', 'n/a')}"]
        for split_info in inputs.split["splits"]:
            line = f"{split_info.get('name')}: {split_info.get('rows', 0):,} rows"
            balance = split_info.get("class_balance")
            if balance:
                balance_str = ", ".join(f"{cls} {share}" for cls, share in balance.items())
                line += f" ({balance_str})"
            split_bullets.append(line)
        sections.append({"title": "Split & validation", "bullets": split_bullets})

    if inputs.training:
        training_bullets: list[str] = []
        if inputs.training.get("task"):
            training_bullets.append(f"Detected task: {inputs.training['task'].title()}")
        winner = inputs.training.get("winner")
        if winner:
            primary = winner.get("metric_primary", {})
            training_bullets.append(
                f"Winner: {winner.get('name', 'Model')} ({primary.get('label', 'score')}={primary.get('value', 'n/a')})"
            )
        leaderboard = inputs.training.get("leaderboard", [])[:3]
        if leaderboard:
            training_bullets.append(
                "Top leaderboard: "
                + "; ".join(
                    f"{entry['name']} ({entry['metric_primary']['label']} {entry['metric_primary']['value']})"
                    for entry in leaderboard
                )
            )
        caveats = inputs.training.get("caveats", [])
        if caveats:
            training_bullets.append("Caveats: " + "; ".join(caveats[:3]))
        sections.append({"title": "Training & selection", "bullets": training_bullets})

    if inputs.visualizations:
        vis_titles = [plot.get("name") for plot in inputs.visualizations if plot.get("name")]
        if vis_titles:
            sections.append(
                {
                    "title": "Visual portfolio",
                    "bullets": ["Charts prepared: " + ", ".join(vis_titles[:6])],
                }
            )

    key_metrics = {
        "rows_processed": cleaned_rows or raw_rows,
        "column_count": dataset.get("columns"),
        "quality_lift": None,
        "modalities": dataset.get("modalities", []),
        "target": target,
    }
    if data_quality.get("raw") is not None and data_quality.get("cleaned") is not None:
        key_metrics["quality_lift"] = data_quality["cleaned"] - data_quality["raw"]

    insights: list[str] = []
    recommendations: list[str] = []

    missing = profiling.get("missing") if profiling else None
    if missing:
        ordered = sorted(
            missing.items(), key=lambda item: _percent_value(item[1]) or 0, reverse=True
        )
        if ordered:
            col, pct = ordered[0]
            insights.append(f"{col} has the highest missingness at {pct}.")
            pct_value = _percent_value(pct)
            if pct_value and pct_value > 20:
                recommendations.append(
                    f"Consider stronger imputation or additional data for {col} (currently {pct} missing)."
                )
    if profiling and profiling.get("duplicates"):
        insights.append(f"Detected {profiling['duplicates']:,} duplicate rows before cleaning.")
    if profiling and profiling.get("outliers"):
        insights.append(f"Flagged roughly {profiling['outliers']:,} numeric outliers in the sample.")

    if inputs.cleaning:
        metrics = inputs.cleaning.get("metrics") or {}
        if metrics.get("rows_retained"):
            insights.append(
                f"Cleaning retained {metrics['rows_retained']} rows after deduplication and imputation."
            )
        if metrics.get("duplicates_removed"):
            insights.append(f"Removed {metrics['duplicates_removed']:,} duplicates during cleaning.")
        if metrics.get("missing_filled"):
            insights.append(
                f"Filled {metrics['missing_filled']:,} missing values across numeric and categorical fields."
            )

    if inputs.feature_engineering:
        created = inputs.feature_engineering.get("total_new") or inputs.feature_engineering.get(
            "features_created"
        )
        if created:
            insights.append(
                f"Feature engineering introduced {created} derived columns (date parts, log/sqrt transforms, text lengths)."
            )

    if inputs.feature_selection:
        selected = inputs.feature_selection.get("selected") or inputs.feature_selection.get(
            "selected_features"
        )
        if selected:
            insights.append(
                f"Feature selection kept {len(selected)} predictors via {inputs.feature_selection.get('strategy', 'variance filtering')}."  
            )

    if inputs.split and inputs.split.get("splits"):
        strategy = inputs.split.get("strategy", "Custom split")
        insights.append(f"Data was partitioned via {strategy}.")
        for split_info in inputs.split["splits"]:
            balance = split_info.get("class_balance")
            if balance:
                balance_str = ", ".join(f"{cls} {share}" for cls, share in list(balance.items())[:3])
                insights.append(f"{split_info.get('name')} split balance â†’ {balance_str}.")

    if inputs.training:
        task = inputs.training.get("task")
        if task:
            insights.append(f"Modeling treated the problem as {task}.")
        winner = inputs.training.get("winner")
        if winner:
            metric = winner.get("metric_primary", {})
            insights.append(
                f"Best model: {winner.get('name', 'Model')} ({metric.get('label', 'score')}={metric.get('value', 'n/a')})."
            )
        duration = inputs.training.get("duration")
        if duration:
            insights.append(f"Total training runtime: {duration:.2f}s.")
        caveats = inputs.training.get("caveats") or []
        recommendations.extend(caveats[:5])
        if not winner:
            recommendations.append(
                "No winning model yet. Review feature selection, ensure the target is configured, or rerun Auto Mode."
            )
    else:
        recommendations.append("Configure a target column and run training to unlock leaderboard metrics.")

    if key_metrics.get("quality_lift") is not None and key_metrics["quality_lift"] < 5:
        recommendations.append(
            "Data Quality Score barely improved; refine cleaning rules or audit upstream data."
        )

    return {
        "generated_at": generated_at,
        "project_name": inputs.project_name,
        "target": target,
        "executive_summary": executive_summary,
        "sections": sections,
        "project_details": project_details,
        "visualizations": inputs.visualizations,
        "dataset_info": dataset,
        "data_quality_scores": data_quality,
        "profiling_details": profiling,
        "cleaning_details": inputs.cleaning,
        "feature_engineering_details": inputs.feature_engineering,
        "feature_selection_details": inputs.feature_selection,
        "split_details": inputs.split,
        "training_details": inputs.training,
        "key_metrics": key_metrics,
        "insights": insights,
        "recommendations": recommendations,
    }


def export_report_pdf(
    report: dict[str, Any],
    output_path: Path,
    df_for_plots: pd.DataFrame | None = None,
) -> None:
    """Render a concise, client-ready analytics PDF."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

    dataset_info = report.get("dataset_info") or {}
    project_name = report.get("project_name") or dataset_info.get("name") or "AutoDA Project"
    dataset_label = dataset_info.get("display_name") or dataset_info.get("name") or project_name
    target_label = report.get("target") or dataset_info.get("target") or "Not configured"
    generated_stamp = report.get("generated_at", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    app_name = "AutoDA"

    split_details = report.get("split_details") or {}
    key_metrics = report.get("key_metrics") or {}
    training_details = report.get("training_details") or {}
    profiling = report.get("profiling_details") or {}
    cleaning_details = report.get("cleaning_details") or {}
    fe_details = report.get("feature_engineering_details") or {}
    fs_details = report.get("feature_selection_details") or {}
    project_details = list(report.get("project_details") or [])
    insights = list(report.get("insights") or [])
    recommendations = list(report.get("recommendations") or [])

    palette_bg = "#f9fbff"
    palette_card = "#ffffff"
    palette_border = "#dfe3f4"
    palette_text = "#1f2438"
    palette_muted = "#5d6488"
    accent = "#37d6c0"
    accent_alt = "#5b6af0"

    def _new_page() -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        fig.patch.set_facecolor(palette_bg)
        ax.set_facecolor(palette_bg)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig, ax

    def _add_footer(fig: plt.Figure, page_no: int) -> None:
        fig.text(
            0.5,
            0.02,
            f"Auto-generated report | {app_name} · Page {page_no}",
            ha="center",
            fontsize=8,
            color=palette_muted,
        )

    def _draw_heading(ax: plt.Axes, title: str, subtitle: str = "", y: float = 0.93) -> None:
        ax.text(0.05, y, title, fontsize=15, fontweight="bold", color=palette_text)
        if subtitle:
            ax.text(0.05, y - 0.035, subtitle, fontsize=11, color=palette_muted)

    def _bullet_block(ax: plt.Axes, bullets: list[str], start_y: float, x: float = 0.05, spacing: float = 0.033) -> float:
        y = start_y
        for bullet in bullets:
            if not bullet:
                continue
            ax.text(x, y, f"• {bullet}", fontsize=10.5, color=palette_text)
            y -= spacing
            if y < 0.08:
                break
        return y

    def _draw_metric_cards(ax: plt.Axes, cards: list[dict[str, str]], start_y: float = 0.38) -> None:
        if not cards:
            return
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        width = (0.9 / len(cards)) - 0.01
        height = 0.16
        x = 0.05
        y = start_y
        for card in cards:
            patch = FancyBboxPatch(
                (x, y - height),
                width,
                height,
                boxstyle="round,pad=0.02",
                facecolor=palette_card,
                edgecolor=palette_border,
                linewidth=1,
                transform=ax.transAxes,
            )
            ax.add_patch(patch)
            ax.text(
                x + 0.015,
                y - 0.045,
                card.get("value", "n/a"),
                fontsize=15,
                fontweight="bold",
                color=palette_text,
                transform=ax.transAxes,
            )
            ax.text(
                x + 0.015,
                y - 0.115,
                card.get("label", ""),
                fontsize=9,
                color=palette_muted,
                transform=ax.transAxes,
            )
            note = card.get("note")
            if note:
                ax.text(
                    x + 0.015,
                    y - 0.14,
                    note,
                    fontsize=8,
                    color=palette_muted,
                    transform=ax.transAxes,
                )
            x += width + 0.015

    def _split_ratio_text() -> str | None:
        splits = split_details.get("splits") or []
        total = sum((split.get("rows") or 0) for split in splits)
        if not splits or not total:
            return None
        return " / ".join(f"{split.get('name', 'Split')}: {split.get('rows', 0)/total:.0%}" for split in splits)

    def _metric_cards() -> list[dict[str, str]]:
        cards: list[dict[str, str]] = []
        raw_rows = dataset_info.get("rows_raw")
        cleaned_rows = dataset_info.get("rows_cleaned")
        if raw_rows or cleaned_rows:
            cards.append(
                {
                    "label": "Rows (raw ? cleaned)",
                    "value": f"{raw_rows or 'n/a'} ? {cleaned_rows or 'n/a'}",
                }
            )
        if dataset_info.get("columns"):
            cards.append({"label": "Columns", "value": str(dataset_info.get("columns"))})
        cards.append({"label": "Target", "value": target_label})
        split_ratio = _split_ratio_text()
        if split_ratio:
            cards.append({"label": "Train / Val / Test", "value": split_ratio})
        winner = training_details.get("winner") or {}
        primary = winner.get("metric_primary", {})
        if winner:
            cards.append(
                {
                    "label": "Best model",
                    "value": winner.get("name", "Model"),
                    "note": f"{primary.get('label', 'score')} {primary.get('value', 'n/a')}",
                }
            )
        dq = report.get("data_quality_scores") or {}
        if dq.get("raw") is not None and dq.get("cleaned") is not None:
            cards.append(
                {
                    "label": "Data quality",
                    "value": f"{dq['raw']} ? {dq['cleaned']}",
                }
            )
        return cards[:6]

    def _summary_bullets() -> list[str]:
        bullets: list[str] = []
        desc = dataset_info.get("description")
        if desc:
            bullets.append(desc)
        rows = dataset_info.get("rows_cleaned") or dataset_info.get("rows_raw")
        cols = dataset_info.get("columns")
        if rows and cols:
            bullets.append(f"Dataset covers {rows:,} rows across {cols} columns.")
        missing = profiling.get("missing") or {}
        if missing:
            top_missing = sorted(
                missing.items(),
                key=lambda pair: _percent_value(pair[1]) or 0,
                reverse=True,
            )[:1]
            if top_missing:
                col, pct = top_missing[0]
                bullets.append(f"Largest gap: {col} has {pct} missing values handled during cleaning.")
        winner = training_details.get("winner") or {}
        primary = winner.get("metric_primary", {})
        if winner:
            bullets.append(
                f"Champion model: {winner.get('name', 'Model')} ({primary.get('label', 'score')} {primary.get('value', 'n/a')})."
            )
        dq = report.get("data_quality_scores") or {}
        if dq.get("raw") is not None and dq.get("cleaned") is not None:
            bullets.append(f"Data quality improved from {dq['raw']} to {dq['cleaned']}.")
        if insights:
            bullets.append(insights[0])
        if recommendations:
            bullets.append(f"Next step: {recommendations[0]}")
        # ensure 5-7 bullets
        extras = [line for line in project_details if line not in bullets]
        for line in extras:
            if len(bullets) >= 6:
                break
            bullets.append(line)
        return bullets[:7]

    def _data_quality_bullets() -> list[str]:
        items: list[str] = []
        missing = profiling.get("missing") or {}
        ordered = sorted(
            missing.items(),
            key=lambda pair: _percent_value(pair[1]) or 0,
            reverse=True,
        )[:3]
        for col, pct in ordered:
            items.append(f"Filled {pct} missing values in {col}.")
        duplicates = cleaning_details.get("duplicates_removed")
        if duplicates:
            items.append(f"Removed {duplicates} duplicate rows.")
        outliers = cleaning_details.get("outliers_handled")
        if outliers:
            items.append(f"Winsorized {outliers} outlier values across numeric columns.")
        conversions = cleaning_details.get("type_conversions")
        if conversions:
            items.append(f"Standardized {conversions} categorical fields.")
        return items[:5]

    def _cleaning_text() -> str:
        parts: list[str] = []
        if cleaning_details.get("missing_filled"):
            parts.append("filled missing values using numeric medians and categorical modes")
        if cleaning_details.get("duplicates_removed"):
            parts.append("dropped duplicate records")
        if cleaning_details.get("outliers_handled"):
            parts.append("capped extreme numeric outliers")
        if cleaning_details.get("type_conversions"):
            parts.append("standardized categorical text fields")
        return "Cleaning: " + ", ".join(parts) + "." if parts else "Cleaning: Data was already well structured."

    def _fe_text() -> str:
        created = fe_details.get("features_created")
        dimension = fe_details.get("dimensionality_change")
        pieces = []
        if created:
            pieces.append(f"created {created} derived features (date parts, log/sqrt transforms, text lengths)")
        if dimension:
            pieces.append(f"dimensionality shifted {dimension}")
        return "Feature engineering: " + ", ".join(pieces) + "." if pieces else "Feature engineering: None required."

    def _fs_text() -> str:
        strategy = fs_details.get("strategy")
        selected = fs_details.get("selected_features")
        text = strategy or "Feature selection skipped."
        if selected:
            text += f" Retained {len(selected)} informative features."
        return text

    def _format_metric(value: Any) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.3f}".rstrip("0").rstrip(".")
        if value is None:
            return "-"
        return str(value)

    def _leaderboard_rows() -> list[list[str]]:
        rows = []
        leaderboard = training_details.get("leaderboard") or []
        for entry in leaderboard[:5]:
            primary = entry.get("metric_primary", {})
            secondary = entry.get("metric_secondary", {})
            rows.append(
                [
                    entry.get("name", "Model"),
                    _format_metric(primary.get("value")),
                    _format_metric(secondary.get("value")) if secondary else "-",
                    _format_metric(entry.get("roc_auc") if training_details.get("task") == "classification" else entry.get("mae")),
                    entry.get("training_time", "-"),
                ]
            )
        return rows

    def _champion_paragraph() -> str:
        winner = training_details.get("winner") or {}
        if not winner:
            return "No winning model yet. Configure a target column and rerun training."
        primary = winner.get("metric_primary", {})
        return (
            f"The champion model is {winner.get('name', 'Model')} with {primary.get('label', 'score')} {primary.get('value', 'n/a')}. "
            "It offered the best balance between accuracy and generalization on the validation split."
        )

    def _perfect_metric_note() -> str | None:
        leaderboard = training_details.get("leaderboard") or []
        for entry in leaderboard:
            for metric in (entry.get("metric_primary", {}).get("value"), entry.get("metric_secondary", {}).get("value"), entry.get("roc_auc")):
                if isinstance(metric, (int, float)) and float(metric) >= 0.999:
                    return "Note: Perfect scores may indicate data leakage or an overly simple evaluation. Please validate with fresh holdout data."
        return None

    def _key_driver_sentences() -> list[str]:
        drivers = training_details.get("key_drivers") or []
        sentences = []
        for feature in drivers[:10]:
            sentences.append(f"{feature}: This feature had one of the largest impacts on the champion model's predictions.")
        return sentences

    def _insight_bullets() -> list[str]:
        if insights:
            return insights[:6]
        fallback = [line for line in project_details if line]
        return fallback[:4]

    metric_cards = _metric_cards()
    summary_bullets = _summary_bullets()
    subtitle = f"Dataset: {dataset_label} | Rows: {dataset_info.get('rows_cleaned', dataset_info.get('rows_raw', 'n/a'))} | Columns: {dataset_info.get('columns', 'n/a')} | Target: {target_label} | Generated: {generated_stamp}"

    page_no = 1
    with PdfPages(output_path) as pdf:
        # Page 1 – Executive summary
        fig, ax = _new_page()
        ax.text(0.05, 0.93, f"{project_name} - Analytics Report", fontsize=22, fontweight="bold", color=palette_text)
        ax.text(0.05, 0.88, subtitle, fontsize=10.5, color=palette_muted)
        ax.text(0.95, 0.93, f"Generated by {app_name}", fontsize=10, color=palette_muted, ha="right")
        metric_anchor = _bullet_block(ax, summary_bullets, start_y=0.8)
        metric_start = max(metric_anchor - 0.02, 0.35)
        _draw_metric_cards(ax, metric_cards, start_y=metric_start)
        _add_footer(fig, page_no)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page_no += 1

        # Page 2 – Dataset & Pipeline
        fig, ax = _new_page()
        _draw_heading(ax, "Dataset overview", "Data quality and pipeline summary")
        dataset_table = [
            ["Name", dataset_label],
            ["Rows (raw)", _format_value(dataset_info.get("rows_raw"))],
            ["Rows (cleaned)", _format_value(dataset_info.get("rows_cleaned"))],
            ["Columns", _format_value(dataset_info.get("columns"))],
            ["Target", target_label],
            ["Modalities", ", ".join(dataset_info.get("modalities", [])) or "Tabular"],
        ]
        table = ax.table(
            cellText=dataset_table,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            colLoc="left",
            bbox=(0.05, 0.48, 0.4, 0.35),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        for cell in table.get_celld().values():
            cell.set_facecolor(palette_card)
            cell.set_edgecolor(palette_border)
            cell._text.set_color(palette_text)
        ax.text(0.52, 0.82, "Data quality", fontsize=12, color=accent)
        dq_y = _bullet_block(ax, _data_quality_bullets(), 0.78, x=0.52)
        ax.text(0.05, 0.4, "Pipeline summary", fontsize=12, color=accent)
        ax.text(0.05, 0.36, _cleaning_text(), fontsize=10.5, color=palette_text)
        ax.text(0.05, 0.32, _fe_text(), fontsize=10.5, color=palette_text)
        ax.text(0.05, 0.28, _fs_text(), fontsize=10.5, color=palette_text)
        splits = split_details.get("splits") or []
        if splits:
            split_rows = []
            total_rows = sum(split.get("rows") or 0 for split in splits) or 1
            for split in splits:
                ratio = (split.get("rows") or 0) / total_rows
                split_rows.append([split.get("name", "Split"), _format_value(split.get("rows")), f"{ratio:.0%}"])
            table = ax.table(
                cellText=split_rows,
                colLabels=["Split", "Rows", "Ratio"],
                cellLoc="left",
                colLoc="left",
                bbox=(0.52, 0.48, 0.43, 0.25),
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9.5)
            for cell in table.get_celld().values():
                cell.set_facecolor(palette_card)
                cell.set_edgecolor(palette_border)
                cell._text.set_color(palette_text)
        _add_footer(fig, page_no)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page_no += 1

        # Page 3 – Model performance
        fig, ax = _new_page()
        _draw_heading(ax, "Model performance", split_details.get("strategy", "Evaluation results"))
        ax.text(0.05, 0.86, _champion_paragraph(), fontsize=10.5, color=palette_text)
        rows = _leaderboard_rows()
        if rows:
            table = ax.table(
                cellText=rows,
                colLabels=["Model", "F1", "Accuracy", "AUC/MAE", "Train time"],
                cellLoc="left",
                colLoc="left",
                bbox=(0.05, 0.38, 0.9, 0.4),
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9.5)
            for (row_idx, col_idx), cell in table.get_celld().items():
                cell.set_facecolor(palette_card)
                cell.set_edgecolor(palette_border)
                cell._text.set_color(palette_text)
                if row_idx == 1:
                    cell.set_facecolor("#e9fbf7")
        note = _perfect_metric_note()
        if note:
            ax.text(0.05, 0.32, note, fontsize=9, color="#c65151")
        _add_footer(fig, page_no)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page_no += 1

        # Page 4 – Key drivers & insights
        fig, ax = _new_page()
        _draw_heading(ax, "Key drivers & insights")
        ax.text(0.05, 0.88, "Top drivers", fontsize=12, color=accent)
        y = _bullet_block(ax, _key_driver_sentences(), 0.84)
        ax.text(0.05, y - 0.05, "Insights", fontsize=12, color=accent_alt)
        _bullet_block(ax, _insight_bullets(), y - 0.09)
        _add_footer(fig, page_no)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page_no += 1

        # Page 5 – Visuals (optional)
        if df_for_plots is not None and not df_for_plots.empty:
            plot_df = df_for_plots.copy()
            if len(plot_df) > 10000:
                plot_df = plot_df.sample(10000, random_state=42)
            plot_df = plot_df.replace([np.inf, -np.inf], np.nan)

            object_cols = plot_df.select_dtypes(include=["object"]).columns
            for col in object_cols:
                sample = plot_df[col].dropna()
                if sample.empty:
                    continue
                coerced = pd.to_numeric(sample, errors="coerce")
                if coerced.notna().mean() > 0.8:
                    plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

            numeric_df = plot_df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
            categorical_candidates = [
                col
                for col in plot_df.columns
                if col not in numeric_df.columns and plot_df[col].nunique(dropna=True) <= 15
            ]

            def _has_variation(series: pd.Series) -> bool:
                return series.nunique(dropna=True) > 1

            if not numeric_df.empty:
                numeric_df = numeric_df.dropna(axis=0, how="all")
                stds = numeric_df.std(numeric_only=True).replace(0, np.nan).dropna()
                hist_col = stds.idxmax() if not stds.empty else numeric_df.columns[0]
                hist_series = numeric_df[hist_col].dropna()
                if not hist_series.empty and _has_variation(hist_series):
                    fig, ax = plt.subplots(figsize=(8, 5))
                    fig.patch.set_facecolor("white")
                    bins = min(40, max(10, hist_series.nunique()))
                    ax.hist(hist_series, bins=bins, color="#3267D6", alpha=0.85)
                    ax.set_title(f"Distribution of {hist_col}")
                    ax.set_xlabel(hist_col)
                    ax.set_ylabel("Frequency")
                    fig.tight_layout()
                    fig.text(0.5, 0.01, f"Takeaway: {hist_col} shows the widest spread among numeric features.", ha="center", fontsize=9)
                    _add_footer(fig, page_no)
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    page_no += 1

                box_col = numeric_df.columns[0]
                fig, ax = plt.subplots(figsize=(6, 5))
                fig.patch.set_facecolor("white")
                ax.boxplot(numeric_df[box_col].dropna(), vert=True, patch_artist=True)
                ax.set_title(f"Box plot of {box_col}")
                ax.set_xticklabels([box_col])
                fig.tight_layout()
                fig.text(0.5, 0.01, f"Takeaway: {box_col} contains mild outliers after cleaning.", ha="center", fontsize=9)
                _add_footer(fig, page_no)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                page_no += 1

                if numeric_df.shape[1] >= 2:
                    corr = numeric_df.corr().fillna(0)
                    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu"))
                    fig.update_layout(title="Correlation heatmap", width=700, height=620)
                    img_bytes = plotly_io.to_image(fig, format="png")
                    image = plt.imread(io.BytesIO(img_bytes))
                    fig_corr, ax_corr = plt.subplots(figsize=(7, 6))
                    ax_corr.imshow(image)
                    ax_corr.axis("off")
                    fig_corr.text(0.5, 0.02, "Takeaway: correlation clusters highlight related metrics.", ha="center", fontsize=9)
                    _add_footer(fig_corr, page_no)
                    pdf.savefig(fig_corr, bbox_inches="tight")
                    plt.close(fig_corr)
                    page_no += 1

            if categorical_candidates:
                cat_col = categorical_candidates[0]
                counts = plot_df[cat_col].value_counts().head(15)
                if not counts.empty:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    fig.patch.set_facecolor("white")
                    counts[::-1].plot(kind="barh", ax=ax, color="#5b6af0")
                    ax.set_title(f"Top categories in {cat_col}")
                    ax.set_xlabel("Count")
                    fig.tight_layout()
                    fig.text(0.5, 0.01, f"Takeaway: {cat_col} is concentrated in the leading categories above.", ha="center", fontsize=9)
                    _add_footer(fig, page_no)
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    page_no += 1

        # Page 6 – Recommendations
        if recommendations:
            fig, ax = _new_page()
            _draw_heading(ax, "Recommendations", "Next steps to drive value")
            _bullet_block(ax, recommendations[:6], 0.86)
            _add_footer(fig, page_no)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
