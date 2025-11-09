"""Context builders translating project state into template-friendly data."""

from __future__ import annotations

import importlib.util
import json
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from flask import session

HAS_STATSMODELS = importlib.util.find_spec("statsmodels.api") is not None

from .pipeline import determine_task
from .models import ProjectMeta
from .state import PROJECT_STORE, ProjectState, get_project_dict

PAGE_META = {
    "overview": {
        "title": "Overview",
        "help": "Monitor pipeline progress, key statuses, and launch Auto Mode.",
    },
    "projects": {
        "title": "Projects",
        "help": "Create and manage analysis workspaces with clear requirements.",
    },
    "workflow": {
        "title": "Workflow Hub",
        "help": "Choose between structured AutoDA and the dedicated NLP pipeline.",
    },
    "ingestion": {
        "title": "Data Ingestion",
        "help": "Upload structured data and configure target detection.",
    },
    "profiling": {
        "title": "Data Profiling",
        "help": "Inspect schema, statistics, and quality signals.",
    },
    "cleaning": {
        "title": "Cleaning",
        "help": "Review automated cleaning decisions with overrides.",
    },
    "feature_engineering": {
        "title": "Feature Engineering",
        "help": "Explore engineered features and modality preparations.",
    },
    "feature_selection": {
        "title": "Feature Selection",
        "help": "Compare selection strategies and manage feature sets.",
    },
    "split_validation": {
        "title": "Split & Validation",
        "help": "Configure data splits and validate class balance.",
    },
    "training_selection": {
        "title": "Training & Best-Fit Selection",
        "help": "Review candidate models, metrics, and explanations.",
    },
    "visualization": {
        "title": "Visualization Hub",
        "help": "Explore generated plots, adjust filters, and download assets.",
    },
    "reports": {
        "title": "Reports",
        "help": "Compile executive summaries and reproducibility appendices.",
    },
    "artifacts": {
        "title": "Artifacts",
        "help": "Download every asset created during Auto Mode.",
    },
    "settings": {
        "title": "Settings",
        "help": "Manage project naming, retention, and notifications.",
    },
    "prediction": {
        "title": "Prediction Sandbox",
        "help": "Score new rows against the selected model.",
    },
    "auto_mode": {
        "title": "Auto Mode",
        "help": "One-click pipeline from profiling through reporting.",
    },
    "web_scraper": {
        "title": "Web Scraper",
        "help": "Capture insights from any webpage to enrich your requirements.",
    },
    "image_hub": {
        "title": "Image Processing Hub",
        "help": "Upload images, stack preprocessing steps, and export augmented datasets.",
    },
}


def breadcrumbs_for(page_key: str) -> list[dict[str, Any]]:
    meta = PAGE_META[page_key]
    return [
        {"label": "Project Home", "endpoint": "main.overview"},
        {"label": meta["title"], "endpoint": None},
    ]


def build_common(project: ProjectState, page_key: str) -> dict[str, Any]:
    project_view = get_project_dict(project)
    return {
        "page": {
            **PAGE_META[page_key],
            "breadcrumbs": breadcrumbs_for(page_key),
            "current_step": project_view["current_phase"],
        },
        "project": project_view,
    }


def build_overview(project: ProjectState) -> dict[str, Any]:
    base = build_common(project, "overview")
    project_view = base["project"]
    datasets = project_view["datasets"]
    cards = [
        {"label": "Project Name", "value": project_view["project_name"]},
        {"label": "Status", "value": project_view["status"]},
        {"label": "Current Phase", "value": project_view["current_phase"]},
    ]
    if datasets:
        primary = datasets[0]
        cards.extend(
            [
                {
                    "label": "Datasets Loaded",
                    "value": ", ".join(d["filename"] for d in datasets),
                },
                {
                    "label": "Rows / Columns",
                    "value": f'{primary.get("rows", "—")} rows · {primary.get("columns", "—")} columns',
                },
                {
                    "label": "Detected Modalities",
                    "value": ", ".join(project_view["detected_modalities"]) or "Tabular",
                },
            ]
        )
    cards.append(
        {
            "label": "Data Quality",
            "value": _quality_label(project_view["data_quality"]),
        }
    )
    base["page"]["cards"] = cards
    base["page"]["quick_actions"] = [
        {"label": "Upload Data", "endpoint": "main.ingestion"},
        {"label": "Run Auto Mode", "endpoint": "main.auto_mode"},
        {"label": "Generate Report", "endpoint": "main.reports"},
    ]
    base["page"]["activity_log"] = project_view["audit_trail"][:10]
    progress = _progress_brief(project_view)
    base["page"]["progress_brief"] = progress
    base["page"]["data_retention"] = "Data stays within this project session. Adjust retention in Settings."
    if project_view["auto_mode"]["errors"]:
        base["page"]["error_guidance"] = [
            {
                "title": err["step"],
                "detail": err["error"],
                "action": "Review and retry",
            }
            for err in project_view["auto_mode"]["errors"]
        ]
    else:
        base["page"]["error_guidance"] = []
    base["page"]["empty_state"] = None
    return base


def build_projects(project: ProjectState | None) -> dict[str, Any]:
    base = build_common(project, "projects")
    projects_view: list[dict[str, Any]] = []
    user_email = session.get("user_email")
    if user_email:
        metas = (
            ProjectMeta.query.filter_by(owner_email=user_email)
            .order_by(ProjectMeta.created_at.desc())
            .all()
        )
        for meta in metas:
            analysis_data = meta.analysis or {}
            analysis_view = {}
            allowed_fields = {"domain", "timeline", "modalities", "stakeholders"}
            if isinstance(analysis_data, dict):
                for key, value in analysis_data.items():
                    if key.lower() not in allowed_fields:
                        continue
                    if isinstance(value, (list, tuple)):
                        parts: list[str] = []
                        for item in value:
                            if isinstance(item, dict):
                                label = (
                                    item.get("label")
                                    or item.get("name")
                                    or item.get("title")
                                    or item.get("step")
                                )
                                parts.append(label or json.dumps(item, sort_keys=True))
                            else:
                                parts.append(str(item))
                        analysis_view[key] = ", ".join(parts)
                    elif isinstance(value, dict):
                        formatted = [f"{k}: {v}" for k, v in value.items()]
                        analysis_view[key] = "; ".join(formatted)
                    else:
                        analysis_view[key] = str(value)
            projects_view.append(
                {
                    "id": str(meta.id),
                    "name": meta.name,
                    "description": meta.description or "No description provided yet.",
                    "status": meta.status or "Not Started",
                    "phase": meta.phase or "Ingestion",
                    "last_action": (
                        (meta.updated_at or meta.created_at).strftime("%Y-%m-%d %H:%M UTC")
                        if meta.updated_at or meta.created_at
                        else "n/a"
                    ),
                    "dataset_count": meta.dataset_count or 0,
                    "artifact_count": meta.artifact_count or 0,
                    "analysis": analysis_view,
                }
            )
    projects_view.sort(key=lambda item: item["name"].lower())
    base["page"]["projects"] = projects_view
    base["page"]["active_project"] = getattr(project, "id", None) if project else None
    return base


def build_workflow(project: ProjectState) -> dict[str, Any]:
    base = build_common(project, "workflow")
    user_email = session.get("user_email")
    recent: list[dict[str, Any]] = []
    if user_email:
        metas = (
            ProjectMeta.query.filter_by(owner_email=user_email)
            .order_by(ProjectMeta.updated_at.desc())
            .limit(3)
            .all()
        )
        for meta in metas:
            recent.append(
                {
                    "id": str(meta.id),
                    "name": meta.name,
                    "status": meta.status or "Not Started",
                    "last_action": (
                        (meta.updated_at or meta.created_at).strftime("%Y-%m-%d %H:%M UTC")
                        if meta.updated_at or meta.created_at
                        else "n/a"
                    ),
                    "artifacts": meta.artifact_count or 0,
                }
            )
    base["page"]["recent_projects"] = recent
    return base


def _quality_label(scores: dict[str, Any]) -> str:
    raw = scores.get("raw")
    cleaned = scores.get("cleaned")
    if raw is None and cleaned is None:
        return "Not yet scored"
    if cleaned is None:
        return f"Raw {raw}"
    if raw is None:
        return f"Cleaned {cleaned}"
    return f"Raw {raw} -> Cleaned {cleaned}"


def _progress_brief(project_view: dict[str, Any]) -> str:
    if not project_view["datasets"]:
        return "Import your first dataset to begin."
    if not project_view["auto_mode"]["completed"]:
        return f"Dataset ready. Auto Mode not yet run. Current phase: {project_view['current_phase']}."
    return "Auto Mode completed. Reports and artifacts ready."


def build_ingestion(project: ProjectState) -> dict[str, Any]:
    base = build_common(project, "ingestion")
    project_view = base.get("project") or {}
    datasets = project_view.get("datasets") or []
    base["page"]["upload_support"] = {
        "Structured": ["CSV", "TSV", "Excel"],
        "Specialized": ["(coming soon)"],
    }
    if datasets:
        base["page"]["dataset_summary"] = datasets
        base["page"]["metrics"] = {
            "files": len(datasets),
            "size": sum(d.get("size_bytes", 0) for d in datasets),
            "rows": datasets[0].get("rows"),
            "columns": datasets[0].get("columns"),
            "quality_score": (project_view.get("data_quality") or {}).get("raw"),
        }
        base["page"]["detected_modalities"] = project_view.get("detected_modalities") or []
        base["page"]["target_column"] = project_view.get("target_column")
        raw_df = project.dataframes.get("raw")
        if raw_df is not None:
            base["page"]["target_options"] = [
                {"value": col, "label": col} for col in raw_df.columns
            ]
        else:
            base["page"]["target_options"] = [
                {"value": col["name"], "label": col["name"]}
                for col in (project_view.get("profiling") or {}).get("schema", [])
            ]
        cleaned_artifact = next(
            (
                {
                    "id": artifact["id"],
                    "name": artifact["name"],
                    "size": artifact.get("size"),
                    "timestamp": artifact.get("timestamp"),
                }
                for artifact in reversed(project.artifacts)
                if artifact.get("type") == "Cleaned dataset" and artifact.get("path")
            ),
            None,
        )
        base["page"]["cleaned_artifact"] = cleaned_artifact
    else:
        base["page"]["dataset_summary"] = []
        base["page"]["metrics"] = None
        base["page"]["target_options"] = []
        base["page"]["cleaned_artifact"] = None
    return base


def build_profiling(project: ProjectState) -> dict[str, Any]:
    base = build_common(project, "profiling")
    project_view = base["project"]
    profile = project_view["profiling"]
    base["page"]["profile"] = profile
    return base


def build_cleaning(project: ProjectState) -> dict[str, Any]:
    base = build_common(project, "cleaning")
    base["page"]["cleaning"] = project.cleaning
    base["page"]["preprocessing"] = project.preprocessing
    base["page"]["artifact_lookup"] = {artifact["id"]: artifact for artifact in project.artifacts}
    return base


def build_feature_engineering(project: ProjectState) -> dict[str, Any]:
    base = build_common(project, "feature_engineering")
    page = base["page"]
    page["feature_engineering"] = project.feature_engineering
    page["preprocessing"] = project.preprocessing
    page["artifact_lookup"] = {artifact["id"]: artifact for artifact in project.artifacts}
    return base


def build_feature_selection(project: ProjectState) -> dict[str, Any]:
    base = build_common(project, "feature_selection")
    page = base["page"]
    page["feature_selection"] = project.feature_selection
    encoded_df = project.dataframes.get("encoded")
    target = project.target_column if encoded_df is not None and project.target_column in encoded_df.columns else None
    feature_options: list[str] = []
    if encoded_df is not None:
        feature_options = [col for col in encoded_df.columns if col != target]
        feature_options.sort()
    page["feature_options"] = feature_options
    page["target"] = target
    page["artifact_lookup"] = {artifact["id"]: artifact for artifact in project.artifacts}
    return base


def build_web_scraper(project: ProjectState) -> dict[str, Any]:
    base = build_common(project, "web_scraper")
    base["page"]["hide_pipeline"] = True
    return base


def build_image_hub(project: ProjectState) -> dict[str, Any]:
    base = build_common(project, "image_hub")
    base["page"]["hide_pipeline"] = True
    base["page"]["sample_sets"] = [
        {"label": "Medical imaging", "size": "512 x 512", "count": 120},
        {"label": "Retail shelves", "size": "640 x 640", "count": 80},
    ]
    return base


def build_split_validation(project: ProjectState) -> dict[str, Any]:
    base = build_common(project, "split_validation")
    page = base["page"]

    split_info = project.split_validation or {}
    target = project.target_column

    def class_balance_from_df(df: pd.DataFrame | None) -> dict[str, str] | None:
        if df is None or getattr(df, "empty", True) or not target or target not in df.columns:
            return None
        counts = df[target].value_counts(normalize=True)
        return {str(cls): f"{pct * 100:.1f}%" for cls, pct in counts.items()}

    def pct_to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(str(value).replace("%", "").strip())
        except ValueError:
            return None

    def preview_from_df(label: str, key: str) -> dict[str, Any] | None:
        df = project.dataframes.get(key)
        if df is None or getattr(df, "empty", True):
            return None
        columns = list(df.columns)[:6]
        sample = df.loc[:, columns].head(5).copy()
        if sample.empty:
            return None
        numeric_cols = sample.select_dtypes(include="number").columns
        if len(numeric_cols):
            sample.loc[:, numeric_cols] = sample.loc[:, numeric_cols].round(4)
        sample = sample.fillna("n/a")
        sample = sample.astype(str).replace("nan", "n/a")
        return {
            "label": label,
            "columns": columns,
            "rows": sample.to_dict(orient="records"),
        }

    if not split_info.get("splits") and not split_info.get("message"):
        fallback_splits: list[dict[str, Any]] = []
        for label, key in (("Train", "train"), ("Validation", "validation"), ("Test", "test")):
            df = project.dataframes.get(key)
            if df is None or getattr(df, "empty", True):
                continue
            fallback_splits.append(
                {
                    "name": label,
                    "rows": len(df),
                    "class_balance": class_balance_from_df(df),
                }
            )
        if fallback_splits:
            strategy = None
            note = None
            if project.preprocessing and project.preprocessing.get("split"):
                split_meta = project.preprocessing["split"]
                strategy = split_meta.get("strategy")
                note = split_meta.get("note")
            split_info = {
                "strategy": strategy or "Train/test split available",
                "note": note,
                "splits": fallback_splits,
            }

    page["split_validation"] = split_info if split_info else None

    previews: list[dict[str, Any]] = []
    for label, key in (("Train", "train"), ("Validation", "validation"), ("Test", "test")):
        preview = preview_from_df(label, key)
        if preview:
            previews.append(preview)
    page["split_previews"] = previews

    validation_notes: list[str] = []
    splits = split_info.get("splits") if split_info else []
    if splits:
        train_info = next((item for item in splits if item.get("name", "").lower() == "train"), None)
        validation_info = next(
            (item for item in splits if item.get("name", "").lower().startswith("val")), None
        )
        if validation_info:
            val_rows = validation_info.get("rows")
            if val_rows is not None:
                validation_notes.append(f"Validation split contains {val_rows} rows.")
            class_balance_val = validation_info.get("class_balance")
            class_balance_train = train_info.get("class_balance") if train_info else None
            if class_balance_val and class_balance_train:
                train_pct = {cls: pct_to_float(value) for cls, value in class_balance_train.items()}
                val_pct = {cls: pct_to_float(value) for cls, value in class_balance_val.items()}
                drift = []
                for cls, pct in val_pct.items():
                    if pct is None:
                        continue
                    baseline = train_pct.get(cls)
                    if baseline is None:
                        continue
                    delta = abs(pct - baseline)
                    if delta > 5:
                        drift.append(f"{cls} differs by {delta:.1f} points")
                if drift:
                    validation_notes.append("Balance check: " + "; ".join(drift))
                else:
                    validation_notes.append("Balance check: validation distribution closely matches training.")
            else:
                validation_notes.append("Balance check: configure a target or rerun Auto Mode to compare class balance.")
        else:
            validation_notes.append("Validation split not generated yet. Create one in Auto Mode or adjust split strategy.")
    else:
        if not previews:
            validation_notes.append("No splits available. Run preprocessing or Auto Mode to create train and test partitions.")

    if target:
        validation_notes.append("Tip: Keep the validation distribution aligned with training to ensure metrics stay reliable.")
    else:
        validation_notes.append("Tip: Configure a target column to unlock validation metrics and model comparisons.")

    page["validation_notes"] = validation_notes
    return base


def build_training(project: ProjectState) -> dict[str, Any]:
    base = build_common(project, "training_selection")
    page = base["page"]
    page["training"] = project.training
    page["artifact_lookup"] = {artifact["id"]: artifact for artifact in project.artifacts}
    page["target_column"] = project.target_column

    candidate_df = project.dataframes.get("encoded")
    if candidate_df is None or getattr(candidate_df, "empty", False):
        candidate_df = project.dataframes.get("raw")
    if candidate_df is not None and not getattr(candidate_df, "empty", False):
        options = [{"value": col, "label": col} for col in candidate_df.columns]
    else:
        profiling = project.profiling or {}
        options = [
            {"value": col["name"], "label": col["name"]}
            for col in profiling.get("schema", [])
        ]
    page["target_options"] = options
    return base


def _prepare_visualizations(project: ProjectState) -> list[dict[str, Any]]:
    df = None
    for key in ("selected", "encoded", "cleaned", "raw"):
        candidate = project.dataframes.get(key)
        if candidate is not None and not getattr(candidate, "empty", False):
            df = candidate
            break

    if df is None or getattr(df, "empty", True):
        return []

    df_vis = df.copy()

    max_rows = 15000
    if len(df_vis) > max_rows:
        df_vis = df_vis.sample(max_rows, random_state=42)

    for col in df_vis.columns:
        if df_vis[col].dtype == object:
            sample = df_vis[col].dropna().head(50)
            if not sample.empty:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    df_vis[col] = pd.to_datetime(df_vis[col], errors="coerce")
    target = project.target_column if project.target_column in df_vis.columns else None
    if target:
        filtered = df_vis.dropna(subset=[target])
        if not filtered.empty:
            df_vis = filtered
    if df_vis.empty:
        return []

    numeric_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df_vis.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    potential_numeric = [
        col
        for col in df_vis.columns
        if col not in numeric_cols and col not in datetime_cols
    ]
    for col in potential_numeric:
        sample = df_vis[col].dropna().head(50)
        if sample.empty:
            continue
        converted = pd.to_numeric(sample, errors="coerce")
        if converted.notna().mean() > 0.8:
            df_vis[col] = pd.to_numeric(df_vis[col], errors="coerce")
            numeric_cols.append(col)

    categorical_cols = [
        col for col in df_vis.columns if col not in numeric_cols and col not in datetime_cols
    ]
    if target in numeric_cols:
        numeric_cols = [col for col in numeric_cols if col != target]

    visuals: list[tuple[int, dict[str, Any]]] = []

    def _style_plotly(fig) -> None:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(9, 12, 18, 0)",
            plot_bgcolor="rgba(9, 12, 18, 0)",
            font=dict(color="#E8ECFC"),
            margin=dict(l=40, r=20, t=60, b=50),
            height=420,
        )
        fig.update_xaxes(
            gridcolor="rgba(255, 255, 255, 0.08)",
            zerolinecolor="rgba(255, 255, 255, 0.12)",
            linecolor="rgba(255, 255, 255, 0.12)",
        )
        fig.update_yaxes(
            gridcolor="rgba(255, 255, 255, 0.08)",
            zerolinecolor="rgba(255, 255, 255, 0.12)",
            linecolor="rgba(255, 255, 255, 0.12)",
        )
        fig.update_traces(
            marker=dict(color="#4CC2C5", line=dict(color="#4CC2C5")),
            line=dict(color="#4CC2C5"),
            selector=dict(mode="lines+markers"),
        )

    def add_visual(
        name: str,
        fig,
        description: str = "",
        insights: list[str] | None = None,
        priority: int = 5,
        chart_type: str = "other",
    ) -> None:
        if fig is None:
            return
        _style_plotly(fig)
        figure_dict = fig.to_dict()
        html = pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs="inline",
            default_width="100%",
            default_height="420px",
        )
        visuals.append(
            (
                priority,
                {
                    "name": name,
                    "description": description,
                    "figure": figure_dict,
                    "html": html,
                    "insights": insights or [],
                    "chart_type": chart_type,
                },
            )
        )

    try:
        numeric_candidates = [col for col in numeric_cols if df_vis[col].nunique(dropna=True) > 1]
        numeric_for_stats = numeric_candidates if numeric_candidates else numeric_cols
        numeric_frame = (
            df_vis[numeric_for_stats].apply(pd.to_numeric, errors="coerce")
            if numeric_for_stats
            else pd.DataFrame()
        )

        if numeric_for_stats:
            def _std(col: str) -> float:
                series = pd.to_numeric(df_vis[col], errors="coerce")
                return float(series.std(skipna=True)) if not series.dropna().empty else 0.0

            hist_col = max(numeric_for_stats, key=_std)
            hist_series = pd.to_numeric(df_vis[hist_col], errors="coerce").dropna()
            if not hist_series.empty:
                stats = [
                    f"Mean: {hist_series.mean():.2f}",
                    f"Median: {hist_series.median():.2f}",
                    f"Std dev: {hist_series.std():.2f}",
                ]
                fig = px.histogram(
                    df_vis,
                    x=hist_col,
                    nbins=min(40, max(10, hist_series.nunique())),
                    color_discrete_sequence=px.colors.sequential.Sunsetdark,
                )
                fig.update_layout(title=f"Distribution of {hist_col}")
                add_visual(
                    "Histogram",
                    fig,
                    f"Value distribution for {hist_col}.",
                    insights=stats,
                    priority=1,
                    chart_type="histogram",
                )

        box_numeric = numeric_for_stats[0] if numeric_for_stats else None
        box_category = None
        if target and target in df_vis.columns and target not in numeric_cols:
            if df_vis[target].nunique(dropna=True) <= 25:
                box_category = target
        if box_category is None:
            for col in categorical_cols:
                if df_vis[col].nunique(dropna=True) <= 25:
                    box_category = col
                    break
        if box_numeric:
            if box_category:
                box_df = df_vis[[box_category, box_numeric]].dropna()
                if not box_df.empty:
                    fig = px.box(
                        box_df,
                        x=box_category,
                        y=box_numeric,
                        color=box_category,
                        color_discrete_sequence=px.colors.sequential.Purples,
                    )
                    fig.update_layout(title=f"{box_numeric} by {box_category}")
                    medians = box_df.groupby(box_category)[box_numeric].median().sort_values(ascending=False)
                    insights = [
                        f"{cat}: median {medians[cat]:.2f}"
                        for cat in medians.head(3).index
                    ]
                    add_visual(
                        "Box Plot",
                        fig,
                        f"{box_numeric} distribution across {box_category}.",
                        insights=insights,
                        priority=2,
                        chart_type="box",
                    )
            else:
                box_series = pd.to_numeric(df_vis[box_numeric], errors="coerce").dropna()
                if not box_series.empty:
                    fig = px.box(
                        pd.DataFrame({box_numeric: box_series}),
                        y=box_numeric,
                        color_discrete_sequence=px.colors.sequential.Purples,
                    )
                    fig.update_layout(title=f"{box_numeric} spread")
                    iqr = box_series.quantile(0.75) - box_series.quantile(0.25)
                    insights = [
                        f"Median: {box_series.median():.2f}",
                        f"IQR: {iqr:.2f}",
                    ]
                    add_visual(
                        "Box Plot",
                        fig,
                        f"{box_numeric} distribution.",
                        insights=insights,
                        priority=2,
                        chart_type="box",
                    )

        if numeric_frame.shape[1] >= 2:
            corr_matrix = numeric_frame.corr()
            corr_matrix = corr_matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="all").dropna(axis=1, how="all")
            scatter_pair = None
            if corr_matrix.shape[0] >= 2:
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                if upper.notna().any().any():
                    scatter_pair = upper.abs().stack().idxmax()
            if scatter_pair:
                x_col, y_col = scatter_pair
            else:
                columns = list(numeric_frame.columns[:2])
                if len(columns) == 2:
                    x_col, y_col = columns
                else:
                    x_col = y_col = None
            if x_col and y_col:
                scatter_df = numeric_frame[[x_col, y_col]].dropna()
                if not scatter_df.empty and scatter_df[x_col].nunique() > 1 and scatter_df[y_col].nunique() > 1:
                    scatter_kwargs: dict[str, Any] = {
                        "x": x_col,
                        "y": y_col,
                        "opacity": 0.75,
                        "color_discrete_sequence": [px.colors.qualitative.Safe[3]],
                    }
                    if HAS_STATSMODELS:
                        scatter_kwargs["trendline"] = "ols"
                    fig = px.scatter(scatter_df, **scatter_kwargs)
                    corr_value = scatter_df[[x_col, y_col]].corr().iloc[0, 1]
                    insights = [f"Correlation: {corr_value:.3f}"]
                    add_visual(
                        "Scatter Plot",
                        fig,
                        f"{x_col} vs {y_col}.",
                        insights=insights,
                        priority=3,
                        chart_type="scatter",
                    )

        if numeric_frame.shape[1] >= 2:
            heat_cols = list(numeric_frame.columns[:8])
            corr_df = numeric_frame[heat_cols].corr().round(2)
            corr_df = corr_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="all").dropna(axis=1, how="all")
            if corr_df.shape[0] >= 2 and corr_df.shape[1] >= 2:
                strongest = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool)).stack()
                insights = []
                if not strongest.empty:
                    pair = strongest.abs().idxmax()
                    value = strongest[pair]
                    insights.append(f"Strongest pair: {pair[0]} vs {pair[1]} ({value:.3f})")
                fig = px.imshow(
                    corr_df,
                    text_auto=True,
                    color_continuous_scale=px.colors.sequential.Aggrnyl,
                )
                fig.update_layout(title="Correlation heatmap")
                add_visual(
                    "Correlation Heatmap",
                    fig,
                    "Pairwise correlations across numeric features.",
                    insights=insights,
                    priority=4,
                    chart_type="heatmap",
                )

    except Exception:
        # Swallow unexpected visualization errors but keep any charts that did render.
        pass


    visuals_sorted = sorted(visuals, key=lambda item: item[0])
    desired_order = ["histogram", "box", "scatter", "heatmap"]
    type_to_chart: dict[str, dict[str, Any]] = {}
    for _, chart in visuals_sorted:
        chart_type = chart.get("chart_type", "other")
        if chart_type not in type_to_chart:
            type_to_chart[chart_type] = chart

    selected: list[dict[str, Any]] = []
    for chart_type in desired_order:
        chart = type_to_chart.get(chart_type)
        if chart:
            selected.append(chart)

    if len(selected) < len(desired_order):
        for _, chart in visuals_sorted:
            if chart not in selected:
                selected.append(chart)
            if len(selected) >= len(desired_order):
                break

    return selected[: len(desired_order)]


def build_visualization(project: ProjectState) -> dict[str, Any]:
    base = build_common(project, "visualization")
    if project.visualizations is None:
        project.visualizations = _prepare_visualizations(project)
    visuals = project.visualizations or []
    base["page"]["plots"] = visuals
    base["page"]["filters"] = ["Auto-selected insights", "Top metrics"] if visuals else []
    base["page"]["plot_links"] = [
        {"label": plot["name"], "anchor": f"vis-{idx + 1}"}
        for idx, plot in enumerate(visuals)
    ]
    base["page"]["has_visualizations"] = bool(visuals)
    return base


def build_reports(project: ProjectState) -> dict[str, Any]:
    base = build_common(project, "reports")
    base["page"]["reports"] = project.reports
    return base


def build_artifacts(project: ProjectState) -> dict[str, Any]:
    base = build_common(project, "artifacts")
    base["page"]["artifacts"] = project.artifacts
    return base


def build_settings(project: ProjectState) -> dict[str, Any]:
    base = build_common(project, "settings")
    project_view = base["project"]
    base["page"]["project_info"] = {
        "name": project_view["project_name"],
        "description": project_view["description"],
    }
    base["page"]["notification_preferences"] = ["Email when training completes"]
    base["page"]["retention"] = "Data retained for local session only."
    return base


def build_prediction(project: ProjectState) -> dict[str, Any]:
    base = build_common(project, "prediction")
    training = project.training or {}
    base["page"]["prediction"] = {
        "model_version": "Model v1" if training else "No model yet",
        "target": project.target_column,
        "available": bool(training),
    }
    return base


def build_auto_mode(project: ProjectState) -> dict[str, Any]:
    base = build_common(project, "auto_mode")
    base["page"]["auto_mode_steps"] = project.auto_mode["steps"]
    base["page"]["auto_mode_status"] = project.auto_mode
    return base


BUILDERS: dict[str, Callable[[ProjectState], dict[str, Any]]] = {
    "projects": build_projects,
    "overview": build_overview,
    "ingestion": build_ingestion,
    "profiling": build_profiling,
    "cleaning": build_cleaning,
    "feature_engineering": build_feature_engineering,
    "feature_selection": build_feature_selection,
    "split_validation": build_split_validation,
    "training_selection": build_training,
    "visualization": build_visualization,
    "reports": build_reports,
    "artifacts": build_artifacts,
    "settings": build_settings,
    "prediction": build_prediction,
    "workflow": build_workflow,
    "auto_mode": build_auto_mode,
    "web_scraper": build_web_scraper,
    "image_hub": build_image_hub,
}


def build_context(page_key: str, project: ProjectState) -> dict[str, Any]:
    builder = BUILDERS.get(page_key)
    if not builder:
        raise KeyError(f"No builder configured for page {page_key}")
    return builder(project)
