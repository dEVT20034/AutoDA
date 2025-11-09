"""Application routes for the AutoDA platform."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from werkzeug.routing import BuildError
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from .context import build_context, _prepare_visualizations
from .image_hub import (
    AutoDatasetBuilder,
    ImageProcessor,
    create_zip,
    original_glob,
    processed_path_for,
    save_upload,
    session_dir,
    thumbnail_path_for,
)
from .models import ProjectMeta, User, db
from .pipeline import (
    calc_data_quality_score,
    clean_dataframe,
    detect_modalities,
    determine_task,
    format_size,
    load_dataframe,
    profile_dataframe,
    _align_X_y,
    run_auto_mode,
)
from .reporting import ReportInputs, build_report_summary, export_report_pdf
from .state import (
    PROJECT_STORE,
    ProjectState,
    add_artifact,
    create_project,
    get_current_project,
    get_dataframe,
    record_audit,
    record_auto_mode_error,
    record_auto_mode_step,
    reset_auto_mode,
    set_target_column,
    store_dataframe,
    switch_project,
    update_status,
)

main_blueprint = Blueprint("main", __name__)

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
SENDGRID_ENDPOINT = "https://api.sendgrid.com/v3/mail/send"

def current_nav_items() -> list[dict[str, str]]:
    """
    Returns navigation items based on workflow mode.
    
    NLP Workflow (7 steps):
    - Standard ML pipeline for text data
    
    Structured Data Workflow (15 items):
    - Comprehensive ML pipeline with manual control + automation
    """
    mode = session.get("nav_mode", "default")
    nav = [
        {"name": "Home", "endpoint": "main.workflow"},
        {"name": "Projects", "endpoint": "main.projects"},
    ]
    if mode == "nlp":
        # NLP Text Analysis Pipeline (Linear workflow)
        nav += [
            {"name": "Data Ingestion", "endpoint": "nlp.ingestion"},
            {"name": "Text Cleaning & Preprocessing", "endpoint": "nlp.preprocessing"},
            {"name": "NLP Automation", "endpoint": "nlp.automation_stage"},
            {"name": "Feature Selection", "endpoint": "nlp.features"},
            {"name": "Split & Validation", "endpoint": "nlp.split"},
            {"name": "Training & Selection", "endpoint": "nlp.training"},
            {"name": "Reports", "endpoint": "nlp.reports"},
            {"name": "Image Hub", "endpoint": "main.image_hub"},
        ]
    else:
        # Structured Data ML Pipeline (Sequential workflow with automation option)
        nav += [
            {"name": "Overview", "endpoint": "main.overview"},
            {"name": "Data Ingestion", "endpoint": "main.ingestion"},
            {"name": "Data Profiling", "endpoint": "main.profiling"},
            {"name": "Cleaning", "endpoint": "main.cleaning"},
            {"name": "Feature Engineering", "endpoint": "main.feature_engineering"},
            {"name": "Feature Selection", "endpoint": "main.feature_selection"},
            {"name": "Split & Validation", "endpoint": "main.split_validation"},
            {"name": "Training & Selection", "endpoint": "main.training_selection"},
            {"name": "Visualization Hub", "endpoint": "main.visualization"},
            {"name": "Reports", "endpoint": "main.reports"},
            {"name": "Artifacts", "endpoint": "main.artifacts"},
            {"name": "Prediction Sandbox", "endpoint": "main.prediction"},
            {"name": "Settings", "endpoint": "main.settings"},
            {"name": "Web Scraper", "endpoint": "main.web_scraper"},
            {"name": "Image Hub", "endpoint": "main.image_hub"},
        ]
    return nav


def resolve_image_hub_link() -> str:
    try:
        return url_for("main.image_hub")
    except BuildError:
        return url_for("main.workflow") + "#image-hub"


@main_blueprint.before_app_request
def ensure_authenticated():
    exempt_endpoints = {
        "main.signin",
        "main.signup",
        "main.logout",
        "main.root",
        "static",
    }
    endpoint = request.endpoint or ""
    if endpoint.startswith("static") or endpoint in exempt_endpoints:
        return
    if not session.get("user_email"):
        return redirect(url_for("main.signin"))
    protected_pages = {
        "main.overview",
        "main.ingestion",
        "main.profiling",
        "main.cleaning",
        "main.feature_engineering",
        "main.feature_selection",
        "main.split_validation",
        "main.training_selection",
        "main.visualization",
        "main.reports",
        "main.artifacts",
        "main.settings",
        "main.prediction",
        "main.auto_mode",
        "main.run_feature_engineering",
        "main.run_feature_selection",
        "main.run_training_selection",
        "main.preprocess",
        "main.generate_visualizations",
        "main.generate_report",
    }
    if endpoint in protected_pages and get_current_project(allow_none=True) is None:
        return redirect(url_for("main.projects"))


def current_user() -> User | None:
    user_email = session.get("user_email")
    if not user_email:
        return None
    return User.query.get(user_email)


def compute_pipeline_phases(project: ProjectState) -> list[dict[str, str]]:
    if project is None:
        return []
    stages = [
        ("Ingestion", bool(project.datasets)),
        ("Profiling", project.profiling is not None),
        ("Cleaning", project.cleaning is not None),
        ("Feature Engineering", project.feature_engineering is not None),
        ("Feature Selection", project.feature_selection is not None),
        ("Split & Validate", project.split_validation is not None),
    ("Training & Selection", project.training is not None),
    ("Reports", project.reports is not None),
    ]
    phases: list[dict[str, str]] = []
    active_found = False
    for name, completed in stages:
        if completed:
            phases.append({"name": name, "status": "complete"})
        elif not active_found:
            phases.append({"name": name, "status": "in_progress"})
            active_found = True
        else:
            phases.append({"name": name, "status": "pending"})
    return phases


def build_download_menu(project: ProjectState) -> list[dict[str, str]]:
    def _label_for(artifact: dict[str, Any]) -> str:
        name = artifact.get("name") or ""
        art_type = artifact.get("type")
        if name.endswith("_analytics_report.pdf"):
            return "Analytics Report"
        if name.endswith("_cleaned.csv"):
            return "Cleaned Dataset"
        if name.endswith("_encoded.csv"):
            return "Encoded Dataset"
        if name.endswith("_train.csv"):
            return "Train Split"
        if name.endswith("_test.csv"):
            return "Test Split"
        if name.endswith("_predictions.csv"):
            return "Predictions"
        if art_type:
            return art_type
        return name or "Artifact"

    options: list[dict[str, str]] = []
    for artifact in project.artifacts:
        if artifact.get("path"):
            options.append(
                {
                    "label": _label_for(artifact),
                    "path": url_for("main.download_artifact", artifact_id=artifact["id"]),
                }
            )
        elif artifact.get("link") and artifact.get("link") != "#":
            options.append({"label": _label_for(artifact), "path": artifact["link"]})
    if not options:
        options = [
            {"label": "Export Cleaned Data", "path": "#"},
            {"label": "Export Config", "path": "#"},
        ]
    return options


def format_distribution(series: pd.Series) -> dict[str, str]:
    """Return percentage formatting for a normalized distribution."""
    return {str(cls): f"{share:.1%}" for cls, share in series.items()}


def _auto_detect_target_column(df: pd.DataFrame | None) -> str | None:
    """Heuristic to pick a likely target column when none is configured."""
    if df is None or df.empty:
        return None
    columns = list(df.columns)
    if not columns:
        return None

    def valid(col: str) -> bool:
        return df[col].nunique(dropna=True) > 1

    keyword_priority = [
        "target",
        "label",
        "class",
        "outcome",
        "result",
        "status",
        "survived",
        "y",
    ]

    keyword_matches: list[str] = []
    for col in columns:
        name = col.lower().strip()
        if not valid(col):
            continue
        for kw in keyword_priority:
            if kw == name or kw in name:
                keyword_matches.append(col)
                break
    if keyword_matches:
        return keyword_matches[0]

    row_count = len(df)
    max_class_cardinality = max(3, min(20, row_count // 2 if row_count else 10))
    low_cardinality = [
        col
        for col in columns
        if valid(col)
        and df[col].nunique(dropna=True) <= max_class_cardinality
        and "id" not in col.lower()
    ]
    if low_cardinality:
        return low_cardinality[0]

    for col in reversed(columns):
        if valid(col) and "id" not in col.lower() and not col.lower().startswith("id_"):
            return col
    return None


def _run_preprocessing_pipeline(
    project: ProjectState,
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    profile: dict[str, Any],
    cleaning_metrics: dict[str, Any],
    quality_score: int,
) -> dict[str, Any]:
    """Execute preprocessing steps and update project state/artifacts."""
    initial_rows = len(raw_df)
    cleaned_rows = len(cleaned_df)

    project.profiling = profile
    project.cleaning = {
        "metrics": cleaning_metrics,
        "quality_score": quality_score,
    }
    project.feature_engineering = None
    project.feature_selection = None
    project.split_validation = None
    project.training = None
    project.reports = None
    project.preprocessing = None
    project.visualizations = None

    project.data_quality["cleaned"] = quality_score
    store_dataframe(project, "cleaned", cleaned_df)

    preview_df = cleaned_df.head(10)
    preview_cols = list(preview_df.columns)[:10]
    preview_records = preview_df[preview_cols].to_dict(orient="records")

    target = (
        project.target_column
        if project.target_column and project.target_column in cleaned_df.columns
        else None
    )

    modelling_df = cleaned_df.copy()
    if target:
        modelling_df = modelling_df.dropna(subset=[target]).reset_index(drop=True)
    feature_df = modelling_df.drop(columns=[target]) if target else modelling_df.copy()
    target_series = modelling_df[target].reset_index(drop=True) if target else None

    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoded_features = pd.get_dummies(feature_df, drop_first=False)
    encoded_df = encoded_features.copy()
    if target:
        encoded_df[target] = target_series.values

    uploads_dir = Path(current_app.config["UPLOAD_FOLDER"])
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    cleaned_filename = f"{timestamp}_{project.id}_cleaned.csv"
    cleaned_path = uploads_dir / cleaned_filename
    cleaned_df.to_csv(cleaned_path, index=False)
    cleaned_artifact = add_artifact(
        project,
        name=cleaned_filename,
        artifact_type="Cleaned dataset",
        size=format_size(cleaned_path.stat().st_size),
        description="Dataset after automated preprocessing (missing values, duplicates, outliers handled).",
        path=str(cleaned_path),
    )

    encoded_filename = f"{timestamp}_{project.id}_encoded.csv"
    encoded_path = uploads_dir / encoded_filename
    encoded_df.to_csv(encoded_path, index=False)
    encoded_artifact = add_artifact(
        project,
        name=encoded_filename,
        artifact_type="Encoded dataset",
        size=format_size(encoded_path.stat().st_size),
        description="Categorical features encoded with one-hot expansion.",
        path=str(encoded_path),
    )

    if target:
        y = encoded_df[target]
        X = encoded_df.drop(columns=[target])
    else:
        y = None
        X = encoded_df

    split_strategy = "70/15/15 split"
    split_note = ""
    validation_df = encoded_df.iloc[0:0].copy()
    try:
        if len(encoded_df) > 2:
            if y is not None:
                X, y = _align_X_y(X, y)
                stratify = None
                detected_task = determine_task(y)
                if detected_task == "classification" and y.nunique() > 1:
                    stratify = y
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=stratify
                )
                stratify_temp = y_temp if stratify is not None else None
                X_valid, X_test, y_valid, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, random_state=42, stratify=stratify_temp
                )
                train_df = X_train.reset_index(drop=True)
                train_df[target] = y_train.reset_index(drop=True)
                validation_df = X_valid.reset_index(drop=True)
                validation_df[target] = y_valid.reset_index(drop=True)
                test_df = X_test.reset_index(drop=True)
                test_df[target] = y_test.reset_index(drop=True)
                if stratify is not None:
                    split_strategy = "70/15/15 split with stratification"
            else:
                train_df, temp_df = train_test_split(
                    encoded_df, test_size=0.3, random_state=42
                )
                validation_df, test_df = train_test_split(
                    temp_df, test_size=0.5, random_state=42
                )
        else:
            train_df = encoded_df.copy()
            validation_df = encoded_df.iloc[0:0].copy()
            test_df = encoded_df.iloc[0:0].copy()
            split_strategy = "Split skipped (dataset too small)"
            split_note = "Dataset has fewer than 3 rows; validation/test splits omitted."
    except ValueError:
        split_strategy = "80/20 random split"
        split_note = "Validation split skipped due to limited data; using 80/20 holdout."
        if y is not None:
            X, y = _align_X_y(X, y)
            stratify = None
            detected_task = determine_task(y)
            if detected_task == "classification" and y.nunique() > 1:
                stratify = y
                split_strategy = "80/20 stratified by target"
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )
            train_df = X_train.reset_index(drop=True)
            train_df[target] = y_train.reset_index(drop=True)
            validation_df = encoded_df.iloc[0:0].copy()
            test_df = X_test.reset_index(drop=True)
            test_df[target] = y_test.reset_index(drop=True)
        else:
            train_df, test_df = train_test_split(
                encoded_df, test_size=0.2, random_state=42
            )
            validation_df = encoded_df.iloc[0:0].copy()

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    numeric_cols = train_df.select_dtypes(include=["number"]).columns.tolist()
    if target and target in numeric_cols:
        numeric_cols.remove(target)
    scaler_columns = numeric_cols.copy()
    if scaler_columns:
        scaler = StandardScaler()
        train_df.loc[:, scaler_columns] = scaler.fit_transform(train_df[scaler_columns])
        if not validation_df.empty:
            validation_df.loc[:, scaler_columns] = scaler.transform(validation_df[scaler_columns])
        if not test_df.empty:
            test_df.loc[:, scaler_columns] = scaler.transform(test_df[scaler_columns])

    if target and target in train_df.columns:
        detected_task = determine_task(train_df[target])
        if detected_task == "classification" and train_df[target].nunique() > 1:
            counts_before = train_df[target].value_counts(normalize=True, dropna=False)
            majority_share = counts_before.max()
            if majority_share > 0.7:
                majority_class = counts_before.idxmax()
                majority_df = train_df[train_df[target] == majority_class]
                balanced_parts = [majority_df]
                for cls, _ in counts_before.items():
                    if cls == majority_class:
                        continue
                    cls_df = train_df[train_df[target] == cls]
                    if cls_df.empty:
                        continue
                    resampled_cls = resample(
                        cls_df, replace=True, n_samples=len(majority_df), random_state=42
                    )
                    balanced_parts.append(resampled_cls)
                train_df = (
                    pd.concat(balanced_parts, ignore_index=True)
                    .sample(frac=1, random_state=42)
                    .reset_index(drop=True)
                )
                counts_after = train_df[target].value_counts(normalize=True, dropna=False)
                imbalance_info = {
                    "handled": True,
                    "strategy": "Oversampled minority classes to match majority class.",
                    "before": format_distribution(counts_before),
                    "after": format_distribution(counts_after),
                }
            else:
                imbalance_info = {
                    "handled": False,
                    "strategy": "Class distribution within acceptable range; no resampling applied.",
                    "before": format_distribution(counts_before),
                }
        else:
            imbalance_info = {
                "handled": False,
                "strategy": "Imbalance check skipped (regression target or insufficient class variety).",
            }
    else:
        imbalance_info = {
            "handled": False,
            "strategy": "Target column not set; imbalance detection unavailable.",
        }

    train_filename = f"{timestamp}_{project.id}_train.csv"
    train_path = uploads_dir / train_filename
    train_df.to_csv(train_path, index=False)
    train_artifact = add_artifact(
        project,
        name=train_filename,
        artifact_type="Train split",
        size=format_size(train_path.stat().st_size),
        description="Scaled training dataset after preprocessing (with resampling if applied).",
        path=str(train_path),
    )

    validation_artifact = None
    if not validation_df.empty:
        validation_filename = f"{timestamp}_{project.id}_validation.csv"
        validation_path = uploads_dir / validation_filename
        validation_df.to_csv(validation_path, index=False)
        validation_artifact = add_artifact(
            project,
            name=validation_filename,
            artifact_type="Validation split",
            size=format_size(validation_path.stat().st_size),
            description="Validation dataset reserved from preprocessing split.",
            path=str(validation_path),
        )

    test_artifact = None
    if not test_df.empty:
        test_filename = f"{timestamp}_{project.id}_test.csv"
        test_path = uploads_dir / test_filename
        test_df.to_csv(test_path, index=False)
        test_artifact = add_artifact(
            project,
            name=test_filename,
            artifact_type="Test split",
            size=format_size(test_path.stat().st_size),
            description="Scaled evaluation dataset reserved from preprocessing split.",
            path=str(test_path),
        )

    store_dataframe(project, "encoded", encoded_df)
    store_dataframe(project, "train", train_df)
    store_dataframe(project, "validation", validation_df)
    store_dataframe(project, "test", test_df)

    encoded_feature_count = encoded_df.shape[1] - (1 if target else 0)
    original_feature_count = feature_df.shape[1]
    new_feature_count = max(0, encoded_feature_count - original_feature_count)

    total_rows = len(train_df) + len(validation_df) + len(test_df)
    validation_ratio = round(len(validation_df) / total_rows, 3) if total_rows else 0.0
    test_ratio = round(len(test_df) / total_rows, 3) if total_rows else 0.0

    summary = {
        "rows_initial": initial_rows,
        "rows_cleaned": cleaned_rows,
        "rows_removed": initial_rows - cleaned_rows,
        "encoding": {
            "categorical_columns": categorical_cols,
            "original_features": original_feature_count,
            "encoded_features": encoded_feature_count,
            "new_features": new_feature_count,
        },
        "scaling": {
            "columns": scaler_columns,
            "applied": bool(scaler_columns),
            "scaler": "StandardScaler",
        },
        "split": {
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "test_rows": len(test_df),
            "validation_ratio": validation_ratio,
            "test_ratio": test_ratio,
            "strategy": split_strategy,
            "note": split_note,
            "target": target,
        },
        "imbalance": imbalance_info,
        "preview": preview_records,
        "preview_columns": preview_cols,
        "artifacts": {
            "cleaned": cleaned_artifact["id"],
            "encoded": encoded_artifact["id"],
            "train": train_artifact["id"],
            "validation": validation_artifact["id"] if validation_artifact else None,
            "test": test_artifact["id"] if test_artifact else None,
        },
    }

    def balance_for(df: pd.DataFrame) -> dict[str, str] | None:
        if target and target in df.columns and not df.empty:
            counts = df[target].value_counts(normalize=True)
            return format_distribution(counts)
        return None

    split_overview: list[dict[str, Any]] = [
        {"name": "Train", "rows": len(train_df), "class_balance": balance_for(train_df)}
    ]
    if len(validation_df):
        split_overview.append(
            {
                "name": "Validation",
                "rows": len(validation_df),
                "class_balance": balance_for(validation_df),
            }
        )
    split_overview.append(
        {
            "name": "Test",
            "rows": len(test_df),
            "class_balance": balance_for(test_df),
        }
    )

    project.preprocessing = summary
    project.split_validation = {
        "strategy": split_strategy,
        "note": split_note,
        "splits": split_overview,
    }
    return summary


def _project_report_inputs(project: ProjectState) -> tuple[ReportInputs, pd.DataFrame | None] | None:
    if project.cleaning is None and project.training is None and project.profiling is None:
        return None
    raw_df = get_dataframe(project, "raw")
    cleaned_df = get_dataframe(project, "cleaned")
    base_df = cleaned_df if cleaned_df is not None and not cleaned_df.empty else raw_df
    if base_df is None or base_df.empty:
        return None

    dataset_meta = project.datasets[0] if project.datasets else {}
    dataset_name = dataset_meta.get("filename") or project.project_name
    dataset = {
        "name": dataset_name,
        "rows_raw": int(len(raw_df)) if raw_df is not None else None,
        "rows_cleaned": int(len(cleaned_df)) if cleaned_df is not None else int(len(base_df)),
        "columns": int(base_df.shape[1]),
        "columns_cleaned": int(cleaned_df.shape[1]) if cleaned_df is not None else None,
        "modalities": project.detected_modalities or [],
    }

    profiling = project.profiling
    if profiling is None and project.preprocessing:
        profiling = project.preprocessing.get("profile")

    inputs = ReportInputs(
        project_name=project.project_name,
        target=project.target_column,
        dataset=dataset,
        profiling=profiling,
        cleaning=project.cleaning,
        feature_engineering=project.feature_engineering,
        feature_selection=project.feature_selection,
        split=project.split_validation,
        training=project.training,
        data_quality=project.data_quality,
        visualizations=project.visualizations,
    )
    return inputs, base_df


def _persist_report_pdf(
    project: ProjectState,
    summary: dict[str, Any],
    df_for_plots: pd.DataFrame | None = None,
) -> tuple[Path, dict[str, Any]]:
    uploads_dir = Path(current_app.config["UPLOAD_FOLDER"])
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{project.id}_analytics_report.pdf"
    report_path = uploads_dir / filename
    export_report_pdf(summary, report_path, df_for_plots)
    artifact = add_artifact(
        project,
        name=filename,
        artifact_type="Report",
        size=format_size(report_path.stat().st_size),
        description="Comprehensive AutoDA analytics report.",
        path=str(report_path),
    )
    return report_path, artifact

def _apply_feature_engineering(cleaned_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    "Create additional derived features from the cleaned dataset."
    engineered = cleaned_df.copy()
    new_columns: list[str] = []
    numeric_new = 0
    datetime_new = 0

    datetime_cols = engineered.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
    for col in datetime_cols:
        dt_series = engineered[col]
        engineered[f"{col}_year"] = dt_series.dt.year
        engineered[f"{col}_month"] = dt_series.dt.month
        engineered[f"{col}_day"] = dt_series.dt.day
        engineered[f"{col}_dow"] = dt_series.dt.dayofweek
        new_columns.extend([f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dow"])
        datetime_new += 4

    numeric_cols = engineered.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        squared_name = f"{col}_squared"
        engineered[squared_name] = engineered[col] ** 2
        new_columns.append(squared_name)
        numeric_new += 1
        if (engineered[col] >= 0).all():
            log_name = f"{col}_log1p"
            engineered[log_name] = np.log1p(engineered[col])
            new_columns.append(log_name)
            numeric_new += 1

    new_columns = list(dict.fromkeys(new_columns))

    info = {
        "new_columns": new_columns,
        "numeric_new": numeric_new,
        "datetime_new": datetime_new,
    }
    return engineered, info


def _auto_select_features(encoded_df: pd.DataFrame, target: str | None) -> tuple[list[str], dict[str, float]]:
    "Select informative features automatically using model-based importances or variance."
    features_only = encoded_df.drop(columns=[target]) if target and target in encoded_df.columns else encoded_df
    if features_only.empty:
        return [], {}

    if target and target in encoded_df.columns:
        y = encoded_df[target]
        task = determine_task(y)
        X = features_only
        if task == "classification":
            model = RandomForestClassifier(n_estimators=300, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        selected: list[str] = []
        scores: dict[str, float] = {}
        for idx in order:
            score = float(importances[idx])
            if score <= 0:
                continue
            feature_name = X.columns[idx]
            selected.append(feature_name)
            scores[feature_name] = score
            if len(selected) >= 20:
                break
        if not selected:
            selected = list(X.columns[: min(20, X.shape[1])])
            scores = {feat: 0.0 for feat in selected}
        return selected, scores

    variances = features_only.var().sort_values(ascending=False)
    selected = list(variances.head(min(20, len(variances))).index)
    scores = variances.head(min(20, len(variances))).to_dict()
    return selected, scores


def _train_best_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target: str,
) -> tuple[dict[str, Any], Any, list[str]]:
    """Train candidate models, return training summary, best estimator, and feature order."""
    if target not in train_df.columns:
        raise ValueError("Target column missing from training split.")

    X_train = train_df.drop(columns=[target]).reset_index(drop=True)
    y_train = train_df[target].reset_index(drop=True)
    if y_train.dropna().empty:
        raise ValueError("Training target contains no values.")

    task = determine_task(y_train)
    if task == "classification" and y_train.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")

    use_validation = validation_df is not None and not validation_df.empty
    if use_validation:
        if target not in validation_df.columns:
            use_validation = False
        else:
            y_valid = validation_df[target]
            if task == "classification" and y_valid.nunique() < 2:
                use_validation = False

    if use_validation:
        X_valid = validation_df.drop(columns=[target]).reset_index(drop=True)
        y_valid = validation_df[target].reset_index(drop=True)
    else:
        stratify = y_train if task == "classification" and y_train.nunique() > 1 else None
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=stratify
        )
        X_train = X_train.reset_index(drop=True)
        X_valid = X_valid.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_valid = y_valid.reset_index(drop=True)

    X_train, y_train = _align_X_y(X_train, y_train)
    X_valid, y_valid = _align_X_y(X_valid, y_valid)

    feature_columns = list(X_train.columns)
    X_valid = X_valid.reindex(columns=feature_columns, fill_value=0)

    candidates: list[tuple[str, Any]] = []
    if task == "classification":
        candidates = [
            ("Logistic Classifier", LogisticRegression(max_iter=1000)),
            ("Random Forest Classifier", RandomForestClassifier(n_estimators=300, random_state=42)),
            ("Gradient Boost Classifier", GradientBoostingClassifier(random_state=42)),
        ]
    elif task == "regression":
        candidates = [
            ("Linear Regression", LinearRegression()),
            ("Random Forest Regressor", RandomForestRegressor(n_estimators=300, random_state=42)),
            ("Gradient Boost Regressor", GradientBoostingRegressor(random_state=42)),
        ]
    else:
        raise ValueError("Unsupported task for manual training. Configure a target column.")

    leaderboard: list[dict[str, Any]] = []
    key_drivers: list[str] = []
    caveats: list[str] = []
    best_entry: dict[str, Any] | None = None
    best_model: Any | None = None
    best_score: float = float("-inf")

    overall_start = perf_counter()
    for name, estimator in candidates:
        model = clone(estimator)
        try:
            model_start = perf_counter()
            model.fit(X_train, y_train)
            preds = model.predict(X_valid)
            train_time = perf_counter() - model_start

            if task == "classification":
                f1 = f1_score(y_valid, preds, average="weighted", zero_division=0)
                accuracy = accuracy_score(y_valid, preds)
                precision = precision_score(y_valid, preds, average="weighted", zero_division=0)
                recall = recall_score(y_valid, preds, average="weighted", zero_division=0)
                proba = None
                if hasattr(model, "predict_proba") and y_valid.nunique() == 2:
                    try:
                        proba = model.predict_proba(X_valid)[:, 1]
                        roc_auc = roc_auc_score(y_valid, proba)
                    except Exception:
                        roc_auc = None
                else:
                    roc_auc = None
                entry = {
                    "name": name,
                    "metric_primary": {"label": "F1", "value": round(f1, 3)},
                    "metric_secondary": {"label": "Accuracy", "value": round(accuracy, 3)},
                    "training_time": f"{train_time:.2f}s",
                    "precision": round(precision, 3),
                    "recall": round(recall, 3),
                    "roc_auc": round(roc_auc, 3) if roc_auc is not None else None,
                }
                score = f1
                feature_source = (
                    model.feature_importances_
                    if hasattr(model, "feature_importances_")
                    else model.coef_[0] if hasattr(model, "coef_") else None
                )
                if feature_source is not None:
                    importances = np.array(feature_source)
                    top_idx = np.argsort(np.abs(importances))[::-1][:5]
                    key_drivers_candidate = [
                        feature_columns[idx]
                        for idx in top_idx
                        if idx < len(feature_columns) and abs(importances[idx]) > 0
                    ]
                else:
                    key_drivers_candidate = []
            else:  # regression
                r2 = r2_score(y_valid, preds)
                rmse = float(np.sqrt(mean_squared_error(y_valid, preds)))
                mae = mean_absolute_error(y_valid, preds)
                entry = {
                    "name": name,
                    "metric_primary": {"label": "R^2", "value": round(r2, 3)},
                    "metric_secondary": {"label": "RMSE", "value": round(rmse, 3)},
                    "training_time": f"{train_time:.2f}s",
                    "mae": round(mae, 3),
                }
                score = r2
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    top_idx = np.argsort(importances)[::-1][:5]
                    key_drivers_candidate = [
                        feature_columns[idx]
                        for idx in top_idx
                        if idx < len(feature_columns) and importances[idx] > 0
                    ]
                elif hasattr(model, "coef_"):
                    coefs = model.coef_
                    coefs = coefs if np.ndim(coefs) == 1 else coefs[0]
                    top_idx = np.argsort(np.abs(coefs))[::-1][:5]
                    key_drivers_candidate = [
                        feature_columns[idx] for idx in top_idx if idx < len(feature_columns)
                    ]
                else:
                    key_drivers_candidate = []

            leaderboard.append(entry)
            if score > best_score:
                best_score = score
                best_entry = entry
                best_model = model
                key_drivers = key_drivers_candidate
        except Exception as exc:  # pragma: no cover - defensive
            caveats.append(f"{name} failed: {exc}")

    duration = perf_counter() - overall_start
    if best_entry is None or best_model is None:
        raise ValueError("No candidates produced a valid model. Review feature selection or target settings.")

    summary = {
        "task": task,
        "leaderboard": leaderboard,
        "winner": best_entry,
        "key_drivers": key_drivers,
        "caveats": caveats,
        "duration": duration,
    }
    return summary, best_model, feature_columns


def _evaluate_on_test(
    model: Any,
    task: str,
    feature_columns: list[str],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
) -> tuple[dict[str, Any] | None, pd.DataFrame | None]:
    """Fit best model on combined data and evaluate on test split."""
    if model is None:
        return None, None

    combined_frames = [train_df]
    if validation_df is not None and not validation_df.empty:
        combined_frames.append(validation_df)
    combined_df = pd.concat(combined_frames, ignore_index=True)

    X_combined = combined_df.drop(columns=[target])
    X_combined = X_combined.reindex(columns=feature_columns, fill_value=0)
    y_combined = combined_df[target]
    X_combined, y_combined = _align_X_y(X_combined, y_combined)
    model.fit(X_combined, y_combined)

    if test_df is None or test_df.empty or target not in test_df.columns:
        return None, None

    X_test = test_df.drop(columns=[target])
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)
    y_test = test_df[target]
    X_test, y_test = _align_X_y(X_test, y_test)
    preds = model.predict(X_test)

    metrics: dict[str, Any]
    if task == "classification":
        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        precision = precision_score(y_test, preds, average="weighted", zero_division=0)
        recall = recall_score(y_test, preds, average="weighted", zero_division=0)
        metrics = {
            "scores": [
                {"label": "Accuracy", "value": round(accuracy, 3)},
                {"label": "F1", "value": round(f1, 3)},
                {"label": "Precision", "value": round(precision, 3)},
                {"label": "Recall", "value": round(recall, 3)},
            ],
            "roc_auc": None,
        }
        if hasattr(model, "predict_proba") and y_test.nunique() == 2:
            try:
                proba = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = round(roc_auc_score(y_test, proba), 3)
            except Exception:
                metrics["roc_auc"] = None
        matrix = confusion_matrix(y_test, preds)
        labels = [str(label) for label in sorted(pd.Series(y_test).unique())]
        metrics["confusion"] = {"labels": labels, "matrix": matrix.tolist()}
    else:
        r2 = r2_score(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = mean_absolute_error(y_test, preds)
        metrics = {
            "scores": [
                {"label": "R^2", "value": round(r2, 3)},
                {"label": "RMSE", "value": round(rmse, 3)},
                {"label": "MAE", "value": round(mae, 3)},
            ]
        }

    preview_df = pd.DataFrame(
        {
            "Actual": y_test.reset_index(drop=True),
            "Predicted": pd.Series(preds).reset_index(drop=True),
        }
    )
    if task == "classification" and hasattr(model, "predict_proba") and y_test.nunique() == 2:
        try:
            proba = model.predict_proba(X_test)[:, 1]
            preview_df["Probability"] = proba
        except Exception:
            pass

    numeric_cols = preview_df.select_dtypes(include="number").columns
    if len(numeric_cols):
        preview_df.loc[:, numeric_cols] = preview_df.loc[:, numeric_cols].round(4)

    return metrics, preview_df


def shared_context(page_key: str) -> dict[str, Any]:
    # Only set default mode if not already set, don't override NLP mode
    if not session.get("nav_mode"):
        session["nav_mode"] = "default"
    
    project = get_current_project(allow_none=True)
    if project is None:
        project = ProjectState(id="placeholder")
    page_context = build_context(page_key, project)
    return {
        **page_context,
        "nav_items": current_nav_items(),
        "pipeline_phases": compute_pipeline_phases(project),
        "download_options": build_download_menu(project),
        "artifacts": project.artifacts,
        "active_endpoint": page_key,
    }


@main_blueprint.route("/artifacts/<artifact_id>/download")
def download_artifact(artifact_id: str):
    project = get_current_project()
    artifact = next((item for item in project.artifacts if item.get("id") == artifact_id), None)
    if not artifact or not artifact.get("path"):
        abort(404)
    file_path = Path(artifact["path"])
    if not file_path.exists() or not file_path.is_file():
        abort(404)
    return send_from_directory(file_path.parent, file_path.name, as_attachment=True)


@main_blueprint.route("/sign-up", methods=["GET", "POST"], endpoint="signup")
def signup():
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""
        if not name or not email or not password:
            flash("Please complete all required fields.", "warning")
        elif password != confirm:
            flash("Passwords do not match.", "danger")
        elif len(password) < 8:
            flash("Password must be at least 8 characters long.", "warning")
        elif User.query.filter_by(email=email).first():
            flash("An account with that email already exists.", "danger")
        else:
            user = User(name=name, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            session["user_email"] = user.email
            session["user_name"] = user.name
            session.pop("autoda_project_id", None)
            flash("Account created. Welcome to AutoDA!", "success")
            return redirect(url_for("main.workflow"))
    return render_template("signup.html")


@main_blueprint.route("/sign-in", methods=["GET", "POST"], endpoint="signin")
def signin():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            flash("Invalid email or password.", "danger")
        else:
            session["user_email"] = user.email
            session["user_name"] = user.name
            session.pop("autoda_project_id", None)
            flash("Signed in successfully.", "success")
            return redirect(url_for("main.workflow"))
    return render_template("signin.html")


@main_blueprint.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("Signed out. See you soon.", "info")
    return redirect(url_for("main.signin"))


# Endpoint registered as main.contact_submit automatically
@main_blueprint.route("/contact", methods=["POST"])
def contact_submit():
    name = (request.form.get("name") or "").strip()
    email = (request.form.get("email") or "").strip()
    subject = (request.form.get("subject") or "AutoDA contact form submission").strip()
    message = (request.form.get("message") or "").strip()
    redirect_to = request.form.get("redirect_to") or url_for("main.root")
    parsed = urlparse(redirect_to)
    if parsed.netloc and parsed.netloc != request.host:
        redirect_to = url_for("main.root")
    if not email or not message:
        flash("Please provide both email and a message.", "warning")
        return redirect(redirect_to)
    if _send_contact_email(name, email, subject, message):
        flash("Message sent. We'll reach out soon.", "success")
    else:
        flash("Unable to send your message right now. Please try again later.", "danger")
    return redirect(redirect_to)


@main_blueprint.route("/projects", methods=["GET", "POST"])
def projects():
    if request.args.get("workflow") == "nlp":
        session["nav_mode"] = "nlp"
    else:
        # Always use default mode for projects unless explicitly requesting NLP workflow
        session["nav_mode"] = "default"
    project = get_current_project(allow_none=True)
    if request.method == "POST":
        action = request.form.get("action")
        if action == "create":
            name = (request.form.get("project_name") or "").strip()
            description = (request.form.get("project_description") or "").strip()
            if not name:
                flash("Provide a project name to create a new workspace.", "warning")
            else:
                modalities = [item for item in request.form.getlist("modalities") if item]
                analysis = {
                    "domain": (request.form.get("domain") or "").strip(),
                    "stakeholders": (request.form.get("stakeholders") or "").strip(),
                    "analysis_goals": (request.form.get("analysis_goals") or "").strip(),
                    "success_metrics": (request.form.get("success_metrics") or "").strip(),
                    "data_sources": (request.form.get("data_sources") or "").strip(),
                    "timeline": (request.form.get("timeline") or "").strip(),
                    "notes": (request.form.get("project_notes") or "").strip(),
                }
                analysis = {k: v for k, v in analysis.items() if v}
                if modalities:
                    analysis["modalities"] = modalities
                new_project = create_project(
                    name,
                    description,
                    analysis,
                    owner_id=session.get("user_email"),
                )
                flash(f"Project “{new_project.project_name}” created and set as active.", "success")
                return redirect(url_for("main.overview"))
        elif action == "switch":
            target_id = request.form.get("project_id")
            try:
                switched = switch_project(target_id)
                flash(f"Switched to project “{switched.project_name}”.", "success")
            except KeyError:
                flash("Unable to locate the requested project.", "danger")
            return redirect(url_for("main.overview"))
        elif action == "delete":
            target_id = request.form.get("project_id")
            meta = ProjectMeta.query.get(int(target_id)) if target_id else None
            if meta and meta.owner_email == session.get("user_email"):
                PROJECT_STORE.pop(str(meta.id), None)
                db.session.delete(meta)
                db.session.commit()
                if session.get("autoda_project_id") == str(meta.id):
                    session.pop("autoda_project_id", None)
                flash("Project deleted.", "success")
            else:
                flash("Unable to locate the requested project.", "danger")
            return redirect(url_for("main.projects"))
    return render_template("projects.html", **shared_context("projects"))


@main_blueprint.route("/")
def root():
    if session.get("user_email"):
        return redirect(url_for("main.workflow"))
    return render_template("landing.html", image_hub_link=resolve_image_hub_link())


@main_blueprint.route("/overview")
def overview():
    return render_template("overview.html", **shared_context("overview"))


@main_blueprint.route("/workflow")
def workflow():
    session["nav_mode"] = "default"
    context = shared_context("workflow")
    auto_link = url_for("main.projects")
    try:
        nlp_link = url_for("main.projects", workflow="nlp")
        nlp_available = True
    except BuildError:
        nlp_link = "/projects?workflow=nlp"
        nlp_available = False
    return render_template(
        "workflow_select.html",
        auto_link=auto_link,
        nlp_link=nlp_link,
        nlp_preview_link=nlp_link,
        nlp_available=nlp_available,
        image_hub_link=resolve_image_hub_link(),
        **context,
    )


@main_blueprint.route("/web-scraper")
def web_scraper():
    session["nav_mode"] = "default"
    context = shared_context("web_scraper")
    scrape_result = session.get("web_scrape_result")
    dataset_meta = session.get("web_scrape_dataset")
    return render_template(
        "web_scraper.html",
        web_scrape=scrape_result,
        web_scrape_dataset=dataset_meta,
        **context,
    )


def _gemini_api_key() -> str | None:
    return (
        current_app.config.get("GEMINI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or "AIzaSyDAfM9_fUuDvlIJJu9VhNq19EQJAcep3BE"
    )


def _contact_api_key() -> str | None:
    return current_app.config.get("CONTACT_EMAIL_API_KEY") or os.environ.get("CONTACT_EMAIL_API_KEY")


def _send_contact_email(name: str, sender_email: str, subject: str, message: str) -> bool:
    api_key = _contact_api_key()
    if not api_key:
        current_app.logger.warning("CONTACT_EMAIL_API_KEY not configured.")
        return False
    payload = {
        "personalizations": [
            {
                "to": [{"email": "devt@gmail.com"}],
                "subject": subject or "New AutoDA contact submission",
            }
        ],
        "from": {"email": sender_email or "noreply@autoda.local", "name": name or "AutoDA Contact"},
        "content": [
            {
                "type": "text/plain",
                "value": message or "No message provided.",
            }
        ],
    }
    try:
        response = requests.post(
            SENDGRID_ENDPOINT,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        response.raise_for_status()
        return True
    except Exception as exc:  # pragma: no cover - network failure
        current_app.logger.warning("Unable to send contact email: %s", exc)
        return False


def _build_dataset_with_gemini(page_text: str, url: str, goal: str) -> list[dict[str, Any]] | None:
    api_key = _gemini_api_key()
    if not api_key:
        return None
    cleaned_text = page_text.strip()
    if not cleaned_text:
        return None
    # Gemini input limit - keep prompt concise
    snippet = cleaned_text[:8000]
    prompt = (
        "You are an assistant that extracts structured insights from webpages for data analysis.\n"
        "Given the page content below, return ONLY valid JSON: an array of objects with keys "
        '"headline" (string), "detail" (string <= 200 chars), and "category" chosen from '
        '["Overview","Feature","Pricing","Benefit","Other"]. Include 5-10 of the most actionable items.\n'
        f"URL: {url}\nGoal: {goal or 'General content extraction'}\n"
        "Page content:\n"
        f"<<<{snippet}>>>"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    try:
        response = requests.post(
            f"{GEMINI_ENDPOINT}?key={api_key}",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        text_output = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
    except Exception as exc:  # pragma: no cover - network/parse errors
        current_app.logger.warning("Gemini dataset generation failed: %s", exc)
        return None

    if not text_output:
        return None
    text_output = text_output.strip()
    # Remove markdown fences if present
    if text_output.startswith("```"):
        text_output = text_output.lstrip("`")
        if "\n" in text_output:
            text_output = text_output.split("\n", 1)[1]
        text_output = text_output.rstrip("`")
    try:
        insights = json.loads(text_output)
    except json.JSONDecodeError:
        current_app.logger.warning("Gemini returned non-JSON payload.")
        return None
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(insights, start=1):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "insight_id": idx,
                "headline": (item.get("headline") or item.get("title") or "").strip() or f"Insight {idx}",
                "detail": (item.get("detail") or item.get("description") or "").strip(),
                "category": (item.get("category") or "Other").strip() or "Other",
                "source_url": url,
                "goal": goal or "General content extraction",
            }
        )
    return rows or None


@main_blueprint.route("/web-scrape", methods=["POST"])
def web_scrape_helper():
    url = (request.form.get("scrape_url") or "").strip()
    goal = (request.form.get("scrape_goal") or "").strip()
    api_header = (request.form.get("scrape_api_header") or "").strip()
    api_key = (request.form.get("scrape_api_key") or "").strip()

    if not url:
        flash("Provide a website URL to scrape.", "warning")
        return redirect(url_for("main.web_scraper"))

    headers = {
        "User-Agent": "AutoDA-Web-Agent/1.0 (+https://autoda.example)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if api_key:
        header_name = api_header or "Authorization"
        headers[header_name] = api_key

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.HTTPError as exc:
        if response.status_code in (401, 403) and not api_key:
            flash("Access denied. Provide an API key or credentials to continue.", "warning")
        else:
            flash(f"Web scraping failed: {exc}", "danger")
        return redirect(url_for("main.web_scraper"))
    except Exception as exc:  # pragma: no cover - network errors
        flash(f"Unable to reach the website: {exc}", "danger")
        return redirect(url_for("main.web_scraper"))

    soup = BeautifulSoup(response.text, "html.parser")

    def _collect(selector: str, limit: int = 3) -> list[str]:
        items: list[str] = []
        for node in soup.select(selector):
            text = node.get_text(strip=True)
            if text:
                items.append(text)
            if len(items) >= limit:
                break
        return items

    paragraphs = _collect("p", 4)
    headings = _collect("h1, h2, h3", 4)
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_desc = meta_tag["content"].strip()

    summary_text = " ".join(paragraphs)[:900]

    goal_label = goal or "General content extraction"
    combined_text = " ".join(paragraphs + headings)
    if not combined_text.strip():
        combined_text = soup.get_text(" ", strip=True)[:12000]
    dataset_rows = _build_dataset_with_gemini(combined_text, url, goal_label) or []
    if not dataset_rows:
        for idx, heading in enumerate(headings, start=1):
            dataset_rows.append(
                {
                    "section": "heading",
                    "sequence": idx,
                    "content": heading,
                    "source_url": url,
                    "goal": goal_label,
                }
            )
        for idx, highlight in enumerate(paragraphs, start=1):
            dataset_rows.append(
                {
                    "section": "highlight",
                    "sequence": idx,
                    "content": highlight,
                    "source_url": url,
                    "goal": goal_label,
                }
            )
        if summary_text:
            dataset_rows.append(
                {
                    "section": "summary",
                    "sequence": 1,
                    "content": summary_text,
                    "source_url": url,
                    "goal": goal_label,
                }
            )

    dataset_info: dict[str, Any] | None = None
    if dataset_rows:
        upload_root = Path(current_app.config["UPLOAD_FOLDER"]) / "web_scrapes"
        upload_root.mkdir(parents=True, exist_ok=True)
        dataset_filename = f"web_scrape_{uuid4().hex}.csv"
        dataset_path = upload_root / dataset_filename
        pd.DataFrame(dataset_rows).to_csv(dataset_path, index=False, encoding="utf-8-sig")
        dataset_info = {
            "filename": dataset_filename,
            "path": str(dataset_path),
            "rows": len(dataset_rows),
        }
        session["web_scrape_dataset"] = dataset_info
    else:
        session.pop("web_scrape_dataset", None)

    session["web_scrape_result"] = {
        "url": url,
        "goal": goal_label,
        "title": (soup.title.string.strip() if soup.title and soup.title.string else "Untitled"),
        "description": meta_desc or "No meta description available.",
        "headings": headings,
        "highlights": paragraphs,
        "summary": summary_text,
        "requires_api": bool(api_key),
        "dataset_ready": bool(dataset_rows),
        "dataset_rows": len(dataset_rows),
    }
    flash("Scraping completed. Review the insights below.", "success")
    return redirect(url_for("main.web_scraper"))


@main_blueprint.route("/web-scrape/dataset")
def download_web_scrape_dataset():
    dataset_meta: dict[str, Any] | None = session.get("web_scrape_dataset")
    if not dataset_meta:
        flash("Run the scraper to generate a dataset first.", "warning")
        return redirect(url_for("main.web_scraper"))
    dataset_path = Path(dataset_meta["path"])
    if not dataset_path.exists():
        flash("Dataset file not found. Please rerun the scraper.", "warning")
        session.pop("web_scrape_dataset", None)
        return redirect(url_for("main.web_scraper"))
    return send_file(dataset_path, as_attachment=True, download_name=dataset_meta.get("filename"))


@main_blueprint.route("/image-hub")
def image_hub():
    return render_template("image_hub.html", **shared_context("image_hub"))


@main_blueprint.route("/image-hub/upload", methods=["POST"])
def image_hub_upload():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "Select at least one image."}), 400
    session_id = request.form.get("session_id") or uuid4().hex
    upload_root = Path(current_app.config["UPLOAD_FOLDER"])
    uploaded: list[dict[str, Any]] = []
    for file in files:
        try:
            meta = save_upload(file, upload_root, session_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        meta["thumbnail_url"] = url_for(
            "main.image_hub_preview", session_id=session_id, variant="thumb", image_id=meta["id"]
        )
        meta["original_url"] = url_for(
            "main.image_hub_preview", session_id=session_id, variant="original", image_id=meta["id"]
        )
        uploaded.append(meta)
    return jsonify({"session_id": session_id, "images": uploaded})


@main_blueprint.route("/image-hub/process", methods=["POST"])
def image_hub_process():
    payload = request.get_json(silent=True) or {}
    session_id: str | None = payload.get("session_id")
    image_ids: list[str] = payload.get("image_ids") or []
    operations: dict[str, Any] = payload.get("operations") or {}
    if not session_id or not image_ids:
        return jsonify({"error": "Session and image IDs are required."}), 400
    upload_root = Path(current_app.config["UPLOAD_FOLDER"])
    processor = ImageProcessor(operations)
    results: list[dict[str, Any]] = []
    for image_id in image_ids:
        original = original_glob(upload_root, session_id, image_id)
        if original is None:
            continue
        processed = processed_path_for(upload_root, session_id, image_id)
        outcome = processor.process(original, processed)
        results.append(
            {
                "id": image_id,
                "processed_url": url_for(
                    "main.image_hub_preview", session_id=session_id, variant="processed", image_id=image_id
                ),
                "original_url": url_for(
                    "main.image_hub_preview", session_id=session_id, variant="original", image_id=image_id
                ),
                "metrics": outcome["metrics"],
            }
        )
    return jsonify({"results": results})


@main_blueprint.route("/image-hub/preview/<session_id>/<variant>/<image_id>")
def image_hub_preview(session_id: str, variant: str, image_id: str):
    upload_root = Path(current_app.config["UPLOAD_FOLDER"])
    path: Path | None
    if variant == "thumb":
        path = thumbnail_path_for(upload_root, session_id, image_id)
    elif variant == "processed":
        path = processed_path_for(upload_root, session_id, image_id)
        if not path.exists():
            path = original_glob(upload_root, session_id, image_id)
    else:
        path = original_glob(upload_root, session_id, image_id)
    if path is None or not path.exists():
        abort(404)
    return send_file(path)


@main_blueprint.route("/image-hub/download", methods=["POST"])
def image_hub_download():
    payload = request.get_json(silent=True) or {}
    session_id: str | None = payload.get("session_id")
    image_ids: list[str] = payload.get("image_ids") or []
    if not session_id or not image_ids:
        return jsonify({"error": "Missing session or images."}), 400
    upload_root = Path(current_app.config["UPLOAD_FOLDER"])
    zip_path = create_zip(upload_root, session_id, image_ids)
    return send_file(
        zip_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"autoda_image_hub_{session_id}.zip",
    )


@main_blueprint.route("/image-hub/auto", methods=["POST"])
def image_hub_auto_mode():
    payload = request.get_json(silent=True) or {}
    session_id: str | None = payload.get("session_id")
    requested = int(payload.get("count") or 0)
    width = int(payload.get("width") or 512)
    height = int(payload.get("height") or 512)
    preset = payload.get("preset") or "balanced"
    if not session_id:
        return jsonify({"error": "Upload images before running Auto Mode."}), 400
    if requested <= 0:
        return jsonify({"error": "Provide a positive dataset size."}), 400
    upload_root = Path(current_app.config["UPLOAD_FOLDER"])
    builder = AutoDatasetBuilder(upload_root, session_id)
    try:
        zip_path, generated = builder.build(requested, {"width": width, "height": height, "preset": preset})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        current_app.logger.exception("Image hub auto mode failed")
        return jsonify({"error": f"Auto Mode failed: {exc}"}), 500
    return jsonify(
        {
            "count": generated,
            "zip_url": url_for("main.image_hub_dataset", session_id=session_id, filename=zip_path.name),
        }
    )


@main_blueprint.route("/image-hub/dataset/<session_id>/<filename>")
def image_hub_dataset(session_id: str, filename: str):
    upload_root = Path(current_app.config["UPLOAD_FOLDER"])
    path = session_dir(upload_root, session_id) / filename
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="application/zip", as_attachment=True, download_name=filename)


@main_blueprint.route("/ingestion", methods=["GET", "POST"])
def ingestion():
    project = get_current_project()
    if request.method == "POST":
        file = request.files.get("data_file")
        if not file or file.filename == "":
            flash("Select a file before uploading.", "warning")
            return redirect(url_for("main.ingestion"))

        original_name = file.filename
        safe_name = secure_filename(original_name)
        if not safe_name:
            safe_name = f"dataset_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv"
        stamped_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{safe_name}"
        upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
        upload_path = upload_dir / stamped_name
        file.save(upload_path)

        try:
            df = load_dataframe(upload_path)
        except Exception as exc:  # broad to surface user-friendly message
            upload_path.unlink(missing_ok=True)
            flash(f"Unable to read file: {exc}", "danger")
            record_audit(project, "Upload failed", str(exc))
            return redirect(url_for("main.ingestion"))

        size_bytes = upload_path.stat().st_size
        project.datasets = [
            {
                "filename": original_name,
                "path": str(upload_path),
                "uploaded_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                "rows": len(df),
                "columns": df.shape[1],
                "size_bytes": size_bytes,
                "size": format_size(size_bytes),
                "source": "Upload",
            }
        ]
        project.detected_modalities = detect_modalities(df)
        project.data_quality["raw"] = calc_data_quality_score(df)
        project.profiling = None  # reset until Auto Mode
        project.cleaning = None
        project.feature_engineering = None
        project.feature_selection = None
        project.split_validation = None
        project.training = None
        project.reports = None
        project.preprocessing = None
        project.visualizations = None
        project.artifacts = []
        reset_auto_mode(project)

        project.dataframes.clear()
        store_dataframe(project, "raw", df)
        update_status(project, "In Progress", "Profiling")
        record_audit(
            project,
            "Upload Data",
            f"{original_name} uploaded with {len(df)} rows and {df.shape[1]} columns",
        )
        flash("Upload succeeded. Configure target and run Auto Mode when ready.", "success")
        return redirect(url_for("main.ingestion"))

    return render_template("ingestion.html", **shared_context("ingestion"))


@main_blueprint.route("/set-target", methods=["POST"])
def set_target():
    project = get_current_project()
    target_column = request.form.get("target_column") or None
    set_target_column(project, target_column)
    record_audit(project, "Target updated", f"Target column set to {target_column or 'None'}")
    flash("Target configuration updated.", "success")
    return redirect(url_for("main.ingestion"))


@main_blueprint.route("/preprocess", methods=["POST"])
def preprocess():
    project = get_current_project()
    raw_df = get_dataframe(project, "raw")
    if raw_df is None or raw_df.empty:
        flash("Upload a dataset with at least one row before preprocessing.", "warning")
        return redirect(url_for("main.ingestion"))

    try:
        profile = profile_dataframe(raw_df)
        cleaned_df, cleaning_metrics = clean_dataframe(raw_df)
    except Exception as exc:  # pragma: no cover - defensive safeguard
        flash(f"Preprocessing failed: {exc}", "danger")
        record_audit(project, "Preprocessing failed", str(exc))
        return redirect(url_for("main.ingestion"))

    cleaned_quality = calc_data_quality_score(cleaned_df)
    summary = _run_preprocessing_pipeline(
        project,
        raw_df,
        cleaned_df,
        profile,
        cleaning_metrics,
        cleaned_quality,
    )

    reset_auto_mode(project)
    update_status(project, "In Progress", "Cleaning")
    rows_removed = summary["rows_removed"]
    record_audit(
        project,
        "Preprocessing",
        f"Cleaned {summary['rows_cleaned']} of {summary['rows_initial']} rows; encoded {summary['encoding']['encoded_features']} features.",
    )
    flash(
        f"Preprocessing complete. Removed {rows_removed} row{'s' if rows_removed != 1 else ''}; downloads updated.",
        "success",
    )
    return redirect(url_for("main.cleaning"))

@main_blueprint.route("/feature-engineering/run", methods=["POST"])
def run_feature_engineering():
    project = get_current_project()
    raw_df = get_dataframe(project, "raw")
    cleaned_df = get_dataframe(project, "cleaned")
    if raw_df is None or cleaned_df is None:
        flash("Run preprocessing before feature engineering.", "warning")
        return redirect(url_for("main.feature_engineering"))

    engineered_df, info = _apply_feature_engineering(cleaned_df)
    profile = profile_dataframe(engineered_df)
    if project.cleaning and project.cleaning.get("metrics"):
        cleaning_metrics = project.cleaning["metrics"]
    else:
        cleaning_metrics = {
            "rows_retained": f"{len(engineered_df)} of {len(cleaned_df)}",
            "duplicates_removed": 0,
            "missing_filled": 0,
            "outliers_handled": 0,
            "type_conversions": 0,
        }

    quality_score = calc_data_quality_score(engineered_df)
    summary = _run_preprocessing_pipeline(
        project,
        raw_df,
        engineered_df,
        profile,
        cleaning_metrics,
        quality_score,
    )

    reset_auto_mode(project)

    feature_summary = {
        "new_columns": info["new_columns"],
        "total_new": len(info["new_columns"]),
        "numeric_new": info["numeric_new"],
        "datetime_new": info["datetime_new"],
        "artifacts": summary["artifacts"],
        "preview": summary["preview"],
        "preview_columns": summary["preview_columns"],
    }
    summary["feature_engineering"] = feature_summary
    project.feature_engineering = feature_summary
    project.preprocessing = summary
    project.feature_selection = None

    record_audit(
        project,
        "Feature engineering",
        "Generated {total} new feature(s).".format(total=feature_summary["total_new"]),
    )

    flash(
        f"Feature engineering created {feature_summary['total_new']} new feature{'s' if feature_summary['total_new'] != 1 else ''}.",
        "success",
    )
    return redirect(url_for("main.feature_engineering"))


@main_blueprint.route("/auto-mode", methods=["GET", "POST"])
def auto_mode():
    project = get_current_project()
    if request.method == "POST":
        raw_df = get_dataframe(project, "raw")
        if raw_df is None:
            flash("Upload data before running Auto Mode.", "warning")
            return redirect(url_for("main.ingestion"))

        start = perf_counter()
        reset_auto_mode(project)
        try:
            record_auto_mode_step(project, "Profile Data", "Profiling raw dataset.")
            target_for_run = project.target_column
            if not target_for_run:
                detected_target = _auto_detect_target_column(raw_df)
                if detected_target:
                    set_target_column(project, detected_target)
                    target_for_run = detected_target
                    record_auto_mode_step(
                        project,
                        "Detect Target",
                        f"Selected '{detected_target}' as the target column.",
                    )
                    record_audit(
                        project,
                        "Target auto-detected",
                        f"Auto Mode selected {detected_target} as the target column.",
                    )
                else:
                    record_auto_mode_step(
                        project,
                        "Detect Target",
                        "No clear target detected; proceeding in unsupervised mode.",
                    )
            result = run_auto_mode(raw_df, project.project_name, target_for_run)
        except Exception as exc:
            record_auto_mode_error(project, "Auto Mode failure", str(exc))
            flash(f"Auto Mode failed: {exc}", "danger")
            return redirect(url_for("main.auto_mode"))

        summary = _run_preprocessing_pipeline(
            project,
            raw_df,
            result.cleaned_df,
            result.profile,
            result.cleaned["metrics"],
            result.cleaned["quality_score"],
        )
        record_auto_mode_step(project, "Clean Data", "Applied default cleaning policies.")

        record_auto_mode_step(project, "Feature Engineering", "Generated baseline features.")
        fe_metrics = result.engineered.get("metrics", {}) if isinstance(result.engineered, dict) else {}
        engineered_df = result.engineered_df
        cleaned_columns = set(result.cleaned_df.columns)
        new_columns = [col for col in engineered_df.columns if col not in cleaned_columns]
        numeric_new = sum(int(pd.api.types.is_numeric_dtype(engineered_df[col])) for col in new_columns)
        datetime_new = sum(int(pd.api.types.is_datetime64_any_dtype(engineered_df[col])) for col in new_columns)
        preview_df = engineered_df.head(10)
        preview_columns = list(preview_df.columns)[:10]
        preview_records = (
            preview_df[preview_columns].to_dict(orient="records") if preview_columns else []
        )
        feature_summary = {
            "new_columns": new_columns,
            "total_new": len(new_columns),
            "numeric_new": numeric_new,
            "datetime_new": datetime_new,
            "artifacts": summary.get("artifacts", {}),
            "preview": preview_records,
            "preview_columns": preview_columns,
        }
        summary["feature_engineering"] = feature_summary
        project.feature_engineering = feature_summary
        store_dataframe(project, "engineered", engineered_df)

        record_auto_mode_step(project, "Feature Selection", "Applied variance filter.")
        auto_selection = result.selected if isinstance(result.selected, dict) else {}
        selected_features = auto_selection.get("selected_features", [])
        selected_filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{project.id}_auto_selected.csv"
        selected_path = Path(current_app.config["UPLOAD_FOLDER"]) / selected_filename
        result.selected_df.to_csv(selected_path, index=False)
        selected_artifact = add_artifact(
            project,
            name=selected_filename,
            artifact_type="Selected feature dataset",
            size=format_size(selected_path.stat().st_size),
            description="Auto Mode feature subset used during modeling.",
            path=str(selected_path),
        )
        feature_selection_summary = {
            "method": auto_selection.get("strategy", "Auto Mode"),
            "selected": selected_features,
            "count": len(selected_features),
            "artifact": selected_artifact["id"],
            "importance": {},
            "target": project.target_column if project.target_column in result.selected_df.columns else None,
        }
        summary["feature_selection"] = feature_selection_summary
        project.feature_selection = feature_selection_summary
        store_dataframe(project, "selected", result.selected_df)

        project.split_validation = result.split
        record_auto_mode_step(project, "Split & Validate", "Built train/validation/test splits.")

        project.training = result.training
        record_auto_mode_step(project, "Training & Selection", "Compared candidate models.")

        project.reports = result.reports
        report_detail = "Generated summary outputs."
        df_for_plots = result.selected_df if not result.selected_df.empty else result.cleaned_df
        try:
            _, report_artifact = _persist_report_pdf(project, result.reports, df_for_plots)
            report_detail = f"Generated summary outputs and exported {report_artifact['name']}."
        except Exception as exc:  # pragma: no cover - defensive guard
            record_auto_mode_error(project, "Reports", f"PDF export failed: {exc}")
            flash(f"Report PDF export failed: {exc}", "warning")
        record_auto_mode_step(project, "Reports", report_detail)

        project.auto_mode["completed"] = True
        project.auto_mode["last_run"] = datetime.utcnow()

        duration = perf_counter() - start
        project.total_processing_time = f"{duration:.2f}s"
        update_status(project, "Complete", "Reports")

        flash("Auto Mode completed successfully.", "success")
        record_audit(project, "Auto Mode", "End-to-end pipeline completed.")
        return redirect(url_for("main.reports"))

    return render_template("auto_mode.html", **shared_context("auto_mode"))


@main_blueprint.route("/profiling")
def profiling():
    return render_template("profiling.html", **shared_context("profiling"))


@main_blueprint.route("/cleaning")
def cleaning():
    return render_template("cleaning.html", **shared_context("cleaning"))


@main_blueprint.route("/feature-engineering")
def feature_engineering():
    return render_template("feature_engineering.html", **shared_context("feature_engineering"))


@main_blueprint.route("/feature-selection")
def feature_selection():
    return render_template("feature_selection.html", **shared_context("feature_selection"))


@main_blueprint.route("/feature-selection/run", methods=["POST"])
def run_feature_selection():
    project = get_current_project()
    encoded_df = get_dataframe(project, "encoded")
    if encoded_df is None or encoded_df.empty:
        flash("Run preprocessing or feature engineering before selecting features.", "warning")
        return redirect(url_for("main.feature_selection"))

    target = project.target_column if project.target_column and project.target_column in encoded_df.columns else None
    available_features = [col for col in encoded_df.columns if col != target]
    if not available_features:
        flash("No eligible features available for selection.", "warning")
        return redirect(url_for("main.feature_selection"))

    action = request.form.get("action", "manual")
    if action == "auto":
        selected_features, importance_scores = _auto_select_features(encoded_df, target)
        method = "Auto (model-based importances)"
        if not selected_features:
            flash("Auto selection could not identify informative features.", "warning")
            return redirect(url_for("main.feature_selection"))
    else:
        selected_features = request.form.getlist("selected_features")
        selected_features = [feat for feat in selected_features if feat in available_features]
        selected_features = list(dict.fromkeys(selected_features))
        if not selected_features:
            flash("Select at least one feature or use auto selection.", "warning")
            return redirect(url_for("main.feature_selection"))
        importance_scores = {}
        method = "Manual selection"

    selected_df = encoded_df[selected_features].copy()
    if target:
        selected_df[target] = encoded_df[target]

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    selected_filename = f"{timestamp}_{project.id}_selected.csv"
    selected_path = Path(current_app.config["UPLOAD_FOLDER"]) / selected_filename
    selected_df.to_csv(selected_path, index=False)
    selected_artifact = add_artifact(
        project,
        name=selected_filename,
        artifact_type="Selected feature dataset",
        size=format_size(selected_path.stat().st_size),
        description=f"Feature subset prepared via {method}.",
        path=str(selected_path),
    )

    train_df = get_dataframe(project, "train")
    test_df = get_dataframe(project, "test")
    keep_columns = selected_features + ([target] if target and target in encoded_df.columns else [])
    if train_df is not None and not train_df.empty:
        available_train = [col for col in keep_columns if col in train_df.columns]
        if available_train:
            store_dataframe(project, "train", train_df[available_train])
    if test_df is not None and not test_df.empty:
        available_test = [col for col in keep_columns if col in test_df.columns]
        if available_test:
            store_dataframe(project, "test", test_df[available_test])

    store_dataframe(project, "selected", selected_df)
    project.feature_selection = {
        "method": method,
        "selected": selected_features,
        "count": len(selected_features),
        "artifact": selected_artifact["id"],
        "importance": importance_scores,
        "target": target,
    }

    if project.preprocessing is not None:
        project.preprocessing["feature_selection"] = project.feature_selection

    project.split_validation = None
    project.training = None
    project.reports = None
    reset_auto_mode(project)

    record_audit(
        project,
        "Feature selection",
        f"{method} kept {len(selected_features)} features.",
    )

    flash(
        f"Feature selection applied ({len(selected_features)} feature{'s' if len(selected_features) != 1 else ''}).",
        "success",
    )
    return redirect(url_for("main.feature_selection"))


@main_blueprint.route("/split-validation")
def split_validation():
    return render_template("split_validation.html", **shared_context("split_validation"))


@main_blueprint.route("/training-selection", methods=["GET", "POST"])
def training_selection():
    project = get_current_project()
    if request.method == "POST":
        target_column = request.form.get("target_column") or None
        set_target_column(project, target_column)
        record_audit(
            project,
            "Target updated",
            f"Target column set to {target_column or 'None'} from Training & Selection.",
        )
        flash("Target configuration updated.", "success")
    return render_template("training_selection.html", **shared_context("training_selection"))


@main_blueprint.route("/training-selection/run", methods=["POST"])
def run_training_selection():
    project = get_current_project()
    target = project.target_column
    if not target:
        flash("Set a target column before training models.", "warning")
        return redirect(url_for("main.training_selection"))

    train_df = get_dataframe(project, "train")
    validation_df = get_dataframe(project, "validation")
    test_df = get_dataframe(project, "test")

    if train_df is None or train_df.empty:
        flash("Run preprocessing and feature steps to generate training data before modeling.", "warning")
        return redirect(url_for("main.training_selection"))

    if validation_df is None:
        validation_df = pd.DataFrame(columns=train_df.columns)
    if test_df is None:
        test_df = pd.DataFrame(columns=train_df.columns)

    try:
        training_summary, best_model, feature_columns = _train_best_model(train_df, validation_df, target)
    except Exception as exc:  # pragma: no cover - surfaced to user
        flash(f"Training failed: {exc}", "danger")
        record_audit(project, "Training failed", str(exc))
        return redirect(url_for("main.training_selection"))

    test_metrics, predictions_df = _evaluate_on_test(
        best_model,
        training_summary["task"],
        feature_columns,
        train_df,
        validation_df,
        test_df,
        target,
    )

    predictions_artifact_id: str | None = None
    preview_columns: list[str] = []
    preview_records: list[dict[str, Any]] = []
    if predictions_df is not None and not predictions_df.empty:
        uploads_dir = Path(current_app.config["UPLOAD_FOLDER"])
        predictions_filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{project.id}_predictions.csv"
        predictions_path = uploads_dir / predictions_filename
        predictions_df.to_csv(predictions_path, index=False)
        prediction_artifact = add_artifact(
            project,
            name=predictions_filename,
            artifact_type="Predictions",
            size=format_size(predictions_path.stat().st_size),
            description="Inference results generated from best-fit model on test split.",
            path=str(predictions_path),
        )
        predictions_artifact_id = prediction_artifact["id"]
        preview_df = predictions_df.head(10).copy()
        preview_columns = list(preview_df.columns)
        preview_records = preview_df.to_dict(orient="records")
        store_dataframe(project, "predictions", predictions_df)

    training_summary["test_metrics"] = test_metrics
    training_summary["test_preview_columns"] = preview_columns
    training_summary["test_preview"] = preview_records
    training_summary["predictions_artifact"] = predictions_artifact_id

    project.training = training_summary
    update_status(project, "In Progress", "Training & Selection")
    winner_name = training_summary.get("winner", {}).get("name")
    record_audit(
        project,
        "Training",
        f"Evaluated {len(training_summary.get('leaderboard', []))} candidate models; best model: {winner_name or 'n/a'}.",
    )
    flash("Training completed. Best-fit model selected and evaluated on the test split.", "success")
    return redirect(url_for("main.training_selection"))


@main_blueprint.route("/visualization")
def visualization():
    return render_template("visualization.html", **shared_context("visualization"))


@main_blueprint.route("/visualization/generate", methods=["POST"])
def generate_visualizations():
    project = get_current_project()
    visuals = _prepare_visualizations(project)
    project.visualizations = visuals
    if visuals:
        flash(
            f"Generated {len(visuals)} visualization{'s' if len(visuals) != 1 else ''}.",
            "success",
        )
        record_audit(
            project,
            "Visualization refresh",
            f"Rendered {len(visuals)} chart{'s' if len(visuals) != 1 else ''} for the current dataset.",
        )
    else:
        flash(
            "No charts could be produced. Ensure the dataset has usable numeric columns.",
            "warning",
        )
        record_audit(project, "Visualization refresh", "No charts generated.")
    return redirect(url_for("main.visualization"))


@main_blueprint.route("/reports/generate", methods=["POST"])
def generate_report():
    project = get_current_project()
    prepared = _project_report_inputs(project)
    if prepared is None:
        flash("Run preprocessing or Auto Mode before generating a report.", "warning")
        return redirect(url_for("main.reports"))

    report_inputs, df_for_plots = prepared
    summary = build_report_summary(report_inputs)
    project.reports = summary

    report_path: Path | None = None
    try:
        report_path, artifact = _persist_report_pdf(project, summary, df_for_plots)
    except Exception as exc:  # pragma: no cover - defensive guard
        if report_path:
            report_path.unlink(missing_ok=True)
        flash(f"Report generation failed: {exc}", "danger")
        record_audit(project, "Report generation failed", str(exc))
        return redirect(url_for("main.reports"))

    update_status(project, "Complete", "Reports")
    record_audit(
        project,
        "Report generated",
        f"Analytics report exported as {artifact['name']}.",
    )
    return send_from_directory(report_path.parent, report_path.name, as_attachment=True)


@main_blueprint.route("/reports")
def reports():
    return render_template("reports.html", **shared_context("reports"))


@main_blueprint.route("/artifacts")
def artifacts():
    return render_template("artifacts.html", **shared_context("artifacts"))


@main_blueprint.route("/settings")
def settings():
    return render_template("settings.html", **shared_context("settings"))


@main_blueprint.route("/prediction")
def prediction():
    return render_template("prediction.html", **shared_context("prediction"))
