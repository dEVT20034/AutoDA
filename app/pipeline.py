"""Data pipeline utilities powering AutoDA Auto Mode."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split

from .reporting import ReportInputs, build_report_summary

SUPPORTED_TABULAR_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}


@dataclass
class AutoModeResult:
    profile: dict[str, Any]
    cleaned: dict[str, Any]
    engineered: dict[str, Any]
    selected: dict[str, Any]
    split: dict[str, Any]
    training: dict[str, Any]
    reports: dict[str, Any]
    raw_df: pd.DataFrame
    cleaned_df: pd.DataFrame
    engineered_df: pd.DataFrame
    selected_df: pd.DataFrame


def _normalize_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with timezone-aware datetimes made naive."""
    if df.empty:
        return df
    converted = df.copy()
    tz_cols = converted.select_dtypes(include=["datetimetz"]).columns
    for col in tz_cols:
        converted[col] = converted[col].dt.tz_localize(None)
    return converted


def load_dataframe(file_path: Path) -> pd.DataFrame:
    """Load a tabular dataset from disk into a DataFrame."""
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_TABULAR_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type {ext}. Supported: {', '.join(sorted(SUPPORTED_TABULAR_EXTENSIONS))}"
        )

    def _read_csv(sep: str | None = None) -> pd.DataFrame:
        try:
            return pd.read_csv(
                file_path,
                sep=sep,
                engine="pyarrow",
                dtype_backend="pyarrow",
            )
        except Exception:
            effective_sep = sep if sep is not None else ","
            return pd.read_csv(file_path, sep=effective_sep, engine="python")

    if ext in {".csv", ".txt"}:
        df = _read_csv()
    elif ext == ".tsv":
        df = _read_csv(sep="\t")
    else:
        try:
            df = pd.read_excel(file_path, engine="openpyxl")
        except Exception:
            df = pd.read_excel(file_path)
    return _normalize_datetimes(df)


def detect_modalities(df: pd.DataFrame) -> list[str]:
    modalities = {"Tabular"}
    text_cols = df.select_dtypes(include=["object"]).columns
    if any(df[col].astype(str).str.len().mean() > 40 for col in text_cols):
        modalities.add("Text")
    return sorted(modalities)


def infer_semantic_type(series: pd.Series) -> str:
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return "Unknown"

    email_pattern = re.compile(r".+@.+\..+")
    phone_pattern = re.compile(r"^\+?\d[\d\-\s]{7,}$")
    currency_pattern = re.compile(r"^\$?\d+(\.\d{2})?$")

    if series.dtype.kind in {"M"}:
        return "Date"
    if series.dtype.kind in {"i", "u", "f"}:
        return "Numeric"
    if all(email_pattern.match(val) for val in sample):
        return "Email"
    if all(phone_pattern.match(val) for val in sample):
        return "Phone"
    if all(currency_pattern.match(val) for val in sample):
        return "Currency"
    if sample.str.contains(r"\b[A-Za-z]{2,}\b").mean() > 0.6:
        return "Text"
    return "Categorical"


def calc_data_quality_score(df: pd.DataFrame) -> int:
    if df.empty:
        return 100
    missing_penalty = df.isna().mean().mean() * 100
    duplicate_penalty = (df.duplicated().sum() / len(df) * 100) if len(df) else 0
    mixed_type_penalty = sum(
        series.map(type).nunique() > 3 for _, series in df.items()
    )
    score = 100 - (missing_penalty * 0.6 + duplicate_penalty * 0.3 + mixed_type_penalty * 2)
    return max(0, min(100, int(round(score))))


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    sample_df = df
    if len(df) > 50000:
        sample_df = df.sample(50000, random_state=42)

    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = sample_df.select_dtypes(exclude=[np.number]).columns.tolist()

    schema = []
    for col in sample_df.columns:
        dtype = str(sample_df[col].dtype)
        semantic = infer_semantic_type(sample_df[col])
        schema.append({"name": col, "dtype": dtype, "semantic": semantic})

    numeric_stats = []
    for col in numeric_cols[:8]:
        series = sample_df[col]
        desc = series.describe()
        summary = (
            f"Mean {desc['mean']:.2f} | Median {series.median():.2f} | "
            f"Std {desc['std']:.2f}"
        )
        numeric_stats.append({"label": col, "summary": summary})

    categorical_stats = []
    for col in cat_cols[:8]:
        series = sample_df[col].astype(str)
        cardinality = series.nunique()
        mode = series.mode().iloc[0] if not series.mode().empty else "n/a"
        top_freq = series.value_counts(normalize=True).head(1)
        top_label = top_freq.index[0] if not top_freq.empty else "n/a"
        top_pct = f"{top_freq.iloc[0]*100:.1f}%" if not top_freq.empty else "n/a"
        summary = f"{cardinality} unique | Mode {mode} | Top freq {top_label} {top_pct}"
        categorical_stats.append({"label": col, "summary": summary})

    missing = {
        col: f"{sample_df[col].isna().mean()*100:.1f}%"
        for col in sample_df.columns
        if sample_df[col].isna().any()
    }

    numeric_outliers = 0
    for col in numeric_cols:
        series = sample_df[col].dropna()
        if series.empty:
            continue
        q1, q3 = np.percentile(series, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        numeric_outliers += ((series < lower) | (series > upper)).sum()

    profile = {
        "schema": schema,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "missing": missing,
        "duplicates": int(sample_df.duplicated().sum()),
        "outliers": int(numeric_outliers),
        "text_insights": {
            "columns": [
                col
                for col in cat_cols
                if sample_df[col].astype(str).str.len().mean() > 40
            ],
            "avg_length": {
                col: float(sample_df[col].astype(str).str.len().mean())
                for col in cat_cols
            },
        },
        "visuals": [
            "Missingness matrix",
            "Histogram bundle",
            "Correlation heatmap",
            "Boxplots",
            "Data type distribution",
        ],
    }
    return profile


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict[str, Any]]:
    cleaned = df.copy()
    initial_rows = len(cleaned)
    duplicates = int(cleaned.duplicated().sum())
    if duplicates:
        cleaned = cleaned.drop_duplicates()

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    categorical_cols = cleaned.select_dtypes(include=["object", "category", "bool"]).columns
    datetime_cols = cleaned.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns

    missing_filled = 0
    for col in numeric_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
            missing_filled += cleaned[col].isna().sum()

    for col in categorical_cols:
        if cleaned[col].isna().any():
            mode = cleaned[col].mode().iloc[0] if not cleaned[col].mode().empty else "Unknown"
            cleaned[col] = cleaned[col].fillna(mode)
            missing_filled += cleaned[col].isna().sum()

    for col in datetime_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(method="ffill").fillna(method="bfill")

    outlier_caps = {}
    for col in numeric_cols:
        series = cleaned[col]
        q1, q3 = np.percentile(series, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        capped = series.clip(lower, upper)
        outlier_caps[col] = int((series != capped).sum())
        cleaned[col] = capped

    type_conversions = 0
    for col in categorical_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        type_conversions += 1

    metrics = {
        "rows_retained": f"{len(cleaned):,} of {initial_rows:,}",
        "duplicates_removed": duplicates,
        "missing_filled": missing_filled,
        "outliers_handled": int(sum(outlier_caps.values())),
        "type_conversions": type_conversions,
    }
    return cleaned, metrics


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict[str, Any]]:
    engineered = df.copy()
    new_features = []

    datetime_cols = engineered.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
    for col in datetime_cols:
        engineered[f"{col}_year"] = engineered[col].dt.year
        engineered[f"{col}_month"] = engineered[col].dt.month
        engineered[f"{col}_day"] = engineered[col].dt.day
        engineered[f"{col}_dow"] = engineered[col].dt.dayofweek
        engineered[f"{col}_is_weekend"] = engineered[col].dt.dayofweek.isin({5, 6}).astype(int)
        engineered[f"{col}_month_sin"] = np.sin(2 * np.pi * engineered[col].dt.month / 12)
        engineered[f"{col}_month_cos"] = np.cos(2 * np.pi * engineered[col].dt.month / 12)
        new_features.extend(
            [
                f"{col}_year",
                f"{col}_month",
                f"{col}_day",
                f"{col}_dow",
                f"{col}_is_weekend",
                f"{col}_month_sin",
                f"{col}_month_cos",
            ]
        )

    numeric_cols = engineered.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (engineered[col] > 0).all():
            engineered[f"{col}_log1p"] = np.log1p(engineered[col])
            new_features.append(f"{col}_log1p")
    for col in numeric_cols:
        engineered[f"{col}_sqrt"] = np.sqrt(engineered[col].clip(lower=0))
        new_features.append(f"{col}_sqrt")

    categorical_cols = engineered.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        engineered[f"{col}_length"] = engineered[col].astype(str).str.len()
        new_features.append(f"{col}_length")

    metrics = {
        "features_created": len(new_features),
        "new_features": new_features[:30],
        "dimensionality_change": f"{df.shape[1]} -> {engineered.shape[1]}",
    }
    return engineered, metrics


def prepare_model_matrix(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    aligned = df.dropna(subset=[target]).reset_index(drop=True)
    y = aligned[target].reset_index(drop=True)
    X = aligned.drop(columns=[target]).reset_index(drop=True)
    X = pd.get_dummies(X, drop_first=True)
    X = X.reset_index(drop=True)
    return X, y


def determine_task(y: pd.Series) -> str:
    if y.dtype.kind in {"i", "u", "f"}:
        if y.nunique() <= 15 and not pd.api.types.is_float_dtype(y):
            return "classification"
        if y.dtype.kind in {"f"} and y.nunique() < 10 and sorted(y.unique()) in ([0, 1], [0, 1, 2]):
            return "classification"
        return "regression"
    if y.nunique() <= 20:
        return "classification"
    return "classification"


def _align_X_y(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Ensure feature matrix and target vector share the same length/index."""
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    if len(X) == len(y):
        return X, y
    min_len = min(len(X), len(y))
    return X.iloc[:min_len].reset_index(drop=True), y.iloc[:min_len].reset_index(drop=True)


def select_features(
    df: pd.DataFrame, target: Optional[str]
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    if not target or target not in df.columns:
        metrics = {
            "strategy": "Dimensionality reduction skipped (no target defined).",
            "selected_features": list(df.columns),
        }
        return df, metrics

    X, y = prepare_model_matrix(df, target)
    # Simple variance-based selection: keep columns with non-zero variance
    variances = X.var()
    selected = variances[variances > 0].sort_values(ascending=False)
    top_features = selected.head(min(50, len(selected))).index.tolist()
    if not top_features:
        top_features = list(X.columns[: min(10, len(X.columns))])

    selected_df = pd.concat([X[top_features], y], axis=1).reset_index(drop=True)
    metrics = {
        "strategy": "Filter method via variance threshold",
        "selected_features": top_features,
        "dropped_features": list(set(X.columns) - set(top_features)),
    }
    return selected_df, metrics


def split_dataset(
    df: pd.DataFrame, target: Optional[str]
) -> Tuple[dict[str, Any], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
    if not target or target not in df.columns:
        metrics = {
            "message": "Target not configured. Split skipped.",
            "splits": [],
        }
        return metrics, None, None, None, None

    aligned = df.dropna(subset=[target]).reset_index(drop=True)
    X = aligned.drop(columns=[target]).reset_index(drop=True)
    y = aligned[target].reset_index(drop=True)

    stratify = y if determine_task(y) == "classification" and y.nunique() > 1 else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=stratify
    )
    stratify_temp = y_temp if stratify is not None else None
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=stratify_temp
    )

    metrics = {
        "strategy": "70/15/15 split with stratification" if stratify is not None else "70/15/15 split",
        "splits": [
            {
                "name": "Train",
                "rows": len(X_train),
                "class_balance": _class_balance(y_train),
            },
            {
                "name": "Validation",
                "rows": len(X_valid),
                "class_balance": _class_balance(y_valid),
            },
            {
                "name": "Test",
                "rows": len(X_test),
                "class_balance": _class_balance(y_test),
            },
        ],
    }
    return metrics, X_train.reset_index(drop=True), X_valid.reset_index(drop=True), y_train.reset_index(drop=True), y_valid.reset_index(drop=True)


def _class_balance(y: pd.Series) -> Optional[dict[str, str]]:
    if y is None or y.empty or y.nunique() > 15:
        return None
    counts = y.value_counts(normalize=True)
    return {str(cls): f"{pct*100:.1f}%" for cls, pct in counts.items()}


def train_and_select_model(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
) -> dict[str, Any]:
    X_train, y_train = _align_X_y(X_train, y_train)
    X_valid, y_valid = _align_X_y(X_valid, y_valid)

    task = determine_task(y_train)
    start = perf_counter()
    X_train_enc = pd.get_dummies(X_train, drop_first=True)
    X_valid_enc = pd.get_dummies(X_valid, drop_first=True)
    X_valid_enc = X_valid_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    leaderboard = []
    key_drivers: list[str] = []
    caveats: list[str] = []
    winner: Optional[dict[str, Any]] = None

    if X_train_enc.empty:
        return {
            "task": "insufficient_features",
            "leaderboard": [],
            "winner": None,
            "key_drivers": [],
            "caveats": [
                "No usable features after encoding. Adjust feature selection or include additional fields."
            ],
            "duration": 0,
        }

    if task == "classification":
        models = [
            ("Logistic Classifier", LogisticRegression(max_iter=1000)),
            ("Random Forest Classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
        for name, model in models:
            try:
                model_start = perf_counter()
                model.fit(X_train_enc, y_train)
                preds = model.predict(X_valid_enc)
                proba = getattr(model, "predict_proba", lambda X: None)(X_valid_enc)
                training_time = perf_counter() - model_start
                f1 = f1_score(y_valid, preds, average="weighted")
                accuracy = accuracy_score(y_valid, preds)
                precision = precision_score(y_valid, preds, average="weighted", zero_division=0)
                recall = recall_score(y_valid, preds, average="weighted", zero_division=0)
                roc_auc = None
                if proba is not None and y_valid.nunique() == 2:
                    roc_auc = roc_auc_score(y_valid, proba[:, 1])
                entry = {
                    "name": name,
                    "metric_primary": {"label": "F1", "value": round(f1, 3)},
                    "metric_secondary": {"label": "Accuracy", "value": round(accuracy, 3)},
                    "training_time": f"{training_time:.2f}s",
                    "precision": round(precision, 3),
                    "recall": round(recall, 3),
                    "roc_auc": round(roc_auc, 3) if roc_auc else None,
                }
                leaderboard.append(entry)
                if winner is None or entry["metric_primary"]["value"] > winner["metric_primary"]["value"]:
                    winner = entry
                    if hasattr(model, "feature_importances_"):
                        importances = model.feature_importances_
                        top_idx = np.argsort(importances)[::-1][:5]
                        key_drivers = [
                            X_train_enc.columns[idx]
                            for idx in top_idx
                            if importances[idx] > 0
                        ]
                    elif hasattr(model, "coef_"):
                        coefs = model.coef_[0]
                        top_idx = np.argsort(np.abs(coefs))[::-1][:5]
                        key_drivers = [X_train_enc.columns[idx] for idx in top_idx]
            except Exception as exc:  # pragma: no cover - defensive fallback
                caveats.append(f"{name} failed: {exc}")

    elif task == "regression":
        models = [
            ("Linear Regression", LinearRegression()),
            ("Random Forest Regressor", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]
        for name, model in models:
            try:
                model_start = perf_counter()
                model.fit(X_train_enc, y_train)
                preds = model.predict(X_valid_enc)
                training_time = perf_counter() - model_start
                r2 = r2_score(y_valid, preds)
                rmse = math.sqrt(mean_squared_error(y_valid, preds))
                mae = mean_absolute_error(y_valid, preds)
                entry = {
                    "name": name,
                    "metric_primary": {"label": "R^2", "value": round(r2, 3)},
                    "metric_secondary": {"label": "RMSE", "value": round(rmse, 3)},
                    "training_time": f"{training_time:.2f}s",
                    "mae": round(mae, 3),
                }
                leaderboard.append(entry)
                if winner is None or entry["metric_primary"]["value"] > winner["metric_primary"]["value"]:
                    winner = entry
                    if hasattr(model, "feature_importances_"):
                        importances = model.feature_importances_
                        top_idx = np.argsort(importances)[::-1][:5]
                        key_drivers = [
                            X_train_enc.columns[idx] for idx in top_idx if importances[idx] > 0
                        ]
                    elif hasattr(model, "coef_"):
                        coefs = model.coef_
                        top_idx = np.argsort(np.abs(coefs))[::-1][:5]
                        key_drivers = [X_train_enc.columns[idx] for idx in top_idx]
            except Exception as exc:  # pragma: no cover - defensive fallback
                caveats.append(f"{name} failed: {exc}")
        caveats.append("Check for heteroscedasticity in residuals.")
    else:  # clustering fallback
        kmeans = KMeans(n_clusters=min(5, len(X_train_enc)), random_state=42, n_init=10)
        kmeans.fit(X_train_enc)
        preds = kmeans.predict(X_valid_enc)
        silhouette = silhouette_score(X_valid_enc, preds) if len(X_valid_enc) >= 2 else 0
        leaderboard.append(
            {
                "name": "KMeans clustering",
                "metric_primary": {"label": "Silhouette", "value": round(silhouette, 3)},
                "metric_secondary": {"label": "Clusters", "value": int(kmeans.n_clusters)},
                "training_time": f"{perf_counter() - start:.2f}s",
            }
        )
        winner = leaderboard[0]
        key_drivers = list(X_train_enc.columns[:5])

    duration = perf_counter() - start
    training_summary = {
        "task": task,
        "leaderboard": leaderboard,
        "winner": winner,
        "key_drivers": key_drivers,
        "caveats": caveats,
        "duration": duration,
    }
    return training_summary


def run_auto_mode(
    df: pd.DataFrame, project_name: str, target: Optional[str]
) -> AutoModeResult:
    profile = profile_dataframe(df)
    cleaned_df, cleaning_metrics = clean_dataframe(df)
    engineered_df, fe_metrics = engineer_features(cleaned_df)
    selected_df, selection_metrics = select_features(engineered_df, target)
    split_metrics, X_train, X_valid, y_train, y_valid = split_dataset(selected_df, target)
    if X_train is not None and X_valid is not None:
        training_metrics = train_and_select_model(X_train, X_valid, y_train, y_valid)
    else:
        training_metrics = {
            "task": "unsupervised",
            "leaderboard": [],
            "winner": None,
            "key_drivers": [],
            "caveats": ["Target column not set. Configure a target to enable modeling."],
            "duration": 0,
        }

    dataset_info = {
        "name": project_name,
        "rows_raw": int(len(df)),
        "columns": int(df.shape[1]),
        "rows_cleaned": int(len(cleaned_df)),
        "columns_cleaned": int(cleaned_df.shape[1]),
        "modalities": detect_modalities(df),
    }
    raw_quality = calc_data_quality_score(df)
    cleaned_quality = calc_data_quality_score(cleaned_df)
    data_quality_summary = {
        "raw": raw_quality,
        "cleaned": cleaned_quality,
    }
    cleaned_summary = {
        "metrics": cleaning_metrics,
        "quality_score": cleaned_quality,
    }
    engineered_summary = {
        "total_new": fe_metrics.get("features_created"),
        "new_features": fe_metrics.get("new_features"),
        "dimensionality_change": fe_metrics.get("dimensionality_change"),
        "metrics": fe_metrics,
    }
    selection_summary = {
        **selection_metrics,
        "count": len(selection_metrics.get("selected_features", [])),
    }
    report_inputs = ReportInputs(
        project_name=project_name,
        target=target,
        dataset=dataset_info,
        profiling=profile,
        cleaning=cleaned_summary,
        feature_engineering=engineered_summary,
        feature_selection=selection_summary,
        split=split_metrics,
        training=training_metrics,
        data_quality=data_quality_summary,
    )
    reports = build_report_summary(report_inputs)
    result = AutoModeResult(
        profile=profile,
        cleaned=cleaned_summary,
        engineered=engineered_summary,
        selected=selection_summary,
        split=split_metrics,
        training=training_metrics,
        reports=reports,
        raw_df=df,
        cleaned_df=cleaned_df,
        engineered_df=engineered_df,
        selected_df=selected_df,
    )
    return result


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024**3:
        return f"{num_bytes / 1024**2:.1f} MB"
    return f"{num_bytes / 1024**3:.1f} GB"
