from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import json
import joblib
import pandas as pd
from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from scipy import sparse

from .nlp_pipeline import (
    extract_tfidf_features,
    load_text_dataset,
    preprocess_texts,
    profile_texts,
    train_quick_text_classifier,
)
from .pipeline import format_size
from .state import (
    add_artifact as add_project_artifact,
    get_current_project,
    get_project_dict,
)
from .routes import current_nav_items, build_download_menu


nlp_bp = Blueprint("nlp", __name__, template_folder="../templates", static_folder="../static")


@dataclass
class NLPJob:
    id: str
    raw_path: Optional[Path] = None
    raw_filename: Optional[str] = None
    columns: list[str] = field(default_factory=list)
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    preprocessing_options: dict[str, Any] = field(
        default_factory=lambda: {
            "remove_stopwords": True,
            "lemmatize": True,
            "sample_size": None,
        }
    )
    samples: list[dict[str, str]] = field(default_factory=list)
    profiling: Optional[dict[str, Any]] = None
    feature_summary: Optional[dict[str, Any]] = None
    model_summary: Optional[dict[str, Any]] = None
    cleaned_path: Optional[Path] = None
    vectorizer_path: Optional[Path] = None
    feature_matrix_path: Optional[Path] = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["raw_path"] = str(self.raw_path) if self.raw_path else None
        payload["cleaned_path"] = str(self.cleaned_path) if self.cleaned_path else None
        payload["vectorizer_path"] = str(self.vectorizer_path) if self.vectorizer_path else None
        payload["feature_matrix_path"] = str(self.feature_matrix_path) if self.feature_matrix_path else None
        return payload


NLP_JOB_STORE: dict[str, NLPJob] = {}
def _uploads_dir() -> Path:
    return Path(current_app.config["UPLOAD_FOLDER"])


def _get_job(create: bool = True) -> NLPJob:
    """
    Retrieve or create NLP job with persistence to prevent data loss.
    
    The job is stored in NLP_JOB_STORE (in-memory) but also persisted to
    the project's analysis_requirements field for recovery across requests.
    """
    job_id = session.get("nlp_job_id")
    
    # Check in-memory store first
    if job_id and job_id in NLP_JOB_STORE:
        return NLP_JOB_STORE[job_id]
    
    # Try to restore from project persistence
    if job_id:
        project = get_current_project(allow_none=True)
        if project and isinstance(project.analysis_requirements, dict) and project.analysis_requirements.get("nlp_job_data"):
            try:
                job_data = project.analysis_requirements["nlp_job_data"]
                if not isinstance(job_data, dict):
                    raise ValueError("Invalid job_data format")
                job = NLPJob(
                    id=job_data["id"],
                    raw_path=Path(job_data["raw_path"]) if job_data.get("raw_path") else None,
                    raw_filename=job_data.get("raw_filename"),
                    columns=job_data.get("columns", []),
                    text_column=job_data.get("text_column"),
                    label_column=job_data.get("label_column"),
                    preprocessing_options=job_data.get("preprocessing_options", {
                        "remove_stopwords": True,
                        "lemmatize": True,
                        "sample_size": None,
                    }),
                    samples=job_data.get("samples", []),
                    profiling=job_data.get("profiling"),
                    feature_summary=job_data.get("feature_summary"),
                    model_summary=job_data.get("model_summary"),
                    cleaned_path=Path(job_data["cleaned_path"]) if job_data.get("cleaned_path") else None,
                    vectorizer_path=Path(job_data["vectorizer_path"]) if job_data.get("vectorizer_path") else None,
                    feature_matrix_path=Path(job_data["feature_matrix_path"]) if job_data.get("feature_matrix_path") else None,
                    artifacts=job_data.get("artifacts", []),
                    messages=job_data.get("messages", []),
                )
                NLP_JOB_STORE[job.id] = job
                return job
            except Exception:
                # If restoration fails, create new job below
                pass
    
    if not create:
        raise KeyError("NLP job not found")
    
    # Create new job
    job = NLPJob(id=uuid4().hex)
    NLP_JOB_STORE[job.id] = job
    session["nlp_job_id"] = job.id
    _persist_nlp_job(job)
    return job


def _persist_nlp_job(job: NLPJob) -> None:
    """Save NLP job data to project for persistence across requests."""
    try:
        project = get_current_project(allow_none=True)
        if project:
            # Ensure analysis_requirements is a dict
            if not isinstance(project.analysis_requirements, dict):
                project.analysis_requirements = {}
            project.analysis_requirements["nlp_job_data"] = job.to_dict()
            from .state import _sync_meta
            _sync_meta(project)
    except Exception as e:
        # Fail silently - job is still in memory for this session
        print(f"WARNING: Failed to persist NLP job: {e}")
        pass


def _load_raw_dataframe(job: NLPJob) -> pd.DataFrame:
    if not job.raw_path or not job.raw_path.exists():
        raise FileNotFoundError("Upload a dataset first.")
    if job.raw_path.suffix.lower() == ".csv":
        return pd.read_csv(job.raw_path)
    df, _ = load_text_dataset(job.raw_path, job.raw_path.name)
    return df


def _add_artifact(job: NLPJob, path: Path, label: str) -> dict[str, Any]:
    size = format_size(path.stat().st_size) if path.exists() else "0 B"
    artifact = {
        "name": label,
        "path": str(path),
        "filename": path.name,
        "size": size,
    }
    job.artifacts.append(artifact)

    project = get_current_project(allow_none=True)
    if project:
        add_project_artifact(
            project,
            name=path.name,
            artifact_type=label,
            size=size,
            description=f"{label} generated via NLP automation.",
            path=str(path),
        )
    return artifact


def _nlp_pipeline_status(job: NLPJob) -> tuple[list[dict[str, str]], str]:
    """Build AutoDA-style pipeline rows for the NLP workflow."""

    def _trained(summary: Optional[dict[str, Any]]) -> bool:
        return bool(summary and summary.get("status") == "trained")

    stages = [
        ("Data Ingestion", bool(job.raw_path)),
        ("Text Cleaning & Preprocessing", bool(job.cleaned_path)),
        ("NLP Automation", bool(job.profiling)),
        ("Feature Selection", bool(job.feature_summary)),
        ("Split & Validation", bool(job.feature_summary) and _trained(job.model_summary)),
        ("Training & Selection", _trained(job.model_summary)),
        ("Reports", bool(job.artifacts)),
    ]

    pipeline: list[dict[str, str]] = []
    active_set = False
    current_stage = stages[0][0]
    for idx, (name, complete) in enumerate(stages):
        if complete:
            status = "complete"
        elif not active_set:
            status = "in_progress"
            current_stage = name
            active_set = True
        else:
            status = "pending"
        pipeline.append({"name": name, "status": status})
        if status == "complete":
            current_stage = name
    return pipeline, current_stage


def _render_step(template: str, page_title: str, page_help: str, active_endpoint: str):
    project = get_current_project(allow_none=True)
    if project is None:
        flash("Create or select a project before launching the NLP workflow.", "warning")
        return redirect(url_for("main.projects"))
    session["nav_mode"] = "nlp"
    job = _get_job()
    pipeline_phases, stage_name = _nlp_pipeline_status(job)
    project_view = get_project_dict(project)
    page = {
        "title": page_title,
        "help": page_help,
        "breadcrumbs": [
            {"label": "Workflow Hub", "endpoint": "main.workflow"},
            {"label": page_title, "endpoint": None},
        ],
        "current_step": stage_name,
    }
    return render_template(
        template,
        job=job.to_dict(),
        project=project_view,
        page=page,
        nav_items=current_nav_items(),
        pipeline_phases=pipeline_phases,
        download_options=build_download_menu(project),
        artifacts=project.artifacts,
        active_endpoint=active_endpoint,
        current_stage=stage_name,
    )


@nlp_bp.route("/nlp")
def automation():
    return redirect(url_for("nlp.ingestion"))


@nlp_bp.route("/nlp/ingestion")
def ingestion():
    return _render_step(
        "nlp/steps/ingestion.html",
        "Data Ingestion",
        "Upload CSV/TXT/ZIP text data and register it with this project.",
        "nlp.ingestion",
    )


@nlp_bp.route("/nlp/preprocess")
def preprocessing():
    return _render_step(
        "nlp/steps/preprocess.html",
        "Text Cleaning & Preprocessing",
        "Configure text/label columns, normalization options, and inspect before/after samples.",
        "nlp.preprocessing",
    )


@nlp_bp.route("/nlp/automation-stage")
def automation_stage():
    return _render_step(
        "nlp/steps/automation.html",
        "NLP Automation",
        "Generate profiling statistics, vocabulary insights, and corpus distribution checks.",
        "nlp.automation_stage",
    )


@nlp_bp.route("/nlp/features")
def features():
    return _render_step(
        "nlp/steps/features.html",
        "Feature Selection",
        "Extract TF-IDF features (unigrams + bigrams) ready for downstream modeling.",
        "nlp.features",
    )


@nlp_bp.route("/nlp/split")
def split():
    return _render_step(
        "nlp/steps/split.html",
        "Split & Validation",
        "Review the train/test allocation and validation checks before modeling.",
        "nlp.split",
    )


@nlp_bp.route("/nlp/training")
def training():
    return _render_step(
        "nlp/steps/training.html",
        "Training & Selection",
        "Train a lightweight classifier and inspect performance metrics.",
        "nlp.training",
    )


@nlp_bp.route("/nlp/reports")
def reports():
    return _render_step(
        "nlp/steps/reports.html",
        "Reports",
        "Download every artifact generated during the NLP automation run.",
        "nlp.reports",
    )


@nlp_bp.route("/nlp/reset", methods=["POST"])
def reset_job():
    job_id = session.pop("nlp_job_id", None)
    if job_id and job_id in NLP_JOB_STORE:
        NLP_JOB_STORE.pop(job_id)
    flash("NLP automation workspace reset.", "info")
    return redirect(url_for("nlp.ingestion"))


@nlp_bp.route("/nlp/upload", methods=["POST"])
def upload_text_data():
    job = _get_job()
    file = request.files.get("nlp_file")
    if not file or file.filename == "":
        flash("Select a CSV/TXT/ZIP file containing text data.", "warning")
        return redirect(url_for("nlp.ingestion"))

    safe_name = file.filename.replace(" ", "_")
    upload_path = _uploads_dir() / f"{job.id}_{safe_name}"
    file.save(upload_path)

    try:
        df, metadata = load_text_dataset(upload_path, file.filename)
    except Exception as exc:  # pragma: no cover - user feedback
        upload_path.unlink(missing_ok=True)
        flash(f"Upload failed: {exc}", "danger")
        return redirect(url_for("nlp.ingestion"))

    job.raw_path = upload_path
    job.raw_filename = file.filename
    job.columns = df.columns.tolist()
    job.text_column = metadata.get("text_column") or (job.columns[0] if job.columns else None)
    job.label_column = None
    job.samples = []
    job.profiling = None
    job.feature_summary = None
    job.model_summary = None
    job.cleaned_path = None
    job.vectorizer_path = None
    job.feature_matrix_path = None
    job.artifacts = []
    
    # Persist job data to prevent loss on page navigation
    _persist_nlp_job(job)
    
    flash(f"Uploaded {file.filename} with {len(df)} rows.", "success")
    return redirect(url_for("nlp.preprocessing"))


@nlp_bp.route("/nlp/preprocess", methods=["POST"])
def preprocess_text():
    job = _get_job()
    if not job.raw_path:
        flash("Upload a dataset before preprocessing.", "warning")
        return redirect(url_for("nlp.preprocessing"))

    df = _load_raw_dataframe(job)

    job.text_column = request.form.get("text_column") or job.text_column
    job.label_column = request.form.get("label_column") or None
    if not job.text_column or job.text_column not in df.columns:
        flash("Select a valid text column to preprocess.", "warning")
        return redirect(url_for("nlp.preprocessing"))

    job.preprocessing_options["remove_stopwords"] = request.form.get("remove_stopwords") == "on"
    job.preprocessing_options["lemmatize"] = request.form.get("lemmatize") == "on"
    sample_size = request.form.get("sample_size")
    job.preprocessing_options["sample_size"] = int(sample_size) if sample_size else None

    series = df[job.text_column].astype(str)
    cleaned_series, samples = preprocess_texts(
        series,
        remove_stopwords=job.preprocessing_options["remove_stopwords"],
        lemmatize=job.preprocessing_options["lemmatize"],
    )
    job.samples = samples

    cleaned_df = df.copy()
    cleaned_df[f"{job.text_column}_cleaned"] = cleaned_series
    cleaned_path = _uploads_dir() / f"{job.id}_cleaned.csv"
    cleaned_df.to_csv(cleaned_path, index=False)
    job.cleaned_path = cleaned_path
    _add_artifact(job, cleaned_path, "Cleaned text (CSV)")
    
    # Persist job data
    _persist_nlp_job(job)
    
    flash("Text preprocessing completed.", "success")
    return redirect(url_for("nlp.preprocessing"))


@nlp_bp.route("/nlp/profile", methods=["POST"])
def profile_text():
    job = _get_job()
    if not job.cleaned_path or not job.cleaned_path.exists():
        flash("Run preprocessing before profiling.", "warning")
        return redirect(url_for("nlp.automation_stage"))

    cleaned_df = pd.read_csv(job.cleaned_path)
    cleaned_column = f"{job.text_column}_cleaned"
    profile = profile_texts(
        cleaned_df[cleaned_column],
        sample_size=job.preprocessing_options.get("sample_size"),
    )
    job.profiling = profile

    profile_path = _uploads_dir() / f"{job.id}_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    _add_artifact(job, profile_path, "Profile summary (JSON)")
    
    # Persist job data
    _persist_nlp_job(job)
    
    flash("Generated profiling summary.", "success")
    return redirect(url_for("nlp.automation_stage"))


@nlp_bp.route("/nlp/features", methods=["POST"])
def build_features():
    job = _get_job()
    if not job.cleaned_path or not job.cleaned_path.exists():
        flash("Run preprocessing before extracting features.", "warning")
        return redirect(url_for("nlp.features"))

    max_features = int(request.form.get("max_features") or 5000)
    cleaned_df = pd.read_csv(job.cleaned_path)
    cleaned_column = f"{job.text_column}_cleaned"

    vectorizer, matrix, feature_info = extract_tfidf_features(
        cleaned_df[cleaned_column],
        max_features=max_features,
    )
    job.feature_summary = feature_info

    feature_matrix_path = _uploads_dir() / f"{job.id}_tfidf_features.npz"
    sparse.save_npz(feature_matrix_path, matrix)
    job.feature_matrix_path = feature_matrix_path
    _add_artifact(job, feature_matrix_path, "TF-IDF Features (NPZ)")

    vectorizer_path = _uploads_dir() / f"{job.id}_vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_path)
    job.vectorizer_path = vectorizer_path
    _add_artifact(job, vectorizer_path, "TF-IDF Vectorizer (PKL)")
    
    # Persist job data
    _persist_nlp_job(job)
    
    flash(f"Extracted TF-IDF features ({feature_info['feature_count']} terms).", "success")
    return redirect(url_for("nlp.features"))


@nlp_bp.route("/nlp/train", methods=["POST"])
def train_text_model():
    job = _get_job()
    if not job.vectorizer_path or not job.feature_matrix_path:
        flash("Extract features before training.", "warning")
        return redirect(url_for("nlp.training"))

    if not job.label_column:
        flash("Select a label column to train a classifier.", "warning")
        return redirect(url_for("nlp.training"))

    cleaned_df = pd.read_csv(job.cleaned_path)
    vectorizer = joblib.load(job.vectorizer_path)

    metrics, model = train_quick_text_classifier(
        vectorizer,
        cleaned_df[f"{job.text_column}_cleaned"],
        cleaned_df[job.label_column],
    )
    job.model_summary = metrics
    if model and metrics.get("status") == "trained":
        model_path = _uploads_dir() / f"{job.id}_text_model.pkl"
        joblib.dump(model, model_path)
        _add_artifact(job, model_path, "Text model (PKL)")
    
    # Persist job data
    _persist_nlp_job(job)
    
    flash("Training finished.", "success")
    return redirect(url_for("nlp.training"))


@nlp_bp.route("/nlp/download/<path:filename>")
def download_artifact(filename: str):
    job = _get_job()
    for artifact in job.artifacts:
        if Path(artifact["path"]).name == filename:
            return send_file(artifact["path"], as_attachment=True)
    flash("Artifact not found.", "warning")
    return redirect(url_for("nlp.reports"))
