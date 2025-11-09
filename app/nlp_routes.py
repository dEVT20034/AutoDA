"""Standalone blueprint powering the NLP Hub workflow.

To enable these routes in the existing Flask application, add the snippet below
to your application factory after creating the `Flask` instance:

    # from .nlp_routes import nlp_bp
    # app.register_blueprint(nlp_bp)

The blueprint is intentionally isolated so that the AutoDA structured pipeline
remains unchanged. All artifacts and audit logs are written to
``artifacts/nlp/<job_id>/`` relative to the project root.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from werkzeug.utils import secure_filename
from scipy import sparse
import joblib

from .nlp_pipeline import (
    SUPPORTED_TEXT_EXTENSIONS,
    ensure_nltk_resources,
    load_text_dataset,
    preprocess_texts,
    profile_texts,
)


nlp_bp = Blueprint("nlp", __name__, template_folder="../templates", static_folder="../static")


@dataclass
class AuditEntry:
    timestamp: str
    step: str
    summary: str


@dataclass
class JobState:
    id: str
    created_at: str
    current_step: str = "ingestion"
    last_action: Optional[str] = None
    completed_steps: set[str] = field(default_factory=set)
    columns: list[str] = field(default_factory=list)
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    data_quality: dict[str, Optional[int]] = field(
        default_factory=lambda: {"raw": None, "cleaned": None}
    )
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    audit: list[AuditEntry] = field(default_factory=list)
    profiling_result: Optional[dict[str, Any]] = None
    cleaning_preview: list[dict[str, str]] = field(default_factory=list)
    features_result: Optional[dict[str, Any]] = None
    training_result: Optional[dict[str, Any]] = None
    text_candidates: list[str] = field(default_factory=list)
    dir: Optional[Path] = None
    dataset_path: Optional[Path] = None
    cleaned_path: Optional[Path] = None
    vectorizer_path: Optional[Path] = None
    feature_matrix_path: Optional[Path] = None

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["completed_steps"] = list(self.completed_steps)
        payload["artifacts"] = self.artifacts
        payload["audit"] = [asdict(entry) for entry in self.audit]
        payload["dir"] = str(self.dir) if self.dir else None
        payload["dataset_path"] = str(self.dataset_path) if self.dataset_path else None
        payload["cleaned_path"] = str(self.cleaned_path) if self.cleaned_path else None
        payload["vectorizer_path"] = str(self.vectorizer_path) if self.vectorizer_path else None
        payload["feature_matrix_path"] = (
            str(self.feature_matrix_path) if self.feature_matrix_path else None
        )
        return payload


JOB_STORE: dict[str, JobState] = {}
STEPS = ["ingestion", "profiling", "cleaning", "features", "train", "reports"]


def _artifact_root() -> Path:
    base = Path(current_app.root_path).parent
    target = base / "artifacts" / "nlp"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _job_dir(job_id: str) -> Path:
    path = _artifact_root() / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_job(job_id: str) -> JobState:
    if job_id in JOB_STORE:
        return JOB_STORE[job_id]
    job_path = _job_dir(job_id)
    job_file = job_path / "job.json"
    if not job_file.exists():
        abort(404, description="Job not found.")
    data = json.loads(job_file.read_text(encoding="utf-8"))
    job = JobState(
        id=data["id"],
        created_at=data["created_at"],
        current_step=data.get("current_step", "ingestion"),
    )
    job.completed_steps = set(data.get("completed_steps", []))
    job.columns = data.get("columns", [])
    job.text_column = data.get("text_column")
    job.label_column = data.get("label_column")
    job.data_quality = data.get("data_quality", {"raw": None, "cleaned": None})
    job.artifacts = data.get("artifacts", [])
    job.audit = [AuditEntry(**entry) for entry in data.get("audit", [])]
    job.profiling_result = data.get("profiling_result")
    job.cleaning_preview = data.get("cleaning_preview", [])
    job.features_result = data.get("features_result")
    job.training_result = data.get("training_result")
    job.text_candidates = data.get("text_candidates", [])
    job.last_action = data.get("last_action")
    if data.get("dir"):
        job.dir = Path(data["dir"])
    else:
        job.dir = job_path
    job.dataset_path = Path(data["dataset_path"]) if data.get("dataset_path") else None
    job.cleaned_path = Path(data["cleaned_path"]) if data.get("cleaned_path") else None
    job.vectorizer_path = Path(data["vectorizer_path"]) if data.get("vectorizer_path") else None
    job.feature_matrix_path = (
        Path(data["feature_matrix_path"]) if data.get("feature_matrix_path") else None
    )
    JOB_STORE[job_id] = job
    return job


def _create_job() -> JobState:
    job_id = uuid4().hex
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    job = JobState(id=job_id, created_at=now, current_step="ingestion")
    job.dir = _job_dir(job_id)
    _persist_job(job)
    JOB_STORE[job_id] = job
    return job


def _persist_job(job: JobState) -> None:
    if not job.dir:
        job.dir = _job_dir(job.id)
    payload = job.to_json()
    (job.dir / "job.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _format_size(num: int) -> str:
    if num is None:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    size = float(num)
    for unit in units:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _artifact_filename(job: JobState, step: str, original: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    safe = secure_filename(original)
    filename = f"{job.id}__{step}__{timestamp}__{safe}"
    return job.dir / filename


def _register_artifact(job: JobState, step: str, name: str, path: Path) -> dict[str, Any]:
    artifact = {
        "name": name,
        "filename": path.name,
        "url": url_for("nlp.download_artifact", job_id=job.id, filename=path.name, _external=True),
        "size": _format_size(path.stat().st_size),
        "ts": _timestamp(),
        "step": step,
    }
    job.artifacts.append(artifact)
    return artifact


def _append_audit(job: JobState, step: str, summary: str) -> None:
    entry = AuditEntry(timestamp=_timestamp(), step=step, summary=summary)
    job.audit.insert(0, entry)
    job.audit = job.audit[:50]


def _update_step(job: JobState, step: str) -> None:
    job.completed_steps.add(step)
    job.last_action = _timestamp()
    for candidate in STEPS:
        if candidate not in job.completed_steps:
            job.current_step = candidate
            break
    else:
        job.current_step = "reports"


def _job_summary(job: JobState) -> dict[str, Any]:
    return {
        "id": job.id,
        "current_step": job.current_step,
        "last_action": job.last_action,
        "data_quality": job.data_quality,
        "artifact_count": len(job.artifacts),
        "completed_steps": sorted(job.completed_steps),
    }


def _audit_payload(job: JobState) -> list[dict[str, Any]]:
    return [asdict(entry) for entry in job.audit[:20]]


@nlp_bp.route("/nlp/")
def home() -> str:
    def _recent_entry(job: JobState) -> dict[str, Any]:
        return {
            "name": f"NLP Job {job.id[:6]}",
            "type": "NLP",
            "last_action": job.last_action or job.created_at,
            "artifacts": len(job.artifacts),
            "status": job.current_step.title(),
            "link": url_for("nlp.pipeline_view", job_id=job.id),
        }

    recent_jobs = sorted(JOB_STORE.values(), key=lambda j: j.last_action or j.created_at, reverse=True)[:3]
    return render_template("nlp/home.html", recent_jobs=[_recent_entry(job) for job in recent_jobs])


@nlp_bp.route("/nlp/pipeline")
def pipeline_new():
    job = _create_job()
    return redirect(url_for("nlp.pipeline_view", job_id=job.id))


@nlp_bp.route("/nlp/pipeline/<job_id>")
def pipeline_view(job_id: str):
    job = _load_job(job_id)
    steps = [
        {"key": "ingestion", "label": "Ingestion", "anchor": "ingestion"},
        {"key": "profiling", "label": "Profiling", "anchor": "profiling"},
        {"key": "cleaning", "label": "Cleaning", "anchor": "cleaning"},
        {"key": "features", "label": "Feature Engineering", "anchor": "features"},
        {"key": "train", "label": "Model Train", "anchor": "train"},
        {"key": "reports", "label": "Reports & Artifacts", "anchor": "reports"},
    ]
    context = {
        "job": {
            "id": job.id,
            "current_step": job.current_step,
            "last_action": job.last_action,
            "data_quality": job.data_quality,
            "artifact_count": len(job.artifacts),
            "columns": job.columns,
            "text_column": job.text_column,
            "label_column": job.label_column,
            "completed_steps": sorted(job.completed_steps),
            "audit": _audit_payload(job),
        },
        "steps": steps,
    }
    return render_template("nlp/pipeline.html", **context)


@nlp_bp.route("/nlp/upload", methods=["POST"])
def upload():
    job_id = request.form.get("job_id")
    if not job_id:
        abort(400, description="job_id is required.")
    job = _load_job(job_id)

    file = request.files.get("nlp_file")
    if not file or file.filename == "":
        abort(400, description="Upload a CSV, TXT, or ZIP file.")

    original_name = file.filename
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_TEXT_EXTENSIONS:
        abort(400, description="Unsupported file type for NLP ingestion.")

    dataset_path = job.dir / f"raw_{secure_filename(original_name)}"
    file.save(dataset_path)

    try:
        df, metadata = load_text_dataset(dataset_path, original_name)
    except Exception as exc:  # pragma: no cover - defensive
        dataset_path.unlink(missing_ok=True)
        abort(400, description=str(exc))

    sample_size = request.form.get("sample_size")
    if sample_size:
        try:
            sample_size_int = max(1, int(sample_size))
            df = df.head(sample_size_int)
        except ValueError:
            pass

    df = df.reset_index(drop=True)
    preview = df.head(5).to_dict(orient="records")
    columns = df.columns.tolist()
    candidates = [
        column for column in columns if pd.api.types.is_string_dtype(df[column]) or df[column].dtype == object
    ]
    text_column = request.form.get("text_column") or (candidates[0] if candidates else None)
    if text_column and text_column not in columns:
        text_column = candidates[0] if candidates else None
    label_column = request.form.get("label_column")
    if label_column and label_column not in columns:
        label_column = None

    dataset_csv = _artifact_filename(job, "ingestion", "dataset.csv")
    df.to_csv(dataset_csv, index=False)
    job.dataset_path = dataset_csv
    job.columns = columns
    job.text_column = text_column
    job.label_column = label_column
    job.text_candidates = candidates
    job.data_quality["raw"] = len(df)
    _register_artifact(job, "ingestion", "Raw dataset (CSV)", dataset_csv)
    _append_audit(job, "ingestion", f"Uploaded {original_name} with {len(df)} rows.")
    _update_step(job, "ingestion")
    _persist_job(job)

    response = {
        "job_id": job.id,
        "step": "ingestion",
        "status": "done",
        "columns": columns,
        "text_column": text_column,
        "label_column": label_column,
        "preview": preview,
        "summary": {
            "Rows": len(df),
            "Columns": metadata.get("columns", columns),
            "Source": metadata.get("source_type", extension.strip(".")),
        },
        "job": _job_summary(job),
        "audit": _audit_payload(job),
    }
    return jsonify(response)


@nlp_bp.route("/nlp/run_step", methods=["POST"])
def run_step():
    payload = request.get_json(silent=True) or {}
    job_id = payload.get("job_id")
    step = payload.get("step")
    options = payload.get("options", {})

    if not job_id or not step:
        abort(400, description="job_id and step are required.")

    job = _load_job(job_id)

    if step == "profiling":
        return _run_profiling(job, options)
    if step == "cleaning":
        return _run_cleaning(job, options)
    if step == "features":
        return _run_features(job, options)
    if step == "train":
        return _run_training(job, options)
    if step == "reports":
        return _run_reports(job)

    abort(400, description="Unsupported step.")


def _ensure_dataset(job: JobState) -> pd.DataFrame:
    if not job.dataset_path or not job.dataset_path.exists():
        abort(400, description="Upload data before running this step.")
    return pd.read_csv(job.dataset_path)


def _ensure_cleaned(job: JobState) -> pd.DataFrame:
    if not job.cleaned_path or not job.cleaned_path.exists():
        abort(400, description="Run cleaning before this step.")
    return pd.read_csv(job.cleaned_path)


def _run_profiling(job: JobState, options: dict[str, Any]):
    df = _ensure_dataset(job)
    text_column = options.get("text_column") or job.text_column
    if not text_column or text_column not in df.columns:
        abort(400, description="Select a text column before profiling.")

    sample_size = options.get("sample_size")
    sample_size = int(sample_size) if sample_size else None
    result = profile_texts(df[text_column], sample_size=sample_size)
    job.profiling_result = result

    profile_path = _artifact_filename(job, "profiling", "profiling.json")
    profile_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    artifact = _register_artifact(job, "profiling", "Profiling report (JSON)", profile_path)
    _append_audit(job, "profiling", "Generated text profiling statistics.")
    _update_step(job, "profiling")
    _persist_job(job)

    response = {
        "job_id": job.id,
        "step": "profiling",
        "status": "done",
        "result": result,
        "artifacts": job.artifacts,
        "job": _job_summary(job),
        "audit": _audit_payload(job),
    }
    return jsonify(response)


def _run_cleaning(job: JobState, options: dict[str, Any]):
    df = _ensure_dataset(job)
    text_column = job.text_column
    if not text_column or text_column not in df.columns:
        abort(400, description="Set a valid text column before cleaning.")

    ensure_nltk_resources()
    remove_stopwords = bool(options.get("remove_stopwords", True))
    lemmatize = bool(options.get("lemmatize", True))
    cleaned_series, examples = preprocess_texts(
        df[text_column],
        remove_stopwords=remove_stopwords,
        lemmatize=lemmatize,
    )
    cleaned_df = df.copy()
    cleaned_df[f"{text_column}_cleaned"] = cleaned_series

    cleaned_path = _artifact_filename(job, "cleaning", "cleaned.csv")
    cleaned_df.to_csv(cleaned_path, index=False)
    job.cleaned_path = cleaned_path
    job.cleaning_preview = examples
    job.data_quality["cleaned"] = int(cleaned_series.astype(bool).sum())

    artifact = _register_artifact(job, "cleaning", "Cleaned dataset (CSV)", cleaned_path)
    _append_audit(
        job,
        "cleaning",
        f"Cleaned {len(cleaned_df)} documents (stopwords={remove_stopwords}, lemmatize={lemmatize}).",
    )
    _update_step(job, "cleaning")
    _persist_job(job)

    if options.get("preview_only"):
        summary = f"Preview generated for {len(examples)} documents."
    else:
        summary = f"Cleaning completed for {len(cleaned_df)} rows."

    response = {
        "job_id": job.id,
        "step": "cleaning",
        "status": "done",
        "result": {
            "examples": examples,
            "summary": summary,
            "artifacts": [artifact],
        },
        "artifacts": job.artifacts,
        "job": _job_summary(job),
        "audit": _audit_payload(job),
    }
    return jsonify(response)


def _run_features(job: JobState, options: dict[str, Any]):
    df = _ensure_cleaned(job)
    text_column = job.text_column
    if not text_column or f"{text_column}_cleaned" not in df.columns:
        abort(400, description="Run cleaning to produce a cleaned text column.")

    cleaned_series = df[f"{text_column}_cleaned"].fillna("")
    max_features = int(options.get("max_features", 5000))
    min_df = int(options.get("min_df", 1))
    max_df = min(100, int(options.get("max_df", 100)))
    ngram_range = options.get("ngram_range", "1,2")
    try:
        ngram_values = tuple(int(val) for val in str(ngram_range).split(","))
        if len(ngram_values) == 1:
            ngram_values = (ngram_values[0], ngram_values[0])
    except Exception:
        ngram_values = (1, 2)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_values,
        min_df=min_df,
        max_df=max_df / 100 if max_df else 1.0,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(cleaned_series)
    feature_count = len(vectorizer.get_feature_names_out())
    sparsity = (
        1.0 - (matrix.count_nonzero() / (matrix.shape[0] * feature_count))
        if matrix.shape[0] and feature_count
        else 1.0
    )

    feature_matrix_path = _artifact_filename(job, "features", "tfidf_features.npz")
    sparse.save_npz(feature_matrix_path, matrix)
    vectorizer_path = _artifact_filename(job, "features", "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    job.vectorizer_path = vectorizer_path
    job.feature_matrix_path = feature_matrix_path

    top_weights = np.asarray(matrix.mean(axis=0)).ravel()
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = np.argsort(top_weights)[::-1][:20]
    top_terms = [
        {"term": feature_names[idx], "score": round(float(top_weights[idx]), 6)}
        for idx in sorted_indices
        if idx < len(feature_names)
    ]
    summary = {
        "documents": int(matrix.shape[0]),
        "feature_count": int(feature_count),
        "sparsity": sparsity,
        "top_terms": top_terms,
    }
    job.features_result = summary

    artifact_matrix = _register_artifact(job, "features", "TF-IDF features (NPZ)", feature_matrix_path)
    artifact_vectorizer = _register_artifact(job, "features", "TF-IDF vectorizer (PKL)", vectorizer_path)
    _append_audit(
        job,
        "features",
        f"Extracted {feature_count} TF-IDF features with n-gram range {ngram_values}.",
    )
    _update_step(job, "features")
    _persist_job(job)

    response = {
        "job_id": job.id,
        "step": "features",
        "status": "done",
        "result": {
            "summary": f"Extracted {feature_count} features across {matrix.shape[0]} documents.",
            "top_terms": top_terms,
            "artifacts": [artifact_matrix, artifact_vectorizer],
        },
        "artifacts": job.artifacts,
        "job": _job_summary(job),
        "audit": _audit_payload(job),
    }
    return jsonify(response)


def _run_training(job: JobState, options: dict[str, Any]):
    if not job.vectorizer_path or not job.vectorizer_path.exists():
        abort(400, description="Extract features before training.")

    if not job.label_column:
        abort(400, description="Select a label column to train a classifier.")

    df = _ensure_cleaned(job)
    if job.label_column not in df.columns:
        abort(400, description="Label column not found in dataset.")

    cleaned_column = f"{job.text_column}_cleaned"
    if cleaned_column not in df.columns:
        abort(400, description="Cleaned text column missing. Run cleaning first.")

    vectorizer = joblib.load(job.vectorizer_path)
    labels = df[job.label_column].dropna().astype(str)
    texts = df.loc[labels.index, cleaned_column].fillna("")
    if labels.nunique() < 2:
        abort(400, description="Need at least two label values to train a model.")

    X = vectorizer.transform(texts)
    test_size = float(options.get("test_size", 0.2))
    random_state = int(options.get("random_state", 42))
    stratify = labels if labels.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state, stratify=stratify
    )
    if X_train.shape[0] < 2 or X_test.shape[0] < 1:
        abort(400, description="Not enough samples after train/test split.")

    model_type = options.get("model_type", "logreg")
    if model_type == "svm":
        model = LinearSVC()
    elif model_type == "mlp":
        hidden_units = int(options.get("hidden_units", 128))
        epochs = int(options.get("epochs", 10))
        model = MLPClassifier(
            hidden_layer_sizes=(hidden_units,),
            max_iter=epochs,
            random_state=random_state,
        )
    else:
        model = LogisticRegression(max_iter=300)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
    }
    labels_order = list(sorted(set(y_test) | set(predictions)))
    cm = confusion_matrix(y_test, predictions, labels=labels_order)
    misclassified = []
    for text_value, true, pred in zip(texts.loc[y_test.index], y_test, predictions):
        if true != pred and len(misclassified) < 3:
            misclassified.append({"text": text_value[:280], "true": true, "pred": pred})

    model_path = _artifact_filename(job, "train", f"{model_type}_model.pkl")
    joblib.dump(model, model_path)
    artifact = _register_artifact(job, "train", "Trained model (PKL)", model_path)

    result = {
        "metrics": metrics,
        "confusion": cm.tolist(),
        "confusion_labels": labels_order,
        "misclassified_samples": misclassified,
        "artifacts": [artifact],
    }
    job.training_result = result
    _append_audit(
        job,
        "train",
        f"Trained {model_type} model (accuracy {metrics['accuracy']:.3f}).",
    )
    _update_step(job, "train")
    _persist_job(job)

    response = {
        "job_id": job.id,
        "step": "train",
        "status": "done",
        "result": result,
        "artifacts": job.artifacts,
        "job": _job_summary(job),
        "audit": _audit_payload(job),
    }
    return jsonify(response)


def _run_reports(job: JobState):
    report = {
        "job_id": job.id,
        "generated_at": _timestamp(),
        "steps_completed": sorted(job.completed_steps),
        "metrics": job.training_result.get("metrics") if job.training_result else None,
        "profiling": job.profiling_result,
        "features": job.features_result,
    }
    report_path = _artifact_filename(job, "reports", "report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    artifact = _register_artifact(job, "reports", "NLP summary (JSON)", report_path)
    _append_audit(job, "reports", "Generated consolidated NLP report.")
    _update_step(job, "reports")
    _persist_job(job)

    response = {
        "job_id": job.id,
        "step": "reports",
        "status": "done",
        "result": report,
        "artifacts": job.artifacts,
        "job": _job_summary(job),
        "audit": _audit_payload(job),
    }
    return jsonify(response)


@nlp_bp.route("/nlp/artifacts/<job_id>")
def list_artifacts(job_id: str):
    job = _load_job(job_id)
    return jsonify({"artifacts": job.artifacts})


@nlp_bp.route("/nlp/artifacts/<job_id>/download/<path:filename>")
def download_artifact(job_id: str, filename: str):
    job = _load_job(job_id)
    target = job.dir / Path(filename).name
    if not target.exists():
        abort(404, description="Artifact not found.")
    return send_file(target, as_attachment=True, download_name=target.name)
