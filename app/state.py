"""In-memory project state management for AutoDA."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from flask import session

from .models import ProjectMeta, db


@dataclass
class ProjectState:
    """Container for a project's evolving analytics state."""

    id: str
    project_name: str = "Untitled AutoDA Project"
    description: str = ""
    status: str = "Not Started"
    current_phase: str = "Ingestion"
    datasets: list[dict[str, Any]] = field(default_factory=list)
    detected_modalities: list[str] = field(default_factory=list)
    total_processing_time: Optional[str] = None
    last_action: Optional[datetime] = None
    data_quality: dict[str, Optional[int]] = field(
        default_factory=lambda: {"raw": None, "cleaned": None}
    )
    audit_trail: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    auto_mode: dict[str, Any] = field(
        default_factory=lambda: {
            "completed": False,
            "last_run": None,
            "steps": [],
            "errors": [],
        }
    )
    target_column: Optional[str] = None
    dataframes: dict[str, Any] = field(default_factory=dict)
    profiling: Optional[dict[str, Any]] = None
    cleaning: Optional[dict[str, Any]] = None
    feature_engineering: Optional[dict[str, Any]] = None
    feature_selection: Optional[dict[str, Any]] = None
    split_validation: Optional[dict[str, Any]] = None
    training: Optional[dict[str, Any]] = None
    reports: Optional[dict[str, Any]] = None
    preprocessing: Optional[dict[str, Any]] = None
    visualizations: Optional[list[dict[str, Any]]] = None
    analysis_requirements: dict[str, Any] = field(default_factory=dict)
    owner_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a view-friendly dictionary (excluding raw DataFrames)."""
        serializable = {
            "id": self.id,
            "project_name": self.project_name,
            "description": self.description,
            "status": self.status,
            "current_phase": self.current_phase,
            "datasets": self.datasets,
            "detected_modalities": self.detected_modalities,
            "total_processing_time": self.total_processing_time,
            "last_action": (
                self.last_action.strftime("%Y-%m-%d %H:%M UTC")
                if self.last_action
                else None
            ),
            "data_quality": self.data_quality,
            "audit_trail": self.audit_trail,
            "artifacts": self.artifacts,
            "auto_mode": {
                "completed": self.auto_mode["completed"],
                "last_run": (
                    self.auto_mode["last_run"].strftime("%Y-%m-%d %H:%M UTC")
                    if self.auto_mode["last_run"]
                    else None
                ),
                "steps": self.auto_mode["steps"],
                "errors": self.auto_mode["errors"],
            },
            "target_column": self.target_column,
            "profiling": self.profiling,
            "cleaning": self.cleaning,
            "feature_engineering": self.feature_engineering,
            "feature_selection": self.feature_selection,
            "split_validation": self.split_validation,
            "training": self.training,
            "reports": self.reports,
            "preprocessing": self.preprocessing,
            "visualizations": self.visualizations,
            "analysis_requirements": self.analysis_requirements,
            "owner_id": self.owner_id,
        }
        return serializable


PROJECT_STORE: dict[str, ProjectState] = {}


def _register_project_state(project: ProjectState) -> ProjectState:
    PROJECT_STORE[project.id] = project
    return project


def _project_from_meta(meta: ProjectMeta) -> ProjectState:
    project_id = str(meta.id)
    if project_id in PROJECT_STORE:
        return PROJECT_STORE[project_id]
    project = ProjectState(id=project_id)
    project.project_name = meta.name
    project.description = meta.description or ""
    project.status = meta.status or "Not Started"
    project.current_phase = meta.phase or "Ingestion"
    project.last_action = meta.updated_at or meta.created_at
    analysis = meta.analysis or {}
    # Ensure analysis is a dict
    if not isinstance(analysis, dict):
        analysis = {}
    # Restore datasets from analysis JSON if available
    if "datasets" in analysis:
        project.datasets = analysis["datasets"]
        # Remove datasets from analysis_requirements to avoid duplication
        analysis_copy = analysis.copy()
        analysis_copy.pop("datasets", None)
        project.analysis_requirements = analysis_copy
    else:
        project.analysis_requirements = analysis
    project.owner_id = meta.owner_email
    _register_project_state(project)
    return project


def get_current_project(allow_none: bool = False) -> Optional[ProjectState]:
    """Retrieve the project associated with the current session."""
    user_email = session.get("user_email")
    project_id = session.get("autoda_project_id")

    meta: Optional[ProjectMeta] = None
    if project_id:
        meta = ProjectMeta.query.get(int(project_id))
        if meta and meta.owner_email != user_email:
            meta = None

    if not meta and user_email:
        meta = (
            ProjectMeta.query.filter_by(owner_email=user_email)
            .order_by(ProjectMeta.updated_at.desc())
            .first()
        )
        if meta:
            session["autoda_project_id"] = str(meta.id)

    if meta:
        return _project_from_meta(meta)

    if allow_none:
        return None

    raise RuntimeError("No active project configured for this session.")


def get_project_dict(project: ProjectState) -> dict[str, Any]:
    """Return a sanitized view of the project."""
    return project.to_dict()


def update_status(project: ProjectState, status: str, phase: Optional[str] = None) -> None:
    project.status = status
    if phase:
        project.current_phase = phase
    project.last_action = datetime.utcnow()
    _sync_meta(project)


def record_audit(project: ProjectState, action: str, detail: str) -> None:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    project.audit_trail.insert(
        0,
        {
            "time": timestamp,
            "action": action,
            "detail": detail,
        },
    )


def add_artifact(
    project: ProjectState,
    name: str,
    artifact_type: str,
    size: str,
    description: str,
    link: Optional[str] = None,
    path: Optional[str] = None,
) -> dict[str, Any]:
    artifact_id = uuid4().hex
    artifact: dict[str, Any] = {
        "id": artifact_id,
        "name": name,
        "type": artifact_type,
        "size": size,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "description": description,
    }
    if link:
        artifact["link"] = link
    if path:
        artifact["path"] = path
    project.artifacts.append(artifact)
    _sync_meta(project)
    return artifact


def reset_auto_mode(project: ProjectState) -> None:
    project.auto_mode = {
        "completed": False,
        "last_run": None,
        "steps": [],
        "errors": [],
    }


def record_auto_mode_step(project: ProjectState, step: str, detail: str) -> None:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    project.auto_mode["steps"].append(
        {
            "step": step,
            "timestamp": timestamp,
            "detail": detail,
        }
    )
    project.last_action = datetime.utcnow()
    _sync_meta(project)


def record_auto_mode_error(project: ProjectState, step: str, error: str) -> None:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    project.auto_mode["errors"].append(
        {
            "step": step,
            "timestamp": timestamp,
            "error": error,
        }
    )
    project.last_action = datetime.utcnow()
    _sync_meta(project)


def set_target_column(project: ProjectState, column_name: Optional[str]) -> None:
    project.target_column = column_name
    project.last_action = datetime.utcnow()
    _sync_meta(project)


def store_dataframe(project: ProjectState, key: str, df: Any) -> None:
    """Store DataFrame in memory cache."""
    project.dataframes[key] = df
    project.last_action = datetime.utcnow()
    _sync_meta(project)


def get_dataframe(project: ProjectState, key: str):
    """
    Retrieve DataFrame from memory cache or reload from disk if not cached.
    
    This fixes the data persistence issue where dataframes were lost between requests
    because the PROJECT_STORE is in-memory only and ProjectState is recreated from DB.
    """
    # Check memory cache first
    if key in project.dataframes:
        return project.dataframes[key]
    
    # If not in cache, try to reload from disk based on key
    if key == "raw" and project.datasets:
        # Load raw dataset from file path stored in project.datasets
        from pathlib import Path
        from .pipeline import load_dataframe
        
        dataset_path = project.datasets[0].get("path")
        if dataset_path and Path(dataset_path).exists():
            try:
                df = load_dataframe(Path(dataset_path))
                # Cache it for future access in this session
                project.dataframes[key] = df
                return df
            except Exception as e:
                # Log the error for debugging
                print(f"ERROR reloading dataframe from {dataset_path}: {e}")
                import traceback
                traceback.print_exc()
                # If file can't be loaded, return None
                return None
        else:
            print(f"WARNING: Dataset path does not exist or is empty. Path: {dataset_path}, Datasets: {project.datasets}")
    
    # For other keys (cleaned, engineered, etc.) that aren't persisted to disk,
    # they'll be None if not in cache - this is expected behavior
    return None


def _sync_meta(project: ProjectState) -> None:
    try:
        meta = ProjectMeta.query.get(int(project.id))
        if not meta:
            return
        meta.name = project.project_name
        meta.description = project.description
        meta.status = project.status
        meta.phase = project.current_phase
        meta.dataset_count = len(project.datasets)
        meta.artifact_count = len(project.artifacts)
        # Store datasets in analysis JSON for persistence
        # Ensure we have a proper dict to work with
        if not isinstance(project.analysis_requirements, dict):
            project.analysis_requirements = {}
        analysis_data = project.analysis_requirements.copy()
        analysis_data["datasets"] = project.datasets
        meta.analysis = analysis_data
        meta.updated_at = datetime.utcnow()
        db.session.commit()
    except Exception:
        db.session.rollback()


def create_project(
    project_name: str,
    description: str = "",
    analysis: Optional[dict[str, Any]] = None,
    owner_id: Optional[str] = None,
) -> ProjectState:
    owner_email = owner_id or session.get("user_email")
    meta = ProjectMeta(
        owner_email=owner_email,
        name=project_name.strip() or "Untitled AutoDA Project",
        description=description.strip(),
        analysis=analysis or {},
    )
    db.session.add(meta)
    db.session.commit()

    project = ProjectState(id=str(meta.id))
    project.project_name = meta.name
    project.description = meta.description
    project.status = meta.status
    project.current_phase = meta.phase
    project.analysis_requirements = analysis or {}
    project.owner_id = owner_email
    project.last_action = datetime.utcnow()
    _register_project_state(project)

    session["autoda_project_id"] = project.id
    return project


def switch_project(project_id: str) -> ProjectState:
    user_email = session.get("user_email")
    meta = ProjectMeta.query.get(int(project_id))
    if meta and meta.owner_email == user_email:
        session["autoda_project_id"] = str(meta.id)
        project = _project_from_meta(meta)
        project.last_action = datetime.utcnow()
        return project
    raise KeyError(f"No project found with id {project_id}")
