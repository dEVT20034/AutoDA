from __future__ import annotations

from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import JSON
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


class User(db.Model):
    email = db.Column(db.String(255), primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    projects = db.relationship(
        "ProjectMeta",
        backref="owner",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class ProjectMeta(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    owner_email = db.Column(db.String(255), db.ForeignKey("user.email"), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, default="")
    status = db.Column(db.String(64), default="Not Started")
    phase = db.Column(db.String(64), default="Ingestion")
    dataset_count = db.Column(db.Integer, default=0)
    artifact_count = db.Column(db.Integer, default=0)
    analysis = db.Column(JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    seed = db.Column(db.Boolean, default=False)
