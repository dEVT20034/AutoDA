from pathlib import Path
from urllib.parse import quote_plus

import pymysql
from flask import Flask

from .models import db


MYSQL_SETTINGS = {
    "host": "localhost",
    "user": "root",
    "password": "2034",
    "database": "autoda",
}


def ensure_mysql_database() -> None:
    connection = pymysql.connect(
        host=MYSQL_SETTINGS["host"],
        user=MYSQL_SETTINGS["user"],
        password=MYSQL_SETTINGS["password"],
        autocommit=True,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{MYSQL_SETTINGS['database']}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            )
    finally:
        connection.close()


def create_app():
    """Application factory for AutoDA."""
    base_dir = Path(__file__).resolve().parent.parent
    uploads_dir = base_dir / "uploads"
    uploads_dir.mkdir(exist_ok=True)

    app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static"),
        static_url_path="/static",
    )
    app.config["APP_NAME"] = "AutoDA"
    app.config["UPLOAD_FOLDER"] = str(uploads_dir)
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 200  # 200 MB
    app.secret_key = "autoda-dev-secret"

    ensure_mysql_database()
    password = quote_plus(MYSQL_SETTINGS["password"])
    app.config["SQLALCHEMY_DATABASE_URI"] = (
        f"mysql+pymysql://{MYSQL_SETTINGS['user']}:{password}@{MYSQL_SETTINGS['host']}/{MYSQL_SETTINGS['database']}?charset=utf8mb4"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)
    with app.app_context():
        db.create_all()

    from .routes import main_blueprint
    from .nlp_automation import nlp_bp

    app.register_blueprint(main_blueprint)
    app.register_blueprint(nlp_bp)

    return app
