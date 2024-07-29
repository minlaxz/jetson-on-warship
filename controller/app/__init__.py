"""Initialize Flask app."""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def create_app():
    """Construct the core application."""
    _app = Flask(__name__, instance_relative_config=False)
    _app.config.from_object("config.Config")

    # Initialize Database Plugin
    db.init_app(_app)

    with _app.app_context():
        from . import routes  # noqa: F401

        db.create_all()

        return _app
