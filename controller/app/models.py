"""Database models."""

from . import db


class Record(db.Model):
    __tablename__ = "lightstack-users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), index=True, unique=True, nullable=False)
    plate = db.Column(db.String(255), index=True, unique=True, nullable=False)
    car = db.Column(db.String(255), index=True, unique=True, nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(db.DateTime, server_default=db.func.now(), server_onupdate=db.func.now())

    def __repr__(self):
        return f"<User id={self.id}, name={self.name}, plate={self.plate}>"