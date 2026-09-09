from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


engine_options: dict[str, object] = {"pool_pre_ping": True}
if settings.database_url.startswith("sqlite"):
    engine_options["connect_args"] = {"check_same_thread": False}
else:
    engine_options.update({"pool_size": 10, "max_overflow": 20, "pool_recycle": 1800})

engine = create_engine(settings.database_url, **engine_options)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


def get_db():
    database = SessionLocal()
    try:
        yield database
    finally:
        database.close()


def initialize_database() -> None:
    from app import models  # noqa: F401 - registers all table metadata

    settings.data_root.mkdir(parents=True, exist_ok=True)
    if settings.auto_create_tables:
        Base.metadata.create_all(bind=engine)


def database_is_ready(session: Session) -> bool:
    from sqlalchemy import text

    session.execute(text("SELECT 1"))
    return True
