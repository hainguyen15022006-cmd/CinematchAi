"""
db.py — Database connection setup.
Uses SQLAlchemy to work with SQLite without writing raw SQL.
"""

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# check_same_thread=False: required for SQLite when handling concurrent requests
connect_args = (
    {"check_same_thread": False}
    if settings.DATABASE_URL.startswith("sqlite")
    else {}
)
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base is the declarative base class that every model (table) inherits from
Base = declarative_base()

def get_db():
    """
    FastAPI dependency: opens one DB session per request and always
    closes it once the request finishes (even when an error occurs).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Columns added after the first databases were created. create_all() only
# creates missing tables, so these are applied by hand. The MVP has no
# migration tool and the columns are all nullable, so adding them is safe
# to repeat on every start.
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "recommendation_runs": {
        "strategy": "VARCHAR",
        "model_name": "VARCHAR",
        "top_k": "INTEGER",
    },
    "run_items": {
        "minimum_score": "FLOAT",
        "disagreement": "FLOAT",
        "misery_warning": "BOOLEAN",
        "member_scores_json": "TEXT",
        "explanations_json": "TEXT",
    },
}


def ensure_runtime_schema() -> None:
    """Add the late nullable columns to an older database in place."""
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    for table_name, columns in _ADDED_COLUMNS.items():
        if table_name not in table_names:
            continue
        existing = {
            column["name"] for column in inspector.get_columns(table_name)
        }
        missing = {
            name: sql_type
            for name, sql_type in columns.items()
            if name not in existing
        }
        if not missing:
            continue
        with engine.begin() as connection:
            for name, sql_type in missing.items():
                # Table and column names are fixed constants above, never
                # user input, so this cannot be injected into.
                connection.execute(
                    text(
                        f"ALTER TABLE {table_name} "
                        f"ADD COLUMN {name} {sql_type}"
                    )
                )
