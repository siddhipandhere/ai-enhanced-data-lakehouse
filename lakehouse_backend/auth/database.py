"""SQLite user store via SQLAlchemy. Kept separate from the Bronze/Silver/
Gold data lakehouse tables — this is application state, not lakehouse data."""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

import config

engine = create_engine(
    f"sqlite:///{config.USERS_DB_PATH}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a request-scoped DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
