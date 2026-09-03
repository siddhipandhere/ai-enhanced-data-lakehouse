"""
FastAPI entrypoint — the "next pass" the README pointed to: login/auth
layer + bulk multi-format upload, sitting in front of the existing
ingestion -> Bronze/Silver/Gold -> embeddings -> agents pipeline.

Run with:
    uvicorn main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth.database import Base, engine
from auth.routes import router as auth_router
from upload.routes import router as upload_router
from datasets.routes import router as datasets_router
from pipeline.routes import router as pipeline_router
from sql.routes import router as sql_router
from search.routes import router as search_router
from reports.routes import router as reports_router

# Creates users.db / the `users` table on first run if it doesn't exist yet.
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI-Enhanced Data Lakehouse — API",
    description="Auth + bulk multi-format dataset upload, backed by a "
                 "local-mode Spark/Delta Lake medallion pipeline (Option D).",
    version="1.0.0",
)

# Streamlit/JS frontend runs on a different port during local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the actual frontend origin before deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(upload_router)
app.include_router(datasets_router)
app.include_router(pipeline_router)
app.include_router(sql_router)
app.include_router(search_router)
app.include_router(reports_router)


@app.get("/health")
def health():
    return {"status": "ok"}
