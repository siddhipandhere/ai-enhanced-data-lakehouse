"""
Vector embedding generation (report Algorithm 6: NLP / Sentence Transformers).
"""
import numpy as np
import config
from utils.logger import get_logger

logger = get_logger("embedder")

_model = None


def get_model():
    # Imported lazily, not at module level: sentence_transformers pulls in
    # sklearn -> scipy, which (as of this codebase's pinned pyspark/pyarrow
    # versions) requires a different NumPy major version than the
    # ingestion/Spark path. Deferring the import until embeddings are
    # actually needed means app startup -- and therefore ingestion, which
    # runs first in the pipeline -- no longer forces that NumPy version
    # before it's necessary.
    from sentence_transformers import SentenceTransformer
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        # local_files_only=True skips a live network call to the Hugging
        # Face Hub that SentenceTransformer otherwise makes on every
        # first load per process (visible as "sending unauthenticated
        # requests to the HF Hub" in the logs) even when the model is
        # already fully cached locally. On a slow/rate-limited
        # connection that round-trip alone can take well over a minute.
        # Try the fast local-only path first; only fall back to a real
        # (network) load on the genuine first-ever run, when nothing is
        # cached yet.
        try:
            _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, local_files_only=True)
        except Exception:
            logger.info("Model not cached locally yet -- downloading (one-time only)")
            _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """
    Encodes a batch of strings into L2-normalized embedding vectors
    (normalization lets FAISS inner-product search double as cosine
    similarity search).

    Encodes in chunks of `batch_size` rather than passing the entire
    list to model.encode() in one call. On large datasets (tens of
    thousands of rows) encoding everything at once holds all of that
    batch's intermediate activations in memory simultaneously -- on a
    machine already running PySpark's JVM for the ingestion steps that
    ran just before this, that combined memory pressure can exceed a
    small/auto-managed Windows page file and crash with
    "The paging file is too small for this operation to complete."
    Chunking keeps peak memory bounded regardless of total row count.
    """
    model = get_model()
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        all_vectors.append(model.encode(chunk, convert_to_numpy=True, show_progress_bar=False))
    vectors = np.vstack(all_vectors) if all_vectors else np.empty((0, config.EMBEDDING_DIM), dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return (vectors / norms).astype("float32")


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]