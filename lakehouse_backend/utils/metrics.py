"""
Evaluation metrics (report Section 3.3): precision, recall, F1,
query execution time, throughput, MRR, storage efficiency, cosine similarity.

These are used for benchmarking the system against a labeled test set,
not needed for a single ad-hoc query.
"""

import time
from pathlib import Path

import numpy as np


def precision(true_positives: int, false_positives: int) -> float:
    denom = true_positives + false_positives
    return round(true_positives / denom, 4) if denom else 0.0


def recall(true_positives: int, false_negatives: int) -> float:
    denom = true_positives + false_negatives
    return round(true_positives / denom, 4) if denom else 0.0


def f1_score(p: float, r: float) -> float:
    denom = p + r
    return round(2 * p * r / denom, 4) if denom else 0.0


def mean_reciprocal_rank(ranks: list[int]) -> float:
    """ranks: the 1-indexed rank of the first relevant result per query (0 = not found)."""
    reciprocals = [1 / r for r in ranks if r > 0]
    return round(sum(reciprocals) / len(ranks), 4) if ranks else 0.0


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return round(float(np.dot(vec_a, vec_b) / denom), 4) if denom else 0.0


def storage_efficiency(original_size_bytes: int, stored_size_bytes: int) -> float:
    return round(original_size_bytes / stored_size_bytes, 2) if stored_size_bytes else 0.0


class QueryTimer:
    """Context manager for measuring Query Execution Time (QET) in milliseconds."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = round((time.perf_counter() - self._start) * 1000, 2)


def throughput_records_per_second(record_count: int, elapsed_seconds: float) -> float:
    return round(record_count / elapsed_seconds, 2) if elapsed_seconds else 0.0
