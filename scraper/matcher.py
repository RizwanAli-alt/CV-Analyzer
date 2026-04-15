"""
Job-to-Profile Match Scorer (SCR-FR-02).

Calculates a semantic similarity match percentage between each scraped
job listing and the user's extracted CV skills.

SRS SCR-FR-02 Business Rules:
    BR-1: Match score calculated using cosine similarity between
          job requirement embeddings and user skill embeddings.
    BR-2: Score displayed as a percentage (0–100%) alongside each listing.
    BR-3: Listings sorted by match score descending by default.
    BR-4: Auto-Apply triggered only for listings meeting user-defined threshold.

Uses the same sentence-transformers model as the CV analyzer (similarity.py)
for consistency — all-MiniLM-L6-v2 (SRS §7.2.3).

Falls back to TF-IDF keyword overlap scoring when sentence-transformers
is not installed, giving a degraded but non-zero result.
"""

import logging
import re
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    _ST_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    cosine_similarity = None
    np = None
    _ST_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed. "
        "Falling back to TF-IDF keyword match scoring. "
        "Run: pip install sentence-transformers scikit-learn"
    )

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model_cache = None   # singleton — load once per process


def _get_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not _ST_AVAILABLE:
        return None
    try:
        _model_cache = SentenceTransformer(MODEL_NAME)
        logger.info(f"Loaded sentence-transformers model: {MODEL_NAME}")
        return _model_cache
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def _skills_to_text(extracted_skills: Dict) -> str:
    """Convert skill dict to a flat text string for embedding."""
    technical = extracted_skills.get("technical", [])
    soft = extracted_skills.get("soft", [])
    return " ".join(technical + soft)


def _tfidf_fallback_score(job_text: str, skills_text: str) -> float:
    """
    Keyword overlap score when sentence-transformers is unavailable.
    Returns 0–100.
    """
    if not job_text or not skills_text:
        return 0.0

    job_words = set(re.findall(r"\b\w{3,}\b", job_text.lower()))
    skill_words = set(re.findall(r"\b\w{3,}\b", skills_text.lower()))

    if not skill_words:
        return 0.0

    overlap = len(job_words & skill_words)
    # Normalise: overlap / sqrt(|job| * |skills|)  (Dice-like)
    score = overlap / (len(job_words) ** 0.5 + 1e-6) * 10
    return min(round(score, 1), 100.0)


def score_listing(job_text: str, user_skills: Dict) -> float:
    """
    Calculate match score between a job listing and user skills.

    Args:
        job_text: Combined job title + description text
        user_skills: {"technical": [...], "soft": [...]}

    Returns:
        Match percentage 0–100 (SCR-FR-02 BR-2)
    """
    if not job_text or not user_skills:
        return 0.0

    skills_text = _skills_to_text(user_skills)
    if not skills_text.strip():
        return 0.0

    model = _get_model()

    if model is None:
        return _tfidf_fallback_score(job_text, skills_text)

    try:
        job_embedding = model.encode([job_text[:1000]], show_progress_bar=False)
        skill_embedding = model.encode([skills_text], show_progress_bar=False)
        sim = float(cosine_similarity(job_embedding, skill_embedding)[0][0])
        return round(max(0.0, min(sim * 100, 100.0)), 1)
    except Exception as e:
        logger.error(f"Embedding similarity failed: {e}")
        return _tfidf_fallback_score(job_text, skills_text)


def score_and_sort(listings, user_skills: Dict) -> list:
    """
    Score all listings against user skills and return sorted descending.

    SCR-FR-02 BR-1 + BR-3.

    Args:
        listings: List[JobListing]
        user_skills: {"technical": [...], "soft": [...]}

    Returns:
        Listings sorted by match_score descending
    """
    from .base import JobListing

    scored = []
    for listing in listings:
        job_text = f"{listing.title} {listing.company} {listing.description}"
        listing.match_score = score_listing(job_text, user_skills)
        scored.append(listing)

    # SCR-FR-02 BR-3: sort descending
    scored.sort(key=lambda j: j.match_score, reverse=True)
    logger.info(
        f"Scored {len(scored)} listings. "
        f"Top match: {scored[0].match_score}% — {scored[0].title}"
        if scored else "No listings to score."
    )
    return scored


def is_match_available() -> bool:
    """True if sentence-transformers is installed (semantic mode)."""
    return _ST_AVAILABLE