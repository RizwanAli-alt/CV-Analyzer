"""
Semantic Similarity Analysis Module.

FIX (Issue #9 — import re placed after class that uses it):
  'import re' was placed at the bottom of the file with a comment saying it was
  intentional. The _chunk() static method called re.split() — this worked in
  CPython only because the import executed before the method was ever called,
  but it is non-standard and breaks static analysis / linters. Moved import re
  to the top of the file where it belongs.

FIX (Issue #5 — NullHandler):
  Added logging.NullHandler() for Django logging safety.

Other fixes retained from previous version:
  - sentence-transformers missing from requirements handled gracefully
  - Public function accepts optional job_description (SRS requirement)
  - Uses built-in sample JD when none provided
"""

import re  # ← FIXED: was at bottom of file; moved to top
import logging
from typing import Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Safe for library / Django use

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    _ST_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    cosine_similarity = None   # type: ignore
    np = None                  # type: ignore
    _ST_AVAILABLE = False
    logger.warning(
        "sentence-transformers or scikit-learn not installed. "
        "Similarity scoring will be unavailable. "
        "Run: pip install sentence-transformers scikit-learn"
    )

SAMPLE_JOB_DESCRIPTION = """
We are looking for a skilled software professional with:
- Strong technical background and programming skills (Python, JavaScript, Java)
- Experience with modern web frameworks (Django, React, Node.js)
- Database knowledge: SQL, PostgreSQL, MongoDB
- Cloud and DevOps: AWS, Docker, Kubernetes, CI/CD
- Problem-solving and analytical abilities
- Team collaboration and communication skills
- Degree in Computer Science, Software Engineering, or related field
- Portfolio, GitHub profile, or open-source contributions
- Understanding of software architecture and design patterns
- Version control with Git
"""

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Maximum characters accepted from a user-supplied job description
MAX_JD_LENGTH = 5_000


class SimilarityAnalyzer:
    """Analyze semantic similarity between a CV and a job description."""

    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        if not _ST_AVAILABLE:
            return None
        try:
            model = SentenceTransformer(MODEL_NAME)
            logger.info(f"Loaded sentence-transformers model: {MODEL_NAME}")
            return model
        except Exception as e:
            logger.error(f"Error loading sentence-transformers model: {e}")
            return None

    @property
    def model_available(self) -> bool:
        return self.model is not None

    def calculate_similarity(
        self, cv_text: str, job_description: Optional[str] = None
    ) -> float:
        """
        Calculate semantic similarity between CV and job description.

        Args:
            cv_text: Full CV text
            job_description: Job description text. Falls back to sample if None.

        Returns:
            Similarity percentage (0.0–100.0), or 0 if model unavailable.
        """
        # Guard: empty CV text → return 0 immediately (not ~50%)
        if not cv_text or not cv_text.strip():
            return 0.0

        if self.model is None:
            logger.warning("Similarity model not available — returning 0.")
            return 0.0

        # Sanitise and cap job description length
        if job_description and job_description.strip():
            jd = job_description.strip()[:MAX_JD_LENGTH]
        else:
            jd = SAMPLE_JOB_DESCRIPTION

        try:
            cv_chunks = self._chunk(cv_text, max_len=256)
            jd_chunks = self._chunk(jd, max_len=256)

            if not cv_chunks or not jd_chunks:
                return 0.0

            cv_emb = self.model.encode(cv_chunks, show_progress_bar=False)
            jd_emb = self.model.encode(jd_chunks, show_progress_bar=False)

            sim_matrix = cosine_similarity(cv_emb, jd_emb)
            avg_sim = float(np.mean(np.max(sim_matrix, axis=1)))
            pct = round(avg_sim * 100, 1)

            logger.info(f"Similarity score: {pct}%")
            return pct

        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0

    @staticmethod
    def _chunk(text: str, max_len: int = 256) -> list:
        """
        Split text into sentence chunks within max_len characters.

        Uses re (imported at top of file) to split on sentence boundaries.
        """
        if not text or not text.strip():
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], ""
        for sent in sentences:
            if len(current) + len(sent) < max_len:
                current += sent + " "
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = sent + " "
        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [text[:max_len]]


def calculate_similarity(
    cv_text: str, job_description: Optional[str] = None
) -> float:
    """
    Calculate semantic similarity between CV and job description.

    Args:
        cv_text: Full CV text
        job_description: Optional job description. Uses built-in sample if None.

    Returns:
        Similarity percentage (0.0–100.0)
    """
    analyzer = SimilarityAnalyzer()
    return analyzer.calculate_similarity(cv_text, job_description)


def is_similarity_available() -> bool:
    """Check whether the similarity model is installed and loadable."""
    return _ST_AVAILABLE