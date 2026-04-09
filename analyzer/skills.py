"""
Skill extraction module.

Strategy (SRS CV-FR-01: spaCy NLP + Hugging Face Transformers + keyword matching):
  1. Keyword matching against skills_db.json      — fast, high precision
  2. spaCy NER (if installed)                     — catches skills not in the DB
  3. Hugging Face Transformer NER pipeline        — SRS-required second ML pass
     (dslim/bert-base-NER, falls back gracefully if not available)
  4. Results are merged and deduplicated

FIX (Issue #1 — SRS CV-FR-01 compliance):
  Previous version was missing the Hugging Face Transformer extraction pass
  entirely. SRS Table 3 explicitly requires "spaCy AND Hugging Face Transformer
  models" for NLP-based skill extraction. Added _load_hf_ner() and _hf_extract()
  using dslim/bert-base-NER. Falls back gracefully when transformers/torch
  is not installed and logs a clear warning.

FIX (Issue #5 — NullHandler):
  Added logging.NullHandler() so this module is safe to import under Django
  without producing 'No handlers could be found' warnings.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Safe for library / Django use

# ---------------------------------------------------------------------------
# Optional spaCy import
# ---------------------------------------------------------------------------
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    spacy = None  # type: ignore
    _SPACY_AVAILABLE = False
    logger.warning(
        "spaCy not installed. Falling back to keyword-only extraction. "
        "Run: pip install spacy && python -m spacy download en_core_web_sm"
    )

# ---------------------------------------------------------------------------
# Optional Hugging Face Transformers import  (SRS CV-FR-01 requirement)
# ---------------------------------------------------------------------------
try:
    from transformers import pipeline as hf_pipeline
    _HF_AVAILABLE = True
except ImportError:
    hf_pipeline = None  # type: ignore
    _HF_AVAILABLE = False
    logger.warning(
        "Hugging Face transformers not installed — SRS CV-FR-01 requires it. "
        "Run: pip install transformers torch"
    )

# Default skill DB path — resolved relative to this file
_DEFAULT_SKILL_DB = Path(__file__).parent.parent / "models" / "skill_db.json"

# HF model: lightweight BERT fine-tuned on CoNLL-2003 NER
# Recognises PER, ORG, LOC, MISC — ORG and MISC capture most tech names in CVs
_HF_NER_MODEL = "dslim/bert-base-NER"
_HF_TECH_LABELS = {"ORG", "MISC"}   # entity_group values after aggregation

# Heuristic pattern — tokens that look like technology / tool names:
#   CamelCase (Vue.js), ALL-CAPS short (AWS, CSS), mixed alphanumeric (Node.js)
_TECH_TOKEN_RE = re.compile(
    r"^(?:[A-Z][a-zA-Z0-9.#+\-/]{1,}|[A-Z]{2,10}|[a-z]+[0-9]+[a-zA-Z0-9]*)$"
)


class SkillExtractor:
    """
    Extract skills from CV text using three complementary strategies:
      1. Keyword matching   — high precision, database-driven, always runs
      2. spaCy NER          — catches technology names not in the skill DB
      3. HF BERT NER        — SRS CV-FR-01 Transformer pass
    """

    def __init__(self, skill_db_path: Optional[str] = None):
        self.skill_db = self._load_skill_db(skill_db_path)
        self._nlp = self._load_spacy()
        self._hf_ner = self._load_hf_ner()

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_skill_db(self, skill_db_path: Optional[str]) -> dict:
        path = Path(skill_db_path) if skill_db_path else _DEFAULT_SKILL_DB
        try:
            with open(path, "r", encoding="utf-8") as f:
                db = json.load(f)
            logger.info(f"Loaded skill database from {path}")
            # Deduplicate lists (fixes duplicate entries in original JSON)
            db["technical"] = list(dict.fromkeys(db.get("technical", [])))
            db["soft"] = list(dict.fromkeys(db.get("soft", [])))
            return db
        except FileNotFoundError:
            logger.warning(f"Skill database not found at {path}")
            return {"technical": [], "soft": []}
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in skill database at {path}")
            return {"technical": [], "soft": []}

    def _load_spacy(self):
        """Load spaCy model if available."""
        if not _SPACY_AVAILABLE:
            return None
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model: en_core_web_sm")
            return nlp
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            return None

    def _load_hf_ner(self):
        """
        Load Hugging Face NER pipeline (SRS CV-FR-01 requirement).

        Uses dslim/bert-base-NER — a BERT model fine-tuned on CoNLL-2003 that
        reliably detects ORG and MISC entities, which in CV text correspond
        to companies, tools, frameworks, and technology names.

        aggregation_strategy="simple" merges sub-word tokens (##suffix pieces)
        into whole words before returning entity spans.

        Falls back gracefully if transformers / torch is not installed.
        """
        if not _HF_AVAILABLE:
            return None
        try:
            ner = hf_pipeline(
                "ner",
                model=_HF_NER_MODEL,
                aggregation_strategy="simple",  # merges ##sub-word tokens
                device=-1,                       # CPU-only; safe default
            )
            logger.info(f"Loaded Hugging Face NER model: {_HF_NER_MODEL}")
            return ner
        except Exception as e:
            logger.error(
                f"Failed to load Hugging Face NER model '{_HF_NER_MODEL}': {e}. "
                "Extraction will proceed without the Transformer pass."
            )
            return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extract skills from text using all available strategies.

        Args:
            text: CV full text

        Returns:
            {"technical": [...], "soft": [...]}
        """
        technical: set = set()
        soft: set = set()

        try:
            # Step 1: keyword matching (high precision, always runs)
            technical.update(self._keyword_match(text, self.skill_db.get("technical", [])))
            soft.update(self._keyword_match(text, self.skill_db.get("soft", [])))

            # Step 2: spaCy NER augmentation
            if self._nlp is not None:
                ner_technical, ner_soft = self._spacy_extract(text)
                technical.update(ner_technical)
                soft.update(ner_soft)

            # Step 3: Hugging Face BERT NER (SRS CV-FR-01)
            if self._hf_ner is not None:
                hf_technical = self._hf_extract(text)
                technical.update(hf_technical)

        except Exception as e:
            logger.error(f"Error during skill extraction: {e}")

        return {
            "technical": sorted(technical),
            "soft": sorted(soft),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _keyword_match(self, text: str, keywords: List[str]) -> set:
        """Word-boundary keyword matching (case-insensitive)."""
        matched: set = set()
        text_lower = text.lower()
        for keyword in keywords:
            try:
                pattern = rf"\b{re.escape(keyword.lower())}\b"
                if re.search(pattern, text_lower):
                    matched.add(keyword)
            except Exception as e:
                logger.debug(f"Keyword match error for '{keyword}': {e}")
        return matched

    def _spacy_extract(self, text: str):
        """
        Use spaCy to find additional skill-like entities not in the DB.
        Only processes first 10k chars to keep inference fast.
        """
        doc = self._nlp(text[:10_000])

        known_technical = {s.lower() for s in self.skill_db.get("technical", [])}
        known_soft = {s.lower() for s in self.skill_db.get("soft", [])}

        extra_technical: set = set()
        extra_soft: set = set()

        for ent in doc.ents:
            token = ent.text.strip()
            token_lower = token.lower()

            if len(token) < 2 or len(token) > 40:
                continue

            if ent.label_ in ("ORG", "PRODUCT"):
                if token_lower not in known_technical and token_lower not in known_soft:
                    if _TECH_TOKEN_RE.match(token) or token.isupper():
                        extra_technical.add(token)

        return extra_technical, extra_soft

    def _hf_extract(self, text: str) -> set:
        """
        Hugging Face BERT NER extraction pass (SRS CV-FR-01).

        Runs dslim/bert-base-NER on the first ~600 words of the CV
        (approximates the 512-token BERT limit). Filters entity spans whose
        aggregated label is ORG or MISC and whose surface form matches the
        technology-name heuristic (_TECH_TOKEN_RE).

        Sub-word artefacts (tokens starting with '##') are discarded.
        Entities already captured by keyword matching are skipped to avoid
        duplicates.

        Returns:
            set of additional technical skill strings
        """
        known_technical = {s.lower() for s in self.skill_db.get("technical", [])}
        known_soft = {s.lower() for s in self.skill_db.get("soft", [])}
        extra: set = set()

        try:
            # Slice to ~600 words ≈ 512 BERT tokens to stay within model limit
            truncated = " ".join(text.split()[:600])
            entities = self._hf_ner(truncated)

            for ent in entities:
                word = ent.get("word", "").strip()
                label = ent.get("entity_group", "")

                # Discard sub-word tokenisation artefacts
                if word.startswith("##") or len(word) < 2 or len(word) > 40:
                    continue

                # Only ORG and MISC labels are relevant for technical skills
                if label not in _HF_TECH_LABELS:
                    continue

                word_lower = word.lower()
                # Skip anything already found by keyword matching
                if word_lower in known_technical or word_lower in known_soft:
                    continue

                # Apply technology-name heuristic
                if _TECH_TOKEN_RE.match(word) or word.isupper():
                    extra.add(word)
                    logger.debug(f"HF NER skill: '{word}' ({label})")

        except Exception as e:
            logger.error(f"Hugging Face NER extraction failed: {e}")

        return extra


# ---------------------------------------------------------------------------
# Module-level availability flags (used by app.py sidebar status display)
# ---------------------------------------------------------------------------

def is_hf_available() -> bool:
    """Return True if Hugging Face transformers is installed."""
    return _HF_AVAILABLE


# ---------------------------------------------------------------------------
# Public module-level function
# ---------------------------------------------------------------------------

def extract_skills(text: str, skill_db_path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Extract skills from CV text using all available strategies.

    Args:
        text: CV text
        skill_db_path: Optional path to skill database JSON

    Returns:
        {"technical": [...], "soft": [...]}
    """
    extractor = SkillExtractor(skill_db_path)
    skills = extractor.extract(text)
    logger.info(
        f"Extracted {len(skills['technical'])} technical and "
        f"{len(skills['soft'])} soft skills "
        f"(spaCy={'on' if _SPACY_AVAILABLE else 'off'}, "
        f"HF NER={'on' if _HF_AVAILABLE else 'off'})"
    )
    return skills