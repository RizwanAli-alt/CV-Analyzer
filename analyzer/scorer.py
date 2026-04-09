"""
CV Quality Scoring Engine.

FIX (Issue #2 — SRS CV-FR-02 weight compliance):
  SRS Table 4 defines 4 weighted scoring dimensions:
    keywords (30%), completeness (25%), skill density (25%), formatting (20%)

  The previous version used 6 independent components whose sub-scores did NOT
  map onto those 4 SRS buckets, causing a discrepancy between what the SRS
  specifies and what was being computed.

  This version explicitly computes the 4 SRS-defined dimensions, each built
  from the relevant sub-components below, so both the weights AND the dimension
  names match the SRS exactly:

    ┌─────────────────────────────────────────────────────────────────┐
    │  SRS Dimension   Weight   Sub-components (internal)             │
    ├─────────────────────────────────────────────────────────────────┤
    │  keywords         0.30    action verbs + keyword density        │
    │  completeness     0.25    section headers + contact + education │
    │  skill_density    0.25    technical skills + soft skills        │
    │  formatting       0.20    structure + bullets + links           │
    └─────────────────────────────────────────────────────────────────┘

  The finer-grained sub-scores (experience_score, education_score, etc.) are
  STILL returned in the breakdown dict for the UI to display — they are now
  clearly labelled as "UI detail only" and do NOT affect the weighted total.
  The weighted total is computed solely from the 4 SRS dimensions.

FIX (Issue #5 — NullHandler):
  Added logging.NullHandler() so the module is safe under Django.
"""

import re
import logging
from typing import Dict
from analyzer.parser import get_text_statistics

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Safe for library / Django use

# ---------------------------------------------------------------------------
# SRS-compliant scoring weights  (must sum to 1.0  — SRS Table 4 / CV-FR-02)
# ---------------------------------------------------------------------------
SRS_WEIGHTS = {
    "keywords":     0.30,   # SRS: keyword relevance dimension
    "completeness": 0.25,   # SRS: section completeness dimension
    "skill_density":0.25,   # SRS: skill density dimension
    "formatting":   0.20,   # SRS: formatting assessment dimension
}

assert abs(sum(SRS_WEIGHTS.values()) - 1.0) < 1e-9, "SRS weights must sum to 1.0"


class CVScorer:
    """
    Score CV quality across the four SRS-defined dimensions (CV-FR-02).

    Each dimension is scored 0–100 internally, then multiplied by its SRS
    weight to produce the weighted 0–100 total.
    """

    def score(self, text: str, extracted_skills: Dict) -> Dict:
        """
        Calculate comprehensive CV score (0–100).

        Args:
            text: Full CV text
            extracted_skills: {"technical": [...], "soft": [...]}

        Returns:
            {"score": float, "breakdown": {...}}
            where breakdown contains:
              - The 4 SRS dimension scores  (drive the weighted total)
              - Additional UI detail scores  (for display only, not weighted)
              - Text statistics
        """
        # ── 4 SRS-defined dimension scores (each 0–100) ─────────────────────
        dim_keywords     = self._dim_keywords(text)
        dim_completeness = self._dim_completeness(text)
        dim_skill_density= self._dim_skill_density(extracted_skills)
        dim_formatting   = self._dim_formatting(text)

        # ── Weighted total per SRS Table 4 formula ───────────────────────────
        total = (
            dim_keywords      * SRS_WEIGHTS["keywords"]
            + dim_completeness  * SRS_WEIGHTS["completeness"]
            + dim_skill_density * SRS_WEIGHTS["skill_density"]
            + dim_formatting    * SRS_WEIGHTS["formatting"]
        )

        # ── Extra UI detail sub-scores (NOT used in weighted total) ──────────
        ui_experience   = self._ui_experience(text)
        ui_education    = self._ui_education(text)
        ui_projects     = self._ui_projects(text)

        stats = get_text_statistics(text)

        breakdown = {
            # --- SRS-defined dimension scores (drive the total) ---
            "keywords_score":      round(dim_keywords,      1),
            "completeness_score":  round(dim_completeness,  1),
            "skill_density_score": round(dim_skill_density, 1),
            "formatting_score":    round(dim_formatting,    1),

            # --- UI detail scores (display only, not weighted) ---
            # Kept for backward-compat with suggestions.py and app.py
            "skills_relevance_score": round(dim_skill_density, 1),  # alias
            "experience_score":       round(ui_experience,  1),
            "education_score":        round(ui_education,   1),
            "keyword_density_score":  round(dim_keywords,   1),      # alias
            "projects_score":         round(ui_projects,    1),

            # --- Text statistics ---
            "word_count":      stats["word_count"],
            "character_count": stats["character_count"],
            "sentence_count":  stats["sentence_count"],
        }

        return {"score": round(total, 1), "breakdown": breakdown}

    # =========================================================================
    # SRS Dimension 1 — Keywords (30%)
    # Measures ATS-friendliness: action verbs, quantified results, keyword density
    # =========================================================================

    def _dim_keywords(self, text: str) -> float:
        """
        SRS Dimension: Keyword Relevance (0–100).

        Evaluates action verbs, quantified achievements, and word-count sweet-spot.
        """
        words = text.split()
        if not words:
            return 0.0

        score = 0.0

        # Word count sweet spot (300–1000 words is ATS-optimal)
        if 300 <= len(words) <= 1000:
            score += 30
        elif 150 <= len(words) < 300 or 1000 < len(words) <= 2000:
            score += 15

        # Action verbs (+5 each, capped at 40)
        action_verbs = [
            "led", "managed", "developed", "implemented", "created", "designed",
            "improved", "achieved", "increased", "reduced", "built", "launched",
            "delivered", "optimized", "collaborated", "mentored",
        ]
        verb_hits = sum(1 for v in action_verbs if re.search(rf"\b{v}\b", text, re.I))
        score += min(verb_hits * 5, 40)

        # Quantified results (+5 each, capped at 30)
        quantified = re.findall(
            r"\b\d+\%|\$\d+|\d+[xX]|\d+\s*(?:days|weeks|months|years|hours|users|clients)\b",
            text, re.I,
        )
        score += min(len(quantified) * 5, 30)

        return min(score, 100.0)

    # =========================================================================
    # SRS Dimension 2 — Completeness (25%)
    # Measures section coverage + contact information presence
    # =========================================================================

    def _dim_completeness(self, text: str) -> float:
        """
        SRS Dimension: Section Completeness (0–100).

        Checks for the standard CV sections required by ATS and recruiters.
        """
        score = 0.0

        # Required sections (12 pts each, 5 sections = 60 pts max)
        sections = {
            "experience":     r"\b(experience|employment|work history|work experience)\b",
            "education":      r"\b(education|academic|university|college|degree)\b",
            "skills":         r"\b(skills?|competencies|technical skills)\b",
            "summary":        r"\b(summary|objective|profile|about)\b",
            "projects":       r"\b(projects?|portfolio)\b",
        }
        for label, pattern in sections.items():
            if re.search(pattern, text, re.I):
                score += 12

        # Contact completeness (40 pts total)
        if re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text):
            score += 15  # email
        if re.search(r"(\+?\d[\d\s\-().]{7,}\d)", text):
            score += 10  # phone
        if re.search(r"(linkedin\.com|github\.com|portfolio|https?://)", text, re.I):
            score += 15  # online presence

        return min(score, 100.0)

    # =========================================================================
    # SRS Dimension 3 — Skill Density (25%)
    # Measures how many and how varied the skills are
    # =========================================================================

    def _dim_skill_density(self, extracted_skills: Dict) -> float:
        """
        SRS Dimension: Skill Density (0–100).

        Technical skills: each worth 6 pts, capped at 60.
        Soft skills:      each worth 8 pts, capped at 40.
        """
        tech_count = len(extracted_skills.get("technical", []))
        soft_count = len(extracted_skills.get("soft", []))

        tech_score = min(tech_count * 6, 60)
        soft_score = min(soft_count * 8, 40)

        return tech_score + soft_score

    # =========================================================================
    # SRS Dimension 4 — Formatting (20%)
    # Measures visual structure, readability, and ATS-parse-friendliness
    # =========================================================================

    def _dim_formatting(self, text: str) -> float:
        """
        SRS Dimension: Formatting Assessment (0–100).
        """
        score = 0.0

        if "\n" in text:
            score += 20     # multi-line content (not a one-block blob)

        if "•" in text or re.search(r"\n\s*[-*]\s", text):
            score += 25     # bullet points present

        if re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text):
            score += 20     # email visible

        if re.search(r"(\+?\d[\d\s\-().]{7,}\d)", text):
            score += 15     # phone visible

        if re.search(r"(https?://|github\.com|linkedin\.com|portfolio)", text, re.I):
            score += 20     # online presence / links

        return min(score, 100.0)

    # =========================================================================
    # UI Detail Sub-scores  (NOT used in SRS weighted total — display only)
    # =========================================================================

    def _ui_experience(self, text: str) -> float:
        """Experience sub-score for UI display (not SRS-weighted)."""
        score = 0.0
        if re.search(r"\b(experience|employment|work history|work experience)\b", text, re.I):
            score += 40
        date_ranges = re.findall(
            r"\d{4}\s*[-–]\s*(?:\d{4}|present|current|now)", text, re.I
        )
        score += min(len(date_ranges) * 15, 45)
        quantified = re.findall(r"\b\d+\s*[%$]|\$\s*\d+|\d+[xX]\b", text)
        score += min(len(quantified) * 5, 15)
        return min(score, 100.0)

    def _ui_education(self, text: str) -> float:
        """Education sub-score for UI display (not SRS-weighted)."""
        score = 0.0
        if re.search(r"\b(education|academic|university|college|degree)\b", text, re.I):
            score += 30
        degrees = ["bachelor", "master", "phd", "diploma", "associate",
                   r"b\.sc", r"m\.sc", "bs", "ms"]
        for deg in degrees:
            if re.search(rf"\b{deg}\b", text, re.I):
                score += 15
                break
        if re.search(r"\bgpa\b.*?[\d.]+", text, re.I):
            score += 10
        if re.search(r"\b(20\d{2}|19\d{2})\b", text):
            score += 10
        fields = ["computer science", "software engineering", "information technology",
                  "data science", "electrical", "mathematics"]
        for field in fields:
            if field in text.lower():
                score += 15
                break
        return min(score, 100.0)

    def _ui_projects(self, text: str) -> float:
        """Projects sub-score for UI display (not SRS-weighted)."""
        if re.search(r"\b(projects?|portfolio|personal projects?)\b", text, re.I):
            return 100.0
        return 0.0


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def calculate_score(text: str, extracted_skills: Dict) -> Dict:
    """
    Calculate CV quality score using the 4 SRS-defined weighted dimensions.

    Args:
        text: CV text
        extracted_skills: {"technical": [...], "soft": [...]}

    Returns:
        {"score": float, "breakdown": {...}}
    """
    scorer = CVScorer()
    result = scorer.score(text, extracted_skills)
    logger.info(
        f"CV scored: {result['score']}/100 "
        f"(SRS weights — keywords:{SRS_WEIGHTS['keywords']}, "
        f"completeness:{SRS_WEIGHTS['completeness']}, "
        f"skill_density:{SRS_WEIGHTS['skill_density']}, "
        f"formatting:{SRS_WEIGHTS['formatting']})"
    )
    return result