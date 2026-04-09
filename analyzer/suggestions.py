"""
Suggestion Generation Engine.

FIX (Issue #10 — filename typo):
  File was named 'suggerstions.py' (extra 'r'). app.py imports from
  'analyzer.suggestions' which caused a ModuleNotFoundError. Renamed to
  suggestions.py. This is the correct canonical file going forward.

FIX (scorer.py key alignment after Issue #2 fix):
  scorer.py now returns both SRS dimension keys AND backward-compat aliases:
    - "keyword_density_score"  → alias for "keywords_score"
    - "skills_relevance_score" → alias for "skill_density_score"
    - "experience_score", "education_score", "projects_score" (UI detail scores)
  All lookups here use the backward-compat alias keys so no changes are needed
  to the suggestion logic itself.

FIX (Issue #5 — NullHandler):
  Added logging.NullHandler() for Django safety.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Safe for library / Django use


class SuggestionGenerator:
    """Generate actionable, prioritized CV improvement suggestions."""

    def generate(
        self, score_result: Dict, extracted_skills: Dict, gaps: Dict
    ) -> List[Dict]:
        suggestions = []
        suggestions.extend(self._high_priority(score_result, extracted_skills, gaps))
        suggestions.extend(self._medium_priority(score_result, extracted_skills, gaps))
        suggestions.extend(self._low_priority(score_result, extracted_skills, gaps))
        return suggestions

    # ------------------------------------------------------------------
    # Priority buckets
    # ------------------------------------------------------------------

    def _high_priority(self, score_result, extracted_skills, gaps) -> List[Dict]:
        suggestions = []
        bd = score_result.get("breakdown", {})
        score = score_result.get("score", 0)

        if score < 50:
            suggestions.append({
                "priority": "High",
                "category": "Overall",
                "message": (
                    f"Your CV score is {score}/100 — below 50. Consider a comprehensive "
                    "revision: add more skills, quantify achievements, and improve structure."
                ),
            })

        if len(extracted_skills.get("technical", [])) < 3:
            suggestions.append({
                "priority": "High",
                "category": "Skills",
                "message": (
                    "Fewer than 3 technical skills detected. Explicitly list programming "
                    "languages, frameworks, tools, and databases in a dedicated Skills section."
                ),
            })

        top_missing = gaps.get("high_priority_skills", [])[:3]
        if top_missing:
            suggestions.append({
                "priority": "High",
                "category": "Skill Gap",
                "message": (
                    f"High-demand skills missing from your CV: {', '.join(top_missing)}. "
                    "Adding or learning these will significantly improve your market fit."
                ),
            })

        if bd.get("formatting_score", 0) < 40:
            suggestions.append({
                "priority": "High",
                "category": "Formatting",
                "message": (
                    "CV formatting needs improvement. Add clear section headers, bullet points, "
                    "and ensure your email, phone, and LinkedIn are visible at the top."
                ),
            })

        if bd.get("experience_score", 0) < 30:
            suggestions.append({
                "priority": "High",
                "category": "Experience",
                "message": (
                    "Experience section is weak or missing. Include job titles, company names, "
                    "date ranges (e.g. 2022–2024), and bullet-pointed achievements."
                ),
            })

        return suggestions

    def _medium_priority(self, score_result, extracted_skills, gaps) -> List[Dict]:
        suggestions = []
        bd = score_result.get("breakdown", {})
        score = score_result.get("score", 0)

        if 50 <= score < 75:
            suggestions.append({
                "priority": "Medium",
                "category": "Overall",
                "message": (
                    "Good foundation — but there's room to grow. Add measurable results "
                    "(e.g. 'reduced load time by 40%') to push your score above 75."
                ),
            })

        if len(extracted_skills.get("soft", [])) < 2:
            suggestions.append({
                "priority": "Medium",
                "category": "Soft Skills",
                "message": (
                    "Add soft skills (leadership, communication, teamwork, problem-solving). "
                    "These are detected from your CV text — mention them naturally."
                ),
            })

        # Uses backward-compat alias key set by scorer.py
        if bd.get("keyword_density_score", 0) < 40:
            suggestions.append({
                "priority": "Medium",
                "category": "Keywords",
                "message": (
                    "Low keyword density. Use strong action verbs (led, built, optimized) "
                    "and include industry-specific terms to improve ATS ranking."
                ),
            })

        if bd.get("education_score", 0) < 30:
            suggestions.append({
                "priority": "Medium",
                "category": "Education",
                "message": (
                    "Enhance your education section: include degree name, institution, "
                    "graduation year, and any relevant coursework or GPA."
                ),
            })

        emerging = gaps.get("emerging_missing_skills", [])[:2]
        if emerging:
            suggestions.append({
                "priority": "Medium",
                "category": "Emerging Skills",
                "message": (
                    f"Consider picking up these emerging skills: {', '.join(emerging)}. "
                    "They are gaining rapid market traction."
                ),
            })

        return suggestions

    def _low_priority(self, score_result, extracted_skills, gaps) -> List[Dict]:
        suggestions = []
        bd = score_result.get("breakdown", {})
        score = score_result.get("score", 0)

        if score >= 75:
            suggestions.append({
                "priority": "Low",
                "category": "Polish",
                "message": (
                    "Great CV score! Fine-tune by adding portfolio links, GitHub profile, "
                    "or open-source contributions to stand out further."
                ),
            })

        if bd.get("projects_score", 0) == 0:
            suggestions.append({
                "priority": "Low",
                "category": "Projects",
                "message": (
                    "No projects section detected. Add 2–3 personal or academic projects "
                    "with the technologies used and measurable outcomes."
                ),
            })

        if bd.get("word_count", 0) < 300:
            suggestions.append({
                "priority": "Low",
                "category": "Content Length",
                "message": (
                    f"CV is short ({bd.get('word_count', 0)} words). "
                    "Aim for 300–1000 words for better ATS coverage and recruiter impact."
                ),
            })

        return suggestions


def generate_suggestions(
    score_result: Dict, extracted_skills: Dict, gaps: Dict
) -> List[Dict]:
    """
    Generate improvement suggestions.

    Args:
        score_result: Output from calculate_score()
        extracted_skills: Output from extract_skills()
        gaps: Output from detect_skill_gaps()

    Returns:
        List of {"priority": str, "category": str, "message": str}
    """
    gen = SuggestionGenerator()
    suggestions = gen.generate(score_result, extracted_skills, gaps)
    logger.info(f"Generated {len(suggestions)} suggestions")
    return suggestions