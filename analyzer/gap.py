"""
Skill Gap Detection Module.

Identifies missing high-demand skills compared to market requirements.

Fix: market_skills.json path is resolved relative to this file so it works
regardless of the working directory (important for Django migration later).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MARKET_SKILLS = Path(__file__).parent.parent / "models" / "market_skills.json"


class GapDetector:
    """Detect skill gaps against market demand."""

    def __init__(self, market_skills_path: Optional[str] = None):
        self.market_skills = self._load_market_skills(market_skills_path)

    def _load_market_skills(self, market_skills_path: Optional[str]) -> dict:
        path = Path(market_skills_path) if market_skills_path else _DEFAULT_MARKET_SKILLS
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded market skills from {path}")
            return data
        except FileNotFoundError:
            logger.warning(f"Market skills file not found at {path}")
            return {"high_demand": [], "emerging": []}
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in market skills at {path}")
            return {"high_demand": [], "emerging": []}

    def detect_gaps(self, user_skills: Dict) -> Dict:
        """
        Detect skill gaps for user.

        Args:
            user_skills: {"technical": [...], "soft": [...]}

        Returns:
            Gap analysis dictionary
        """
        if isinstance(user_skills, dict):
            user_technical = {s.lower() for s in user_skills.get("technical", [])}
            user_soft = {s.lower() for s in user_skills.get("soft", [])}
        else:
            user_technical = set()
            user_soft = set()

        user_all = user_technical | user_soft

        market_high = {s.lower() for s in self.market_skills.get("high_demand", [])}
        market_emerging = {s.lower() for s in self.market_skills.get("emerging", [])}
        market_all = market_high | market_emerging

        missing_skills = sorted(market_all - user_all)
        high_priority_missing = sorted(market_high - user_all)
        emerging_missing = sorted(market_emerging - user_all)

        coverage = (
            round(len(user_all & market_high) / len(market_high) * 100, 1)
            if market_high
            else 0.0
        )

        return {
            "missing_skills": missing_skills,
            "high_priority_skills": high_priority_missing,
            "emerging_missing_skills": emerging_missing,
            "user_has_high_demand_count": len(user_all & market_high),
            "total_high_demand": len(market_high),
            "coverage_percentage": coverage,
        }


def detect_skill_gaps(extracted_skills: Dict) -> Dict:
    """
    Detect skill gaps.

    Args:
        extracted_skills: {"technical": [...], "soft": [...]}

    Returns:
        Gap analysis dictionary
    """
    detector = GapDetector()
    gaps = detector.detect_gaps(extracted_skills)
    logger.info(
        f"Gap analysis: {len(gaps['missing_skills'])} missing, "
        f"{gaps['coverage_percentage']}% high-demand coverage"
    )
    return gaps