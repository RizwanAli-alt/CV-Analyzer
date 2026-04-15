"""
Scraper Orchestrator — Unified Pipeline.

Coordinates scraping across all portals, match scoring, and caching.
This is the single entry point called by the Streamlit app.

SRS compliance:
    SCR-FR-01: Multi-portal scraping with rate limiting, failure isolation
    SCR-FR-02: Match scoring via matcher.py, sorted descending
    BR-2: Cache integration with 6-hour TTL
    BR-3: Each portal failure is isolated — others continue
"""

import logging
import time
from typing import List, Optional, Dict

from .base import JobListing
from .linkedin import LinkedInScraper
from .indeed import IndeedScraper
from .rozee import RozeeScraper
from .matcher import score_and_sort, is_match_available
from .cache import get_cache
from .mock_data import get_mock_listings

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Portal name → scraper class
_PORTAL_CLASSES = {
    "LinkedIn": LinkedInScraper,
    "Indeed": IndeedScraper,
    "Rozee.pk": RozeeScraper,
}


class ScraperOrchestrator:
    """
    Runs the full scrape → score → sort pipeline.

    Usage:
        orchestrator = ScraperOrchestrator(skill_db=skill_db)
        results = orchestrator.run(
            query="Python Developer",
            location="Islamabad",
            portals=["LinkedIn", "Rozee.pk"],
            user_skills={"technical": [...], "soft": [...]},
            max_results=30,
            demo_mode=False,
        )
    """

    def __init__(self, skill_db: Optional[dict] = None):
        self._skill_db = skill_db or {}
        self._cache = get_cache()
        self._run_log: List[dict] = []   # per-run scraper health log (for monitoring)

    def run(
        self,
        query: str,
        location: str = "",
        portals: Optional[List[str]] = None,
        user_skills: Optional[Dict] = None,
        max_per_portal: int = 15,
        demo_mode: bool = False,
        force_refresh: bool = False,
    ) -> List[JobListing]:
        """
        Main pipeline: scrape → cache → score → sort.

        Args:
            query: Job search keywords (e.g. "Python Developer")
            location: City / country filter
            portals: List of portals to query (default: all three)
            user_skills: CV analyzer output for match scoring
            max_per_portal: Max listings per portal
            demo_mode: Use mock data instead of live scraping
            force_refresh: Bypass cache and re-scrape

        Returns:
            List[JobListing] sorted by match_score descending
        """
        portals = portals or ["LinkedIn", "Indeed", "Rozee.pk"]
        user_skills = user_skills or {}
        self._run_log = []

        if not demo_mode:
            # Check cache first
            if not force_refresh:
                cached = self._cache.get(query, location, portals)
                if cached is not None:
                    logger.info(f"[Orchestrator] Serving {len(cached)} listings from cache")
                    return score_and_sort(cached, user_skills)

            # Live scrape
            all_listings = self._scrape_all_portals(query, location, portals, max_per_portal)

            # Store in cache
            if all_listings:
                self._cache.set(query, location, portals, all_listings)
        else:
            # Demo mode — use mock data
            all_listings = get_mock_listings(
                query=query,
                location=location,
                portals=portals,
                max_results=max_per_portal * len(portals),
            )
            logger.info(f"[Orchestrator] Demo mode: {len(all_listings)} mock listings")

        # Score and sort (SCR-FR-02)
        if user_skills:
            all_listings = score_and_sort(all_listings, user_skills)
        else:
            all_listings.sort(key=lambda j: j.scraped_at, reverse=True)

        return all_listings

    def _scrape_all_portals(
        self, query: str, location: str, portals: List[str], max_per_portal: int
    ) -> List[JobListing]:
        """
        Scrape each portal independently.
        SCR-FR-01 BR-3: failures are isolated — one portal crashing
        does not stop others.
        """
        all_listings: List[JobListing] = []

        for portal_name in portals:
            scraper_cls = _PORTAL_CLASSES.get(portal_name)
            if scraper_cls is None:
                logger.warning(f"[Orchestrator] Unknown portal: {portal_name}")
                continue

            start_time = time.monotonic()
            try:
                scraper = scraper_cls(skill_db=self._skill_db)
                listings = scraper.scrape(query, location, max_results=max_per_portal)
                elapsed = round(time.monotonic() - start_time, 1)

                self._run_log.append({
                    "portal": portal_name,
                    "status": "success",
                    "count": len(listings),
                    "elapsed_s": elapsed,
                    "error": None,
                })

                logger.info(f"[Orchestrator] {portal_name}: {len(listings)} listings in {elapsed}s")
                all_listings.extend(listings)

            except Exception as e:
                elapsed = round(time.monotonic() - start_time, 1)
                # SCR-FR-01 BR-3: log and continue — do not crash pipeline
                logger.error(f"[Orchestrator] {portal_name} scraper failed: {e}")
                self._run_log.append({
                    "portal": portal_name,
                    "status": "error",
                    "count": 0,
                    "elapsed_s": elapsed,
                    "error": str(e),
                })

        return all_listings

    def get_run_log(self) -> List[dict]:
        """Return per-portal health log for the scraper monitor UI."""
        return self._run_log

    def cache_stats(self) -> dict:
        """Return current cache statistics."""
        return self._cache.stats()

    def clear_cache(self) -> int:
        """Clear all cached listings and return count removed."""
        return self._cache.invalidate()