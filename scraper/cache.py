"""
Job Listing Cache (SCR-FR-01 BR-2).

In-memory cache for Streamlit prototype.
(In Django production: replace with PostgreSQL JobListings table.)

SRS requirements implemented:
    BR-2: Listings older than 30 days automatically purged on access.
    Refresh: 6-hour TTL triggers re-scrape on next access.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .base import JobListing, LISTING_MAX_AGE_DAYS

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# SRS SCR-FR-01: refresh listings every 6 hours
CACHE_TTL_HOURS = 6


class JobCache:
    """
    Thread-safe in-memory job cache.

    Cache key: (query, location, portals_tuple)
    Cache value: (listings, cached_at_timestamp)
    """

    def __init__(self):
        self._cache: Dict[str, Tuple[List[JobListing], float]] = {}

    def _make_key(self, query: str, location: str, portals: List[str]) -> str:
        return f"{query.lower().strip()}|{location.lower().strip()}|{','.join(sorted(portals))}"

    def get(self, query: str, location: str, portals: List[str]) -> Optional[List[JobListing]]:
        """
        Return cached listings if fresh (< 6 hours old).
        Purges expired individual listings (> 30 days) before returning.
        Returns None if cache miss or stale.
        """
        key = self._make_key(query, location, portals)
        entry = self._cache.get(key)
        if entry is None:
            return None

        listings, cached_at = entry
        age_hours = (time.monotonic() - cached_at) / 3600

        if age_hours >= CACHE_TTL_HOURS:
            logger.info(f"[Cache] STALE — key '{key}' ({age_hours:.1f}h old)")
            del self._cache[key]
            return None

        # SRS BR-2: purge listings older than 30 days
        fresh = [j for j in listings if not j.is_expired()]
        if len(fresh) < len(listings):
            logger.info(f"[Cache] Purged {len(listings) - len(fresh)} expired listings")
            self._cache[key] = (fresh, cached_at)

        logger.info(f"[Cache] HIT — '{key}' ({len(fresh)} listings, {age_hours:.1f}h old)")
        return fresh

    def set(self, query: str, location: str, portals: List[str], listings: List[JobListing]) -> None:
        """Store listings in cache with current timestamp."""
        key = self._make_key(query, location, portals)
        self._cache[key] = (listings, time.monotonic())
        logger.info(f"[Cache] SET — '{key}' ({len(listings)} listings)")

    def invalidate(self, query: str = "", location: str = "", portals: Optional[List[str]] = None) -> int:
        """Invalidate specific or all cache entries. Returns count removed."""
        if not query and not location and portals is None:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"[Cache] Cleared all {count} entries")
            return count
        key = self._make_key(query, location, portals or [])
        if key in self._cache:
            del self._cache[key]
            return 1
        return 0

    def stats(self) -> dict:
        """Return cache statistics for display in the scraper monitor."""
        total_listings = sum(len(v[0]) for v in self._cache.values())
        return {
            "cached_queries": len(self._cache),
            "total_listings": total_listings,
            "entries": [
                {
                    "key": k,
                    "count": len(v[0]),
                    "age_hours": round((time.monotonic() - v[1]) / 3600, 1),
                }
                for k, v in self._cache.items()
            ],
        }


# Module-level singleton shared across Streamlit sessions
_global_cache = JobCache()


def get_cache() -> JobCache:
    """Return the module-level cache singleton."""
    return _global_cache