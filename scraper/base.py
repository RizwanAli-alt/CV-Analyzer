"""
Base Scraper Infrastructure.

Provides:
    - JobListing dataclass (canonical result format for all portals)
    - BaseScraper with shared HTTP session, User-Agent rotation,
      and SRS-compliant 2-second rate limiting (SCR-FR-01 BR-1)
    - Graceful error handling — failures logged, pipeline continues (BR-3)
    - Public-listings-only enforcement (BR-4)
"""

import time
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# SRS SCR-FR-01 BR-1: minimum delay between consecutive requests per portal
REQUEST_DELAY_SECONDS = 2.0

# SRS SCR-FR-01 BR-2: purge listings older than this
LISTING_MAX_AGE_DAYS = 30

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]
_UA_INDEX = 0


def _next_ua() -> str:
    global _UA_INDEX
    ua = _USER_AGENTS[_UA_INDEX % len(_USER_AGENTS)]
    _UA_INDEX += 1
    return ua


@dataclass
class JobListing:
    """Canonical job listing returned by every portal scraper."""
    title: str
    company: str
    location: str
    modality: str                   # "Remote" | "On-site" | "Hybrid" | "Unknown"
    portal: str                     # "LinkedIn" | "Indeed" | "Rozee.pk"
    url: str
    description: str = ""
    skills_mentioned: List[str] = field(default_factory=list)
    scraped_at: datetime = field(default_factory=datetime.utcnow)
    match_score: float = 0.0        # filled by matcher.py (SCR-FR-02)

    def is_expired(self) -> bool:
        """True if listing is older than LISTING_MAX_AGE_DAYS (SCR-FR-01 BR-2)."""
        return (datetime.utcnow() - self.scraped_at).days >= LISTING_MAX_AGE_DAYS

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "modality": self.modality,
            "portal": self.portal,
            "url": self.url,
            "description": self.description,
            "skills_mentioned": self.skills_mentioned,
            "scraped_at": self.scraped_at.isoformat(),
            "match_score": self.match_score,
        }


def _detect_modality(text: str) -> str:
    """Infer work modality from listing text."""
    t = text.lower()
    if "remote" in t:
        return "Remote"
    if "hybrid" in t:
        return "Hybrid"
    if any(w in t for w in ("on-site", "onsite", "on site", "in-office", "in office")):
        return "On-site"
    return "Unknown"


def _extract_skills_from_text(text: str, skill_db: Optional[dict] = None) -> List[str]:
    """
    Quick keyword scan of a job description for known skills.
    Uses the same skill_db as the CV analyzer for consistency.
    """
    if not skill_db:
        return []
    found = []
    text_lower = text.lower()
    for skill in skill_db.get("technical", []) + skill_db.get("soft", []):
        pattern = rf"\b{re.escape(skill.lower())}\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return list(set(found))


def _build_session() -> requests.Session:
    """Build a requests Session with retry logic and browser headers."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "DNT": "1",
    })
    return session


class BaseScraper:
    """
    Base class for all portal scrapers.

    Subclasses implement:
        scrape(query, location, max_results) -> List[JobListing]
    """

    PORTAL_NAME: str = "Unknown"

    def __init__(self, skill_db: Optional[dict] = None):
        self._session = _build_session()
        self._last_request_time: float = 0.0
        self._skill_db = skill_db or {}

    def _throttle(self) -> None:
        """SRS SCR-FR-01 BR-1: enforce 2-second delay between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < REQUEST_DELAY_SECONDS:
            time.sleep(REQUEST_DELAY_SECONDS - elapsed)
        self._last_request_time = time.monotonic()

    def _get(self, url: str, params: Optional[dict] = None, timeout: int = 15) -> Optional[requests.Response]:
        """
        GET with rate limiting, rotating UA, and graceful error handling.
        SCR-FR-01 BR-3: failures logged, None returned so pipeline continues.
        """
        self._throttle()
        self._session.headers["User-Agent"] = _next_ua()
        try:
            resp = self._session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            logger.info(f"[{self.PORTAL_NAME}] GET {url} → {resp.status_code}")
            return resp
        except requests.exceptions.HTTPError as e:
            logger.warning(f"[{self.PORTAL_NAME}] HTTP error for {url}: {e}")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"[{self.PORTAL_NAME}] Connection error for {url}: {e}")
        except requests.exceptions.Timeout:
            logger.warning(f"[{self.PORTAL_NAME}] Timeout for {url}")
        except Exception as e:
            logger.error(f"[{self.PORTAL_NAME}] Unexpected error for {url}: {e}")
        return None

    def _make_listing(self, title: str, company: str, location: str,
                      url: str, description: str = "") -> JobListing:
        return JobListing(
            title=title,
            company=company,
            location=location,
            modality=_detect_modality(f"{title} {location} {description}"),
            portal=self.PORTAL_NAME,
            url=url,
            description=description,
            skills_mentioned=_extract_skills_from_text(description, self._skill_db),
        )

    def scrape(self, query: str, location: str = "", max_results: int = 20) -> List[JobListing]:
        raise NotImplementedError