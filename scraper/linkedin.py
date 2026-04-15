"""
LinkedIn Job Scraper (SCR-FR-01).

Scrapes publicly accessible job listings from LinkedIn /jobs/search.
No authentication required — public listings only (SCR-FR-01 BR-4).

LinkedIn renders most content server-side in the public job search,
so BeautifulSoup4 is sufficient for the initial listing cards.
Full job descriptions require a follow-up request per listing.

Rate limiting: 2-second delay enforced by BaseScraper._throttle() (BR-1).
Failures: logged and skipped — pipeline continues (BR-3).
"""

import logging
import re
from typing import List, Optional
from bs4 import BeautifulSoup
from .base import BaseScraper, JobListing

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# LinkedIn public job search (no auth required)
_SEARCH_URL = "https://www.linkedin.com/jobs/search/"


class LinkedInScraper(BaseScraper):
    """Scrape LinkedIn public job listings."""

    PORTAL_NAME = "LinkedIn"

    def scrape(self, query: str, location: str = "", max_results: int = 20) -> List[JobListing]:
        """
        Scrape LinkedIn job search results.

        Args:
            query: Job title / skill keywords
            location: City, country, or 'Remote'
            max_results: Maximum listings to return

        Returns:
            List of JobListing objects
        """
        listings: List[JobListing] = []
        start = 0
        page_size = 25  # LinkedIn returns up to 25 per page

        while len(listings) < max_results:
            params = {
                "keywords": query,
                "location": location,
                "start": start,
                "trk": "public_jobs_jobs-search-bar_search-submit",
                "position": 1,
                "pageNum": 0,
            }

            resp = self._get(_SEARCH_URL, params=params)
            if resp is None:
                logger.warning(f"[LinkedIn] Failed to fetch page start={start}")
                break

            page_listings = self._parse_search_page(resp.text)
            if not page_listings:
                break

            for listing in page_listings:
                if len(listings) >= max_results:
                    break
                # Optionally fetch individual job description
                detail = self._fetch_description(listing.url)
                if detail:
                    listing.description = detail
                    from .base import _extract_skills_from_text, _detect_modality
                    listing.skills_mentioned = _extract_skills_from_text(detail, self._skill_db)
                    listing.modality = _detect_modality(f"{listing.title} {listing.location} {detail}")
                listings.append(listing)

            start += page_size
            if len(page_listings) < page_size:
                break  # no more pages

        logger.info(f"[LinkedIn] Scraped {len(listings)} listings for '{query}' in '{location}'")
        return listings

    def _parse_search_page(self, html: str) -> List[JobListing]:
        """Parse the LinkedIn job search results page."""
        soup = BeautifulSoup(html, "html.parser")
        listings = []

        # LinkedIn public search uses these classes (as of 2024-2025)
        cards = soup.find_all("div", class_=re.compile(r"job-search-card|base-card"))
        if not cards:
            # Fallback: try list items
            cards = soup.find_all("li", class_=re.compile(r"jobs-search__results"))

        for card in cards:
            try:
                listing = self._parse_card(card)
                if listing:
                    listings.append(listing)
            except Exception as e:
                logger.debug(f"[LinkedIn] Card parse error: {e}")
                continue

        return listings

    def _parse_card(self, card) -> Optional[JobListing]:
        """Extract fields from a single job card element."""
        # Title
        title_el = (
            card.find("h3", class_=re.compile(r"base-search-card__title|job-search-card__title"))
            or card.find("span", class_="sr-only")
        )
        if not title_el:
            return None
        title = title_el.get_text(strip=True)

        # Company
        company_el = card.find(
            "h4", class_=re.compile(r"base-search-card__subtitle")
        ) or card.find("a", class_=re.compile(r"job-search-card__company"))
        company = company_el.get_text(strip=True) if company_el else "Unknown Company"

        # Location
        location_el = card.find(
            "span", class_=re.compile(r"job-search-card__location|base-search-card__metadata")
        )
        location = location_el.get_text(strip=True) if location_el else ""

        # URL
        link_el = card.find("a", href=re.compile(r"/jobs/view/"))
        url = ""
        if link_el:
            href = link_el.get("href", "")
            # Strip tracking params
            url = href.split("?")[0] if "?" in href else href
            if not url.startswith("http"):
                url = "https://www.linkedin.com" + url

        if not title or not url:
            return None

        return self._make_listing(title, company, location, url)

    def _fetch_description(self, job_url: str) -> str:
        """Fetch full job description from individual listing page."""
        if not job_url:
            return ""
        resp = self._get(job_url)
        if not resp:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        desc_el = soup.find("div", class_=re.compile(r"description__text|show-more-less-html"))
        if desc_el:
            return desc_el.get_text(separator=" ", strip=True)[:3000]
        return ""