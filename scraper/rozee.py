"""
Rozee.pk Job Scraper (SCR-FR-01).

Scrapes publicly accessible job listings from Rozee.pk — Pakistan's
largest job portal. Included per SRS SCR-FR-01 requirement.

Rozee.pk serves job cards server-side in HTML, making BeautifulSoup4
sufficient. No authentication required (BR-4).

Rate limiting: 2-second delay enforced by BaseScraper._throttle() (BR-1).
Failures: logged and skipped — pipeline continues (BR-3).
"""

import logging
import re
from typing import List, Optional
from urllib.parse import urljoin, quote_plus
from bs4 import BeautifulSoup
from .base import BaseScraper, JobListing

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_BASE_URL = "https://www.rozee.pk"
_SEARCH_URL = "https://www.rozee.pk/job/jsearch/q/{query}"
_SEARCH_CITY_URL = "https://www.rozee.pk/job/jsearch/q/{query}/fc/1/fpth/{city}"


class RozeeScraper(BaseScraper):
    """Scrape Rozee.pk public job listings."""

    PORTAL_NAME = "Rozee.pk"

    def scrape(self, query: str, location: str = "", max_results: int = 20) -> List[JobListing]:
        """
        Scrape Rozee.pk job search results.

        Args:
            query: Job title / keywords
            location: City in Pakistan (Karachi, Lahore, Islamabad, etc.)
            max_results: Maximum listings to return

        Returns:
            List of JobListing objects
        """
        listings: List[JobListing] = []
        page = 1

        # Build search URL
        query_slug = quote_plus(query.lower().replace(" ", "-"))
        if location:
            city_slug = location.lower().replace(" ", "-")
            base_url = _SEARCH_CITY_URL.format(query=query_slug, city=city_slug)
        else:
            base_url = _SEARCH_URL.format(query=query_slug)

        while len(listings) < max_results:
            url = base_url if page == 1 else f"{base_url}/pg/{page}"
            resp = self._get(url)
            if resp is None:
                logger.warning(f"[Rozee.pk] Failed to fetch page {page}")
                break

            page_listings = self._parse_search_page(resp.text)
            if not page_listings:
                break

            for listing in page_listings:
                if len(listings) >= max_results:
                    break
                listings.append(listing)

            page += 1
            if len(page_listings) < 10:
                break

        logger.info(f"[Rozee.pk] Scraped {len(listings)} listings for '{query}' in '{location}'")
        return listings

    def _parse_search_page(self, html: str) -> List[JobListing]:
        """Parse Rozee.pk search results page."""
        soup = BeautifulSoup(html, "html.parser")
        listings = []

        # Rozee.pk job cards (2024 layout)
        cards = soup.find_all("div", class_=re.compile(r"job|listing|jlisting"))
        if not cards:
            # Try alternate selectors
            cards = soup.find_all("li", class_=re.compile(r"job"))

        for card in cards:
            try:
                listing = self._parse_card(card)
                if listing:
                    listings.append(listing)
            except Exception as e:
                logger.debug(f"[Rozee.pk] Card parse error: {e}")
                continue

        return listings

    def _parse_card(self, card) -> Optional[JobListing]:
        """Extract fields from a single Rozee.pk job card."""
        # Title
        title_el = (
            card.find("h3")
            or card.find("h2")
            or card.find("a", class_=re.compile(r"title|job-title"))
        )
        if not title_el:
            return None
        title = title_el.get_text(strip=True)
        if not title or len(title) < 3:
            return None

        # Company
        company_el = (
            card.find("span", class_=re.compile(r"company|employer"))
            or card.find("div", class_=re.compile(r"company"))
        )
        company = company_el.get_text(strip=True) if company_el else "Unknown Company"

        # Location
        location_el = (
            card.find("span", class_=re.compile(r"location|city"))
            or card.find("div", class_=re.compile(r"location"))
        )
        location = location_el.get_text(strip=True) if location_el else "Pakistan"

        # URL
        link_el = title_el if title_el.name == "a" else title_el.find("a")
        if not link_el:
            link_el = card.find("a", href=re.compile(r"/job/|/apply/"))
        url = ""
        if link_el:
            href = link_el.get("href", "")
            url = href if href.startswith("http") else urljoin(_BASE_URL, href)

        # Description snippet
        desc_el = card.find("div", class_=re.compile(r"desc|summary|detail"))
        description = desc_el.get_text(separator=" ", strip=True)[:500] if desc_el else ""

        if not title:
            return None

        # Generate a placeholder URL if none found
        if not url:
            url = f"https://www.rozee.pk/job/jsearch/q/{title.lower().replace(' ', '-')}"

        return self._make_listing(title, company, location, url, description)