"""
Indeed Job Scraper (SCR-FR-01).

Scrapes publicly accessible job listings from Indeed /jobs.
No authentication required — public listings only (SCR-FR-01 BR-4).

Indeed serves its search results server-side (HTML) for the initial
listing cards, making BeautifulSoup4 sufficient.

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

_BASE_URL = "https://www.indeed.com"
_SEARCH_URL = "https://www.indeed.com/jobs"


class IndeedScraper(BaseScraper):
    """Scrape Indeed public job listings."""

    PORTAL_NAME = "Indeed"

    def scrape(self, query: str, location: str = "", max_results: int = 20) -> List[JobListing]:
        """
        Scrape Indeed job search results.

        Args:
            query: Job title / keywords
            location: City or 'remote'
            max_results: Maximum listings to return

        Returns:
            List of JobListing objects
        """
        listings: List[JobListing] = []
        start = 0
        page_size = 15  # Indeed default

        while len(listings) < max_results:
            params = {
                "q": query,
                "l": location,
                "start": start,
                "fromage": "30",    # only last 30 days (aligns with BR-2 purge)
            }
            if not location:
                params.pop("l")

            resp = self._get(_SEARCH_URL, params=params)
            if resp is None:
                logger.warning(f"[Indeed] Failed to fetch page start={start}")
                break

            page_listings = self._parse_search_page(resp.text)
            if not page_listings:
                break

            for listing in page_listings:
                if len(listings) >= max_results:
                    break
                listings.append(listing)

            start += page_size
            if len(page_listings) < page_size:
                break

        logger.info(f"[Indeed] Scraped {len(listings)} listings for '{query}' in '{location}'")
        return listings

    def _parse_search_page(self, html: str) -> List[JobListing]:
        """Parse Indeed search results page."""
        soup = BeautifulSoup(html, "html.parser")
        listings = []

        # Indeed uses td.resultContent or div.job_seen_beacon (2024 layout)
        cards = soup.find_all("div", class_=re.compile(r"job_seen_beacon|jobsearch-SerpJobCard"))
        if not cards:
            cards = soup.find_all("td", class_="resultContent")

        for card in cards:
            try:
                listing = self._parse_card(card)
                if listing:
                    listings.append(listing)
            except Exception as e:
                logger.debug(f"[Indeed] Card parse error: {e}")
                continue

        return listings

    def _parse_card(self, card) -> Optional[JobListing]:
        """Extract fields from a single Indeed job card."""
        # Title
        title_el = (
            card.find("h2", class_=re.compile(r"jobTitle"))
            or card.find("a", {"data-jk": True})
        )
        if not title_el:
            return None
        title = title_el.get_text(strip=True).replace("new", "").strip()

        # Company
        company_el = card.find(
            "span", {"data-testid": "company-name"}
        ) or card.find("span", class_=re.compile(r"companyName"))
        company = company_el.get_text(strip=True) if company_el else "Unknown Company"

        # Location
        location_el = card.find(
            "div", {"data-testid": "text-location"}
        ) or card.find("div", class_=re.compile(r"companyLocation"))
        location = location_el.get_text(strip=True) if location_el else ""

        # URL — job key is embedded in the card
        job_id_el = card.find("a", {"data-jk": True}) or card.find("a", id=re.compile(r"job_"))
        url = ""
        if job_id_el:
            jk = job_id_el.get("data-jk") or job_id_el.get("id", "").replace("job_", "")
            if jk:
                url = f"https://www.indeed.com/viewjob?jk={jk}"
        if not url:
            link = card.find("a", href=re.compile(r"/rc/clk|/viewjob"))
            if link:
                href = link.get("href", "")
                url = urljoin(_BASE_URL, href.split("?")[0])

        # Description snippet
        snippet_el = card.find("div", class_=re.compile(r"job-snippet|summary"))
        description = snippet_el.get_text(separator=" ", strip=True)[:500] if snippet_el else ""

        if not title or not url:
            return None

        return self._make_listing(title, company, location, url, description)