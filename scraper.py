"""
scraper.py – SHL product catalog scraper (improved)

Improvements over original:
- Pagination handling: crawls all catalog pages, not just page 1
- Retry logic with exponential backoff (rate limiting protection)
- Richer data extraction: description, test_type parsed from page content
- Progress reporting with tqdm
- Deduplication by URL
- Saves both CSV and JSON
- Polite crawling: configurable delay between requests

Usage:
    python scraper.py
    python scraper.py --max-pages 20 --delay 1.5
"""

import argparse
import csv
import json
import logging
import sys
import time
from urllib.parse import urljoin, urlencode

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("shl_scraper")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
DEFAULT_OUTPUT_CSV = "shl_catalog.csv"
DEFAULT_OUTPUT_JSON = "shl_catalog.json"
DEFAULT_MAX_PAGES = 32
DEFAULT_DELAY = 1.0  # seconds between requests
DEFAULT_TIMEOUT = 15

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; SHLCatalogBot/1.0; "
        "assignment-research-scraper)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# Test type keywords for classification
_TEST_TYPE_KEYWORDS = {
    "Cognitive": ["cognitive", "numerical", "verbal", "reasoning", "verify", "inductive", "deductive"],
    "Personality": ["personality", "behavioral", "behaviour", "opq", "motivational"],
    "Technical": ["coding", "technical", "programming", "simulation", "software"],
    "Situational Judgment": ["situational", "sjt", "judgment", "scenario"],
    "Skills": ["skills", "typing", "data entry", "simulation", "clerical"],
}


def _classify_test_type(name: str, description: str) -> str:
    """Heuristically classify test type from name and description."""
    text = (name + " " + description).lower()
    for test_type, keywords in _TEST_TYPE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return test_type
    return "Personality"  # SHL default for job-focused solutions


def _fetch_with_retry(url: str, max_retries: int = 3, delay: float = 1.0) -> requests.Response | None:
    """Fetch URL with exponential backoff on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 429:
                wait = delay * (2 ** attempt)
                logger.warning(f"Rate limited (429). Waiting {wait:.1f}s...")
                time.sleep(wait)
            else:
                logger.warning(f"HTTP {resp.status_code} for {url}")
                return None
        except requests.RequestException as e:
            logger.warning(f"Request error (attempt {attempt}/{max_retries}): {e}")
            time.sleep(delay * attempt)
    logger.error(f"Failed to fetch after {max_retries} attempts: {url}")
    return None


def _parse_product_list_page(html: str) -> list[dict]:
    """
    Parse a catalog listing page and extract product links + basic metadata.

    SHL's catalog page uses a table/grid with each product having a link.
    """
    soup = BeautifulSoup(html, "html.parser")
    products = []

    # SHL catalog renders products in a table with class "custom-table" or similar
    # We look for all links pointing to /products/product-catalog/view/
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/product-catalog/view/" not in href:
            continue

        name = link.get_text(strip=True)
        if not name or len(name) < 3:
            continue

        full_url = urljoin(BASE_URL, href)
        products.append({"name": name, "url": full_url})

    return products


def _parse_product_detail_page(url: str, name: str, delay: float) -> dict:
    """
    Fetch and parse a product detail page to extract description and test type.
    """
    time.sleep(delay)
    resp = _fetch_with_retry(url)
    if not resp:
        return {"name": name, "url": url, "description": "", "test_type": "Unknown", "content": ""}

    soup = BeautifulSoup(resp.text, "html.parser")

    # Try to extract description from meta description tag
    description = ""
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        description = meta_desc["content"].strip()

    # Fallback: first paragraph in main content area
    if not description:
        for tag in soup.find_all(["p", "div"], class_=lambda c: c and "description" in c.lower()):
            text = tag.get_text(strip=True)
            if len(text) > 30:
                description = text[:600]
                break

    # Fallback: first meaningful paragraph in body
    if not description:
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 50:
                description = text[:600]
                break

    # Classify test type
    test_type = _classify_test_type(name, description)

    # Full page text (for content column, truncated for storage)
    full_text = soup.get_text(separator=" ", strip=True)[:2000]

    return {
        "name": name,
        "url": url,
        "description": description,
        "test_type": test_type,
        "content": full_text,
    }


def scrape_catalog(
    max_pages: int = DEFAULT_MAX_PAGES,
    delay: float = DEFAULT_DELAY,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    output_json: str = DEFAULT_OUTPUT_JSON,
    scrape_details: bool = True,
) -> list[dict]:
    """
    Full catalog scraping pipeline.

    1. Paginate through catalog listing pages
    2. Collect all product URLs
    3. Fetch each product detail page
    4. Save CSV + JSON

    Returns list of product dicts.
    """
    logger.info(f"Starting SHL catalog scrape (max_pages={max_pages}, delay={delay}s)")

    # ── Step 1: Collect all product URLs from listing pages ───
    all_products_stub = {}  # url → {name, url}
    page = 1

    while page <= max_pages:
        # SHL catalog pagination uses ?start=N or ?page=N
        page_url = CATALOG_URL if page == 1 else f"{CATALOG_URL}?start={(page-1)*12}"
        logger.info(f"Scraping listing page {page}: {page_url}")

        resp = _fetch_with_retry(page_url)
        if not resp:
            logger.warning(f"Failed to fetch page {page}. Stopping pagination.")
            break

        products_on_page = _parse_product_list_page(resp.text)

        if not products_on_page:
            logger.info(f"No products found on page {page}. Reached end of catalog.")
            break

        new_count = 0
        for p in products_on_page:
            if p["url"] not in all_products_stub:
                all_products_stub[p["url"]] = p
                new_count += 1

        logger.info(f"Page {page}: found {len(products_on_page)} products ({new_count} new). Total: {len(all_products_stub)}")

        if new_count == 0:
            logger.info("No new products on this page. Stopping.")
            break

        time.sleep(delay)
        page += 1

    logger.info(f"Listing crawl complete. Found {len(all_products_stub)} unique products.")

    if not all_products_stub:
        logger.error("No products found. Check the catalog URL and selectors.")
        return []

    # ── Step 2: Fetch detail pages ────────────────────────────
    all_products = []
    stubs = list(all_products_stub.values())

    for i, stub in enumerate(stubs, 1):
        logger.info(f"[{i}/{len(stubs)}] Fetching detail: {stub['name'][:60]}")
        if scrape_details:
            product = _parse_product_detail_page(stub["url"], stub["name"], delay)
        else:
            product = {
                "name": stub["name"],
                "url": stub["url"],
                "description": "",
                "test_type": _classify_test_type(stub["name"], ""),
                "content": "",
            }
        all_products.append(product)

    # ── Step 3: Save CSV ───────────────────────────────────────
    logger.info(f"Saving {len(all_products)} products to {output_csv}")
    fieldnames = ["name", "url", "description", "test_type", "content"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in all_products:
            writer.writerow({k: p.get(k, "") for k in fieldnames})

    # ── Step 4: Save JSON ──────────────────────────────────────
    logger.info(f"Saving {len(all_products)} products to {output_json}")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_products, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Scrape complete. {len(all_products)} products saved.")
    return all_products


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape SHL product catalog.")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help="Delay between requests in seconds")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--no-details", action="store_true",
                        help="Skip detail page scraping (faster, less info)")
    args = parser.parse_args()

    scrape_catalog(
        max_pages=args.max_pages,
        delay=args.delay,
        output_csv=args.output_csv,
        output_json=args.output_json,
        scrape_details=not args.no_details,
    )
