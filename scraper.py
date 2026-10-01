"""Scrape Codeforces problems and editorials into structured JSON files.

The complete scraping flow stays in one file so that it is easy to study and
explain during a project walkthrough or interview.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import cloudscraper
from bs4 import BeautifulSoup, Tag
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait


BASE_URL = "https://codeforces.com"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "problems"
REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRIES = 3

LOGGER = logging.getLogger("codeforces_scraper")
PROBLEM_LINK_PATTERN = re.compile(
    r"/(?:contest/\d+/problem|problemset/problem/\d+)/[A-Za-z0-9]+"
)


def create_http_client() -> Any:
    """Create one reusable browser-like HTTP client for all requests."""
    client = cloudscraper.create_scraper()
    client.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 Chrome/129.0 Safari/537.36"
            )
        }
    )
    return client


def fetch_html(client: Any, url: str) -> str:
    """Download a page, retrying temporary failures with a short backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.text
        except Exception as error:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Could not download {url}") from error

            wait_seconds = attempt * 2
            LOGGER.warning(
                "Request failed for %s. Retrying in %s seconds...",
                url,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Could not download {url}")


class PageFetcher:
    """Fetch pages using fast HTTP requests or a real Selenium browser."""

    def __init__(self, browser: str = "http") -> None:
        self.browser = browser
        self.http_client = create_http_client()
        self.driver: webdriver.Chrome | None = None

        if browser == "selenium":
            options = Options()
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--window-size=1920,1080")
            options.add_argument(
                "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 Chrome/129.0 Safari/537.36"
            )
            self.driver = webdriver.Chrome(options=options)
            self.driver.set_page_load_timeout(REQUEST_TIMEOUT_SECONDS)

    def get_html(self, url: str) -> str:
        """Return page HTML from the configured fetching method."""
        if self.driver is None:
            return fetch_html(self.http_client, url)

        try:
            self.driver.get(url)

            expected_selector = "body"
            if "/problemset/page/" in url:
                expected_selector = "table.problems"
            elif (
                "/problemset/problem/" in url
                or "/contest/" in url and "/problem/" in url
            ):
                expected_selector = ".problem-statement"
            elif "/blog/entry/" in url:
                expected_selector = ".ttypography"

            WebDriverWait(self.driver, REQUEST_TIMEOUT_SECONDS).until(
                expected_conditions.presence_of_element_located(
                    (By.CSS_SELECTOR, expected_selector)
                )
            )
            return self.driver.page_source
        except (TimeoutException, WebDriverException) as error:
            LOGGER.warning(
                "Selenium could not load expected content from %s; "
                "using the HTTP fallback (%s)",
                url,
                type(error).__name__,
            )
            return fetch_html(self.http_client, url)

    def close(self) -> None:
        """Close the Selenium browser when one was created."""
        if self.driver is not None:
            self.driver.quit()


def normalize_text(element: Tag | None) -> str:
    """Convert an HTML section to readable text while preserving line breaks."""
    if element is None:
        return ""

    section = BeautifulSoup(str(element), "html.parser")
    for line_break in section.find_all("br"):
        line_break.replace_with("\n")

    lines = []
    previous_line_was_empty = False

    for raw_line in section.get_text("\n").splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()

        if line:
            lines.append(line)
            previous_line_was_empty = False
        elif lines and not previous_line_was_empty:
            lines.append("")
            previous_line_was_empty = True

    return "\n".join(lines).strip()


def preformatted_text(element: Tag) -> str:
    """Extract a sample or code block without removing its indentation."""
    block = BeautifulSoup(str(element), "html.parser")
    for line_break in block.find_all("br"):
        line_break.replace_with("\n")
    return block.get_text().strip("\n")


def section_text(section: Tag | None) -> str:
    """Extract a named statement section without its displayed heading."""
    if section is None:
        return ""

    cleaned_section = BeautifulSoup(str(section), "html.parser")
    heading = cleaned_section.select_one(".section-title")
    if heading:
        heading.decompose()

    return normalize_text(cleaned_section)


def extract_description(statement: Tag) -> str:
    """Extract only the main description, excluding other statement sections."""
    cleaned_statement = BeautifulSoup(str(statement), "html.parser")

    for selector in (
        ".header",
        ".input-specification",
        ".output-specification",
        ".sample-tests",
        ".note",
    ):
        for section in cleaned_statement.select(selector):
            section.decompose()

    for script in cleaned_statement.find_all("script"):
        script.decompose()

    return normalize_text(cleaned_statement)


def extract_samples(statement: Tag) -> list[dict[str, str]]:
    """Return sample inputs and outputs as paired structured values."""
    sample_section = statement.select_one(".sample-tests")
    if sample_section is None:
        return []

    inputs = [
        preformatted_text(item.find("pre"))
        for item in sample_section.select(".input")
        if item.find("pre") is not None
    ]
    outputs = [
        preformatted_text(item.find("pre"))
        for item in sample_section.select(".output")
        if item.find("pre") is not None
    ]

    return [
        {"input": sample_input, "output": sample_output}
        for sample_input, sample_output in zip(inputs, outputs)
    ]


def remove_title_prefix(title: str) -> str:
    """Convert a title such as 'A. Watermelon' to 'Watermelon'."""
    return re.sub(r"^[A-Z0-9]+\.\s*", "", title).strip()


def extract_limit(statement: Tag, css_class: str, label: str) -> str:
    """Extract a time or memory limit without its heading text."""
    section = statement.select_one(f".{css_class}")
    if section is None:
        return ""

    value = normalize_text(section)
    return re.sub(rf"^{re.escape(label)}\s*", "", value, flags=re.IGNORECASE)


def find_editorial_url(problem_page: BeautifulSoup) -> str | None:
    """Find the official tutorial link displayed on a problem page."""
    for link in problem_page.find_all("a", href=True):
        link_text = normalize_text(link).lower()
        if "tutorial" in link_text or "editorial" in link_text:
            return urljoin(BASE_URL, link["href"])
    return None


def is_problem_heading(element: Tag) -> bool:
    """Return whether an editorial element starts another problem section."""
    return any(
        PROBLEM_LINK_PATTERN.search(link.get("href", ""))
        for link in element.find_all("a", href=True)
    )


def extract_editorial_section(
    editorial_page: BeautifulSoup,
    problem_reference: dict[str, Any],
) -> tuple[str, list[str]]:
    """Extract only the tutorial section belonging to one problem."""
    content = editorial_page.select_one(".ttypography")
    if content is None:
        raise ValueError("Tutorial content was not found")

    contest_id = problem_reference["contest_id"]
    problem_index = problem_reference["problem_index"]
    expected_links = (
        f"/contest/{contest_id}/problem/{problem_index}",
        f"/problemset/problem/{contest_id}/{problem_index}",
    )

    problem_link = content.find(
        "a",
        href=lambda href: href
        and any(expected_link in href for expected_link in expected_links),
    )
    if problem_link is None:
        raise ValueError("This problem's section was not found in the tutorial")

    section_start: Tag = problem_link
    while section_start.parent is not content:
        if not isinstance(section_start.parent, Tag):
            raise ValueError("Could not identify the tutorial section")
        section_start = section_start.parent

    section_elements: list[Tag] = []
    current_element: Tag | None = section_start

    while current_element is not None:
        if current_element is not section_start and (
            current_element.name == "hr" or is_problem_heading(current_element)
        ):
            break

        section_elements.append(current_element)
        current_element = current_element.find_next_sibling()

    section_html = "".join(str(element) for element in section_elements)
    section = BeautifulSoup(section_html, "html.parser")
    code_blocks = [
        preformatted_text(code_block)
        for code_block in section.find_all("pre")
        if preformatted_text(code_block)
    ]
    content_text = normalize_text(section)

    if not content_text:
        raise ValueError("The extracted tutorial section is empty")

    return content_text, code_blocks


def scrape_editorial(
    fetcher: PageFetcher,
    editorial_url: str,
    problem_reference: dict[str, Any],
    page_cache: dict[str, BeautifulSoup],
) -> dict[str, Any]:
    """Download an editorial once, then reuse it for other contest problems."""
    if editorial_url not in page_cache:
        page_cache[editorial_url] = BeautifulSoup(
            fetcher.get_html(editorial_url),
            "html.parser",
        )

    editorial_page = page_cache[editorial_url]
    content, code_blocks = extract_editorial_section(
        editorial_page,
        problem_reference,
    )

    return {
        "available": True,
        "url": editorial_url,
        "content": content,
        "code_blocks": code_blocks,
    }


def collect_problem_links(
    fetcher: PageFetcher,
    start_page: int,
    page_count: int,
    delay_seconds: float,
) -> list[dict[str, Any]]:
    """Collect unique problem URLs and list-page metadata."""
    problems: dict[str, dict[str, Any]] = {}

    for page_number in range(start_page, start_page + page_count):
        page_url = f"{BASE_URL}/problemset/page/{page_number}"
        LOGGER.info("Reading problem list page %s", page_number)
        soup = BeautifulSoup(fetcher.get_html(page_url), "html.parser")
        problem_rows = soup.select("table.problems tr")
        if not problem_rows:
            raise ValueError(
                f"No problems were found on page {page_number}. "
                "Codeforces may have returned a verification page."
            )

        for row in problem_rows:
            link = row.select_one('a[href*="/problemset/problem/"]')
            if link is None:
                continue

            problem_url = urljoin(BASE_URL, link.get("href", ""))
            url_parts = problem_url.rstrip("/").split("/")
            if len(url_parts) < 2:
                continue

            contest_id, problem_index = url_parts[-2], url_parts[-1]
            problem_id = f"{contest_id}{problem_index}"
            tags = [
                normalize_text(tag)
                for tag in row.select('a.notice[href*="/problemset?tags="]')
            ]

            rating_element = row.select_one("span.ProblemRating")
            rating_text = normalize_text(rating_element)
            rating = int(rating_text) if rating_text.isdigit() else None

            problems[problem_id] = {
                "problem_id": problem_id,
                "contest_id": contest_id,
                "problem_index": problem_index,
                "url": problem_url,
                "tags": tags,
                "rating": rating,
            }

        if page_number < start_page + page_count - 1:
            time.sleep(delay_seconds)

    return list(problems.values())


def scrape_problem(
    fetcher: PageFetcher,
    problem_reference: dict[str, Any],
) -> dict[str, Any]:
    """Download and parse one complete Codeforces problem statement."""
    soup = BeautifulSoup(
        fetcher.get_html(problem_reference["url"]),
        "html.parser",
    )
    statement = soup.select_one(".problem-statement")
    if statement is None:
        raise ValueError("Problem statement was not found on the page")

    raw_title = normalize_text(statement.select_one(".title"))
    editorial_url = find_editorial_url(soup)

    problem = {
        **problem_reference,
        "title": remove_title_prefix(raw_title),
        "time_limit": extract_limit(statement, "time-limit", "time limit per test"),
        "memory_limit": extract_limit(
            statement,
            "memory-limit",
            "memory limit per test",
        ),
        "description": extract_description(statement),
        "input": section_text(statement.select_one(".input-specification")),
        "output": section_text(statement.select_one(".output-specification")),
        "examples": extract_samples(statement),
        "note": section_text(statement.select_one(".note")),
        "editorial": {
            "available": False,
            "url": editorial_url,
            "content": "",
            "code_blocks": [],
        },
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }

    if not problem["title"] or not problem["description"]:
        raise ValueError("The parsed problem is missing its title or description")

    return problem


def save_problem(problem: dict[str, Any], output_directory: Path) -> Path:
    """Save one problem as readable UTF-8 JSON."""
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{problem['problem_id']}.json"

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(problem, output_file, ensure_ascii=False, indent=2)

    return output_path


def parse_arguments() -> argparse.Namespace:
    """Read simple command-line options for the scraper."""
    parser = argparse.ArgumentParser(description="Scrape Codeforces problems")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--max-problems", type=int, default=5)
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--browser",
        choices=("http", "selenium"),
        default="http",
        help="Use fast HTTP requests or a headless Selenium Chrome browser",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download problems again even when their JSON files already exist",
    )
    return parser.parse_args()


def run_scraper(arguments: argparse.Namespace, fetcher: PageFetcher) -> None:
    """Run the problem-list, scraping and saving stages."""
    discovered_problems = collect_problem_links(
        fetcher,
        start_page=arguments.start_page,
        page_count=arguments.pages,
        delay_seconds=arguments.delay,
    )

    problem_references = []
    for problem_reference in discovered_problems:
        output_path = arguments.output_dir / f"{problem_reference['problem_id']}.json"
        if output_path.exists() and not arguments.overwrite:
            LOGGER.info("Skipping existing problem %s", problem_reference["problem_id"])
            continue

        problem_references.append(problem_reference)
        if len(problem_references) == arguments.max_problems:
            break

    LOGGER.info("Found %s problems to scrape", len(problem_references))
    saved_count = 0
    editorial_page_cache: dict[str, BeautifulSoup] = {}

    for position, problem_reference in enumerate(problem_references, start=1):
        problem_id = problem_reference["problem_id"]

        try:
            problem = scrape_problem(fetcher, problem_reference)

            editorial_url = problem["editorial"]["url"]
            if editorial_url:
                try:
                    if editorial_url not in editorial_page_cache:
                        time.sleep(arguments.delay)
                    problem["editorial"] = scrape_editorial(
                        fetcher,
                        editorial_url,
                        problem_reference,
                        editorial_page_cache,
                    )
                    LOGGER.info("Found editorial for %s", problem_id)
                except Exception as error:
                    LOGGER.warning("Could not parse editorial for %s: %s", problem_id, error)

            output_path = save_problem(problem, arguments.output_dir)
            saved_count += 1
            LOGGER.info("Saved %s to %s", problem_id, output_path)
        except Exception as error:
            LOGGER.error("Could not scrape %s: %s", problem_id, error)

        if position < len(problem_references):
            time.sleep(arguments.delay)

    LOGGER.info("Finished: saved %s of %s problems", saved_count, len(problem_references))


def main() -> None:
    """Read options, create the selected fetcher and run the scraper."""
    arguments = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if arguments.pages < 1 or arguments.max_problems < 1:
        raise SystemExit("--pages and --max-problems must be at least 1")

    fetcher = PageFetcher(browser=arguments.browser)
    try:
        run_scraper(arguments, fetcher)
    finally:
        fetcher.close()


if __name__ == "__main__":
    main()
