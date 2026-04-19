from __future__ import annotations

import os
from pathlib import Path
import requests
import xml.etree.ElementTree as ET

os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
)

SITEMAP_URL = "https://www.tbank.ru/sitemap/static/help.xml"
EXCLUDED_TAGS = (
    "/eng/",
    "/uz/",
    "/kz/",
    "/kg/",
    "/am/",
    "/tj/",
    "/bank/help/apps/",
)
ALLOWED_ROOTS = (
    "https://www.tbank.ru/bank/help/",
    "https://www.tbank.ru/business/help/",
    "https://www.tbank.ru/invest/help/",
    "https://www.tbank.ru/insurance/help/",
    "https://www.tbank.ru/travel/help/",
    "https://www.tbank.ru/mobile-operator/help/",
)


def fetch_urls(sitemap_url: str = SITEMAP_URL) -> list[str]:
    response = requests.get(
        sitemap_url,
        timeout=30,
        headers={"User-Agent": os.environ["USER_AGENT"]},
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)

    urls: list[str] = []
    seen: set[str] = set()
    for element in root.iter():
        if not element.tag.endswith("loc") or not element.text:
            continue

        url = element.text.strip()
        if any(tag in url for tag in EXCLUDED_TAGS):
            continue
        if not any(url.startswith(prefix) for prefix in ALLOWED_ROOTS):
            continue
        if url in seen:
            continue

        seen.add(url)
        urls.append(url)

    return urls


def main() -> None:
    urls = fetch_urls()
    output_path = Path(__file__).with_name("urls.txt")
    output_path.write_text("\n".join(urls), encoding="utf-8")
    print(f"Saved {len(urls)} urls to {output_path}")


if __name__ == "__main__":
    main()
