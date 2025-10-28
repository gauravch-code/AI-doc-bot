import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import re

corpus_data = []
scraped_urls = set()

LIBS_CONFIG = {
    "pandas": {
        "base_url": "https://pandas.pydata.org/docs/",
        "seed_urls": [
            "https://pandas.pydata.org/docs/user_guide/index.html",
            "https://pandas.pydata.org/docs/reference/index.html"
        ],
        "content_selector": ('main', {'class': 'bd-main'}), # tag, {attributes}
        "link_selector": ('main', {'class': 'bd-main'}),    # Area to find links in
        "source_logic": lambda url: "api_spec" if "/reference/" in url else ("tutorial" if "/user_guide/" in url else "other")
    },
    "numpy": {
        "base_url": "https://numpy.org/doc/stable/",
        "seed_urls": [
            "https://numpy.org/doc/stable/user/index.html", # User Guide
            "https://numpy.org/doc/stable/reference/index.html" # API Reference
        ],
        "content_selector": ('div', {'role': 'main'}), # Found via inspection
        "link_selector": ('div', {'role': 'main'}),
        "source_logic": lambda url: "api_spec" if "/reference/" in url else ("tutorial" if "/user/" in url else "other")
    },
    "scikit-learn": {
        "base_url": "https://scikit-learn.org/stable/",
        "seed_urls": [
            "https://scikit-learn.org/stable/user_guide.html",
            "https://scikit-learn.org/stable/modules/classes.html" # API
        ],
        "content_selector": ('div', {'role': 'main'}), # Found via inspection
        "link_selector": ('div', {'role': 'main'}),
        "source_logic": lambda url: "api_spec" if "/modules/classes.html" in url or "/modules/generated/" in url else ("tutorial" if "/user_guide.html" in url or "/tutorial/" in url else "other")
    }
}

# --- Helper Functions ---

def get_soup(url):
    """Fetches and parses a URL, returns a BeautifulSoup object."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        # Use lxml for potentially faster parsing if installed (pip install lxml)
        return BeautifulSoup(response.text, 'lxml' if 'lxml' in globals() else 'html.parser')
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def find_doc_links(index_url, base_url, link_selector_tag, link_selector_attrs):
    """Finds documentation links on an index page for a specific library."""
    links_to_scrape = []
    soup = get_soup(index_url)
    if not soup: return links_to_scrape

    link_area = soup.find(link_selector_tag, link_selector_attrs)
    if not link_area:
        print(f"Warning: Could not find link area ({link_selector_tag}, {link_selector_attrs}) for {index_url}")
        return links_to_scrape

    for a_tag in link_area.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(index_url, href).split('#')[0] # Use index_url as base

        # Stay within the library's base URL and avoid already scraped URLs
        if full_url.startswith(base_url) and full_url not in scraped_urls:
             # Basic check to avoid non-HTML files
            parsed_url = urlparse(full_url)
            if '.' not in os.path.basename(parsed_url.path) or \
               os.path.basename(parsed_url.path).endswith(('.html', '.htm')):
                links_to_scrape.append(full_url)
                scraped_urls.add(full_url)

    return links_to_scrape

def scrape_doc_page(url, library_name, content_selector_tag, content_selector_attrs, source_logic_func):
    """Scrapes a single doc page, labels it, and adds to corpus."""
    soup = get_soup(url)
    if not soup: return

    content_area = soup.find(content_selector_tag, content_selector_attrs)
    if not content_area: return

    # Clean text: remove script/style tags, excessive whitespace
    for element in content_area(["script", "style"]):
        element.decompose()
    text = re.sub(r'\s+', ' ', content_area.get_text(separator=' ', strip=True))

    source_label = source_logic_func(url)

    if text and source_label != "other":
        corpus_data.append({
            "library": library_name, # <-- NEW METADATA
            "source": source_label,
            "text": text,
            "url": url
        })
        print(f"[{library_name}] Scraped and labeled: {url} as {source_label}")
    elif source_label == "other":
        print(f"[{library_name}] Skipping (labeled 'other'): {url}")

def scrape_stackoverflow(tag, library_name, limit=50):
    """Scrapes Stack Overflow for questions tagged with 'tag'."""
    print(f"\nScraping Stack Overflow for tag '{tag}'...")
    so_url = f"https://stackoverflow.com/questions/tagged/{tag}?sort=newest&pageSize={limit}"
    soup = get_soup(so_url)
    if not soup: return

    questions = soup.find_all('div', class_='s-post-summary')
    count = 0
    for q in questions:
        title_tag = q.find('h3', class_='s-post-summary--content-title')
        title = title_tag.get_text(strip=True) if title_tag else ""

        excerpt_tag = q.find('div', class_='s-post-summary--content-excerpt')
        excerpt = excerpt_tag.get_text(strip=True) if excerpt_tag else ""

        url_tag = title_tag.find('a', href=True) if title_tag else None
        url = urljoin("https://stackoverflow.com", url_tag['href']) if url_tag else "N/A"

        full_text = f"{title} {excerpt}"

        if full_text and url != "N/A" and url not in scraped_urls:
            corpus_data.append({
                "library": library_name, # <-- NEW METADATA
                "source": "community",
                "text": full_text,
                "url": url
            })
            scraped_urls.add(url)
            print(f"[StackOverflow/{tag}] Scraped: {title}")
            count += 1
    print(f"Finished scraping {count} questions for tag '{tag}'.")


# --- Main Execution ---
import os # Need this for the path check later

def main():
    print("Starting scraper for Pandas, NumPy, Scikit-learn...")
    MAX_PAGES_PER_LIB = 50 # Limit pages per library for faster runs

    # --- Scrape Official Docs ---
    for lib, config in LIBS_CONFIG.items():
        print(f"\n--- Processing {lib.upper()} ---")
        links_to_scrape = []
        for url in config['seed_urls']:
            print(f"Finding links on index page: {url}")
            links_to_scrape.extend(find_doc_links(
                url,
                config['base_url'],
                config['link_selector'][0],
                config['link_selector'][1]
            ))

        print(f"Found {len(links_to_scrape)} unique documentation links for {lib}.")
        links_to_process = list(set(links_to_scrape))[:MAX_PAGES_PER_LIB] # Use set to ensure unique, then slice
        print(f"--- Scraping first {len(links_to_process)} {lib} doc pages ---")

        for i, url in enumerate(links_to_process):
            scrape_doc_page(
                url,
                lib,
                config['content_selector'][0],
                config['content_selector'][1],
                config['source_logic']
            )
            time.sleep(0.1) # Be polite

    # --- Scrape Stack Overflow ---
    scrape_stackoverflow("pandas", "pandas")
    scrape_stackoverflow("numpy", "numpy")
    scrape_stackoverflow("scikit-learn", "scikit-learn")

    # --- Save to File ---
    print(f"\nScraping complete. Total documents collected: {len(corpus_data)}")

    # Filter out any entries that might have been skipped or failed (no text)
    final_corpus = [doc for doc in corpus_data if doc.get("text")]
    print(f"Filtered corpus size (valid entries): {len(final_corpus)}")

    with open('corpus.json', 'w', encoding='utf-8') as f:
        json.dump(final_corpus, f, indent=2, ensure_ascii=False)

    print("Corpus saved to corpus.json!")

if __name__ == "__main__":
    main()