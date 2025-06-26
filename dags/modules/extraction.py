import ssl
import urllib.request
import feedparser
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
import os
import requests
from typing import List
from newsapi import NewsApiClient
from modules.filter import rule_based_filter
from airflow.providers.postgres.hooks.postgres import PostgresHook
from urllib.parse import urlparse, parse_qs
from modules.utils import get_last_dag_run_date
import time


# ------- Models ---------- #
@dataclass
class Article:
    title: str
    link: str
    published: str
    content: str

    def __str__(self):
        return (
            f"Title: {self.title}\n"
            f"Link: {self.link}\n"
            f"Published: {self.published}\n"
            f"Content: {self.content}\n"
            + "-" * 40
        )


# -------- Extraction Constants ------------ #

QUERY_LIST = [
    '("elephant" OR "elephants") AND (ivory OR tusks) AND ("illegal trade" OR trafficking OR smuggling OR seizure OR confiscation)',
]

QUERY_LIST_all = [
    '("elephant" OR "elephants") AND (ivory OR tusks) AND ("illegal trade" OR trafficking OR smuggling OR seizure OR confiscation)',
]

GOOGLE_ALERT_RSS: Dict[str, str] = {
    'ivory': 'https://www.google.com/alerts/feeds/09542737377863196634/7152410951420299179',
    'elephant tusks': 'https://www.google.com/alerts/feeds/09542737377863196634/13970440859733604322'
}

# ---- Additional News API Keys & RSS Feeds ----
CURRENTS_API_KEY = "ZWdnRwV7_YtoJm_LwMt5eVkFayV2CDxcKQwDRKwkNQ-gpxra"
GUARDIAN_API_KEY = "eeb5b860-3b48-46bb-86c3-c34dd946c9c3"
MEDIASTACK_API_KEY  = "51aefb0ae9e5947d6b0e456265802c08"
NYTIMES_SECTIONS = [
    "Africa",
    "Americas",
    "AsiaPacific",
    "EnergyEnvironment",
    "Europe",
    "MiddleEast",
    "World",
    "US",
    "Science",
    "Climate",
    "Business",
    "Travel",
]
NYTIMES_RSS_FEEDS = [
    f"https://rss.nytimes.com/services/xml/rss/nyt/{section}.xml"
    for section in NYTIMES_SECTIONS
]


def db_insert(articles: List[Article]) -> None:
    """
    Inserts list of articles into the postgres database
    """
    pg_hook = PostgresHook(postgres_conn_id="postgres_default")
    insert_sql = """
        INSERT INTO articles (title, link, published, summary)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (link) DO NOTHING;
    """

    for article in articles:
        try:
            pg_hook.run(insert_sql, parameters=(article.title, article.link, article.published, article.content))
            print(f"Inserted article: {article.title}")
        except Exception as e:
            print(f"Error inserting article '{article.title}': {e}")

    print("Finished storing articles into PostgreSQL.")


def fetch_google_alerts() -> None:
    """
    Fetches from predefined Google Alert RSS feeds and inserts them.
    """
    print("Started fetching Google Alerts feeds…")
    articles: List[Article] = []
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    for token, feed_url in GOOGLE_ALERT_RSS.items():
        try:
            req = urllib.request.Request(
                feed_url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
                    )
                }
            )
            with urllib.request.urlopen(req, context=context, timeout=30) as response:
                data = response.read()

            print("Data read, now parsing…")
            feed = feedparser.parse(data)
            print("Parsing done…")

            if feed.bozo:
                print(f"Error parsing feed for '{token}': {feed.bozo_exception}")
                continue

            for entry in feed.entries:
                title = entry.get('title', 'No title')
                published = entry.get('published', 'No publication date')
                content_data = entry.get('content', 'No content')

                google_link = entry.get('link', 'No link')
                parsed_url = urlparse(google_link)
                qs = parse_qs(parsed_url.query)
                link = qs.get('url', [google_link])[0]

                if isinstance(content_data, list):
                    content = " ".join(
                        item.get("value", "") if isinstance(item, dict) else str(item)
                        for item in content_data
                    )
                else:
                    content = content_data

                article = Article(title=title, link=link, published=published, content=content)
                articles.append(article)

        except urllib.error.URLError as e:
            print(f"Failed to fetch feed for '{token}': {e}")
        time.sleep(1.1)

    filter_articles = rule_based_filter(articles)
    db_insert(filter_articles)


def fetch_news_api() -> None:
    """
    Fetches articles from the NewsAPI for each query in QUERY_LIST and inserts them into PostgreSQL.
    """
    os.environ['NO_PROXY'] = '*'
    last_run_time = get_last_dag_run_date()
    print(f"the last run time was {last_run_time}")

    newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY", ""))

    articles_to_insert: List[Article] = []

    for query in QUERY_LIST:
        page = 1
        while True:
            try:
                res = newsapi.get_everything(
                    q=query,
                    from_param=last_run_time,
                    to=datetime.utcnow().isoformat(),
                    sort_by='relevancy',
                    page_size=100,
                    page=page
                )
            except Exception as e:
                print(f"Error fetching from NewsAPI: {e}")
                break

            if res.get("status") != "ok":
                print(f"Error: status is '{res.get('status')}' not 'ok'")
                break

            articles = res.get("articles", [])
            total_count = res.get("totalResults", 0)

            print(f"  Page {page}, articles on this page: {len(articles)}, total estimated: {total_count}")

            if not articles:
                break

            for article in articles:
                title = article.get("title", "No Title")
                link = article.get("url", "No URL")
                published_at = article.get("publishedAt", "No Publish Date")
                content = article.get("description", "")

                new_article = Article(
                    title=title,
                    link=link,
                    published=published_at,
                    content=content
                )
                articles_to_insert.append(new_article)

            if len(articles) < 100:
                break

            page += 1
            time.sleep(1.1)

    print("Finished fetching NewsAPI articles. Now inserting them into PostgreSQL...")
    filter_articles = rule_based_filter(articles_to_insert)
    db_insert(filter_articles)


def fetch_nytimes_rss() -> None:
    """
    Fetches from NYTimes RSS feeds for the specified sections
    and inserts them.
    """
    print("Started fetching NYTimes RSS feeds…")
    articles: List[Article] = []

    for feed_url in NYTIMES_RSS_FEEDS:
        print(f"  → loading section feed: {feed_url.split('/')[-1].replace('.xml','')}")
        feed = feedparser.parse(feed_url)
        if feed.bozo:
            print(f"NYTimes feed parse error: {feed.bozo_exception}")
            continue

        for entry in feed.entries:
            title     = entry.get("title", "No Title")
            link      = entry.get("link", "No URL")
            published = entry.get("published", datetime.utcnow().isoformat())
            content   = entry.get("summary", "")

            articles.append(Article(
                title=title,
                link=link,
                published=published,
                content=content
            ))

        time.sleep(1.1)

    filtered = rule_based_filter(articles)
    db_insert(filtered)
    print("Finished NYTimes RSS ingestion.")

def fetch_currents_news() -> None:
    """
    Calls the Currents API search endpoint for targeted illegal ivory/elephant trade stories
    and inserts results.
    """
    print("Started fetching Currents API articles on illegal ivory/elephant trade…")
    endpoint = "https://api.currentsapi.services/v1/search"
    last_run = get_last_dag_run_date()
    
    query = (
        '"illegal ivory" OR '
        '"illegal ivory extraction" OR '
        '"ivory trafficking" OR '
        '"illegal elephant trade" OR '
        '"elephant trafficking" OR '
        '"elephant smuggling"'
    )
    
    params = {
        "apiKey": CURRENTS_API_KEY,
        "language": "en",
        "keywords": query,
        "start_date": last_run,
        "end_date": datetime.utcnow().isoformat(),
        "page_size": 50,
        "page_number": 1
    }
    
    all_articles: List[Article] = []
    while True:
        try:
            resp = requests.get(endpoint, params=params, timeout=10).json()
        except Exception as e:
            print(f"Error calling Currents API (page {params['page_number']}): {e}")
            break
        
        if resp.get("status") != "ok":
            print(f"Currents API returned status {resp.get('status')}")
            break
        
        news_items = resp.get("news", [])
        if not news_items:
            break
        
        # collect into Article objects
        for item in news_items:
            all_articles.append(Article(
                title=item.get("title", ""),
                link=item.get("url", ""),
                published=item.get("published", ""),
                content=item.get("description", "")
            ))
        
        print(f"Fetched Currents page {params['page_number']}, items: {len(news_items)}")
        
        if len(news_items) < params["page_size"]:
            break
        
        params["page_number"] += 1
        time.sleep(1.1)
    
    filtered = rule_based_filter(all_articles)
    db_insert(filtered)
    print("Finished Currents API ingestion.")


def fetch_guardian_news() -> None:
    """
    Calls The Guardian Content API search endpoint for targeted illegal ivory/elephant trade stories,
    using the same params you tested in the explorer UI (q, api-key, order-by, page-size, etc.)
    """
    print("Started fetching The Guardian articles on illegal ivory/elephant trade…")
    endpoint = "https://content.guardianapis.com/search"
    last_run = get_last_dag_run_date()

    query = (
        '"illegal ivory" OR '
        '"illegal ivory extraction" OR '
        '"ivory trafficking" OR '
        '"illegal elephant trade" OR '
        '"elephant trafficking" OR '
        '"elephant smuggling"'
    )

    params = {
        "api-key": GUARDIAN_API_KEY,
        "q": query,
        "from-date": last_run,
        "page-size": 50,
        "page": 1,
        "order-by": "relevance",
        "show-fields": "trailText"
    }

    resp0 = requests.get(endpoint, params=params, timeout=10).json().get("response", {})
    if resp0.get("status") != "ok":
        print(f"Initial Guardian API call failed: {resp0.get('status')}")
        return

    total_pages = resp0.get("pages", 1)
    print(f"Guardian API: {resp0.get('total')} articles across {total_pages} pages")

    articles: List[Article] = []

    for pg in range(1, total_pages + 1):
        params["page"] = pg
        try:
            body = requests.get(endpoint, params=params, timeout=10).json().get("response", {})
        except Exception as e:
            print(f"Error on page {pg}: {e}")
            break

        if body.get("status") != "ok":
            print(f"Guardian API page {pg} bad status: {body.get('status')}")
            break

        results = body.get("results", [])
        if not results:
            break

        for item in results:
            fields = item.get("fields", {})
            articles.append(Article(
                title=item.get("webTitle", ""),
                link=item.get("webUrl", ""),
                published=item.get("webPublicationDate", ""),
                content=fields.get("trailText", "")
            ))

        print(f"Fetched page {pg}/{total_pages}, items: {len(results)}")
        time.sleep(1.1)

    filtered = rule_based_filter(articles)
    db_insert(filtered)
    print("Finished Guardian API ingestion.")

def fetch_mediastack_news() -> None:
    """
    Calls the Mediastack API for targeted illegal ivory/elephant trade stories
    and inserts results into PostgreSQL.
    """
    print("Started fetching Mediastack API articles on illegal ivory/elephant trade…")
    endpoint = "http://api.mediastack.com/v1/news"
    last_run = get_last_dag_run_date()

    # comma-separated keywords to include; excludes none
    keywords = ",".join([
        "illegal ivory",
        "illegal ivory extraction",
        "ivory trafficking",
        "illegal elephant trade",
        "elephant trafficking",
        "elephant smuggling",
    ])

    limit  = 100
    offset = 0
    all_articles: List[Article] = []

    while True:
        params = {
            "access_key": MEDIASTACK_API_KEY,
            "keywords":   keywords,
            "languages":  "en",
            "limit":      limit,
            "offset":     offset,
            "sort":       "published_desc"
        }

        try:
            resp = requests.get(endpoint, params=params, timeout=10).json()
        except Exception as e:
            print(f"Error calling Mediastack API (offset {offset}): {e}")
            break

        pagination = resp.get("pagination", {})
        data       = resp.get("data", [])
        if not data:
            break

        for item in data:
            # optionally you could parse item["published_at"] and skip anything before last_run
            all_articles.append(Article(
                title     = item.get("title", ""),
                link      = item.get("url", ""),
                published = item.get("published_at", ""),
                content   = item.get("description", "")
            ))

        print(f"Fetched Mediastack offset {offset}, items: {len(data)}")
        total = pagination.get("total", 0)
        offset += limit
        if offset >= total:
            break

        time.sleep(1.2)  # throttle

    filtered = rule_based_filter(all_articles)
    db_insert(filtered)
    print("Finished Mediastack API ingestion.")