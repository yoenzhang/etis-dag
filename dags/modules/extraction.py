import ssl
import urllib.request
import feedparser
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
import os
from typing import List
from newsapi import NewsApiClient
from modules.filter import rule_based_filter
from airflow.providers.postgres.hooks.postgres import PostgresHook
from urllib.parse import urlparse, parse_qs
from modules.utils import get_last_dag_run_date


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
    '("illegal wildlife trade" OR "wildlife trafficking" OR "wildlife smuggling") AND (elephant OR elephants OR tusks OR ivory)',
    '(ivory OR tusks) AND (seizure OR confiscation OR smuggling OR trafficking) AND (illegal OR "black market" OR '
    'poaching)',
    '(elephant OR elephants) AND poaching'
]

GOOGLE_ALERT_RSS: Dict[str, str] = {
    'ivory': 'https://www.google.com/alerts/feeds/09542737377863196634/7152410951420299179',
    'elephant tusks': 'https://www.google.com/alerts/feeds/09542737377863196634/13970440859733604322'
}


def db_insert(articles : List[Article]) -> None:
    """
    Inserts list of articles into the postgres database
    """
    
    # Now, store the articles into the PostgreSQL table
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
    Fetches articles from Google Alert feeds and writes them to a PostgreSQL table.
    """
    # Bypass any proxy issues (e.g., on macOS in Airflow tasks)
    os.environ['NO_PROXY'] = '*'
    context = ssl._create_unverified_context()
    articles: List[Article] = []
    
    print("Started fetching Google Alerts!")

    for token, feed_url in GOOGLE_ALERT_RSS.items():
        try:
            print("About to open URL:", feed_url)
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

            # Loop through each entry in the feed and create an Article instance
            for entry in feed.entries:
                title = entry.get('title', 'No title')
                published = entry.get('published', 'No publication date')
                content_data = entry.get('content', 'No content')

                # extracting the link from google linke 
                # google.com/url=XXX 
                google_link = entry.get('link', 'No link')
                parsed_url = urlparse(google_link)
                query_params = parse_qs(parsed_url.query)
                link = query_params.get("url", [""])[0] # Extract the actual URL

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
            

    # Now, we conduct pre liminary filtering and thenstore the articles into the PostgreSQL table
    filter_articles = rule_based_filter(articles)
    db_insert(filter_articles)



def fetch_news_api() -> None:
    """
    Fetches articles from the NewsAPI for each query in QUERY_LIST and inserts them into PostgreSQL.
    """
    # Bypass any proxy issues if needed
    os.environ['NO_PROXY'] = '*'
    
    # Retrieving last run time
    last_run_time = get_last_dag_run_date()
    
    print(f"the last run time was {last_run_time}")
    
    # Load API key (from environment variable or fallback)
    api_key = os.getenv('NEWS_API_KEY', '93f0ba1f1c4c498aabe43c6bbd531355')
    newsapi = NewsApiClient(api_key=api_key)
    
    articles_to_insert: List[Article] = []

    # Define your date range (adjust as needed)
    today = datetime.today().strftime('%Y-%m-%d')
    from_param = last_run_time
    to_param = today

    for query in QUERY_LIST_all:
        print(f"\nFetching articles for query: {query}\n")
        page = 1

        while True:
            try:
                res = newsapi.get_everything(
                    q=query,
                    from_param=from_param,
                    to=to_param,
                    sort_by='relevancy',
                    page_size=100,  # Maximum allowed by NewsAPI
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

            # Convert each API article to our unified Article class
            for article in articles:
                title = article.get("title", "No Title")
                link = article.get("url", "No URL")
                published_at = article.get("publishedAt", "No Publish Date")
                content = article.get("description", "No Content")

                new_article = Article(
                    title=title,
                    link=link,
                    published=published_at,
                    content=content  # NewsAPI does not always provide a summary
                )
                articles_to_insert.append(new_article)

            # If the number of articles returned is less than the page size, we can assume we've reached the end.
            if len(articles) < 100:
                break

            page += 1

    print("Finished fetching NewsAPI articles. Now inserting them into PostgreSQL...")

    # Now, we conduct pre liminary filtering and thenstore the articles into the PostgreSQL table
    filter_articles = rule_based_filter(articles_to_insert)
    db_insert(filter_articles)
    
    