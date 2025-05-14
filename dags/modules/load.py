# modules/scrape.py
import urllib.parse
from bs4 import BeautifulSoup
from airflow.providers.postgres.hooks.postgres import PostgresHook
from modules.utils import fetch_url_body

def resolve_google_redirect(link: str) -> str:
    """
    If the link is a Google redirect, parse out the real 'url' parameter.
    Otherwise, return the original link.
    """
    parsed = urllib.parse.urlparse(link)
    if "google.com" in parsed.netloc and "/url" in parsed.path:
        qs = urllib.parse.parse_qs(parsed.query)
        if "url" in qs and len(qs["url"]) > 0:
            return qs["url"][0]
    return link

def scrape_full_articles() -> None:
    """
    1. Pull all articles from DB that have content_full IS NULL.
    2. For each article, fetch the HTML from its link, parse out the main text.
    3. Update the DB record with the scraped full text.
    """
    pg_hook = PostgresHook(postgres_conn_id="postgres_default")
    
    select_sql = """
        SELECT id, link
        FROM articles
        WHERE ml_class=1 and content_full is null
    """
    rows = pg_hook.get_records(select_sql)
    articles_to_scrape = [(r[0], r[1]) for r in rows]
    print(f"Found {len(articles_to_scrape)} article(s) to scrape.")
    
    for article_id, link in articles_to_scrape:
        print(f"Scraping article ID={article_id} Link={link}")
        try:
            real_link = resolve_google_redirect(link)
            print(f"  -> Real link: {real_link}")
            
            data = fetch_url_body(real_link)
            if data:
                soup = BeautifulSoup(data, "html.parser")
                paragraphs = soup.find_all("p")
                full_text = "\n".join(p.get_text() for p in paragraphs)
                
                update_sql = """
                    UPDATE articles
                    SET content_full = %s
                    WHERE id = %s
                """
                pg_hook.run(update_sql, parameters=(full_text, article_id))
                print(f"Updated article ID={article_id} with full text.")
            else:
                print(f"Failed to fetch or parse article ID={article_id}")
        except Exception as e:
            print(f"Error scraping article {article_id}: {e}")
    
    print("Scraping job complete.")
