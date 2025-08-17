# etis_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# Acquisition functions
from modules.extraction import (
    fetch_google_alerts,
    fetch_news_api,
    fetch_nytimes_rss,
    fetch_currents_news,
    fetch_guardian_news,
    fetch_mediastack_news
)
# Classification
from modules.ml_classify import apply_ml_classifier
# Scraping full articles
from modules.load import scrape_full_articles
# LLM extraction
from modules.nlp import extract_ivory_info_for_articles
# Negative example collection
from modules.negative_collector import collect_all_negative_examples

default_args = {
    "owner": "etis_team",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": True,
    "email": ["alerts@etis.org"],
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(minutes=5),
}

with DAG(
    dag_id="etis_dag",
    default_args=default_args,
    description="ETIS ingestion: acquisition → ML → scrape → LLM extraction → negative collection",
    schedule_interval="@daily",
    catchup=False,
) as dag:

    # ────────────── Acquisition ──────────────
    fetch_google_alerts_task = PythonOperator(
        task_id="fetch_google_alerts",
        python_callable=fetch_google_alerts,
    )

    fetch_news_api_task = PythonOperator(
        task_id="fetch_news_api",
        python_callable=fetch_news_api,
    )

    fetch_nytimes_rss_task = PythonOperator(
        task_id="fetch_nytimes_rss",
        python_callable=fetch_nytimes_rss,
    )

    fetch_currents_news_task = PythonOperator(
        task_id="fetch_currents_news",
        python_callable=fetch_currents_news,
    )

    fetch_guardian_news_task = PythonOperator(
        task_id="fetch_guardian_news",
        python_callable=fetch_guardian_news,
    )

    fetch_mediastack_task = PythonOperator(
        task_id="fetch_mediastack_news",
        python_callable=fetch_mediastack_news,
    )

    # ──────────── ML Classification ────────────
    apply_ml_classifier_task = PythonOperator(
        task_id="apply_ml_classifier",
        python_callable=apply_ml_classifier,
    )

    # ──────────── Scrape Full Text ────────────
    scrape_full_articles_task = PythonOperator(
        task_id="scrape_full_articles",
        python_callable=scrape_full_articles,
    )

    # ──────────── LLM Extraction ────────────
    extract_ivory_info_task = PythonOperator(
        task_id="extract_ivory_info_for_articles",
        python_callable=extract_ivory_info_for_articles,
    )

    # ──────────── Negative Collection ────────────
    collect_negative_examples_task = PythonOperator(
        task_id="collect_negative_examples",
        python_callable=collect_all_negative_examples,
    )

    # ─────────────── Dependencies ───────────────
    [
        fetch_google_alerts_task,
        fetch_news_api_task,
        fetch_nytimes_rss_task,
        fetch_currents_news_task,
        fetch_guardian_news_task,
        fetch_mediastack_task,
    ] >> apply_ml_classifier_task \
      >> scrape_full_articles_task \
      >> extract_ivory_info_task \
      >> collect_negative_examples_task
