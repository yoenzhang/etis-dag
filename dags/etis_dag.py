# etis_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from modules.extraction import fetch_google_alerts, fetch_news_api
from modules.load import scrape_full_articles
from modules.ml_classify import apply_ml_classifier
from modules.nlp import extract_ivory_info_for_articles

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 0,
    'retry_delay': timedelta(minutes=1),
    'execution_timeout': timedelta(minutes=5),
}

with DAG(
    dag_id='etis_dag',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
) as dag:

    fetch_google_alerts_task = PythonOperator(
        task_id='fetch_google_alerts',
        python_callable=fetch_google_alerts
    )

    fetch_news_api_task = PythonOperator(
        task_id='fetch_news_api',
        python_callable=fetch_news_api
    )
    
    scrape_full_articles_task = PythonOperator(
        task_id="scrape_full_articles",
        python_callable=scrape_full_articles
    )

    apply_ml_classifier = PythonOperator(
        task_id="apply_ml_classifier", 
        python_callable=apply_ml_classifier
    )
    
    extract_ivory_info_for_articles = PythonOperator(
        task_id="extract_ivory_info_for_articles", 
        python_callable=extract_ivory_info_for_articles
    )
    

    [fetch_google_alerts_task, fetch_news_api_task] >> apply_ml_classifier >> scrape_full_articles_task >> extract_ivory_info_for_articles
