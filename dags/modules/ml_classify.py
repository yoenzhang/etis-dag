from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import joblib
from airflow.providers.postgres.hooks.postgres import PostgresHook
import os
import logging

def apply_ml_classifier() -> None:
    """
    Applies ML logistic regression model to the current data to apply pre classfication label
    """
    
    logging.info("Current working directory: %s", os.getcwd())
    logging.info("Files in current directory: %s", os.listdir("."))
    logging.info("Files in current directory: %s", os.listdir("./dags"))


    pg_hook =  PostgresHook(postgres_conn_id="postgres_default")
        
    # importing model pipeline
    pipeline = joblib.load("./dags/data/elephant_ivory_model.joblib")
    # Extract articles that have not been classified yet
    select_sql = """
        SELECT id, title, summary
        FROM articles 
        WHERE ml_class IS NULL;
    """
    rows = pg_hook.get_records(select_sql)
    articles_to_classify = [(r[0], r[1], r[2]) for r in rows]
    
    for article_id, title, summary in articles_to_classify:
        combined_text = f"{title}   {summary} "
        
        
        # run prediction -- 1 being relevant and 0 being irrelevant 
        prediction = pipeline.predict([combined_text]) #  array of [1] or [0]
        prob = pipeline.predict_proba([combined_text])  # [[prob_of_0, prob_of_1]]
        
        
        # Convert NumPy dtypes to native Python types
        prediction_value = int(prediction[0])
        prob_of_1 = float(prob[0][1])

        print(
            f"Classifying '{title}' as {prediction_value} with confidence {prob_of_1}"
        )
        
        # Updating database with ML prediction 
        update_sql = """
            UPDATE articles
            SET ml_class = %s
            WHERE id = %s;
        """
        pg_hook.run(update_sql, parameters=(prediction_value, article_id))
    

    
    