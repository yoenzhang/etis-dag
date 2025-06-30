import os
import logging
import joblib
from airflow.providers.postgres.hooks.postgres import PostgresHook
from modules.negative_collector import negative_collector

def apply_ml_classifier() -> None:
    """
    Applies ML classifier to unclassified articles using the new pipeline approach.
    Uses separate vectorizer, classifier, and threshold components.
    """
    logging.info("Starting ML classification with new pipeline...")
    
    try:
        vectorizer = joblib.load("./dags/data/ivory_vectorizer.joblib")
        classifier = joblib.load("./dags/data/ivory_classifier.joblib")
        threshold = joblib.load("./dags/data/ivory_threshold.joblib")
        logging.info("Loaded new pipeline components successfully")
        
    except Exception as e:
        logging.error(f"Failed to load ML pipeline components: {e}")
        raise
    
    pg_hook = PostgresHook(postgres_conn_id="postgres_default")
    
    # Get unclassified articles
    select_sql = """
        SELECT id, title, summary
        FROM articles 
        WHERE ml_class IS NULL;
    """
    rows = pg_hook.get_records(select_sql)
    articles_to_classify = [(r[0], r[1], r[2]) for r in rows]
    
    logging.info(f"Found {len(articles_to_classify)} articles to classify")
    
    for article_id, title, summary in articles_to_classify:
        try:
            # Prepare text (same as training)
            combined_text = f"{title or ''}   {summary or ''} "
            
            # Transform text using vectorizer
            features = vectorizer.transform([combined_text])
            
            # Get probability predictions
            prob = classifier.predict_proba(features)
            prob_of_1 = float(prob[0][1])
            
            # Apply threshold to get final prediction
            prediction_value = 1 if prob_of_1 >= threshold else 0
            
            # Update database
            update_sql = """
                UPDATE articles
                SET ml_class = %s
                WHERE id = %s;
            """
            pg_hook.run(update_sql, parameters=(prediction_value, article_id))
            
            # Collect negative examples for articles classified as 0
            if prediction_value == 0:
                # Get the full article data for negative collection
                article_sql = """
                    SELECT title, summary, link, published
                    FROM articles
                    WHERE id = %s;
                """
                article_row = pg_hook.get_records(article_sql, parameters=(article_id,))
                if article_row:
                    article_data = article_row[0]
                    negative_collector.add_ml_negative_article(
                        title=article_data[0] or '',
                        link=article_data[2] or '',
                        published=article_data[3] or '',
                        summary=article_data[1] or '',
                        probability=prob_of_1,
                        threshold=threshold
                    )
            
            logging.info(f"Classified article {article_id}: prob={prob_of_1:.3f} (threshold={threshold:.3f}) → class={prediction_value}")
            
        except Exception as e:
            logging.error(f"Error classifying article {article_id}: {e}")
            continue
    
    logging.info("ML classification completed successfully")
