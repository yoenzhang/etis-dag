import openai
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.hooks.base import BaseHook 
import psycopg2
from psycopg2 import sql
import time
from modules.prompt import build_prompt
import json
import os
import ssl

# from psycopg2 import sql  # Not needed if we're not constructing dynamic SQL
from modules.prompt import build_prompt

# Load your OpenAI key from env or Airflow connections/variables
openai.api_key = os.getenv("OPENAI_API_KEY") or \
                 (BaseHook.get_connection("openai_default").password
                  if BaseHook.get_connection("openai_default") else None)
                 
                 
def extract_ivory_info_for_articles():
    """Fetch unprocessed ivory-related articles and extract seizure info using GPT-3.5-turbo."""
    
    pg_hook = PostgresHook(postgres_conn_id="postgres_default")
    
    os.environ['NO_PROXY'] = '*'
    context = ssl._create_unverified_context()

    select_sql = """
        SELECT a.id, a.title, a.summary, a.link, a.content_full
        FROM articles a
        LEFT JOIN ivory_extractions ie ON ie.article_id = a.id
        WHERE a.ml_class = 1 
          AND ie.article_id IS NULL;    
    """
    rows = pg_hook.get_records(select_sql)
    articles_to_process = [(r[0], r[1], r[2], r[3], r[4]) for r in rows]

    if not articles_to_process:
        print("No new articles to process.")
        return
    
    for (article_id, title, summary, url, content) in articles_to_process:
        # Build the prompt messages for this article
        messages = build_prompt(title or "", summary or "", content or "", url or "")
                
        # Call the OpenAI ChatCompletion API
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0,  # deterministic output
                max_tokens=512  # enough space for output JSON
            )
        except openai.error.RateLimitError as e:
            print(f"Rate limit error for article {article_id}, retrying in 5 seconds...")
            time.sleep(5)
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0,
                    max_tokens=512
                )
            except Exception as e2:
                print(f"Failed to get response for article {article_id} after retry: {e2}")
                continue  # skip this article on persistent failure
        except Exception as e:
            print(f"OpenAI API error for article {article_id}: {e}")
            continue
        
        # Extract the model's response
        result_text = response['choices'][0]['message']['content'].strip()
        
        # Parse the JSON output
        try:
            data = json.loads(result_text)
        except json.JSONDecodeError:
            # Attempt to isolate JSON substring if model returned extra text
            try:
                start_idx = result_text.index('{')
                end_idx = result_text.rfind('}')
                json_str = result_text[start_idx:end_idx+1]
                data = json.loads(json_str)
            except Exception as parse_err:
                print(f"JSON parse error for article {article_id}: {parse_err}")
                continue
        
        # Ensure all expected keys exist, set them to None if missing
        expected_keys = ["report_date", "seizure_date", "location", 
                         "items_seized", "arrest_count", "amount_approximate", "comment", "url"]
        for key in expected_keys:
            if key not in data:
                data[key] = None
        
        # Skip insertion if the model indicates this isn't an ivory-seizure article
        not_relevant = False
        if (
            data.get("seizure_date") is None
            and data.get("location") is None
            and data.get("items_seized") in (None, [], {})
        ):
            not_relevant = True
        if isinstance(data.get("comment"), str) and "not an ivory" in data["comment"].lower():
            not_relevant = True
        
        if not_relevant:
            print(f"Article {article_id} is not an ivory seizure-related article. Skipping insertion.")
            continue
        
        # Insert the extracted data using a plain string for the INSERT query
        try:
            insert_query = """
                INSERT INTO ivory_extractions 
                (article_id, report_date, seizure_date, location, items_seized, 
                 arrest_count, amount_approximate, comment, url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            # Convert items_seized to JSON string for storage
            items_seized_json = json.dumps(data.get("items_seized")) if data.get("items_seized") else None
            
            params = (
                article_id,
                data["report_date"],
                data["seizure_date"],
                data["location"],
                items_seized_json,
                data["arrest_count"],
                data["amount_approximate"],
                data["comment"],
                data["url"]
            )
            
            pg_hook.run(insert_query, parameters=params)
            print(f"Inserted extraction for article {article_id}")
        except Exception as db_err:
            print(f"Database insertion error for article {article_id}: {db_err}")