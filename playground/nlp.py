# modules/nlp.py
import os
import openai
import json
from typing import Dict, Optional

# Load your OpenAI key from env or Airflow connections/variables

def gpt_extract_ivory_info(article_text: str) -> Optional[Dict]:
    """
    1. Determines if article is relevant to elephant ivory seizures.
    2. If relevant, extracts structured fields.
    3. Regardless of relevance, returns a short 2-3 line summary.
    """

    # Build the user prompt:
    prompt = f"""
      You are a helpful assistant for extracting structured data and a brief summary from text.
      You will receive an article about possible elephant ivory seizures.

      Steps:
      1) Summarize the article in 2-3 lines (max ~50 words).
      2) Decide if it is relevant to an elephant ivory seizure. 
        - If it is irrelevant, return JSON with:
            {{
              "is_relevant": false,
              "brief_summary": "...(2-3 lines)..."
            }}
      3) If relevant, return JSON with these keys (no extras!):
        - is_relevant (true),
        - brief_summary (2-3 lines),
        - report_date,
        - seizure_day,
        - seizure_month,
        - seizure_year,
        - discovered_place,
        - discovered_city,
        - species,
        - raw_pieces,
        - raw_weight,
        - raw_present_amount_unknown,
        - raw_uncertainty,
        - worked_pieces,*
        - worked_weight,
        - worked_present_amount_unknown,
        - worked_uncertainty,
        - ivory_comment,
        - hide_pieces,
        - hide_weight

      Output valid JSON only, with no additional text or commentary.

      Article text:
      \"\"\"{article_text}\"\"\"
    """
    os.environ['NO_PROXY'] = '*'

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4 gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a JSON extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10000,
            request_timeout=60, 
        )

        # The assistant's reply
        content = response['choices'][0]['message']['content'].strip()

        # Parse JSON safely
        data = json.loads(content)
        return data
    except Exception as e:
        print(f"Error in GPT extraction: {e}")
        return None
