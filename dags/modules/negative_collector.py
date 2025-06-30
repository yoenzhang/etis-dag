import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from airflow.providers.postgres.hooks.postgres import PostgresHook
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class NegativeExample:
    title: str
    link: str
    published: str
    summary: str
    rejection_reason: str
    source: str

class NegativeCollector:
    """
    Collects negative examples from various points in the pipeline.
    """
    
    def __init__(self):
        self.negative_examples = []
        # Use a path that exists in the Airflow environment
        self.output_dir = Path("./dags/data")
        # Create the directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_filtered_out_article(self, article, reason: str):
        """Add articles that were filtered out by rule-based filter."""
        negative_example = NegativeExample(
            title=article.title,
            link=article.link,
            published=article.published,
            summary=article.content,
            rejection_reason=reason,
            source="rule_based_filter"
        )
        self.negative_examples.append(negative_example)
    
    def add_ml_negative_article(self, title: str, link: str, published: str, summary: str, probability: float, threshold: float):
        """Add articles that were classified as negative by the ML model."""
        negative_example = NegativeExample(
            title=title,
            link=link,
            published=published,
            summary=summary,
            rejection_reason=f"ML classification negative (prob={probability:.3f}, threshold={threshold:.3f})",
            source="ml_classifier"
        )
        self.negative_examples.append(negative_example)
    
    def add_llm_rejected_article(self, title: str, link: str, published: str, summary: str, llm_comment: str, llm_response: str):
        """Add articles that were rejected by the LLM as not relevant."""
        negative_example = NegativeExample(
            title=title,
            link=link,
            published=published,
            summary=summary,
            rejection_reason=f"LLM rejected: {llm_comment}",
            source="llm_extraction"
        )
        self.negative_examples.append(negative_example)
    
    def collect_ml_negative_articles(self):
        """Collect articles classified as negative (ml_class = 0) by the ML model."""
        try:
            pg_hook = PostgresHook(postgres_conn_id="postgres_default")
            
            select_sql = """
                SELECT id, title, summary, link, published, created_at
                FROM articles 
                WHERE ml_class = 0
                ORDER BY created_at DESC;
            """
            rows = pg_hook.get_records(select_sql)
            
            for row in rows:
                article_id, title, summary, link, published, created_at = row
                negative_example = NegativeExample(
                    title=title or '',
                    link=link or '',
                    published=published or '',
                    summary=summary or '',
                    rejection_reason="ML classification negative",
                    source="ml_classifier"
                )
                self.negative_examples.append(negative_example)
            
            print(f"Collected {len(rows)} ML-negative articles")
            
        except Exception as e:
            print(f"Error collecting ML negative articles: {e}")
    
    def collect_llm_rejected_articles(self):
        """Collect articles rejected by the LLM as not relevant."""
        try:
            pg_hook = PostgresHook(postgres_conn_id="postgres_default")
            
            # Get articles that were processed by LLM but marked as not relevant
            # We can identify these by looking for articles with ml_class=1 but no extraction
            select_sql = """
                SELECT a.id, a.title, a.summary, a.link, a.published, a.created_at
                FROM articles a
                LEFT JOIN ivory_extractions ie ON ie.article_id = a.id
                WHERE a.ml_class = 1 
                  AND ie.article_id IS NULL
                  AND a.created_at >= NOW() - INTERVAL '7 days';
            """
            rows = pg_hook.get_records(select_sql)
            
            for row in rows:
                article_id, title, summary, link, published, created_at = row
                negative_example = NegativeExample(
                    title=title or '',
                    link=link or '',
                    published=published or '',
                    summary=summary or '',
                    rejection_reason="LLM marked as not relevant",
                    source="llm_extraction"
                )
                self.negative_examples.append(negative_example)
            
            print(f"Collected {len(rows)} LLM-rejected articles")
            
        except Exception as e:
            print(f"Error collecting LLM rejected articles: {e}")
    
    def collect_short_content_articles(self):
        """Collect articles that are too short to be useful."""
        try:
            pg_hook = PostgresHook(postgres_conn_id="postgres_default")
            
            select_sql = """
                SELECT id, title, summary, link, published, created_at
                FROM articles 
                WHERE (LENGTH(title) + LENGTH(summary)) < 50
                  AND created_at >= NOW() - INTERVAL '7 days';
            """
            rows = pg_hook.get_records(select_sql)
            
            for row in rows:
                article_id, title, summary, link, published, created_at = row
                negative_example = NegativeExample(
                    title=title or '',
                    link=link or '',
                    published=published or '',
                    summary=summary or '',
                    rejection_reason="Content too short",
                    source="content_filter"
                )
                self.negative_examples.append(negative_example)
            
            print(f"Collected {len(rows)} short content articles")
            
        except Exception as e:
            print(f"Error collecting short content articles: {e}")
    
    def save_negative_examples(self):
        """Save collected negative examples to CSV file."""
        if not self.negative_examples:
            print("No negative examples to save")
            return str(self.output_dir / "extracted_negative_examples.csv")
        
        # Convert to DataFrame
        data = []
        for example in self.negative_examples:
            data.append({
                'title': example.title,
                'link': example.link,
                'published': example.published,
                'summary': example.summary,
                'label': 0,  # All negative examples are labeled as 0
                'rejection_reason': example.rejection_reason,
                'source': example.source,
                'text': f"{example.title} {example.summary}".lower()
            })
        
        df = pd.DataFrame(data)
        
        # Remove duplicates based on link
        df = df.drop_duplicates(subset=['link'])
        
        # Save to file
        output_file = self.output_dir / "extracted_negative_examples.csv"
        df.to_csv(output_file, index=False)
        
        print(f"Saved {len(df)} unique negative examples to {output_file}")
        
        # Print summary
        print("\n=== Negative Examples Summary ===")
        print(f"Total negative examples: {len(df)}")
        print("\nBy source:")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
        
        print("\nBy rejection reason:")
        reason_counts = df['rejection_reason'].value_counts()
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count}")
        
        # Return string instead of PosixPath for XCom compatibility
        return str(output_file)

# Global instance to collect negative examples throughout the pipeline
negative_collector = NegativeCollector()

def collect_all_negative_examples():
    """
    Main function to collect all negative examples from the pipeline.
    This should be called at the end of the DAG run.
    """
    print("=== Collecting Negative Examples ===")
    
    # Collect from different sources
    negative_collector.collect_ml_negative_articles()
    negative_collector.collect_llm_rejected_articles()
    negative_collector.collect_short_content_articles()
    
    # Save all collected examples
    output_file = negative_collector.save_negative_examples()
    
    return output_file 