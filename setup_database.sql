-- Database setup script for ETIS DAG
-- Run this script to create the necessary tables

-- Create articles table
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    link TEXT UNIQUE NOT NULL,
    published TEXT,
    summary TEXT,
    content_full TEXT,
    ml_class INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create ivory_extractions table
CREATE TABLE IF NOT EXISTS ivory_extractions (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id),
    report_date TEXT,
    seizure_date TEXT,
    location TEXT,
    items_seized JSONB,
    arrest_count INTEGER,
    amount_approximate TEXT,
    comment TEXT,
    url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_articles_ml_class ON articles(ml_class);
CREATE INDEX IF NOT EXISTS idx_articles_link ON articles(link);
CREATE INDEX IF NOT EXISTS idx_ivory_extractions_article_id ON ivory_extractions(article_id);

-- Add some sample data for testing (optional)
-- INSERT INTO articles (title, link, published, summary) VALUES 
-- ('Test Article 1', 'https://example.com/1', '2025-01-01', 'This is a test article about ivory trade'),
-- ('Test Article 2', 'https://example.com/2', '2025-01-02', 'Another test article about elephant trafficking'); 