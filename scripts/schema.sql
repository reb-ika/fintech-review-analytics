
-- bank_reviews Database Schema
-- Kifiya AI Training Program Week 2
-- Author: Rebika Woldeyesus

-- Banks table: stores metadata about each bank
CREATE TABLE IF NOT EXISTS banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) NOT NULL UNIQUE,
    app_name VARCHAR(200),
    app_id VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Reviews table: stores all scraped and analyzed review data
CREATE TABLE IF NOT EXISTS reviews (
    review_id SERIAL PRIMARY KEY,
    bank_id INTEGER REFERENCES banks(bank_id),
    review_text TEXT,
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    review_date DATE,
    sentiment_label VARCHAR(20),
    sentiment_score FLOAT,
    identified_theme VARCHAR(100),
    source VARCHAR(50) DEFAULT 'Google Play',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Verification queries
-- SELECT b.bank_name, COUNT(r.review_id) FROM banks b LEFT JOIN reviews r ON b.bank_id = r.bank_id GROUP BY b.bank_name;
-- SELECT b.bank_name, AVG(r.rating), AVG(r.sentiment_score) FROM banks b JOIN reviews r ON b.bank_id = r.bank_id GROUP BY b.bank_name;
