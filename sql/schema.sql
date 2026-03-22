-- Database schema for churn prediction pipeline
-- Production-ready PostgreSQL schema with proper data types and constraints

-- Create database if not exists (handled by Docker initialization)
-- CREATE DATABASE churn_prediction;

-- Main customers table with all customer data
CREATE TABLE IF NOT EXISTS customers (
    -- Primary key and identifiers
    customer_id VARCHAR(20) PRIMARY KEY,

    -- Demographics
    gender VARCHAR(10) CHECK (gender IN ('Male', 'Female', 'Other')),
    senior_citizen INTEGER CHECK (senior_citizen IN (0, 1)),
    partner VARCHAR(5) CHECK (partner IN ('Yes', 'No')),
    dependents VARCHAR(5) CHECK (dependents IN ('Yes', 'No')),

    -- Account information
    tenure INTEGER CHECK (tenure >= 0),
    contract VARCHAR(20) CHECK (contract IN ('Month-to-month', 'One year', 'Two year')),
    payment_method VARCHAR(50),
    paperless_billing VARCHAR(5) CHECK (paperless_billing IN ('Yes', 'No')),

    -- Financial data
    monthly_charges DECIMAL(10,2) CHECK (monthly_charges > 0),
    total_charges DECIMAL(12,2) CHECK (total_charges >= 0),

    -- Service information
    phone_service VARCHAR(5) CHECK (phone_service IN ('Yes', 'No')),
    multiple_lines VARCHAR(20) CHECK (multiple_lines IN ('Yes', 'No', 'No phone service')),
    internet_service VARCHAR(20) CHECK (internet_service IN ('DSL', 'Fiber optic', 'No')),
    online_security VARCHAR(20) CHECK (online_security IN ('Yes', 'No', 'No internet service')),
    online_backup VARCHAR(20) CHECK (online_backup IN ('Yes', 'No', 'No internet service')),
    device_protection VARCHAR(20) CHECK (device_protection IN ('Yes', 'No', 'No internet service')),
    tech_support VARCHAR(20) CHECK (tech_support IN ('Yes', 'No', 'No internet service')),
    streaming_tv VARCHAR(20) CHECK (streaming_tv IN ('Yes', 'No', 'No internet service')),
    streaming_movies VARCHAR(20) CHECK (streaming_movies IN ('Yes', 'No', 'No internet service')),

    -- Target variable
    churn VARCHAR(5) CHECK (churn IN ('Yes', 'No')),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feature store for computed metrics and caching
CREATE TABLE IF NOT EXISTS feature_store (
    customer_id VARCHAR(20) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    feature_name VARCHAR(50) NOT NULL,
    feature_value DECIMAL(12,4),
    feature_type VARCHAR(20) DEFAULT 'numeric',
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,

    PRIMARY KEY (customer_id, feature_name, computed_at)
);

-- Model predictions history for tracking and comparison
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,

    -- Model information
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20),

    -- Prediction results
    churn_probability DECIMAL(5,4) CHECK (churn_probability BETWEEN 0 AND 1),
    predicted_churn INTEGER CHECK (predicted_churn IN (0, 1)),
    risk_segment VARCHAR(20),
    confidence_score DECIMAL(5,4),

    -- Feature importance (top 3 features)
    top_feature_1 VARCHAR(50),
    top_feature_1_importance DECIMAL(6,4),
    top_feature_2 VARCHAR(50),
    top_feature_2_importance DECIMAL(6,4),
    top_feature_3 VARCHAR(50),
    top_feature_3_importance DECIMAL(6,4),

    -- Metadata
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    run_id VARCHAR(100),

    -- Business context
    retention_cost DECIMAL(10,2),
    estimated_clv DECIMAL(12,2)
);

-- Model performance tracking
CREATE TABLE IF NOT EXISTS model_performance (
    performance_id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20),

    -- Performance metrics
    accuracy DECIMAL(6,4),
    precision DECIMAL(6,4),
    recall DECIMAL(6,4),
    f1_score DECIMAL(6,4),
    roc_auc DECIMAL(6,4),

    -- Training details
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    training_size INTEGER,
    test_size INTEGER,
    cv_folds INTEGER,

    -- Hyperparameters (JSON format)
    hyperparameters JSONB,

    -- Validation information
    cross_val_score DECIMAL(6,4),
    validation_method VARCHAR(50),

    -- Run information
    run_id VARCHAR(100),
    run_duration_seconds INTEGER
);

-- Business intelligence aggregations for dashboards
CREATE TABLE IF NOT EXISTS churn_analytics (
    analytics_id SERIAL PRIMARY KEY,
    analysis_date DATE DEFAULT CURRENT_DATE,

    -- Aggregation level
    segment_type VARCHAR(50), -- contract, tenure_group, risk_segment, etc.
    segment_value VARCHAR(50),

    -- Metrics
    total_customers INTEGER,
    churned_customers INTEGER,
    churn_rate DECIMAL(5,4),
    avg_monthly_charges DECIMAL(10,2),
    avg_total_charges DECIMAL(12,2),
    avg_tenure DECIMAL(6,2),

    -- Financial impact
    total_revenue DECIMAL(15,2),
    lost_revenue DECIMAL(15,2),
    retention_opportunity DECIMAL(15,2),

    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data quality tracking
CREATE TABLE IF NOT EXISTS data_quality_log (
    log_id SERIAL PRIMARY KEY,
    table_name VARCHAR(50),
    check_name VARCHAR(100),
    check_result VARCHAR(20), -- PASS, FAIL, WARNING
    error_count INTEGER,
    total_records INTEGER,
    error_details TEXT,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create trigger for automatic updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_customers_updated_at
    BEFORE UPDATE ON customers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE OR REPLACE VIEW customer_summary AS
SELECT
    customer_id,
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    CASE
        WHEN tenure <= 12 THEN '0-12 Months'
        WHEN tenure <= 24 THEN '13-24 Months'
        WHEN tenure <= 48 THEN '25-48 Months'
        WHEN tenure <= 60 THEN '49-60 Months'
        ELSE '60+ Months'
    END AS tenure_group,
    contract,
    payment_method,
    monthly_charges,
    total_charges,
    churn,
    -- Service counts
    (CASE WHEN phone_service = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN internet_service != 'No' THEN 1 ELSE 0 END +
     CASE WHEN online_security = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN online_backup = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN device_protection = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN tech_support = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN streaming_tv = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN streaming_movies = 'Yes' THEN 1 ELSE 0 END) AS total_services,
    created_at
FROM customers;

-- Grant permissions for application user (if using role-based security)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO churn_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO churn_app_user;