-- Performance indexes for churn prediction database
-- Optimized for common query patterns and analytics workloads

-- Primary table indexes for customers
CREATE INDEX IF NOT EXISTS idx_customers_churn
ON customers(churn);

CREATE INDEX IF NOT EXISTS idx_customers_contract
ON customers(contract);

CREATE INDEX IF NOT EXISTS idx_customers_tenure
ON customers(tenure);

CREATE INDEX IF NOT EXISTS idx_customers_payment_method
ON customers(payment_method);

CREATE INDEX IF NOT EXISTS idx_customers_monthly_charges
ON customers(monthly_charges);

-- Composite indexes for common filtering patterns
CREATE INDEX IF NOT EXISTS idx_customers_churn_contract
ON customers(churn, contract);

CREATE INDEX IF NOT EXISTS idx_customers_tenure_charges
ON customers(tenure, monthly_charges);

CREATE INDEX IF NOT EXISTS idx_customers_senior_tenure
ON customers(senior_citizen, tenure);

-- Service-related indexes for feature engineering
CREATE INDEX IF NOT EXISTS idx_customers_internet_service
ON customers(internet_service);

CREATE INDEX IF NOT EXISTS idx_customers_phone_service
ON customers(phone_service);

-- Financial analysis indexes
CREATE INDEX IF NOT EXISTS idx_customers_total_charges_desc
ON customers(total_charges DESC);

-- Feature store indexes
CREATE INDEX IF NOT EXISTS idx_feature_store_customer_feature
ON feature_store(customer_id, feature_name);

CREATE INDEX IF NOT EXISTS idx_feature_store_computed_at
ON feature_store(computed_at DESC);

CREATE INDEX IF NOT EXISTS idx_feature_store_expires_at
ON feature_store(expires_at) WHERE expires_at IS NOT NULL;

-- Predictions table indexes
CREATE INDEX IF NOT EXISTS idx_predictions_customer_model
ON predictions(customer_id, model_name);

CREATE INDEX IF NOT EXISTS idx_predictions_model_date
ON predictions(model_name, prediction_date DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_risk_segment
ON predictions(risk_segment);

CREATE INDEX IF NOT EXISTS idx_predictions_probability
ON predictions(churn_probability DESC);

-- Performance monitoring indexes
CREATE INDEX IF NOT EXISTS idx_model_performance_name_version
ON model_performance(model_name, model_version);

CREATE INDEX IF NOT EXISTS idx_model_performance_training_date
ON model_performance(training_date DESC);

CREATE INDEX IF NOT EXISTS idx_model_performance_accuracy
ON model_performance(accuracy DESC);

-- Analytics table indexes
CREATE INDEX IF NOT EXISTS idx_analytics_date_segment
ON churn_analytics(analysis_date DESC, segment_type, segment_value);

CREATE INDEX IF NOT EXISTS idx_analytics_segment_churn_rate
ON churn_analytics(segment_type, churn_rate DESC);

-- Data quality indexes
CREATE INDEX IF NOT EXISTS idx_data_quality_table_result
ON data_quality_log(table_name, check_result);

CREATE INDEX IF NOT EXISTS idx_data_quality_checked_at
ON data_quality_log(checked_at DESC);

-- Partial indexes for performance on filtered queries
CREATE INDEX IF NOT EXISTS idx_customers_churned_only
ON customers(customer_id, monthly_charges, tenure)
WHERE churn = 'Yes';

CREATE INDEX IF NOT EXISTS idx_customers_high_value
ON customers(customer_id, churn, contract)
WHERE monthly_charges > 50;

CREATE INDEX IF NOT EXISTS idx_customers_long_tenure
ON customers(customer_id, churn, monthly_charges)
WHERE tenure > 24;

-- Functional indexes for computed columns
CREATE INDEX IF NOT EXISTS idx_customers_avg_monthly_spend
ON customers(
    (CASE WHEN tenure > 0 THEN total_charges / tenure ELSE monthly_charges END)
);

CREATE INDEX IF NOT EXISTS idx_customers_tenure_group
ON customers(
    (CASE
        WHEN tenure <= 12 THEN '0-12 Months'
        WHEN tenure <= 24 THEN '13-24 Months'
        WHEN tenure <= 48 THEN '25-48 Months'
        WHEN tenure <= 60 THEN '49-60 Months'
        ELSE '60+ Months'
    END)
);

-- Text search indexes for payment methods
CREATE INDEX IF NOT EXISTS idx_customers_payment_method_gin
ON customers USING gin(to_tsvector('english', payment_method));

-- Statistics and performance notes
-- Run ANALYZE after bulk data loading to update query planner statistics
-- Example: ANALYZE customers;

-- Monitor index usage with:
-- SELECT schemaname, tablename, attname, n_distinct, correlation
-- FROM pg_stats WHERE tablename = 'customers';

-- Check index efficiency with:
-- SELECT indexname, indexsize, indexscan, indextupreads
-- FROM pg_stat_user_indexes WHERE relname = 'customers';