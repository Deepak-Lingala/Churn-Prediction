-- Enhanced PostgreSQL CTE-based feature engineering for Telco churn prediction
-- Combines RFM analysis with Python-based feature engineering
-- Optimized for production database performance with proper indexing

WITH base AS (
    -- Data loading and initial cleaning from customers table
    SELECT
        customer_id,
        gender,
        senior_citizen,
        partner,
        dependents,
        tenure,
        monthly_charges,
        -- Handle null/empty total charges
        CASE
            WHEN total_charges IS NULL OR total_charges = 0
            THEN 0.0
            ELSE total_charges
        END AS total_charges,
        contract,
        payment_method,
        churn,
        -- Service columns for multi-service calculation
        phone_service,
        multiple_lines,
        internet_service,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies,
        paperless_billing,
        -- Create tenure groups (matching Python logic)
        CASE
            WHEN tenure <= 12 THEN '0-12 Months'
            WHEN tenure <= 24 THEN '13-24 Months'
            WHEN tenure <= 48 THEN '25-48 Months'
            WHEN tenure <= 60 THEN '49-60 Months'
            ELSE '60+ Months'
        END AS tenure_group
    FROM customers
),
python_features AS (
    -- Port Python feature engineering logic to SQL
    SELECT
        customer_id,
        gender,
        senior_citizen,
        partner,
        dependents,
        tenure,
        monthly_charges,
        total_charges,
        contract,
        payment_method,
        churn,
        tenure_group,
        phone_service,
        multiple_lines,
        internet_service,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies,
        paperless_billing,

        -- Python feature: avg_monthly_spend
        CASE
            WHEN tenure > 0 THEN total_charges / tenure
            ELSE monthly_charges
        END AS avg_monthly_spend,

        -- Python feature: is_new_customer
        CASE WHEN tenure <= 12 THEN 1 ELSE 0 END AS is_new_customer,

        -- Python feature: tenure_charge_ratio
        tenure / (monthly_charges + 1) AS tenure_charge_ratio,

        -- Python feature: charge_per_tenure_plus1
        total_charges / (tenure + 1) AS charge_per_tenure_plus1,

        -- Python feature: electronic_check_flag
        CASE WHEN payment_method = 'Electronic check' THEN 1 ELSE 0 END AS electronic_check_flag,

        -- Python feature: auto_pay_flag
        CASE WHEN LOWER(payment_method) LIKE '%automatic%' THEN 1 ELSE 0 END AS auto_pay_flag,

        -- Python feature: partner_dependents_flag
        CASE WHEN partner = 'Yes' AND dependents = 'Yes' THEN 1 ELSE 0 END AS partner_dependents_flag,

        -- Python feature: senior_and_new (calculated after is_new_customer)
        CASE WHEN senior_citizen = 1 AND tenure <= 12 THEN 1 ELSE 0 END AS senior_and_new,

        -- Python feature: multi_services (count of active services)
        COALESCE(
            (CASE WHEN phone_service = 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN multiple_lines = 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN internet_service NOT IN ('No', 'No internet service') AND internet_service IS NOT NULL THEN 1 ELSE 0 END) +
            (CASE WHEN online_security = 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN online_backup = 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN device_protection = 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN tech_support = 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN streaming_tv = 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN streaming_movies = 'Yes' THEN 1 ELSE 0 END),
            0
        ) AS multi_services
    FROM base
),
rfm_raw AS (
    -- RFM analysis preparation
    SELECT
        customer_id,
        tenure AS recency_months,
        1::int AS frequency_txn,  -- Simplified for telco (monthly billing)
        total_charges AS monetary_value,
        monthly_charges,
        contract,
        churn,
        tenure_group,
        -- Include Python features
        avg_monthly_spend,
        is_new_customer,
        tenure_charge_ratio,
        charge_per_tenure_plus1,
        electronic_check_flag,
        auto_pay_flag,
        partner_dependents_flag,
        senior_and_new,
        multi_services
    FROM python_features
),
rfm_scored AS (
    -- RFM scoring with quintiles and cohort aggregations
    SELECT
        customer_id,
        recency_months,
        frequency_txn,
        monetary_value,
        monthly_charges,
        contract,
        churn,
        tenure_group,
        avg_monthly_spend,
        is_new_customer,
        tenure_charge_ratio,
        charge_per_tenure_plus1,
        electronic_check_flag,
        auto_pay_flag,
        partner_dependents_flag,
        senior_and_new,
        multi_services,

        -- RFM quintile scoring
        NTILE(5) OVER (ORDER BY recency_months ASC) AS recency_ntile,
        NTILE(5) OVER (ORDER BY frequency_txn DESC) AS frequency_ntile,
        NTILE(5) OVER (ORDER BY monetary_value DESC) AS monetary_ntile,

        -- Cohort benchmarking features
        AVG(monetary_value) OVER (PARTITION BY contract) AS avg_totalcharges_by_contract,
        AVG(monthly_charges) OVER (PARTITION BY tenure_group) AS avg_monthly_by_tenure_group,

        -- Additional cohort analytics
        AVG(multi_services) OVER (PARTITION BY contract) AS avg_services_by_contract,
        AVG(tenure_charge_ratio) OVER (PARTITION BY tenure_group) AS avg_tenure_charge_ratio_by_group
    FROM rfm_raw
),
trend_features AS (
    -- Trend analysis with LAG functions
    SELECT
        customer_id,
        recency_months,
        frequency_txn,
        monetary_value,
        monthly_charges,
        contract,
        churn,
        tenure_group,
        avg_monthly_spend,
        is_new_customer,
        tenure_charge_ratio,
        charge_per_tenure_plus1,
        electronic_check_flag,
        auto_pay_flag,
        partner_dependents_flag,
        senior_and_new,
        multi_services,
        recency_ntile,
        frequency_ntile,
        monetary_ntile,
        avg_totalcharges_by_contract,
        avg_monthly_by_tenure_group,
        avg_services_by_contract,
        avg_tenure_charge_ratio_by_group,

        -- Trend features using window functions
        LAG(monetary_value) OVER (PARTITION BY contract ORDER BY recency_months) AS prev_monetary_same_contract,
        LAG(monthly_charges) OVER (PARTITION BY contract ORDER BY recency_months) AS prev_monthly_same_contract
    FROM rfm_scored
),
enhanced_features AS (
    -- Final feature engineering and enhanced risk scoring
    SELECT
        customer_id,
        recency_months,
        frequency_txn,
        monetary_value,
        monthly_charges,
        contract,
        churn,
        tenure_group,
        avg_monthly_spend,
        is_new_customer,
        tenure_charge_ratio,
        charge_per_tenure_plus1,
        electronic_check_flag,
        auto_pay_flag,
        partner_dependents_flag,
        senior_and_new,
        multi_services,
        recency_ntile,
        frequency_ntile,
        monetary_ntile,
        avg_totalcharges_by_contract,
        avg_monthly_by_tenure_group,
        avg_services_by_contract,
        avg_tenure_charge_ratio_by_group,
        prev_monetary_same_contract,
        prev_monthly_same_contract,

        -- RFM composite score
        (recency_ntile + frequency_ntile + monetary_ntile) AS rfm_score,

        -- Python feature: service_intensity (calculated after avg_monthly_spend)
        monthly_charges / (avg_monthly_spend + 1) AS service_intensity,

        -- Enhanced risk scoring with business rules
        CASE
            WHEN (recency_ntile + frequency_ntile + monetary_ntile) <= 5 THEN 'Critical Risk'
            WHEN (recency_ntile + frequency_ntile + monetary_ntile) <= 7
                 AND electronic_check_flag = 1 THEN 'Very High Risk'
            WHEN (recency_ntile + frequency_ntile + monetary_ntile) <= 7 THEN 'High Risk'
            WHEN (recency_ntile + frequency_ntile + monetary_ntile) <= 9
                 AND is_new_customer = 1 THEN 'Medium-High Risk'
            WHEN (recency_ntile + frequency_ntile + monetary_ntile) <= 11 THEN 'Medium Risk'
            WHEN (recency_ntile + frequency_ntile + monetary_ntile) <= 13 THEN 'Low-Medium Risk'
            ELSE 'Low Risk'
        END AS enhanced_risk_segment,

        -- Simple risk categories for model compatibility
        CASE
            WHEN (recency_ntile + frequency_ntile + monetary_ntile) <= 7 THEN 'High Risk'
            WHEN (recency_ntile + frequency_ntile + monetary_ntile) <= 11 THEN 'Medium Risk'
            ELSE 'Low Risk'
        END AS risk_segment

    FROM trend_features
)
-- Final selection with all engineered features
SELECT
    customer_id,
    -- Original features
    recency_months,
    frequency_txn,
    monetary_value,
    monthly_charges,
    contract,
    tenure_group,
    churn,

    -- Python-ported features
    avg_monthly_spend,
    is_new_customer,
    tenure_charge_ratio,
    charge_per_tenure_plus1,
    service_intensity,
    electronic_check_flag,
    auto_pay_flag,
    partner_dependents_flag,
    senior_and_new,
    multi_services,

    -- RFM features
    recency_ntile,
    frequency_ntile,
    monetary_ntile,
    rfm_score,

    -- Risk segmentation
    risk_segment,
    enhanced_risk_segment,

    -- Cohort features
    avg_totalcharges_by_contract,
    avg_monthly_by_tenure_group,
    avg_services_by_contract,
    avg_tenure_charge_ratio_by_group,

    -- Trend features
    prev_monetary_same_contract,
    prev_monthly_same_contract,

    -- Performance metadata
    CURRENT_TIMESTAMP as feature_computed_at

FROM enhanced_features
ORDER BY
    rfm_score ASC,
    enhanced_risk_segment DESC,
    customer_id;