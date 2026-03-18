-- PostgreSQL CTE-based cohort and RFM feature engineering for Telco churn
-- Includes NTILE(), LAG(), and AVG() OVER (PARTITION BY ...)

WITH base AS (
    SELECT
        customerid,
        gender,
        seniorcitizen,
        partner,
        dependents,
        tenure,
        monthlycharges,
        NULLIF(TRIM(totalcharges), '')::numeric AS totalcharges,
        contract,
        paymentmethod,
        churn,
        CASE
            WHEN tenure <= 12 THEN '0-12 Months'
            WHEN tenure <= 24 THEN '13-24 Months'
            WHEN tenure <= 48 THEN '25-48 Months'
            WHEN tenure <= 60 THEN '49-60 Months'
            ELSE '60+ Months'
        END AS tenure_group
    FROM telco_customers
),
cleaned AS (
    SELECT
        customerid,
        gender,
        seniorcitizen,
        partner,
        dependents,
        tenure,
        monthlycharges,
        COALESCE(totalcharges, 0) AS totalcharges,
        contract,
        paymentmethod,
        churn,
        tenure_group
    FROM base
),
rfm_raw AS (
    SELECT
        customerid,
        tenure AS recency_months,
        1::int AS frequency_txn,
        totalcharges AS monetary_value,
        monthlycharges,
        contract,
        churn,
        tenure_group
    FROM cleaned
),
rfm_scored AS (
    SELECT
        customerid,
        recency_months,
        frequency_txn,
        monetary_value,
        monthlycharges,
        contract,
        churn,
        tenure_group,
        NTILE(5) OVER (ORDER BY recency_months ASC) AS recency_ntile,
        NTILE(5) OVER (ORDER BY frequency_txn DESC) AS frequency_ntile,
        NTILE(5) OVER (ORDER BY monetary_value DESC) AS monetary_ntile,
        AVG(monetary_value) OVER (PARTITION BY contract) AS avg_totalcharges_by_contract,
        AVG(monthlycharges) OVER (PARTITION BY tenure_group) AS avg_monthly_by_tenure_group
    FROM rfm_raw
),
trend_features AS (
    SELECT
        customerid,
        recency_months,
        frequency_txn,
        monetary_value,
        monthlycharges,
        contract,
        churn,
        tenure_group,
        recency_ntile,
        frequency_ntile,
        monetary_ntile,
        avg_totalcharges_by_contract,
        avg_monthly_by_tenure_group,
        LAG(monetary_value) OVER (PARTITION BY contract ORDER BY recency_months) AS prev_monetary_same_contract
    FROM rfm_scored
),
segmented AS (
    SELECT
        customerid,
        recency_months,
        frequency_txn,
        monetary_value,
        monthlycharges,
        contract,
        churn,
        tenure_group,
        recency_ntile,
        frequency_ntile,
        monetary_ntile,
        avg_totalcharges_by_contract,
        avg_monthly_by_tenure_group,
        prev_monetary_same_contract,
        (recency_ntile + frequency_ntile + monetary_ntile) AS rfm_score,
        CASE
            WHEN (recency_ntile + frequency_ntile + monetary_ntile) <= 7 THEN 'High Risk'
            WHEN (recency_ntile + frequency_ntile + monetary_ntile) <= 11 THEN 'Med Risk'
            ELSE 'Low Risk'
        END AS risk_segment
    FROM trend_features
)
SELECT
    customerid,
    recency_months,
    frequency_txn,
    monetary_value,
    monthlycharges,
    contract,
    tenure_group,
    churn,
    recency_ntile,
    frequency_ntile,
    monetary_ntile,
    rfm_score,
    risk_segment,
    avg_totalcharges_by_contract,
    avg_monthly_by_tenure_group,
    prev_monetary_same_contract
FROM segmented
ORDER BY rfm_score ASC, customerid;
