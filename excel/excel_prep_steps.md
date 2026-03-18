# Excel Preparation Workflow (Telco Customer Churn)

Use this process in Excel on the raw Kaggle CSV file.

## 1) Load and clean the raw file
1. Open the raw CSV from Kaggle (`blastchar/telco-customer-churn`).
2. Select all columns and format as Table.
3. Remove duplicates:
   - Data -> Remove Duplicates
   - Use `customerID` as the key.
4. Fix `TotalCharges` blanks:
   - Filter the `TotalCharges` column for blanks.
   - Replace blanks with `0` (or `MonthlyCharges * tenure` if your business rule requires estimated value).
   - Ensure `TotalCharges` is numeric (General/Number format).

## 2) Create tenure_group column
Add a new column `tenure_group` with this Excel formula (assuming `tenure` is in `F2`):

```excel
=IF(F2<=12,"0-12 Months",IF(F2<=24,"13-24 Months",IF(F2<=48,"25-48 Months",IF(F2<=60,"49-60 Months","60+ Months"))))
```

Fill down for all rows.

## 3) Build churn pivot by contract type
1. Insert -> PivotTable (new worksheet).
2. Pivot fields setup:
   - Rows: `Contract`
   - Columns: `Churn`
   - Values: `customerID` (Count)
3. Optional: Add calculated churn rate = `Yes / (Yes + No)`.
4. Save as `cleaned_telco_customer_churn.xlsx`.

This cleaned file can feed downstream SQL/Python workflows.
