"""
Data migration utility for churn prediction pipeline.

This script migrates data from CSV files to PostgreSQL database,
handling data cleaning, transformation, and validation.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import warnings

import pandas as pd
import numpy as np
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from database import DatabaseManager, DatabaseConfig
from config import PipelineConfig, setup_logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def clean_csv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare CSV data for database insertion.

    Args:
        df: Raw CSV DataFrame

    Returns:
        Cleaned DataFrame ready for database insertion
    """
    logger.info(f"Starting data cleaning for {len(df)} records")

    # Create a copy to avoid modifying original
    df_clean = df.copy()

    # Convert to lowercase first, then map to DB schema-compatible snake_case names.
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    column_mapping = {
        'customerid': 'customer_id',
        'seniorcitizen': 'senior_citizen',
        'phoneservice': 'phone_service',
        'multiplelines': 'multiple_lines',
        'internetservice': 'internet_service',
        'onlinesecurity': 'online_security',
        'onlinebackup': 'online_backup',
        'deviceprotection': 'device_protection',
        'techsupport': 'tech_support',
        'streamingtv': 'streaming_tv',
        'streamingmovies': 'streaming_movies',
        'paperlessbilling': 'paperless_billing',
        'paymentmethod': 'payment_method',
        'monthlycharges': 'monthly_charges',
        'totalcharges': 'total_charges',
    }
    df_clean = df_clean.rename(columns=column_mapping)

    # Handle TotalCharges - it comes as string with blanks in CSV
    if 'total_charges' in df_clean.columns:
        # Convert blank strings to NaN, then to numeric
        df_clean['total_charges'] = pd.to_numeric(
            df_clean['total_charges'].replace(' ', np.nan),
            errors='coerce'
        )
        # Fill NaN values with 0 (new customers)
        df_clean['total_charges'] = df_clean['total_charges'].fillna(0.0)
        logger.info("Cleaned TotalCharges column (converted blanks to 0)")

    # Standardize Yes/No values
    yes_no_columns = [
        'partner', 'dependents', 'phone_service', 'paperless_billing', 'churn'
    ]
    for col in yes_no_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map({'Yes': 'Yes', 'No': 'No'})

    # Handle SeniorCitizen (should be 0/1)
    if 'senior_citizen' in df_clean.columns:
        df_clean['senior_citizen'] = df_clean['senior_citizen'].astype(int)

    # Clean service columns - handle "No internet service" / "No phone service"
    service_columns = [
        'multiple_lines', 'online_security', 'online_backup', 'device_protection',
        'tech_support', 'streaming_tv', 'streaming_movies'
    ]
    for col in service_columns:
        if col in df_clean.columns:
            # Keep the original values including "No internet service" etc
            # as they provide business insight
            pass

    # Validate numeric columns
    numeric_columns = ['tenure', 'monthly_charges', 'total_charges']
    for col in numeric_columns:
        if col in df_clean.columns:
            # Ensure non-negative values
            if (df_clean[col] < 0).any():
                logger.warning(f"Found negative values in {col}, setting to 0")
                df_clean[col] = df_clean[col].clip(lower=0)

    # Remove duplicates based on customer_id
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['customer_id'], keep='first')
    final_count = len(df_clean)

    if initial_count != final_count:
        logger.info(f"Removed {initial_count - final_count} duplicate records")

    # Validate required columns
    required_columns = {
        'customer_id', 'churn', 'tenure', 'monthly_charges', 'total_charges',
        'contract', 'senior_citizen', 'payment_method'
    }

    missing_columns = required_columns - set(df_clean.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    logger.info(f"Data cleaning completed. Final dataset: {len(df_clean)} records")
    return df_clean


def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Perform data quality checks on cleaned dataset.

    Args:
        df: Cleaned DataFrame

    Returns:
        Dictionary with validation results
    """
    logger.info("Performing data quality validation")

    results = {
        'total_records': len(df),
        'null_counts': {},
        'duplicates': 0,
        'outliers': {},
        'value_checks': {},
        'errors': []
    }

    # Check for null values
    null_counts = df.isnull().sum()
    results['null_counts'] = {col: int(count) for col, count in null_counts.items() if count > 0}

    # Check for duplicates
    results['duplicates'] = df.duplicated(subset=['customer_id']).sum()

    # Check value distributions
    if 'churn' in df.columns:
        churn_dist = df['churn'].value_counts()
        results['value_checks']['churn_distribution'] = churn_dist.to_dict()
        churn_rate = (churn_dist.get('Yes', 0) / len(df)) * 100
        results['value_checks']['churn_rate_percent'] = round(churn_rate, 2)

    if 'monthly_charges' in df.columns:
        monthly_stats = {
            'min': float(df['monthly_charges'].min()),
            'max': float(df['monthly_charges'].max()),
            'mean': float(df['monthly_charges'].mean()),
            'median': float(df['monthly_charges'].median())
        }
        results['value_checks']['monthly_charges_stats'] = monthly_stats

        # Check for outliers (using IQR method)
        Q1 = df['monthly_charges'].quantile(0.25)
        Q3 = df['monthly_charges'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df['monthly_charges'] < lower_bound) |
                 (df['monthly_charges'] > upper_bound)]
        results['outliers']['monthly_charges'] = len(outliers)

    if 'tenure' in df.columns:
        tenure_stats = {
            'min': int(df['tenure'].min()),
            'max': int(df['tenure'].max()),
            'mean': float(df['tenure'].mean()),
            'zero_tenure_count': int((df['tenure'] == 0).sum())
        }
        results['value_checks']['tenure_stats'] = tenure_stats

    # Business logic validation
    if 'total_charges' in df.columns and 'monthly_charges' in df.columns and 'tenure' in df.columns:
        # Check customers with zero tenure but positive charges
        zero_tenure_with_charges = df[(df['tenure'] == 0) & (df['total_charges'] > 0)]
        if len(zero_tenure_with_charges) > 0:
            results['errors'].append(f"Found {len(zero_tenure_with_charges)} customers with zero tenure but positive total charges")

        # Check if total charges approximately equals monthly * tenure for long-term customers
        long_term = df[df['tenure'] > 6]  # Only check customers with >6 months
        expected_total = long_term['monthly_charges'] * long_term['tenure']
        actual_total = long_term['total_charges']
        significant_diff = abs(expected_total - actual_total) > (expected_total * 0.5)  # 50% tolerance

        if significant_diff.any():
            results['errors'].append(f"Found {significant_diff.sum()} customers with significant discrepancy between monthly charges * tenure and total charges")

    logger.info(f"Data quality validation completed. Found {len(results['errors'])} potential issues")
    return results


def migrate_csv_to_database(
    csv_path: Path,
    db_config: DatabaseConfig,
    table_name: str = 'customers',
    chunk_size: int = 1000
) -> bool:
    """
    Migrate CSV data to PostgreSQL database.

    Args:
        csv_path: Path to CSV file
        db_config: Database configuration
        table_name: Target table name
        chunk_size: Number of records to insert per batch

    Returns:
        True if migration successful, False otherwise
    """
    try:
        # Validate CSV file exists
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        logger.info(f"Starting migration from {csv_path} to table {table_name}")

        # Load and clean data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records from CSV")

        df_clean = clean_csv_data(df)

        # Validate data quality
        quality_results = validate_data_quality(df_clean)
        logger.info(f"Data quality check completed: {quality_results['total_records']} records ready for migration")

        # Log any quality issues
        if quality_results['errors']:
            for error in quality_results['errors']:
                logger.warning(f"Data quality issue: {error}")

        # Initialize database manager
        db_manager = DatabaseManager(db_config)

        # Test connection
        if not db_manager.test_connection():
            raise RuntimeError("Unable to connect to database")

        logger.info("Database connection established")

        # Keep schema objects intact and clear data before re-loading.
        # Using TRUNCATE avoids DROP TABLE issues when views/FKs depend on customers.
        with db_manager.get_connection() as conn:
            conn.execute(text("TRUNCATE TABLE feature_store, predictions, customers RESTART IDENTITY CASCADE"))
            conn.commit()
        logger.info("Cleared target tables: feature_store, predictions, customers")

        # Perform migration in chunks
        total_records = len(df_clean)
        records_migrated = 0

        for i in range(0, total_records, chunk_size):
            chunk = df_clean.iloc[i:i + chunk_size]

            try:
                db_manager.dataframe_to_table(
                    chunk,
                    table_name,
                    if_exists='append',
                    index=False
                )
                records_migrated += len(chunk)
                logger.info(f"Migrated {records_migrated}/{total_records} records ({records_migrated/total_records*100:.1f}%)")

            except Exception as e:
                logger.error(f"Error migrating chunk {i//chunk_size + 1}: {e}")
                raise

        # Verify migration
        with db_manager.get_connection() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
            if result != total_records:
                raise RuntimeError(f"Migration verification failed: expected {total_records}, got {result}")

        logger.info(f"Migration completed successfully: {total_records} records migrated to {table_name}")

        # Log quality summary
        logger.info(f"Migration summary: {quality_results['value_checks']}")

        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def main():
    """Main migration script."""
    parser = argparse.ArgumentParser(description="Migrate CSV data to PostgreSQL database")

    parser.add_argument(
        "--from-csv",
        type=Path,
        required=True,
        help="Path to source CSV file"
    )

    parser.add_argument(
        "--table",
        default="customers",
        help="Target database table name (default: customers)"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of records per insertion batch (default: 1000)"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate data without performing migration"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    try:
        # Create configuration
        config = PipelineConfig.from_csv_migration()
        logger.info(f"Starting migration with configuration: {config.database.host}:{config.database.port}/{config.database.database}")

        if args.validate_only:
            # Only perform validation
            logger.info("Running validation-only mode")
            df = pd.read_csv(args.from_csv)
            df_clean = clean_csv_data(df)
            quality_results = validate_data_quality(df_clean)

            print("\n=== Data Quality Report ===")
            print(f"Total records: {quality_results['total_records']}")
            print(f"Null counts: {quality_results['null_counts']}")
            print(f"Duplicate records: {quality_results['duplicates']}")
            print(f"Outliers: {quality_results['outliers']}")
            print(f"Value checks: {quality_results['value_checks']}")
            if quality_results['errors']:
                print(f"Errors: {quality_results['errors']}")
            else:
                print("No data quality errors found")

        else:
            # Perform full migration
            success = migrate_csv_to_database(
                args.from_csv,
                config.database,
                args.table,
                args.chunk_size
            )

            if success:
                logger.info("Migration completed successfully")
                sys.exit(0)
            else:
                logger.error("Migration failed")
                sys.exit(1)

    except Exception as e:
        logger.error(f"Migration script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()