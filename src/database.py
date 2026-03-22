"""
Database configuration and connection management for the churn prediction pipeline.

This module provides production-ready PostgreSQL connection handling with:
- Environment-based configuration
- Connection pooling for performance
- Context managers for proper resource cleanup
- Error handling and connection validation
"""

import os
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Generator, Optional
import logging

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Configuration class for PostgreSQL database connections."""

    host: str = None
    port: int = None
    database: str = None
    username: str = None
    password: str = None

    # Connection pool settings for production
    pool_size: int = None
    max_overflow: int = None
    pool_timeout: int = None

    def __post_init__(self):
        """Initialize and validate configuration after instantiation."""
        # Load from environment variables with defaults
        self.host = self.host or os.getenv("DB_HOST", "localhost")
        self.port = self.port or int(os.getenv("DB_PORT", "5432"))
        self.database = self.database or os.getenv("DB_NAME", "churn_prediction")
        self.username = self.username or os.getenv("DB_USER", "postgres")
        self.password = self.password or os.getenv("DB_PASSWORD", "password")

        self.pool_size = self.pool_size or int(os.getenv("DB_POOL_SIZE", "5"))
        self.max_overflow = self.max_overflow or int(os.getenv("DB_MAX_OVERFLOW", "10"))
        self.pool_timeout = self.pool_timeout or int(os.getenv("DB_POOL_TIMEOUT", "30"))

        if not self.password and os.getenv("DB_PASSWORD") is None:
            logger.warning("Database password not set. This may cause connection failures.")

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class DatabaseManager:
    """Manages database connections and operations for the churn prediction pipeline."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine: Optional[Engine] = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize SQLAlchemy engine with connection pooling."""
        try:
            self.engine = create_engine(
                self.config.connection_string,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_pre_ping=True,  # Validates connections before use
                echo=False  # Set to True for SQL debugging
            )
            logger.info(f"Database engine initialized for {self.config.host}:{self.config.port}/{self.config.database}")
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Context manager for database connections with automatic cleanup."""
        if not self.engine:
            raise RuntimeError("Database engine not initialized")

        conn = None
        try:
            conn = self.engine.connect()
            logger.debug("Database connection established")
            yield conn
        except SQLAlchemyError as e:
            logger.error(f"Database operation failed: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
                logger.debug("Database connection closed")

    def test_connection(self) -> bool:
        """Test database connectivity."""
        try:
            with self.get_connection() as conn:
                result = conn.execute(text("SELECT 1")).scalar()
                return result == 1
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def read_sql_file(self, sql_file_path: str, **params) -> pd.DataFrame:
        """
        Execute SQL from file and return results as DataFrame.

        Args:
            sql_file_path: Path to SQL file
            **params: Parameters to pass to the SQL query

        Returns:
            DataFrame with query results
        """
        try:
            with open(sql_file_path, 'r') as file:
                sql_query = file.read()

            with self.get_connection() as conn:
                df = pd.read_sql_query(text(sql_query), conn, params=params)
                logger.info(f"Successfully executed SQL file: {sql_file_path}")
                logger.info(f"Retrieved {len(df)} rows with {len(df.columns)} columns")
                return df

        except FileNotFoundError as e:
            logger.error(f"SQL file not found: {sql_file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to execute SQL file {sql_file_path}: {e}")
            raise

    def execute_sql_file(self, sql_file_path: str, **params) -> None:
        """
        Execute SQL file without returning results (for DDL, DML operations).

        Args:
            sql_file_path: Path to SQL file
            **params: Parameters to pass to the SQL query
        """
        try:
            with open(sql_file_path, 'r') as file:
                sql_query = file.read()

            with self.get_connection() as conn:
                conn.execute(text(sql_query), params)
                conn.commit()
                logger.info(f"Successfully executed SQL file: {sql_file_path}")

        except FileNotFoundError as e:
            logger.error(f"SQL file not found: {sql_file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to execute SQL file {sql_file_path}: {e}")
            raise

    def dataframe_to_table(self, df: pd.DataFrame, table_name: str,
                          if_exists: str = 'replace', index: bool = False) -> None:
        """
        Write DataFrame to database table.

        Args:
            df: DataFrame to write
            table_name: Target table name
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
            index: Whether to include DataFrame index
        """
        try:
            with self.get_connection() as conn:
                df.to_sql(table_name, conn, if_exists=if_exists, index=index)
                logger.info(f"Successfully wrote {len(df)} rows to table: {table_name}")
        except Exception as e:
            logger.error(f"Failed to write DataFrame to table {table_name}: {e}")
            raise


# Backwards compatibility functions for the existing codebase
@contextmanager
def get_db_connection(config: DatabaseConfig) -> Generator:
    """
    Legacy function for backwards compatibility.
    Use DatabaseManager.get_connection() for new code.
    """
    db_manager = DatabaseManager(config)
    with db_manager.get_connection() as conn:
        yield conn


def create_database_config() -> DatabaseConfig:
    """Create database configuration from environment variables."""
    return DatabaseConfig()


def validate_database_setup(config: DatabaseConfig) -> bool:
    """Validate that database is accessible and properly configured."""
    db_manager = DatabaseManager(config)
    return db_manager.test_connection()