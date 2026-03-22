"""
Configuration management for the churn prediction pipeline.

This module provides centralized configuration management with:
- Environment-based settings
- Default values for development
- Production-ready configuration patterns
- Integration with database configuration
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
from dotenv import load_dotenv

try:
    from .database import DatabaseConfig
except ImportError:
    from database import DatabaseConfig

# Load environment variables from .env file if it exists
# Explicitly load from parent directory (project root)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)


@dataclass
class ModelConfig:
    """Configuration for machine learning model parameters."""

    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    optimize_metric: Literal["accuracy", "precision", "recall", "f1", "roc_auc"] = os.getenv("OPTIMIZE_METRIC", "accuracy")
    use_smote: bool = os.getenv("USE_SMOTE", "false").lower() == "true"
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))
    cv_folds: int = int(os.getenv("CV_FOLDS", "5"))

    def __post_init__(self):
        """Validate model configuration."""
        if self.test_size <= 0 or self.test_size >= 1:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {self.cv_folds}")


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""

    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = field(init=False)
    sql_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    runs_dir: Path = field(init=False)

    # Legacy CSV path for migration purposes
    csv_data_path: Path = field(init=False)

    def __post_init__(self):
        """Initialize derived paths."""
        self.data_dir = self.base_dir / "data"
        self.sql_dir = self.base_dir / "sql"
        self.outputs_dir = self.base_dir / "outputs"
        self.plots_dir = self.outputs_dir / "plots"
        self.models_dir = self.outputs_dir / "models"
        self.runs_dir = self.outputs_dir / "runs"
        self.csv_data_path = self.data_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

        # Create directories if they don't exist
        for dir_path in [self.outputs_dir, self.plots_dir, self.models_dir, self.runs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineConfig:
    """Main configuration class for the churn prediction pipeline."""

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # Pipeline behavior settings
    use_database: bool = os.getenv("USE_DATABASE", "true").lower() == "true"
    run_name: Optional[str] = os.getenv("RUN_NAME", None)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = os.getenv("LOG_LEVEL", "INFO")

    # Feature engineering settings
    enable_sql_features: bool = os.getenv("ENABLE_SQL_FEATURES", "true").lower() == "true"
    cache_features: bool = os.getenv("CACHE_FEATURES", "true").lower() == "true"

    # Business export settings
    generate_business_exports: bool = os.getenv("GENERATE_BUSINESS_EXPORTS", "true").lower() == "true"
    export_predictions_to_db: bool = os.getenv("EXPORT_PREDICTIONS_TO_DB", "true").lower() == "true"

    @classmethod
    def from_environment(cls, environment: Literal["development", "production", "testing"] = "development"):
        """
        Create configuration from environment with environment-specific defaults.

        Args:
            environment: Target environment (development, production, testing)

        Returns:
            Configured PipelineConfig instance
        """
        config = cls()

        # Environment-specific overrides
        if environment == "production":
            # Production optimizations
            config.model.cv_folds = int(os.getenv("CV_FOLDS", "3"))  # Fewer folds for speed
            config.cache_features = True
            config.export_predictions_to_db = True
            config.log_level = os.getenv("LOG_LEVEL", "WARNING")

        elif environment == "testing":
            # Testing optimizations
            config.model.cv_folds = 2  # Minimal folds for speed
            config.cache_features = False
            config.export_predictions_to_db = False
            config.generate_business_exports = False
            config.log_level = "DEBUG"

        elif environment == "development":
            # Development defaults (already set in dataclass)
            config.log_level = os.getenv("LOG_LEVEL", "INFO")

        return config

    @classmethod
    def from_csv_migration(cls):
        """
        Create configuration for migrating from CSV to database.

        Returns:
            PipelineConfig with settings appropriate for data migration
        """
        config = cls()
        config.use_database = False  # Don't try to read from DB during migration
        config.enable_sql_features = False
        config.cache_features = False
        config.export_predictions_to_db = False
        config.log_level = "INFO"
        return config

    def get_sql_file_path(self, sql_filename: str) -> Path:
        """Get fully qualified path to SQL file."""
        return self.paths.sql_dir / sql_filename

    def get_run_output_dir(self, run_name: Optional[str] = None) -> Path:
        """Get output directory for specific run."""
        run_id = run_name or self.run_name or "default_run"
        run_dir = self.paths.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def validate(self) -> None:
        """Validate the complete configuration."""
        # Validate model config
        self.model.__post_init__()

        # Validate critical paths exist
        if self.use_database:
            # Database validation happens in database.py
            pass
        else:
            # Validate CSV file exists for fallback
            if not self.paths.csv_data_path.exists():
                raise FileNotFoundError(
                    f"CSV data file not found: {self.paths.csv_data_path}. "
                    "Either provide the CSV file or configure database access."
                )

        # Validate SQL files exist if SQL features enabled
        if self.enable_sql_features:
            enhanced_sql_path = self.get_sql_file_path("enhanced_churn_features.sql")
            if not enhanced_sql_path.exists():
                # Check for original SQL file as fallback
                original_sql_path = self.get_sql_file_path("churn_rfm_features.sql")
                if not original_sql_path.exists():
                    raise FileNotFoundError(
                        f"SQL feature file not found: {enhanced_sql_path} or {original_sql_path}"
                    )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for logging/serialization."""
        return {
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "username": self.database.username,
                # Never log passwords
            },
            "model": {
                "random_state": self.model.random_state,
                "optimize_metric": self.model.optimize_metric,
                "use_smote": self.model.use_smote,
                "test_size": self.model.test_size,
                "cv_folds": self.model.cv_folds,
            },
            "pipeline": {
                "use_database": self.use_database,
                "run_name": self.run_name,
                "log_level": self.log_level,
                "enable_sql_features": self.enable_sql_features,
                "cache_features": self.cache_features,
            }
        }


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the pipeline."""
    import logging

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            # Add file handler for production
            # logging.FileHandler("churn_prediction.log")
        ]
    )