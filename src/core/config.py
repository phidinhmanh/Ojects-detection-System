"""Configuration management using Pydantic Settings.

This module provides centralized configuration for the application,
supporting environment variables and .env files.
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class DatabaseConfig(BaseSettings):
    """Database connection configuration."""
    host: str = Field(default="localhost", alias="DB_HOST")
    port: int = Field(default=5432, alias="DB_PORT")
    name: str = Field(default="object_detection", alias="DB_NAME")
    user: str = Field(default="postgres", alias="DB_USER")
    password: str = Field(default="", alias="DB_PASSWORD")

    class Config:
        env_prefix = "DB_"
        extra = "ignore"


class MLflowConfig(BaseSettings):
    """MLflow tracking configuration."""
    tracking_uri: str = Field(default="mlruns", alias="MLFLOW_TRACKING_URI")
    experiment_name: str = Field(
        default="object-detection", alias="MLFLOW_EXPERIMENT"
    )

    class Config:
        env_prefix = "MLFLOW_"
        extra = "ignore"


class ModelConfig(BaseSettings):
    """Model training configuration."""
    architecture: str = Field(default="mobilenetv2_ssd", alias="MODEL_ARCH")
    input_size: int = Field(default=320, alias="MODEL_INPUT_SIZE")
    num_classes: int = Field(default=80, alias="MODEL_NUM_CLASSES")
    batch_size: int = Field(default=16, alias="BATCH_SIZE")
    epochs: int = Field(default=50, alias="EPOCHS")
    learning_rate: float = Field(default=0.001, alias="LEARNING_RATE")

    class Config:
        env_prefix = "MODEL_"
        extra = "ignore"


class PathConfig(BaseSettings):
    """File path configuration."""
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    raw_dir: Path = Field(default=Path("data/raw"), alias="RAW_DIR")
    processed_dir: Path = Field(default=Path("data/processed"), alias="PROCESSED_DIR")
    model_dir: Path = Field(default=Path("model"), alias="MODEL_DIR")
    reports_dir: Path = Field(default=Path("reports"), alias="REPORTS_DIR")

    class Config:
        env_prefix = "PATH_"
        extra = "ignore"

    def ensure_dirs(self) -> None:
        """Create all configured directories if they don't exist."""
        for path in [self.data_dir, self.raw_dir, self.processed_dir, 
                     self.model_dir, self.reports_dir]:
            path.mkdir(parents=True, exist_ok=True)


class APIConfig(BaseSettings):
    """API server configuration."""
    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8080, alias="API_PORT")
    debug: bool = Field(default=False, alias="API_DEBUG")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    class Config:
        env_prefix = "API_"
        extra = "ignore"


class AppConfig(BaseSettings):
    """Root application configuration aggregating all sub-configs."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reset_config() -> None:
    """Reset configuration (useful for testing)."""
    global _config
    _config = None
