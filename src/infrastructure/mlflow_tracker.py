"""MLflow Tracker - Experiment tracking implementation.

Implements IExperimentTracker interface for MLflow-based experiment logging.
"""
import logging
from pathlib import Path
from typing import Optional

import mlflow
from mlflow import MlflowClient

from src.core.entities import TrainingMetrics
from src.core.interfaces import IExperimentTracker
from src.core.config import get_config

logger = logging.getLogger(__name__)


class MLflowTracker(IExperimentTracker):
    """MLflow-based experiment tracking implementation."""

    def __init__(self, tracking_uri: Optional[str] = None, 
                 experiment_name: Optional[str] = None):
        """Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
        """
        self._config = get_config()
        self._tracking_uri = tracking_uri or self._config.mlflow.tracking_uri
        self._experiment_name = experiment_name or self._config.mlflow.experiment_name
        self._run_id: Optional[str] = None

        # Configure MLflow
        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)

    def start_run(self, run_name: str) -> None:
        """Start a new experiment run.
        
        Args:
            run_name: Name for the run
        """
        run = mlflow.start_run(run_name=run_name)
        self._run_id = run.info.run_id
        logger.info(f"Started MLflow run: {run_name} (ID: {self._run_id})")

    def log_params(self, params: dict) -> None:
        """Log hyperparameters.
        
        Args:
            params: Dictionary of parameters to log
        """
        if not self._run_id:
            logger.warning("No active run. Call start_run() first.")
            return
        
        mlflow.log_params(params)
        logger.debug(f"Logged params: {params}")

    def log_metrics(self, metrics: TrainingMetrics) -> None:
        """Log training metrics for an epoch.
        
        Args:
            metrics: TrainingMetrics entity
        """
        if not self._run_id:
            logger.warning("No active run. Call start_run() first.")
            return

        step = metrics.epoch
        mlflow.log_metric("train_loss", metrics.train_loss, step=step)
        
        if metrics.val_loss is not None:
            mlflow.log_metric("val_loss", metrics.val_loss, step=step)
        if metrics.mAP is not None:
            mlflow.log_metric("mAP", metrics.mAP, step=step)
        if metrics.learning_rate is not None:
            mlflow.log_metric("learning_rate", metrics.learning_rate, step=step)

    def log_artifact(self, artifact_path: Path) -> None:
        """Log a file artifact.
        
        Args:
            artifact_path: Path to the artifact file
        """
        if not self._run_id:
            logger.warning("No active run. Call start_run() first.")
            return

        mlflow.log_artifact(str(artifact_path))
        logger.info(f"Logged artifact: {artifact_path}")

    def end_run(self) -> None:
        """End the current experiment run."""
        if self._run_id:
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self._run_id}")
            self._run_id = None
