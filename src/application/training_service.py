"""Training Service - Model training orchestration with experiment tracking.

This service coordinates model training, validation, and experiment logging
using injected dependencies for flexibility and testability.
"""
import logging
from pathlib import Path
from typing import List, Optional, Callable

from src.core.entities import ImageAnnotation, TrainingMetrics
from src.core.interfaces import IModelTrainer, IExperimentTracker
from src.core.config import get_config

logger = logging.getLogger(__name__)


class TrainingService:
    """Orchestrates model training with experiment tracking."""

    def __init__(
        self,
        trainer: IModelTrainer,
        tracker: Optional[IExperimentTracker] = None,
    ):
        """Initialize with training and tracking dependencies.
        
        Args:
            trainer: Model trainer implementation
            tracker: Optional experiment tracker (MLflow, etc.)
        """
        self._trainer = trainer
        self._tracker = tracker
        self._config = get_config()

    def train_model(
        self,
        train_data: List[ImageAnnotation],
        val_data: List[ImageAnnotation],
        run_name: Optional[str] = None,
        on_epoch_end: Optional[Callable[[TrainingMetrics], None]] = None,
    ) -> Path:
        """Train the object detection model.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            run_name: Optional name for the experiment run
            on_epoch_end: Optional callback for epoch completion
            
        Returns:
            Path to the trained model weights
        """
        run_name = run_name or f"train_{self._config.model.architecture}"
        
        # Start experiment tracking
        if self._tracker:
            self._tracker.start_run(run_name)
            self._tracker.log_params({
                "architecture": self._config.model.architecture,
                "input_size": self._config.model.input_size,
                "batch_size": self._config.model.batch_size,
                "epochs": self._config.model.epochs,
                "learning_rate": self._config.model.learning_rate,
                "train_samples": len(train_data),
                "val_samples": len(val_data),
            })

        logger.info(f"Starting training: {run_name}")
        logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

        try:
            # Execute training
            model_path = self._trainer.train(
                train_data=train_data,
                val_data=val_data,
                epochs=self._config.model.epochs,
                batch_size=self._config.model.batch_size,
            )

            # Log final model artifact
            if self._tracker:
                self._tracker.log_artifact(model_path)

            logger.info(f"Training completed. Model saved to: {model_path}")
            return model_path

        finally:
            if self._tracker:
                self._tracker.end_run()

    def evaluate_model(
        self, test_data: List[ImageAnnotation]
    ) -> dict:
        """Evaluate the trained model on test data.
        
        Args:
            test_data: Test dataset for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating on {len(test_data)} test samples")
        metrics = self._trainer.evaluate(test_data)
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
