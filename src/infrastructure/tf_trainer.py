"""TensorFlow Model Trainer - Training implementation.

Implements IModelTrainer for training object detection models with
callbacks and progress tracking.
"""
import logging
from pathlib import Path
from typing import List, Optional, Callable

import tensorflow as tf
from tensorflow import keras

from src.core.entities import ImageAnnotation, TrainingMetrics
from src.core.interfaces import IModelTrainer
from src.core.config import get_config
from src.infrastructure.tf_dataset import TFDatasetBuilder
from src.infrastructure.model_factory import ModelFactory

logger = logging.getLogger(__name__)


class TFModelTrainer(IModelTrainer):
    """TensorFlow-based model trainer implementation."""

    def __init__(
        self,
        images_dir: Path,
        model: Optional[keras.Model] = None,
        on_epoch_end: Optional[Callable[[TrainingMetrics], None]] = None,
    ):
        """Initialize trainer.
        
        Args:
            images_dir: Directory containing training images
            model: Pre-built model (optional, creates new if not provided)
            on_epoch_end: Callback function for epoch completion
        """
        self._images_dir = images_dir
        self._config = get_config()
        self._model = model
        self._on_epoch_end = on_epoch_end
        self._history = None

    def train(
        self,
        train_data: List[ImageAnnotation],
        val_data: List[ImageAnnotation],
        epochs: int,
        batch_size: int,
    ) -> Path:
        """Train the model.
        
        Args:
            train_data: Training annotations
            val_data: Validation annotations
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Path to saved model weights
        """
        # Create model if not provided
        if self._model is None:
            factory = ModelFactory()
            self._model = factory.create_model()

        # Build datasets
        builder = TFDatasetBuilder(self._images_dir)
        train_ds = builder.build_dataset(
            train_data, batch_size, shuffle=True, augment=True
        )
        val_ds = builder.build_dataset(
            val_data, batch_size, shuffle=False, augment=False
        )

        # Setup callbacks
        callbacks = self._create_callbacks()

        # Train model
        logger.info(f"Starting training for {epochs} epochs")
        self._history = self._model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # Save model
        model_path = self._config.paths.model_dir / "detector_model.keras"
        self._model.save(model_path)
        logger.info(f"Model saved to: {model_path}")

        return model_path

    def evaluate(self, test_data: List[ImageAnnotation]) -> dict:
        """Evaluate the model on test data.
        
        Args:
            test_data: Test annotations
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Train first or load a model.")

        builder = TFDatasetBuilder(self._images_dir)
        test_ds = builder.build_dataset(
            test_data, self._config.model.batch_size, shuffle=False
        )

        results = self._model.evaluate(test_ds, verbose=0)
        metrics = dict(zip(self._model.metrics_names, results))

        logger.info(f"Evaluation results: {metrics}")
        return metrics

    def _create_callbacks(self) -> List[keras.callbacks.Callback]:
        """Create training callbacks.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(self._config.paths.model_dir / "checkpoint.keras"),
                save_best_only=True,
                monitor="val_loss",
            ),
        ]

        # Add custom callback for epoch tracking
        if self._on_epoch_end:
            callbacks.append(_MetricsCallback(self._on_epoch_end))

        return callbacks


class _MetricsCallback(keras.callbacks.Callback):
    """Custom callback for logging metrics per epoch."""

    def __init__(self, callback_fn: Callable[[TrainingMetrics], None]):
        super().__init__()
        self._callback_fn = callback_fn

    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}
        metrics = TrainingMetrics(
            epoch=epoch + 1,
            train_loss=logs.get("loss", 0.0),
            val_loss=logs.get("val_loss"),
            learning_rate=float(self.model.optimizer.learning_rate),
        )
        self._callback_fn(metrics)
