"""Prediction Service - Inference orchestration.

This service provides a clean interface for running object detection
predictions on images.
"""
import logging
from pathlib import Path
from typing import List

from src.core.entities import DetectionResult
from src.core.interfaces import IModelInference

logger = logging.getLogger(__name__)


class PredictionService:
    """Handles object detection predictions."""

    def __init__(self, inference: IModelInference):
        """Initialize with inference implementation.
        
        Args:
            inference: Model inference implementation
        """
        self._inference = inference
        self._model_loaded = False

    def load_model(self, model_path: Path) -> None:
        """Load a trained model for inference.
        
        Args:
            model_path: Path to the model file
        """
        logger.info(f"Loading model from: {model_path}")
        self._inference.load_model(model_path)
        self._model_loaded = True
        logger.info("Model loaded successfully")

    def predict(
        self, image_path: Path, confidence_threshold: float = 0.5
    ) -> DetectionResult:
        """Run prediction on a single image.
        
        Args:
            image_path: Path to the input image
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            DetectionResult with detected objects
            
        Raises:
            RuntimeError: If model not loaded
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.debug(f"Running prediction on: {image_path}")
        result = self._inference.predict(image_path, confidence_threshold)
        logger.debug(f"Found {result.num_detections} objects")
        return result

    def predict_batch(
        self, image_paths: List[Path], confidence_threshold: float = 0.5
    ) -> List[DetectionResult]:
        """Run predictions on multiple images.
        
        Args:
            image_paths: List of image paths
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        for path in image_paths:
            result = self.predict(path, confidence_threshold)
            results.append(result)
        return results
