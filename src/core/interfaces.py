"""Abstract interfaces for dependency injection.

This module defines the contracts (interfaces) that infrastructure
implementations must follow, enabling the Dependency Inversion Principle.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Iterator
from pathlib import Path

from .entities import (
    ImageAnnotation,
    DetectionResult,
    BoundingBox,
    DatasetStats,
    TrainingMetrics,
)


class IDataRepository(ABC):
    """Interface for data persistence operations."""

    @abstractmethod
    def save_annotations(self, annotations: List[ImageAnnotation]) -> None:
        """Persist image annotations to storage."""
        pass

    @abstractmethod
    def load_annotations(self) -> List[ImageAnnotation]:
        """Load all image annotations from storage."""
        pass

    @abstractmethod
    def get_annotation_by_filename(self, filename: str) -> Optional[ImageAnnotation]:
        """Retrieve annotation for a specific image file."""
        pass


class IDataExtractor(ABC):
    """Interface for extracting raw data from sources."""

    @abstractmethod
    def extract(self, source_path: Path) -> Iterator[dict]:
        """Extract raw records from a data source.
        
        Yields:
            Dictionary with 'file_name', 'width', 'height', 'raw_labels'
        """
        pass


class IDataTransformer(ABC):
    """Interface for transforming raw data to domain entities."""

    @abstractmethod
    def transform(self, raw_records: Iterator[dict]) -> List[ImageAnnotation]:
        """Transform raw records into ImageAnnotation entities."""
        pass


class IModelTrainer(ABC):
    """Interface for model training operations."""

    @abstractmethod
    def train(
        self,
        train_data: List[ImageAnnotation],
        val_data: List[ImageAnnotation],
        epochs: int,
        batch_size: int,
    ) -> Path:
        """Train the model and return path to saved weights.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Path to the saved model weights
        """
        pass

    @abstractmethod
    def evaluate(self, test_data: List[ImageAnnotation]) -> dict:
        """Evaluate model on test data and return metrics."""
        pass


class IModelInference(ABC):
    """Interface for running model inference."""

    @abstractmethod
    def load_model(self, model_path: Path) -> None:
        """Load a trained model from disk."""
        pass

    @abstractmethod
    def predict(self, image_path: Path, threshold: float = 0.5) -> DetectionResult:
        """Run inference on a single image.
        
        Args:
            image_path: Path to the input image
            threshold: Minimum confidence threshold
            
        Returns:
            DetectionResult with detected bounding boxes
        """
        pass


class IExperimentTracker(ABC):
    """Interface for ML experiment tracking."""

    @abstractmethod
    def start_run(self, run_name: str) -> None:
        """Start a new experiment run."""
        pass

    @abstractmethod
    def log_params(self, params: dict) -> None:
        """Log hyperparameters."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: TrainingMetrics) -> None:
        """Log training metrics for an epoch."""
        pass

    @abstractmethod
    def log_artifact(self, artifact_path: Path) -> None:
        """Log a file artifact (model, report, etc.)."""
        pass

    @abstractmethod
    def end_run(self) -> None:
        """End the current experiment run."""
        pass


class IDataAnalyzer(ABC):
    """Interface for dataset analysis operations."""

    @abstractmethod
    def compute_stats(self, annotations: List[ImageAnnotation]) -> DatasetStats:
        """Compute statistical summary of the dataset."""
        pass

    @abstractmethod
    def generate_report(
        self, stats: DatasetStats, output_path: Path
    ) -> Path:
        """Generate a visual analysis report.
        
        Returns:
            Path to the generated report file
        """
        pass
