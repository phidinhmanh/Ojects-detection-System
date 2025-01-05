"""ETL Service - Orchestrates the Extract, Transform, Load pipeline.

This service follows the Single Responsibility Principle by delegating
actual operations to injected dependencies.
"""
import logging
from pathlib import Path
from typing import List

from src.core.entities import ImageAnnotation
from src.core.interfaces import IDataExtractor, IDataTransformer, IDataRepository

logger = logging.getLogger(__name__)


class ETLService:
    """Orchestrates the ETL pipeline for object detection data.
    
    Uses dependency injection for all external operations,
    making the service testable and following Open/Closed Principle.
    """

    def __init__(
        self,
        extractor: IDataExtractor,
        transformer: IDataTransformer,
        repository: IDataRepository,
    ):
        """Initialize with injected dependencies.
        
        Args:
            extractor: Implementation for data extraction
            transformer: Implementation for data transformation
            repository: Implementation for data persistence
        """
        self._extractor = extractor
        self._transformer = transformer
        self._repository = repository

    def run_pipeline(self, source_path: Path) -> List[ImageAnnotation]:
        """Execute the full ETL pipeline.
        
        Args:
            source_path: Path to the raw data directory
            
        Returns:
            List of processed ImageAnnotation entities
        """
        logger.info(f"Starting ETL pipeline from: {source_path}")

        # Extract
        logger.info("Phase 1: Extracting raw data...")
        raw_records = self._extractor.extract(source_path)

        # Transform
        logger.info("Phase 2: Transforming to domain entities...")
        annotations = self._transformer.transform(raw_records)
        logger.info(f"Transformed {len(annotations)} image annotations")

        # Load
        logger.info("Phase 3: Persisting to storage...")
        self._repository.save_annotations(annotations)

        logger.info("ETL pipeline completed successfully")
        return annotations

    def validate_data(self, annotations: List[ImageAnnotation]) -> dict:
        """Validate the processed annotations.
        
        Args:
            annotations: List of annotations to validate
            
        Returns:
            Validation report dictionary
        """
        issues: List[str] = []
        
        for ann in annotations:
            if ann.width <= 0 or ann.height <= 0:
                issues.append(f"{ann.file_name}: Invalid dimensions")
            
            for box in ann.boxes:
                if box.x_min < 0 or box.y_min < 0:
                    issues.append(f"{ann.file_name}: Negative coordinates")
                if box.x_max > ann.width or box.y_max > ann.height:
                    issues.append(f"{ann.file_name}: Box outside image bounds")

        return {
            "total_images": len(annotations),
            "total_boxes": sum(a.num_objects for a in annotations),
            "issues_count": len(issues),
            "issues": issues[:10],  # Limit to first 10 issues
            "is_valid": len(issues) == 0,
        }
