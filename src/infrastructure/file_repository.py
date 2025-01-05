"""File Repository - JSON-based data persistence.

Implements IDataRepository for storing annotations as JSON files,
avoiding database dependencies for simpler deployment.
"""
import json
import logging
from pathlib import Path
from typing import List, Optional

from src.core.entities import BoundingBox, ImageAnnotation
from src.core.interfaces import IDataRepository

logger = logging.getLogger(__name__)


class FileRepository(IDataRepository):
    """JSON file-based repository for image annotations."""

    def __init__(self, storage_path: Path):
        """Initialize with storage directory.
        
        Args:
            storage_path: Directory to store JSON files
        """
        self._storage_path = storage_path
        self._annotations_file = storage_path / "annotations.json"
        storage_path.mkdir(parents=True, exist_ok=True)

    def save_annotations(self, annotations: List[ImageAnnotation]) -> None:
        """Persist annotations to JSON file.
        
        Args:
            annotations: List of ImageAnnotation entities
        """
        data = [self._to_dict(ann) for ann in annotations]
        
        self._annotations_file.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        logger.info(f"Saved {len(annotations)} annotations to {self._annotations_file}")

    def load_annotations(self) -> List[ImageAnnotation]:
        """Load all annotations from JSON file.
        
        Returns:
            List of ImageAnnotation entities
        """
        if not self._annotations_file.exists():
            logger.warning(f"Annotations file not found: {self._annotations_file}")
            return []

        data = json.loads(self._annotations_file.read_text(encoding="utf-8"))
        annotations = [self._from_dict(item) for item in data]
        logger.info(f"Loaded {len(annotations)} annotations")
        return annotations

    def get_annotation_by_filename(self, filename: str) -> Optional[ImageAnnotation]:
        """Get annotation for a specific image file.
        
        Args:
            filename: Image file name to search for
            
        Returns:
            ImageAnnotation if found, None otherwise
        """
        annotations = self.load_annotations()
        for ann in annotations:
            if ann.file_name == filename:
                return ann
        return None

    def _to_dict(self, annotation: ImageAnnotation) -> dict:
        """Convert ImageAnnotation to dictionary."""
        return {
            "file_name": annotation.file_name,
            "width": annotation.width,
            "height": annotation.height,
            "boxes": [
                {
                    "x_min": box.x_min,
                    "y_min": box.y_min,
                    "width": box.width,
                    "height": box.height,
                    "class_id": box.class_id,
                }
                for box in annotation.boxes
            ],
        }

    def _from_dict(self, data: dict) -> ImageAnnotation:
        """Convert dictionary to ImageAnnotation."""
        boxes = [
            BoundingBox(
                x_min=b["x_min"],
                y_min=b["y_min"],
                width=b["width"],
                height=b["height"],
                class_id=b["class_id"],
            )
            for b in data.get("boxes", [])
        ]
        return ImageAnnotation(
            file_name=data["file_name"],
            width=data["width"],
            height=data["height"],
            boxes=boxes,
        )
