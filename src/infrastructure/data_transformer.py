"""Data Transformer - Converts YOLO format to domain entities.

Implements IDataTransformer for converting normalized YOLO bounding boxes
to pixel-based BoundingBox entities.
"""
import logging
from typing import Iterator, List

from src.core.entities import BoundingBox, ImageAnnotation
from src.core.interfaces import IDataTransformer

logger = logging.getLogger(__name__)


class YOLOTransformer(IDataTransformer):
    """Transforms YOLO format labels to ImageAnnotation entities."""

    def transform(self, raw_records: Iterator[dict]) -> List[ImageAnnotation]:
        """Transform raw YOLO records to ImageAnnotation entities.
        
        YOLO format: class_id cx cy w h (normalized 0-1)
        Output: pixel-based BoundingBox with x_min, y_min, width, height
        
        Args:
            raw_records: Iterator of raw record dictionaries
            
        Returns:
            List of ImageAnnotation entities
        """
        annotations: List[ImageAnnotation] = []

        for record in raw_records:
            boxes = self._parse_labels(
                record["raw_labels"],
                record["width"],
                record["height"],
            )

            annotation = ImageAnnotation(
                file_name=record["file_name"],
                width=record["width"],
                height=record["height"],
                boxes=boxes,
            )
            annotations.append(annotation)

        logger.info(f"Transformed {len(annotations)} images")
        total_boxes = sum(a.num_objects for a in annotations)
        logger.info(f"Total bounding boxes: {total_boxes}")

        return annotations

    def _parse_labels(
        self, raw_labels: List[str], img_w: int, img_h: int
    ) -> List[BoundingBox]:
        """Parse YOLO label lines to BoundingBox entities.
        
        Args:
            raw_labels: List of YOLO format label strings
            img_w: Image width in pixels
            img_h: Image height in pixels
            
        Returns:
            List of BoundingBox entities
        """
        boxes: List[BoundingBox] = []

        for line in raw_labels:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])

                # Convert normalized center coords to pixel top-left coords
                real_w = w * img_w
                real_h = h * img_h
                x_min = (cx * img_w) - (real_w / 2)
                y_min = (cy * img_h) - (real_h / 2)

                box = BoundingBox(
                    x_min=round(x_min, 2),
                    y_min=round(y_min, 2),
                    width=round(real_w, 2),
                    height=round(real_h, 2),
                    class_id=class_id,
                )
                boxes.append(box)
            except (ValueError, IndexError) as e:
                logger.warning(f"Invalid label line '{line}': {e}")
                continue

        return boxes
