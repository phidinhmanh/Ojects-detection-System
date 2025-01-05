"""Domain entities for object detection system.

This module contains the core business entities that represent
the fundamental concepts of the object detection domain.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class DetectionClass(Enum):
    """COCO dataset class labels (subset for demonstration)."""
    PERSON = 0
    BICYCLE = 1
    CAR = 2
    MOTORCYCLE = 3
    AIRPLANE = 4
    BUS = 5
    TRAIN = 6
    TRUCK = 7
    BOAT = 8
    TRAFFIC_LIGHT = 9


@dataclass(frozen=True)
class BoundingBox:
    """Represents a bounding box annotation.
    
    Attributes:
        x_min: Left edge coordinate (pixels)
        y_min: Top edge coordinate (pixels)
        width: Box width (pixels)
        height: Box height (pixels)
        class_id: Object class identifier
        confidence: Detection confidence (0.0-1.0), None for ground truth
    """
    x_min: float
    y_min: float
    width: float
    height: float
    class_id: int
    confidence: Optional[float] = None

    @property
    def x_max(self) -> float:
        """Right edge coordinate."""
        return self.x_min + self.width

    @property
    def y_max(self) -> float:
        """Bottom edge coordinate."""
        return self.y_min + self.height

    @property
    def area(self) -> float:
        """Box area in pixels squared."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Center point (x, y) of the box."""
        return (self.x_min + self.width / 2, self.y_min + self.height / 2)

    def to_yolo_format(self, img_w: int, img_h: int) -> tuple:
        """Convert to normalized YOLO format (cx, cy, w, h)."""
        cx = (self.x_min + self.width / 2) / img_w
        cy = (self.y_min + self.height / 2) / img_h
        return (self.class_id, cx, cy, self.width / img_w, self.height / img_h)


@dataclass
class ImageAnnotation:
    """Represents an image with its annotations.
    
    Attributes:
        file_name: Image file name (without path)
        width: Image width in pixels
        height: Image height in pixels
        boxes: List of bounding box annotations
    """
    file_name: str
    width: int
    height: int
    boxes: List[BoundingBox] = field(default_factory=list)

    @property
    def num_objects(self) -> int:
        """Total number of annotated objects."""
        return len(self.boxes)

    def get_class_counts(self) -> dict[int, int]:
        """Count objects per class."""
        counts: dict[int, int] = {}
        for box in self.boxes:
            counts[box.class_id] = counts.get(box.class_id, 0) + 1
        return counts


@dataclass
class DetectionResult:
    """Result from running object detection on an image.
    
    Attributes:
        image_path: Path to the processed image
        detections: List of detected bounding boxes with confidence
        inference_time_ms: Time taken for inference in milliseconds
    """
    image_path: str
    detections: List[BoundingBox]
    inference_time_ms: float

    @property
    def num_detections(self) -> int:
        """Number of detected objects."""
        return len(self.detections)

    def filter_by_confidence(self, threshold: float) -> List[BoundingBox]:
        """Filter detections by minimum confidence threshold."""
        return [d for d in self.detections if d.confidence and d.confidence >= threshold]


@dataclass
class TrainingMetrics:
    """Metrics from a training epoch.
    
    Attributes:
        epoch: Current epoch number
        train_loss: Training loss value
        val_loss: Validation loss value
        mAP: Mean Average Precision
        learning_rate: Current learning rate
    """
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    mAP: Optional[float] = None
    learning_rate: Optional[float] = None


@dataclass
class DatasetStats:
    """Statistical summary of a dataset.
    
    Attributes:
        total_images: Total number of images
        total_annotations: Total number of bounding boxes
        class_distribution: Count per class ID
        avg_annotations_per_image: Average boxes per image
        image_sizes: List of (width, height) tuples
    """
    total_images: int
    total_annotations: int
    class_distribution: dict[int, int]
    avg_annotations_per_image: float
    image_sizes: List[tuple[int, int]] = field(default_factory=list)
