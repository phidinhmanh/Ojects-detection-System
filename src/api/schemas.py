"""API Request/Response Schemas using Pydantic.

Defines the data transfer objects for API validation and serialization.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBoxResponse(BaseModel):
    """Bounding box detection response."""
    x_min: float = Field(..., description="Left edge coordinate")
    y_min: float = Field(..., description="Top edge coordinate")
    width: float = Field(..., description="Box width")
    height: float = Field(..., description="Box height")
    class_id: int = Field(..., description="Object class ID")
    class_name: str = Field(..., description="Object class name")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")


class PredictionResponse(BaseModel):
    """Object detection prediction response."""
    detections: List[BoundingBoxResponse] = Field(
        default_factory=list, description="List of detected objects"
    )
    num_detections: int = Field(..., description="Number of detected objects")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    image_width: int = Field(..., description="Original image width")
    image_height: int = Field(..., description="Original image height")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy", description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")


class DatasetStatsResponse(BaseModel):
    """Dataset statistics response."""
    total_images: int
    total_annotations: int
    class_distribution: dict
    avg_annotations_per_image: float
