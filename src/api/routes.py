"""API Routes - Endpoint handlers for object detection service.

Defines the REST API endpoints with proper error handling.
"""
import io
import time
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from PIL import Image

from src.api.schemas import (
    PredictionResponse,
    BoundingBoxResponse,
    HealthResponse,
    DatasetStatsResponse,
)
from src.core.config import get_config
from src.infrastructure.file_repository import FileRepository
from src.application.analysis_service import AnalysisService

logger = logging.getLogger(__name__)

router = APIRouter()

# Global state for loaded model
_model = None
_model_loaded = False

# Class names mapping
CLASS_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    4: "airplane", 5: "bus", 6: "train", 7: "truck",
    8: "boat", 9: "traffic_light", 10: "fire_hydrant",
    11: "stop_sign", 12: "parking_meter", 13: "bench",
}


def load_model(model_path: Optional[Path] = None) -> None:
    """Load the TFLite model for inference.
    
    Args:
        model_path: Path to the .tflite model file
    """
    global _model, _model_loaded
    import tensorflow as tf
    
    config = get_config()
    path = model_path or config.paths.model_dir / "detector.tflite"
    
    if path.exists():
        _model = tf.lite.Interpreter(model_path=str(path))
        _model.allocate_tensors()
        _model_loaded = True
        logger.info(f"Model loaded from: {path}")
    else:
        logger.warning(f"Model file not found: {path}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model_loaded,
        version="1.0.0",
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence: float = Query(0.5, ge=0.0, le=1.0, description="Min confidence"),
):
    """Run object detection on an uploaded image.
    
    Args:
        file: Image file (JPEG, PNG)
        confidence: Minimum confidence threshold
        
    Returns:
        PredictionResponse with detected objects
    """
    if not _model_loaded:
        raise HTTPException(503, detail="Model not loaded. Service unavailable.")

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, detail="Only JPEG and PNG images are supported")

    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        orig_width, orig_height = image.size

        start_time = time.time()

        # Run inference (placeholder - actual TFLite inference)
        detections = _run_inference(image, confidence)

        inference_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            detections=detections,
            num_detections=len(detections),
            inference_time_ms=round(inference_time, 2),
            image_width=orig_width,
            image_height=orig_height,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, detail=str(e))


def _run_inference(image: Image.Image, threshold: float) -> list:
    """Run TFLite model inference.
    
    Args:
        image: PIL Image to process
        threshold: Confidence threshold
        
    Returns:
        List of BoundingBoxResponse objects
    """
    import numpy as np
    
    config = get_config()
    input_size = config.model.input_size

    # Preprocess image
    img_resized = image.resize((input_size, input_size))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Get model input/output details
    input_details = _model.get_input_details()
    output_details = _model.get_output_details()

    # Run inference
    _model.set_tensor(input_details[0]["index"], img_array)
    _model.invoke()

    # Process outputs (simplified - actual implementation depends on model)
    # This is a placeholder for demonstration
    detections = []
    
    return detections


@router.get("/stats", response_model=DatasetStatsResponse)
async def get_dataset_stats():
    """Get cached dataset statistics."""
    config = get_config()
    repo = FileRepository(config.paths.processed_dir)
    analyzer = AnalysisService()

    annotations = repo.load_annotations()
    if not annotations:
        raise HTTPException(404, detail="No annotations found")

    stats = analyzer.compute_stats(annotations)
    return DatasetStatsResponse(
        total_images=stats.total_images,
        total_annotations=stats.total_annotations,
        class_distribution=stats.class_distribution,
        avg_annotations_per_image=stats.avg_annotations_per_image,
    )
