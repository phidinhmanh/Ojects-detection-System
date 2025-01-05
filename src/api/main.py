"""FastAPI Application - Main entry point for the API server.

Configures the FastAPI application with CORS, routes, and startup events.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, load_model
from src.core.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    logger.info("Starting Object Detection API...")
    config = get_config()
    config.paths.ensure_dirs()
    
    # Load model on startup
    model_path = config.paths.model_dir / "detector.tflite"
    if model_path.exists():
        load_model(model_path)
    else:
        logger.warning(f"No model found at {model_path}. Run training first.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Object Detection API...")


def create_app() -> FastAPI:
    """Factory function to create the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    config = get_config()

    app = FastAPI(
        title="Object Detection API",
        description="Mobile-optimized object detection service using TensorFlow Lite",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Configure CORS
    origins = config.api.cors_origins.split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router, prefix="/api/v1", tags=["detection"])

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
    )
