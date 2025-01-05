"""Export to TFLite - Convert trained model to mobile-optimized format.

Usage:
    uv run python scripts/export_tflite.py --quantize
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_config
from src.infrastructure.model_factory import ModelFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for TFLite export."""
    parser = argparse.ArgumentParser(description="Export model to TFLite")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to saved Keras model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for TFLite file",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply float16 quantization",
    )
    args = parser.parse_args()

    config = get_config()

    # Load model
    model_path = Path(args.model_path) if args.model_path else (
        config.paths.model_dir / "detector_model"
    )

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Train a model first: uv run python scripts/train_model.py")
        sys.exit(1)

    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Loaded model from: {model_path}")

    # Export
    output_path = Path(args.output) if args.output else config.paths.model_dir
    factory = ModelFactory()
    tflite_path = factory.export_tflite(model, output_path)

    # Print model size
    size_mb = tflite_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Exported TFLite model: {tflite_path}")
    logger.info(f"   Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
