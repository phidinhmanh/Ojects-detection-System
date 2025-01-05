"""Train Model - CLI script for model training with MLflow tracking.

Usage:
    uv run python scripts/train_model.py --epochs 10 --batch-size 16
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_config
from src.application.training_service import TrainingService
from src.infrastructure.file_repository import FileRepository
from src.infrastructure.tf_trainer import TFModelTrainer
from src.infrastructure.mlflow_tracker import MLflowTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description="Train object detection model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--no-track", action="store_true", help="Disable MLflow")
    args = parser.parse_args()

    config = get_config()
    config.paths.ensure_dirs()

    # Override config with CLI args
    config.model.epochs = args.epochs
    config.model.batch_size = args.batch_size

    # Load data
    repository = FileRepository(config.paths.processed_dir)
    annotations = repository.load_annotations()

    if not annotations:
        logger.error("No annotations found. Run ETL pipeline first.")
        sys.exit(1)

    # Split data
    split_idx = int(len(annotations) * (1 - args.val_split))
    train_data = annotations[:split_idx]
    val_data = annotations[split_idx:]

    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Setup components
    images_dir = config.paths.raw_dir / "coco128" / "images" / "train2017"
    trainer = TFModelTrainer(images_dir)
    tracker = None if args.no_track else MLflowTracker()

    # Train
    service = TrainingService(trainer, tracker)
    model_path = service.train_model(train_data, val_data, run_name=args.run_name)

    logger.info(f"✅ Training complete! Model saved to: {model_path}")


if __name__ == "__main__":
    main()
