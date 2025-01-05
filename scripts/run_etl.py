"""Run ETL Pipeline - CLI script for data processing.

Usage:
    uv run python scripts/run_etl.py --source data/raw/coco128
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_config
from src.application.etl_service import ETLService
from src.infrastructure.data_loader import COCO128Extractor
from src.infrastructure.data_transformer import YOLOTransformer
from src.infrastructure.file_repository import FileRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for ETL script."""
    parser = argparse.ArgumentParser(description="Run ETL pipeline")
    parser.add_argument(
        "--source",
        type=str,
        default="data/raw/coco128",
        help="Path to raw data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for processed data",
    )
    args = parser.parse_args()

    config = get_config()
    config.paths.ensure_dirs()

    source_path = Path(args.source)
    output_path = Path(args.output) if args.output else config.paths.processed_dir

    if not source_path.exists():
        logger.error(f"Source path not found: {source_path}")
        logger.info("Run 'uv run python utils/load_coo.py' to download COCO128")
        sys.exit(1)

    # Create dependencies
    extractor = COCO128Extractor()
    transformer = YOLOTransformer()
    repository = FileRepository(output_path)

    # Run ETL
    service = ETLService(extractor, transformer, repository)
    annotations = service.run_pipeline(source_path)

    # Validate
    validation = service.validate_data(annotations)
    logger.info(f"Validation: {validation}")

    if validation["is_valid"]:
        logger.info("✅ ETL completed successfully!")
    else:
        logger.warning(f"⚠️  ETL completed with {validation['issues_count']} issues")


if __name__ == "__main__":
    main()
