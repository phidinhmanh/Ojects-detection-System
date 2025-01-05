"""Analyze Dataset - Generate data analysis report before training.

Usage:
    uv run python scripts/analyze_data.py --open
"""
import argparse
import logging
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_config
from src.application.analysis_service import AnalysisService
from src.infrastructure.file_repository import FileRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for analysis script."""
    parser = argparse.ArgumentParser(description="Analyze dataset")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open report in browser after generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for report",
    )
    args = parser.parse_args()

    config = get_config()
    config.paths.ensure_dirs()

    output_path = Path(args.output) if args.output else config.paths.reports_dir

    # Load processed annotations
    repository = FileRepository(config.paths.processed_dir)
    annotations = repository.load_annotations()

    if not annotations:
        logger.error("No annotations found. Run ETL pipeline first.")
        logger.info("Usage: uv run python scripts/run_etl.py")
        sys.exit(1)

    # Analyze data
    analyzer = AnalysisService()
    stats = analyzer.compute_stats(annotations)

    logger.info(f"Dataset Summary:")
    logger.info(f"  - Total images: {stats.total_images}")
    logger.info(f"  - Total annotations: {stats.total_annotations}")
    logger.info(f"  - Avg objects/image: {stats.avg_annotations_per_image}")
    logger.info(f"  - Classes: {len(stats.class_distribution)}")

    # Generate report
    report_path = analyzer.generate_report(stats, output_path)
    logger.info(f"✅ Report generated: {report_path}")

    if args.open:
        webbrowser.open(f"file://{report_path.absolute()}")


if __name__ == "__main__":
    main()
