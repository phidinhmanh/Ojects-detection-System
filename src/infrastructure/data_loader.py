"""COCO128 Data Loader - Extracts data from YOLO format.

Implements IDataExtractor interface for loading COCO128 dataset
in YOLO format (normalized center coordinates).
"""
import os
import logging
from pathlib import Path
from typing import Iterator
from PIL import Image

from src.core.interfaces import IDataExtractor

logger = logging.getLogger(__name__)


class COCO128Extractor(IDataExtractor):
    """Extracts raw data from COCO128 dataset in YOLO format."""

    def __init__(self, images_subdir: str = "images/train2017",
                 labels_subdir: str = "labels/train2017"):
        """Initialize with dataset subdirectory paths.
        
        Args:
            images_subdir: Subdirectory containing images
            labels_subdir: Subdirectory containing label files
        """
        self._images_subdir = images_subdir
        self._labels_subdir = labels_subdir

    def extract(self, source_path: Path) -> Iterator[dict]:
        """Extract raw records from COCO128 dataset.
        
        Args:
            source_path: Root path to the coco128 folder
            
        Yields:
            Dictionary with file_name, width, height, raw_labels
        """
        images_dir = source_path / self._images_subdir
        labels_dir = source_path / self._labels_subdir

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        label_files = list(labels_dir.glob("*.txt"))
        logger.info(f"Found {len(label_files)} label files")

        for label_path in label_files:
            file_id = label_path.stem
            image_path = images_dir / f"{file_id}.jpg"

            if not image_path.exists():
                logger.warning(f"Image not found for label: {label_path}")
                continue

            try:
                with Image.open(image_path) as img:
                    width, height = img.size

                raw_labels = label_path.read_text().strip().split("\n")

                yield {
                    "file_name": f"{file_id}.jpg",
                    "width": width,
                    "height": height,
                    "raw_labels": raw_labels,
                }
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue
