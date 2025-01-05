"""Analysis Service - Dataset exploration and reporting.

This service analyzes dataset statistics and generates visual reports
to help understand data distribution before training.
"""
import logging
from pathlib import Path
from typing import List
from collections import Counter

from src.core.entities import ImageAnnotation, DatasetStats
from src.core.interfaces import IDataAnalyzer

logger = logging.getLogger(__name__)


class AnalysisService(IDataAnalyzer):
    """Computes dataset statistics and generates analysis reports."""

    # COCO class names for the first 10 classes
    CLASS_NAMES = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
        4: "airplane", 5: "bus", 6: "train", 7: "truck",
        8: "boat", 9: "traffic_light"
    }

    def compute_stats(self, annotations: List[ImageAnnotation]) -> DatasetStats:
        """Compute comprehensive dataset statistics.
        
        Args:
            annotations: List of image annotations
            
        Returns:
            DatasetStats with computed metrics
        """
        logger.info(f"Computing stats for {len(annotations)} images")

        class_counter: Counter = Counter()
        total_boxes = 0
        image_sizes: List[tuple[int, int]] = []

        for ann in annotations:
            total_boxes += ann.num_objects
            image_sizes.append((ann.width, ann.height))
            for box in ann.boxes:
                class_counter[box.class_id] += 1

        avg_per_image = total_boxes / len(annotations) if annotations else 0.0

        return DatasetStats(
            total_images=len(annotations),
            total_annotations=total_boxes,
            class_distribution=dict(class_counter),
            avg_annotations_per_image=round(avg_per_image, 2),
            image_sizes=image_sizes,
        )

    def generate_report(self, stats: DatasetStats, output_path: Path) -> Path:
        """Generate an HTML analysis report with visualizations.
        
        Args:
            stats: Computed dataset statistics
            output_path: Directory to save the report
            
        Returns:
            Path to the generated HTML report
        """
        output_path.mkdir(parents=True, exist_ok=True)
        report_file = output_path / "data_analysis_report.html"

        # Generate class distribution data for chart
        class_labels = [
            self.CLASS_NAMES.get(k, f"class_{k}") 
            for k in sorted(stats.class_distribution.keys())
        ]
        class_counts = [
            stats.class_distribution[k] 
            for k in sorted(stats.class_distribution.keys())
        ]

        html_content = self._build_html_report(stats, class_labels, class_counts)
        
        report_file.write_text(html_content, encoding="utf-8")
        logger.info(f"Report generated: {report_file}")
        return report_file

    def _build_html_report(
        self, stats: DatasetStats, labels: List[str], counts: List[int]
    ) -> str:
        """Build the HTML report content."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Dataset Analysis Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .stat-card {{ background: #f5f5f5; padding: 20px; margin: 10px;
                      border-radius: 8px; display: inline-block; }}
        .chart-container {{ width: 600px; margin: 20px 0; }}
        h1 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>📊 Object Detection Dataset Analysis</h1>
    
    <div class="stat-card">
        <h3>Total Images</h3>
        <p style="font-size: 2em;">{stats.total_images}</p>
    </div>
    <div class="stat-card">
        <h3>Total Annotations</h3>
        <p style="font-size: 2em;">{stats.total_annotations}</p>
    </div>
    <div class="stat-card">
        <h3>Avg Objects/Image</h3>
        <p style="font-size: 2em;">{stats.avg_annotations_per_image}</p>
    </div>

    <h2>Class Distribution</h2>
    <div class="chart-container">
        <canvas id="classChart"></canvas>
    </div>

    <script>
        new Chart(document.getElementById('classChart'), {{
            type: 'bar',
            data: {{
                labels: {labels},
                datasets: [{{
                    label: 'Object Count',
                    data: {counts},
                    backgroundColor: 'rgba(54, 162, 235, 0.6)'
                }}]
            }},
            options: {{ responsive: true }}
        }});
    </script>
</body>
</html>"""
