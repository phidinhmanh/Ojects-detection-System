# 🎯Object Detection System

A production-ready object detection system optimized for mobile deployment, built with **TensorFlow**, **FastAPI**, and **Docker**, designed for deployment on **Google Cloud Platform**.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.20](https://img.shields.io/badge/tensorflow-2.20-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- **Clean Architecture** with SOLID principles for maintainability
- **TensorFlow Lite** optimized models for mobile (MobileNetV2-SSD, EfficientDet-Lite)
- **MLflow** integration for experiment tracking and model versioning
- **FastAPI** REST service with OpenAPI documentation
- **Docker** multi-stage builds for optimized container images
- **GCP Cloud Run** deployment with CI/CD pipeline

---

## 📁 Project Structure

```
├── src/
│   ├── core/                    # Domain Layer
│   │   ├── config.py            # Pydantic Settings configuration
│   │   ├── entities.py          # Domain entities (BoundingBox, etc.)
│   │   └── interfaces.py        # Repository interfaces (DIP)
│   ├── application/             # Application Layer (Use Cases)
│   │   ├── etl_service.py       # ETL pipeline orchestration
│   │   ├── analysis_service.py  # Dataset analysis & reporting
│   │   ├── training_service.py  # Model training orchestration
│   │   └── prediction_service.py# Inference service
│   ├── infrastructure/          # Infrastructure Layer
│   │   ├── data_loader.py       # COCO128 data extractor
│   │   ├── data_transformer.py  # YOLO format transformer
│   │   ├── file_repository.py   # JSON-based annotation storage
│   │   ├── tf_dataset.py        # TensorFlow data pipeline
│   │   ├── model_factory.py     # Model architecture factory
│   │   ├── tf_trainer.py        # TensorFlow model trainer
│   │   └── mlflow_tracker.py    # MLflow experiment tracker
│   └── api/                     # Presentation Layer
│       ├── main.py              # FastAPI application
│       ├── routes.py            # API endpoints
│       └── schemas.py           # Pydantic request/response models
├── scripts/
│   ├── run_etl.py               # Run ETL pipeline
│   ├── analyze_data.py          # Generate data analysis report
│   ├── train_model.py           # Train object detection model
│   └── export_tflite.py         # Export model to TFLite format
├── data/
│   ├── raw/                     # Raw dataset (COCO128)
│   └── processed/               # Processed annotations
├── model/                       # Saved models and TFLite files
├── reports/                     # Analysis reports
├── Dockerfile                   # Multi-stage container build
├── docker-compose.yml           # Local development setup
├── cloudbuild.yaml              # GCP Cloud Build CI/CD
└── pyproject.toml               # UV package configuration
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [UV package manager](https://github.com/astral-sh/uv)
- Docker (for containerized deployment)
- Google Cloud SDK (for GCP deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/object-detection-mobile.git
cd object-detection-mobile

# Install dependencies with UV
uv sync

# Download COCO128 dataset
uv run python utils/load_coo.py
```

### Run the Pipeline

```bash
# 1. Process dataset (ETL)
uv run python scripts/run_etl.py --source data/raw/coco128

# 2. Analyze data (generates HTML report)
uv run python scripts/analyze_data.py --open

# 3. Train model (with MLflow tracking)
uv run python scripts/train_model.py --epochs 10 --batch-size 16

# 4. Export to TFLite for mobile
uv run python scripts/export_tflite.py
```

---

## 🐳 Docker Deployment

### Local Development

```bash
# Start API server + MLflow UI
docker-compose up

# Access services:
# - API: http://localhost:8080/api/v1/health
# - MLflow: http://localhost:5000
# - Swagger: http://localhost:8080/docs
```

### Build Production Image

```bash
docker build -t object-detector:latest .
docker run -p 8080:8080 object-detector:latest
```

---

## ☁️ Google Cloud Deployment

### One-Command Deploy

```bash
gcloud run deploy object-detector \
    --source . \
    --region us-central1 \
    --memory 2Gi \
    --allow-unauthenticated
```

### CI/CD with Cloud Build

```bash
# Connect repository and trigger builds automatically
gcloud builds submit --config cloudbuild.yaml
```

---

## 📡 API Reference

### Health Check
```http
GET /api/v1/health
```
**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "version": "1.0.0"
}
```

### Object Detection
```http
POST /api/v1/predict
Content-Type: multipart/form-data
```
| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | File | Image file (JPEG/PNG) |
| `confidence` | float | Min confidence threshold (0.0-1.0) |

**Response:**
```json
{
    "detections": [
        {
            "x_min": 100.5,
            "y_min": 50.2,
            "width": 200.0,
            "height": 150.0,
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.92
        }
    ],
    "num_detections": 1,
    "inference_time_ms": 45.2,
    "image_width": 640,
    "image_height": 480
}
```

### Dataset Statistics
```http
GET /api/v1/stats
```

---

## ⚙️ Configuration

Create a `.env` file from the template:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | 8080 | API server port |
| `MODEL_ARCH` | mobilenetv2_ssd | Model architecture |
| `MODEL_INPUT_SIZE` | 320 | Input image size |
| `BATCH_SIZE` | 16 | Training batch size |
| `EPOCHS` | 50 | Training epochs |
| `MLFLOW_TRACKING_URI` | mlruns | MLflow tracking location |

---

## 📊 MLflow Experiment Tracking

View training experiments in MLflow UI:

```bash
# Start MLflow server
uv run mlflow ui --port 5000

# Or use docker-compose
docker-compose up mlflow
```

Access at: http://localhost:5000

Tracked metrics:
- `train_loss`, `val_loss` - Training/validation loss per epoch
- `mAP` - Mean Average Precision
- `learning_rate` - Learning rate schedule

---

## 🏗️ Architecture

This project follows **Clean Architecture** principles:

```
┌─────────────────────────────────────────┐
│           Presentation Layer            │
│  (FastAPI routes, Pydantic schemas)     │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│          Application Layer              │
│  (ETL, Training, Prediction Services)   │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│         Infrastructure Layer            │
│  (TensorFlow, MLflow, Repositories)     │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│            Domain Layer                 │
│  (Entities, Interfaces, Config)         │
└─────────────────────────────────────────┘
```

**SOLID Principles Applied:**
- **S**: Each service has a single responsibility
- **O**: New model architectures via factory pattern
- **L**: All repositories satisfy interface contracts
- **I**: Specific interfaces (IDataExtractor, IModelTrainer)
- **D**: Services depend on abstractions, not implementations

---

## 🧪 Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
```

---

## 📱 Mobile Integration

The exported TFLite model (`model/detector.tflite`) can be integrated into:

- **Android**: Use TensorFlow Lite Android Support Library
- **iOS**: Use TensorFlow Lite Swift/Objective-C APIs
- **Flutter**: Use `tflite_flutter` package
- **React Native**: Use `react-native-tflite` package

Example model size: ~8-15 MB (quantized)

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
