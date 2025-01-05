"""Model Factory - Creates object detection models.

Factory pattern for instantiating various object detection architectures
optimized for mobile deployment.
"""
import logging
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from src.core.config import get_config

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating object detection models."""

    SUPPORTED_ARCHITECTURES = ["mobilenetv2_ssd", "efficientdet_lite"]

    def __init__(self, num_classes: Optional[int] = None):
        """Initialize factory.
        
        Args:
            num_classes: Number of object classes to detect
        """
        self._config = get_config()
        self._num_classes = num_classes or self._config.model.num_classes
        self._input_size = self._config.model.input_size

    def create_model(self, architecture: Optional[str] = None) -> Model:
        """Create a detection model by architecture name.
        
        Args:
            architecture: Model architecture name
            
        Returns:
            Compiled Keras model
        """
        arch = architecture or self._config.model.architecture
        
        if arch == "mobilenetv2_ssd":
            return self._create_mobilenetv2_ssd()
        elif arch == "efficientdet_lite":
            return self._create_efficientdet_lite()
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def _create_mobilenetv2_ssd(self) -> Model:
        """Create a MobileNetV2-based SSD model.
        
        Returns:
            Compiled Keras model for object detection
        """
        logger.info("Creating MobileNetV2-SSD model")
        
        # Base feature extractor
        base_model = keras.applications.MobileNetV2(
            input_shape=(self._input_size, self._input_size, 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False  # Freeze initially

        # Build detection head
        inputs = keras.Input(shape=(self._input_size, self._input_size, 3))
        features = base_model(inputs, training=False)
        
        # Detection layers
        x = layers.Conv2D(256, 3, padding="same", activation="relu")(features)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        
        # Output heads: classification + bounding box regression
        # Simplified output for demonstration
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        
        # Multi-task output
        class_output = layers.Dense(
            self._num_classes, activation="softmax", name="class_output"
        )(x)
        bbox_output = layers.Dense(4, activation="sigmoid", name="bbox_output")(x)

        model = Model(inputs=inputs, outputs=[class_output, bbox_output])
        
        model.compile(
            optimizer=keras.optimizers.Adam(self._config.model.learning_rate),
            loss={
                "class_output": "sparse_categorical_crossentropy",
                "bbox_output": "mse",
            },
            metrics={"class_output": "accuracy"},
        )

        logger.info(f"Model created with {model.count_params():,} parameters")
        return model

    def _create_efficientdet_lite(self) -> Model:
        """Create an EfficientDet-Lite model (simplified).
        
        Returns:
            Compiled Keras model
        """
        logger.info("Creating EfficientDet-Lite model")
        
        # Use EfficientNetB0 as backbone
        base_model = keras.applications.EfficientNetB0(
            input_shape=(self._input_size, self._input_size, 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False

        inputs = keras.Input(shape=(self._input_size, self._input_size, 3))
        features = base_model(inputs, training=False)
        
        # BiFPN-like feature fusion (simplified)
        x = layers.Conv2D(128, 1, padding="same", activation="relu")(features)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        
        class_output = layers.Dense(
            self._num_classes, activation="softmax", name="class_output"
        )(x)
        bbox_output = layers.Dense(4, activation="sigmoid", name="bbox_output")(x)

        model = Model(inputs=inputs, outputs=[class_output, bbox_output])
        
        model.compile(
            optimizer=keras.optimizers.Adam(self._config.model.learning_rate),
            loss={
                "class_output": "sparse_categorical_crossentropy",
                "bbox_output": "mse",
            },
        )

        return model

    def export_tflite(self, model: Model, output_path: Path) -> Path:
        """Export model to TensorFlow Lite format.
        
        Args:
            model: Trained Keras model
            output_path: Directory to save the .tflite file
            
        Returns:
            Path to the exported .tflite file
        """
        output_path.mkdir(parents=True, exist_ok=True)
        tflite_path = output_path / "detector.tflite"

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()
        tflite_path.write_bytes(tflite_model)

        logger.info(f"Exported TFLite model to: {tflite_path}")
        return tflite_path
