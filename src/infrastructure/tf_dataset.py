import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import pad_sequences

from src.core.entities import ImageAnnotation
from src.core.config import get_config

logger = logging.getLogger(__name__)

class TFDatasetBuilder:
    def __init__(self, images_dir: Path, input_size: Optional[int] = None):
        self._images_dir = images_dir
        self._config = get_config()
        self._input_size = input_size or self._config.model.input_size

    def build_dataset(
        self,
        annotations: List[ImageAnnotation],
        batch_size: int,
        shuffle: bool = True,
        augment: bool = False,
    ) -> tf.data.Dataset:
        image_paths = []
        all_boxes = []
        all_classes = []

        for ann in annotations:
            image_paths.append(str(self._images_dir / ann.file_name))
            
            image_boxes = []
            image_classes = []
            for box in ann.boxes:
                image_boxes.append([
                    box.x_min / ann.width,
                    box.y_min / ann.height,
                    box.x_max / ann.width,
                    box.y_max / ann.height,
                ])
                image_classes.append(box.class_id)
            
            if not image_boxes:
                image_boxes.append([0.0, 0.0, 0.0, 0.0])
                image_classes.append(-1)
            
            all_boxes.append(image_boxes)
            all_classes.append(image_classes)

        # 1. Padding: Most Keras detection heads need a fixed shape (e.g., 1 box per image in your factory)
        # If your model only supports 1 object (based on your GlobalAveragePooling), 
        # we take the first box only.
        final_boxes = np.array([b[0] for b in all_boxes], dtype="float32")
        final_classes = np.array([c[0] for c in all_classes], dtype="int32")

        # 2. Correct from_tensor_slices syntax (passed as a single tuple)
        dataset = tf.data.Dataset.from_tensor_slices((
            image_paths, 
            {"class_output": final_classes, "bbox_output": final_boxes}
        ))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(annotations))

        # 3. Map preprocessing
        dataset = dataset.map(
            lambda x, y: (self._load_and_preprocess(x, augment), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def _load_and_preprocess(self, image_path: tf.Tensor, augment: bool) -> tf.Tensor:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self._input_size, self._input_size])
        image = tf.cast(image, tf.float32) / 255.0

        if augment:
            image = self._apply_augmentation(image)
        return image

    def _apply_augmentation(self, image: tf.Tensor) -> tf.Tensor:
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_flip_left_right(image)
        return tf.clip_by_value(image, 0.0, 1.0)