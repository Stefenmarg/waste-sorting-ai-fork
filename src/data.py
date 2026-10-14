"""tf.data pipeline. Images flow as RAW 0-255 pixels; backbone-specific scaling lives in the model."""
from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import keras
import tensorflow as tf
from keras import layers

from config import Config

logger = logging.getLogger(__name__)

# FIX: single source of truth for image extensions.
# Original: class-weight counting used (.png,.jpg,.jpeg) while XAI sample collection used
# only *.jpg/*.png (so .jpeg files were skipped) - the two disagreed.
IMG_EXTS = (".png", ".jpg", ".jpeg")
AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class Datasets:
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    class_names: List[str]
    class_weights: Optional[Dict[int, float]]


class OptimizedDataLoader:
    def __init__(self, config: Config):
        self.config = config

    def create_augmentation_layer(self) -> keras.Sequential:
        # FIX: RandomBrightness gets an explicit value_range matching the 0-255 data.
        # Original: data was scaled to [0,1] but RandomBrightness defaults to value_range
        # (0, 255), so factor 0.15 added up to ~38 to pixels in [0,1] -> images saturated to
        # white/black during training while val/test images were normal.
        return keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.15),
                layers.RandomZoom(0.15),
                layers.RandomContrast(0.15),
                layers.RandomBrightness(0.15, value_range=(0, 255)),
            ],
            name="augmentation",
        )

    def _load_split(self, split: str, shuffle: bool) -> Tuple[tf.data.Dataset, List[str], List[str]]:
        cfg = self.config
        # FIX: keras.utils.image_dataset_from_directory (Keras 3 location), unbatched.
        # Unbatched so that caching and shuffling happen per IMAGE, not per frozen batch.
        ds = keras.utils.image_dataset_from_directory(
            os.path.join(cfg.data_root, split),
            labels="inferred",
            label_mode="int",
            image_size=cfg.img_size,
            batch_size=None,
            shuffle=shuffle,
            seed=cfg.seed if shuffle else None,
        )
        class_names = list(ds.class_names)
        file_paths = list(ds.file_paths)  # the exact files Keras decided to load
        # FIX: cache compact uint8 images, round (not truncate) when casting.
        # Original: cached float32 /255 images AFTER batching. float32 is 4x the RAM, and
        # caching batches freezes their composition, so every epoch saw identical batches
        # (shuffle only ever happened once, before the cache).
        ds = ds.map(
            lambda x, y: (tf.saturate_cast(tf.round(x), tf.uint8), y),
            num_parallel_calls=AUTOTUNE,
        ).cache()
        return ds, class_names, file_paths

    def load_data(self) -> Datasets:
        cfg = self.config
        logger.info("Loading data...")

        # train is globally shuffled once (so cached order is not class-sorted), then
        # re-shuffled every epoch below.
        train_raw, class_names, train_files = self._load_split("train", shuffle=True)
        val_raw, val_names, _ = self._load_split("val", shuffle=False)
        test_raw, test_names, _ = self._load_split("test", shuffle=False)

        # FIX (new): verify label spaces match across splits.
        # Original: no check. If val/ had a missing or extra class folder, indices silently
        # shifted and metrics were computed against the wrong labels.
        for split, names in (("val", val_names), ("test", test_names)):
            if names != class_names:
                raise ValueError(
                    f"Class mismatch: train={class_names} vs {split}={names}. "
                    "Every split needs the same class folders."
                )
        logger.info("Classes found: %s", class_names)

        augmentation = self.create_augmentation_layer()
        to_float = lambda x, y: (tf.cast(x, tf.float32), y)  # noqa: E731

        train_ds = (
            train_raw.shuffle(cfg.shuffle_buffer, seed=cfg.seed, reshuffle_each_iteration=True)
            .batch(cfg.batch_size)
            .map(to_float, num_parallel_calls=AUTOTUNE)
            .map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
            .prefetch(AUTOTUNE)
        )
        val_ds = val_raw.batch(cfg.batch_size).map(to_float, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
        test_ds = test_raw.batch(cfg.batch_size).map(to_float, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

        class_weights = None
        if cfg.use_class_weights:
            # FIX: count from the files Keras actually loaded.
            # Original: re-listed directories with its own extension filter, which could
            # disagree with the dataset (and so give wrong weights).
            counts = Counter(Path(p).parent.name for p in train_files)
            n_cls, total = len(class_names), sum(counts.values())
            class_weights = {
                i: total / (n_cls * max(counts.get(c, 0), 1)) for i, c in enumerate(class_names)
            }
            logger.info("Class distribution: %s", {c: counts.get(c, 0) for c in class_names})
            logger.info("Class weights: %s", {k: round(v, 3) for k, v in class_weights.items()})

        return Datasets(train_ds, val_ds, test_ds, class_names, class_weights)
