import os
import gc
import json
import time
import logging
from pathlib import Path
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

import numpy as np
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
    CSVLogger,
    Callback,
)
from tensorflow.keras.applications import (
    ResNet50,
    EfficientNetV2B0,
    EfficientNetV2B3,
    MobileNetV2,
    DenseNet121,
    ConvNeXtBase,
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from lime import lime_image
import warnings
import cv2


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# GPU CONFIGURATION
# ============================================================================

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ Found {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"training_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class Config:
    """Central configuration with validation"""

    data_root: str = os.environ.get("WASTE_DATA_ROOT", "./data")
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    fine_tune_lr: float = 1e-5
    dropout_rate: float = 0.3
    output_dir: str = "optimized_outputs"

    num_workers: int = min(mp.cpu_count(), 8)

    models_to_train: List[str] = field(
        default_factory=lambda: [
            "ResNet50",
            "EfficientNetV2B0",
            "DenseNet121",
            "MobileNetV2",
        ]
    )

    xai_samples_per_class: int = 3
    xai_methods: List[str] = field(
        default_factory=lambda: [
            "grad_cam",
            "lime",
            "occlusion",
            "integrated_gradients",
        ]
    )

    use_mixed_precision: bool = False
    use_class_weights: bool = True
    use_focal_loss: bool = False
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25

    use_tta: bool = False
    tta_steps: int = 5

    warmup_epochs: int = 3
    gradient_clip_value: float = 1.0

    # Training phases
    phase1_epochs: int = 15
    fine_tune_from_layer_percent: float = 0.8  # Freeze 80% of base layers

    # Occlusion settings
    occlusion_size: int = 30
    occlusion_stride: int = 15
    occlusion_batch_size: int = 32

    def __post_init__(self):
        """Validate configuration and create directories"""
        # Validate data directory exists
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(
                f"Data directory not found: {self.data_root}\n"
                f"Please set WASTE_DATA_ROOT environment variable or place data in ./data/\n"
                f"Expected structure:\n"
                f"  {self.data_root}/\n"
                f"    ├── train/class_name/\n"
                f"    ├── val/class_name/\n"
                f"    └── test/class_name/"
            )

        # Check for required subdirectories
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(self.data_root, split)
            if not os.path.exists(split_dir):
                raise FileNotFoundError(
                    f"Missing required directory: {split_dir}\n"
                    f"Expected structure: {self.data_root}/{{train,val,test}}/class_name/"
                )

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        for subdir in [
            "models",
            "confusion_matrices",
            "training_curves",
            "xai_visualizations",
            "per_class_metrics",
            "logs",
            "checkpoints",
        ]:
            os.makedirs(f"{self.output_dir}/{subdir}", exist_ok=True)

        logger.info("✓ Configuration validated")
        logger.info(f"  Data root: {self.data_root}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Models to train: {', '.join(self.models_to_train)}")


config = Config()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def set_seed(seed: int = 42):
    """Ensure reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    logger.info(f"✓ Random seed set to {seed}")


def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for handling class imbalance"""

    def focal_loss_fixed(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * keras.backend.log(y_pred)
        weight = alpha * y_true * keras.backend.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return keras.backend.sum(loss, axis=-1)

    return focal_loss_fixed


# ============================================================================
# CALLBACKS
# ============================================================================


class WarmUpLearningRate(Callback):
    """Learning rate warmup callback"""

    def __init__(self, warmup_epochs, initial_lr, target_lr, verbose=0):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (
                epoch / self.warmup_epochs
            )
            self.model.optimizer.learning_rate.assign(lr)
            if self.verbose:
                logger.info(f"Epoch {epoch + 1}: Warmup LR = {lr:.6f}")


class ProgressCallback(Callback):
    """Custom progress callback with epoch timing"""

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        logger.info(f"\n{self.model_name} - Epoch {epoch + 1}")

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.epoch_start_time
        logger.info(
            f"{self.model_name} - Epoch {epoch + 1} completed in {duration:.1f}s - "
            f"loss: {logs.get('loss', 0):.4f} - "
            f"accuracy: {logs.get('accuracy', 0):.4f} - "
            f"val_loss: {logs.get('val_loss', 0):.4f} - "
            f"val_accuracy: {logs.get('val_accuracy', 0):.4f}"
        )


# ============================================================================
# DATA LOADER
# ============================================================================


class OptimizedDataLoader:
    """Optimized data loading with tf.data pipeline"""

    def __init__(self, config: Config):
        self.config = config
        self.AUTOTUNE = tf.data.AUTOTUNE

    def create_augmentation_layer(self) -> keras.Sequential:
        """Moderate augmentation for stability"""
        return keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.15),
                layers.RandomZoom(0.15),
                layers.RandomContrast(0.15),
                layers.RandomBrightness(0.15),
            ],
            name="augmentation",
        )

    def load_data(self) -> Tuple:
        """Load data with optimized pipeline"""
        logger.info("Loading data with optimized pipeline...")

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(self.config.data_root, "train"),
            image_size=self.config.img_size,
            batch_size=self.config.batch_size,
            shuffle=True,
            seed=42,
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(self.config.data_root, "val"),
            image_size=self.config.img_size,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(self.config.data_root, "test"),
            image_size=self.config.img_size,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        class_names = train_ds.class_names
        logger.info(f"✓ Classes found: {class_names}")

        normalization_layer = layers.Rescaling(1.0 / 255)
        augmentation = self.create_augmentation_layer()

        train_ds = train_ds.map(
            lambda x, y: (normalization_layer(x), y), num_parallel_calls=self.AUTOTUNE
        ).cache()  # Cache normalized data

        train_ds = train_ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=self.AUTOTUNE,
        ).prefetch(buffer_size=self.AUTOTUNE)

        val_ds = (
            val_ds.map(
                lambda x, y: (normalization_layer(x), y),
                num_parallel_calls=self.AUTOTUNE,
            )
            .cache()
            .prefetch(buffer_size=self.AUTOTUNE)
        )

        test_ds = (
            test_ds.map(
                lambda x, y: (normalization_layer(x), y),
                num_parallel_calls=self.AUTOTUNE,
            )
            .cache()
            .prefetch(buffer_size=self.AUTOTUNE)
        )

        logger.info("✓ Data pipeline optimized with caching and prefetching")

        class_weights = None
        if self.config.use_class_weights:
            train_dir = os.path.join(self.config.data_root, "train")
            class_counts = {}

            for class_idx, class_name in enumerate(class_names):
                class_dir = os.path.join(train_dir, class_name)
                count = len(
                    [
                        f
                        for f in os.listdir(class_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    ]
                )
                class_counts[class_idx] = count

            total = sum(class_counts.values())
            class_weights = {
                i: total / (len(class_counts) * count)
                for i, count in class_counts.items()
            }

            logger.info(
                f"✓ Class distribution: {dict(zip(class_names, class_counts.values()))}"
            )
            logger.info(f"✓ Class weights: {class_weights}")

        return train_ds, val_ds, test_ds, class_names, class_weights


# ============================================================================
# MODEL BUILDER
# ============================================================================


class ModelBuilder:
    """Advanced model builder with progressive fine-tuning"""

    BASE_MODELS = {
        "ResNet50": ResNet50,
        "EfficientNetV2B0": EfficientNetV2B0,
        "EfficientNetV2B3": EfficientNetV2B3,
        "MobileNetV2": MobileNetV2,
        "DenseNet121": DenseNet121,
        "ConvNeXtBase": ConvNeXtBase,
    }

    @staticmethod
    def build_model(arch: str, num_classes: int, config: Config) -> Tuple[Model, Model]:
        """Build model with improved architecture"""
        logger.info(f"Building {arch} model...")

        base_model = None
        for attempt in range(3):
            try:
                base_model = ModelBuilder.BASE_MODELS[arch](
                    include_top=False,
                    weights="imagenet",
                    input_shape=config.img_size + (3,),
                )
                logger.info(f"✓ Loaded pretrained ImageNet weights for {arch}")
                break
            except Exception as e:
                if attempt < 2:
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in 2s..."
                    )
                    time.sleep(2)
                else:
                    logger.warning(
                        f"Failed to load pretrained weights after 3 attempts: {e}"
                    )
                    logger.warning("Training from scratch (random initialization)...")
                    base_model = ModelBuilder.BASE_MODELS[arch](
                        include_top=False,
                        weights=None,
                        input_shape=config.img_size + (3,),
                    )

        base_model.trainable = False

        inputs = keras.Input(shape=config.img_size + (3,))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(config.dropout_rate)(x)
        x = layers.Dense(
            512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(config.dropout_rate * 0.5)(x)
        x = layers.Dense(
            256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(config.dropout_rate * 0.5)(x)
        outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

        model = keras.Model(inputs, outputs, name=arch)

        if config.use_focal_loss:
            loss = focal_loss(
                gamma=config.focal_loss_gamma, alpha=config.focal_loss_alpha
            )
            logger.info(
                f"✓ Using Focal Loss (gamma={config.focal_loss_gamma}, alpha={config.focal_loss_alpha})"
            )
        else:
            loss = "sparse_categorical_crossentropy"

        optimizer = keras.optimizers.Adam(
            learning_rate=config.learning_rate, clipnorm=config.gradient_clip_value
        )

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                "accuracy",
                keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
                keras.metrics.SparseCategoricalCrossentropy(name="ce_loss"),
            ],
        )

        logger.info(f"\n{arch} Architecture Summary:")
        model.summary(print_fn=lambda x: logger.info(x))

        total_params = model.count_params()
        trainable_params = sum(
            [keras.backend.count_params(w) for w in model.trainable_weights]
        )
        logger.info(f"Total params: {total_params:,}")
        logger.info(f"Trainable params: {trainable_params:,}")
        logger.info(f"Non-trainable params: {total_params - trainable_params:,}")

        return model, base_model


# ============================================================================
# TEST-TIME AUGMENTATION
# ============================================================================


def predict_with_tta(model, images, config: Config) -> np.ndarray:
    """Predict with test-time augmentation"""
    if not config.use_tta:
        return model.predict(images, verbose=0)

    predictions = []

    predictions.append(model.predict(images, verbose=0))

    for _ in range(config.tta_steps - 1):
        aug_images = images.numpy() if hasattr(images, "numpy") else images

        if np.random.random() > 0.5:
            aug_images = np.flip(aug_images, axis=2)

        # Random brightness adjustment
        brightness_factor = 0.9 + np.random.random() * 0.2
        aug_images = aug_images * brightness_factor
        aug_images = np.clip(aug_images, 0, 1)

        predictions.append(model.predict(aug_images, verbose=0))

    # Average predictions
    return np.mean(predictions, axis=0)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================


def train_single_model(args):
    """Train a single model with two-phase approach"""
    arch, train_ds, val_ds, test_ds, class_names, class_weights, config = args

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus, True)
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")

    logger.info(f"\n{'=' * 80}\nTraining {arch}...\n{'=' * 80}")

    model, base_model = ModelBuilder.build_model(arch, len(class_names), config)

    warmup_callback = WarmUpLearningRate(
        warmup_epochs=config.warmup_epochs,
        initial_lr=config.learning_rate * 0.1,
        target_lr=config.learning_rate,
        verbose=1,
    )

    progress_callback = ProgressCallback(arch)

    callbacks = [
        warmup_callback,
        progress_callback,
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
            cooldown=2,
        ),
        ModelCheckpoint(
            f"{config.output_dir}/checkpoints/{arch}_phase1_best.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        TensorBoard(
            log_dir=f"{config.output_dir}/logs/{arch}_{datetime.now():%Y%m%d-%H%M%S}",
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        ),
        CSVLogger(
            f"{config.output_dir}/logs/{arch}_training_log.csv",
            separator=",",
            append=False,
        ),
    ]

    logger.info(f"\n{arch}: PHASE 1 - Training classifier head (frozen backbone)...")
    logger.info(f"Training for {config.phase1_epochs} epochs")

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.phase1_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0,
    )

    logger.info(f"\n{arch}: PHASE 2 - Fine-tuning with unfrozen backbone...")
    base_model.trainable = True

    freeze_until = int(len(base_model.layers) * config.fine_tune_from_layer_percent)
    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
    logger.info(
        f"Unfrozen {trainable_layers}/{len(base_model.layers)} layers in base model"
    )

    # Recompile with lower learning rate
    if config.use_focal_loss:
        loss = focal_loss(gamma=config.focal_loss_gamma, alpha=config.focal_loss_alpha)
    else:
        loss = "sparse_categorical_crossentropy"

    optimizer_finetune = keras.optimizers.Adam(
        learning_rate=config.fine_tune_lr, clipnorm=config.gradient_clip_value
    )

    model.compile(
        optimizer=optimizer_finetune,
        loss=loss,
        metrics=[
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
            keras.metrics.SparseCategoricalCrossentropy(name="ce_loss"),
        ],
    )

    callbacks_phase2 = [
        progress_callback,
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.0001,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=5,
            min_lr=1e-8,
            verbose=1,
            cooldown=2,
        ),
        ModelCheckpoint(
            f"{config.output_dir}/models/{arch}_best_model.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        TensorBoard(
            log_dir=f"{config.output_dir}/logs/{arch}_finetune_{datetime.now():%Y%m%d-%H%M%S}",
            histogram_freq=1,
        ),
    ]

    phase2_epochs = config.epochs - config.phase1_epochs
    logger.info(f"Fine-tuning for {phase2_epochs} additional epochs")

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.phase1_epochs + phase2_epochs,
        initial_epoch=config.phase1_epochs,
        class_weight=class_weights,
        callbacks=callbacks_phase2,
        verbose=0,
    )

    logger.info(f"\n{arch}: Evaluating on test set...")

    y_true = []
    y_pred_list = []

    for x_batch, y_batch in tqdm(test_ds, desc=f"Evaluating {arch}"):
        y_true.extend(y_batch.numpy())

        if config.use_tta:
            predictions = predict_with_tta(model, x_batch, config)
        else:
            predictions = model.predict(x_batch, verbose=0)

        y_pred_list.extend(predictions)

    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_list)
    y_pred = np.argmax(y_pred_proba, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0
    )

    logger.info(f"\n{arch} Test Results:")
    logger.info(
        f"  Accuracy: {report['accuracy']:.4f} ({report['accuracy'] * 100:.2f}%)"
    )
    logger.info(f"  Macro Avg Precision: {report['macro avg']['precision']:.4f}")
    logger.info(f"  Macro Avg Recall: {report['macro avg']['recall']:.4f}")
    logger.info(f"  Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
    logger.info(f"  Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")

    combined_history = {
        "loss": history1.history["loss"] + history2.history["loss"],
        "val_loss": history1.history["val_loss"] + history2.history["val_loss"],
        "accuracy": history1.history["accuracy"] + history2.history["accuracy"],
        "val_accuracy": history1.history["val_accuracy"]
        + history2.history["val_accuracy"],
    }

    keras.backend.clear_session()
    del model
    del base_model
    gc.collect()

    return {
        "arch": arch,
        "model_path": f"{config.output_dir}/models/{arch}_best_model.h5",
        "history": combined_history,
        "accuracy": report["accuracy"],
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "cm": cm,
        "report": report,
        "y_pred": y_pred,
        "y_true": y_true,
        "y_pred_proba": y_pred_proba,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "per_class_support": support,
    }


# ============================================================================
# SEQUENTIAL TRAINER (PROPER IMPLEMENTATION)
# ============================================================================


class SequentialTrainer:
    """Sequential model training (for single GPU)"""

    def __init__(self, config: Config):
        self.config = config

    def train_all_models(self, train_ds, val_ds, test_ds, class_names, class_weights):
        """Train multiple models sequentially"""
        logger.info(
            f"\nTraining {len(self.config.models_to_train)} models sequentially..."
        )
        logger.info("Note: Sequential training is optimal for single GPU setups")

        results = {}

        for i, arch in enumerate(self.config.models_to_train, 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(
                f"Training model {i}/{len(self.config.models_to_train)}: {arch}"
            )
            logger.info(f"{'=' * 80}\n")

            args = (
                arch,
                train_ds,
                val_ds,
                test_ds,
                class_names,
                class_weights,
                self.config,
            )
            result = train_single_model(args)
            results[result["arch"]] = result

            gc.collect()
            if gpus:
                tf.keras.backend.clear_session()

        logger.info("✓ All models trained successfully")
        return results


# ============================================================================
# XAI EXPLAINER
# ============================================================================


class XAIExplainer:
    """Comprehensive XAI explanation generator with optimizations"""

    def __init__(self, model_path: str, class_names: List[str], config: Config):
        self.model = keras.models.load_model(model_path, compile=False)
        self.class_names = class_names
        self.config = config
        self.grad_cam_model = self._build_grad_cam_model()
        self.lime_explainer = lime_image.LimeImageExplainer()

    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, "grad_cam_model") and self.grad_cam_model:
            del self.grad_cam_model
        if hasattr(self, "model"):
            del self.model
        keras.backend.clear_session()

    def _build_grad_cam_model(self):
        """Build Grad-CAM model"""
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                last_conv_layer = layer.name
                break

        if last_conv_layer is None:
            logger.warning("No convolutional layer found for Grad-CAM")
            return None

        logger.info(f"Using layer '{last_conv_layer}' for Grad-CAM")

        return keras.Model(
            [self.model.inputs],
            [self.model.get_layer(last_conv_layer).output, self.model.output],
        )

    def grad_cam(self, img_array: np.ndarray, class_idx: int) -> np.ndarray:
        """Generate Grad-CAM heatmap"""
        if self.grad_cam_model is None:
            return np.zeros(self.config.img_size)

        img_array_expanded = np.expand_dims(img_array, axis=0)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_cam_model(img_array_expanded)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)

        heatmap = cv2.resize(heatmap.numpy(), self.config.img_size)

        return heatmap

    def occlusion_sensitivity(
        self, img_array: np.ndarray, class_idx: int
    ) -> np.ndarray:
        """Generate occlusion sensitivity map with batched predictions (OPTIMIZED)"""
        h, w = self.config.img_size

        baseline_pred = self.model.predict(np.expand_dims(img_array, 0), verbose=0)[
            class_idx
        ]

        occluded_images = []
        positions = []

        for y in range(0, h - self.config.occlusion_size, self.config.occlusion_stride):
            for x in range(
                0, w - self.config.occlusion_size, self.config.occlusion_stride
            ):
                occluded = img_array.copy()
                occluded[
                    y : y + self.config.occlusion_size,
                    x : x + self.config.occlusion_size,
                ] = 0
                occluded_images.append(occluded)
                positions.append((y, x))

        occluded_batch = np.array(occluded_images)
        predictions = self.model.predict(
            occluded_batch, batch_size=self.config.occlusion_batch_size, verbose=0
        )

        sensitivity_map = np.zeros((h, w))
        for (y, x), pred in zip(positions, predictions):
            sensitivity = baseline_pred - pred[class_idx]
            sensitivity_map[
                y : y + self.config.occlusion_size, x : x + self.config.occlusion_size
            ] = max(sensitivity_map[y, x], sensitivity)

        if sensitivity_map.max() > sensitivity_map.min():
            sensitivity_map = (sensitivity_map - sensitivity_map.min()) / (
                sensitivity_map.max() - sensitivity_map.min()
            )

        return sensitivity_map

    def integrated_gradients(
        self, img_array: np.ndarray, class_idx: int, steps: int = 50
    ) -> np.ndarray:
        """Generate Integrated Gradients attribution"""
        baseline = np.zeros_like(img_array)

        alphas = np.linspace(0, 1, steps)
        interpolated_images = np.array(
            [baseline + alpha * (img_array - baseline) for alpha in alphas]
        )

        gradients = []
        for img in interpolated_images:
            img_batch = np.expand_dims(img, 0)
            with tf.GradientTape() as tape:
                img_tensor = tf.Variable(img_batch, dtype=tf.float32)
                predictions = self.model(img_tensor)
                loss = predictions[:, class_idx]

            grads = tape.gradient(loss, img_tensor)
            gradients.append(grads.numpy())

        avg_gradients = np.mean(gradients, axis=0)
        integrated_grads = (img_array - baseline) * avg_gradients

        attribution = np.sum(np.abs(integrated_grads), axis=-1)

        if attribution.max() > attribution.min():
            attribution = (attribution - attribution.min()) / (
                attribution.max() - attribution.min()
            )

        return attribution

    def lime_explain(self, img_array: np.ndarray, class_idx: int) -> np.ndarray:
        """Generate LIME explanation"""

        def predict_fn(images):
            return self.model.predict(images, verbose=0)

        img_uint8 = (img_array * 255).astype(np.uint8)

        explanation = self.lime_explainer.explain_instance(
            img_uint8, predict_fn, top_labels=1, num_samples=300, random_seed=42
        )

        temp, mask = explanation.get_image_and_mask(
            class_idx, positive_only=True, num_features=5, hide_rest=False
        )

        return mask.astype(np.float32)

    def explain_all(self, img_array: np.ndarray, true_class: str) -> Dict:
        """Generate all explanations for an image"""
        pred_proba = self.model.predict(np.expand_dims(img_array, 0), verbose=0)
        pred_idx = int(np.argmax(pred_proba))
        pred_class = self.class_names[pred_idx]
        confidence = float(pred_proba[pred_idx])

        explanations = {
            "true_class": true_class,
            "pred_class": pred_class,
            "confidence": confidence,
        }

        if "grad_cam" in self.config.xai_methods:
            try:
                explanations["grad_cam"] = self.grad_cam(img_array, pred_idx)
            except Exception as e:
                logger.warning(f"Grad-CAM failed: {e}")
                explanations["grad_cam"] = None

        if "occlusion" in self.config.xai_methods:
            try:
                explanations["occlusion"] = self.occlusion_sensitivity(
                    img_array, pred_idx
                )
            except Exception as e:
                logger.warning(f"Occlusion failed: {e}")
                explanations["occlusion"] = None

        if "integrated_gradients" in self.config.xai_methods:
            try:
                explanations["integrated_gradients"] = self.integrated_gradients(
                    img_array, pred_idx
                )
            except Exception as e:
                logger.warning(f"Integrated Gradients failed: {e}")
                explanations["integrated_gradients"] = None

        if "lime" in self.config.xai_methods:
            try:
                explanations["lime"] = self.lime_explain(img_array, pred_idx)
            except Exception as e:
                logger.warning(f"LIME failed: {e}")
                explanations["lime"] = None

        return explanations


# ============================================================================
# XAI VISUALIZATION (MULTIPROCESSING)
# ============================================================================


def process_single_xai_worker(args):
    """Worker function for multiprocessing XAI generation"""
    model_path, img_path, class_name, sample_idx, output_dir, class_names, config = args

    try:
        explainer = XAIExplainer(model_path, class_names, config)

        img = Image.open(img_path).convert("RGB").resize(config.img_size)
        img_array = np.array(img) / 255.0

        explanations = explainer.explain_all(img_array, class_name)

        n_methods = sum(
            1
            for k in explanations
            if k not in ["true_class", "pred_class", "confidence"]
            and explanations[k] is not None
        )

        fig, axes = plt.subplots(1, n_methods + 1, figsize=(5 * (n_methods + 1), 5))
        if n_methods == 0:
            axes = [axes]

        axes.imshow(img)
        axes.set_title(
            f"Original\nTrue: {explanations['true_class']}\n"
            f"Pred: {explanations['pred_class']}\n"
            f"Conf: {explanations['confidence']:.3f}",
            fontsize=10,
        )
        axes.axis("off")

        ax_idx = 1
        for method in ["grad_cam", "occlusion", "integrated_gradients", "lime"]:
            if method in explanations and explanations[method] is not None:
                axes[ax_idx].imshow(img)

                if method == "lime":
                    axes[ax_idx].imshow(explanations[method], cmap="jet", alpha=0.3)
                else:
                    axes[ax_idx].imshow(explanations[method], cmap="jet", alpha=0.5)

                axes[ax_idx].set_title(method.replace("_", " ").title(), fontsize=10)
                axes[ax_idx].axis("off")
                ax_idx += 1

        plt.tight_layout()
        output_path = f"{output_dir}/{class_name}_sample_{sample_idx}_xai.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        del explainer
        keras.backend.clear_session()

        return output_path

    except Exception as e:
        logger.error(f"XAI generation failed for {img_path}: {e}")
        return None


def generate_xai_visualizations_parallel(
    model_path: str,
    test_dir: str,
    class_names: List[str],
    model_name: str,
    config: Config,
):
    """Generate XAI visualizations using multiprocessing (FIXED)"""
    logger.info(f"Generating XAI visualizations for {model_name}...")

    output_dir = f"{config.output_dir}/xai_visualizations/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Collect sample paths
    sample_paths = []
    for cls in class_names:
        cls_path = Path(test_dir) / cls
        if cls_path.exists():
            imgs = sorted(list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png")))
            for idx, img_path in enumerate(imgs[: config.xai_samples_per_class]):
                sample_paths.append((str(img_path), cls, idx))

    logger.info(
        f"Processing {len(sample_paths)} images with {len(config.xai_methods)} XAI methods"
    )

    args_list = [
        (model_path, img_path, cls, idx, output_dir, class_names, config)
        for img_path, cls, idx in sample_paths
    ]

    num_processes = min(config.num_workers, len(sample_paths))
    logger.info(f"Using {num_processes} parallel workers")

    with mp.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_xai_worker, args_list),
                total=len(args_list),
                desc="Generating XAI visualizations",
            )
        )

    successful = sum(1 for r in results if r is not None)
    logger.info(f"✓ XAI visualization complete: {successful}/{len(results)} successful")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def plot_training_curves(results: Dict, config: Config):
    """Plot individual training curves for each model"""
    logger.info("Generating individual training curves...")

    # Create individual training curve for each model
    for name, res in results.items():
        history = res["history"]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Loss plot
        epochs = range(1, len(history["loss"]) + 1)
        axes[0].plot(
            epochs,
            history["loss"],
            "b-",
            label="Training Loss",
            linewidth=2.5,
            marker="o",
            markersize=4,
        )
        axes[0].plot(
            epochs,
            history["val_loss"],
            "r--",
            label="Validation Loss",
            linewidth=2.5,
            marker="s",
            markersize=4,
        )
        axes[0].set_title(
            f"{name}\nLoss During Training", fontsize=16, fontweight="bold", pad=20
        )
        axes[0].set_xlabel("Training Epoch", fontsize=13)
        axes[0].set_ylabel("Loss Value (Lower is Better)", fontsize=13)
        axes[0].legend(fontsize=11, loc="upper right")
        axes[0].grid(True, alpha=0.3, linestyle="--")

        # Accuracy plot
        axes[1].plot(
            epochs,
            history["accuracy"],
            "b-",
            label="Training Accuracy",
            linewidth=2.5,
            marker="o",
            markersize=4,
        )
        axes[1].plot(
            epochs,
            history["val_accuracy"],
            "r--",
            label="Validation Accuracy",
            linewidth=2.5,
            marker="s",
            markersize=4,
        )
        axes[1].set_title(
            f"{name}\nAccuracy During Training", fontsize=16, fontweight="bold", pad=20
        )
        axes[1].set_xlabel("Training Epoch", fontsize=13)
        axes[1].set_ylabel("Accuracy (Higher is Better)", fontsize=13)
        axes[1].set_ylim([0, 1.05])
        axes[1].legend(fontsize=11, loc="lower right")
        axes[1].grid(True, alpha=0.3, linestyle="--")

        # Add final values as text
        final_train_acc = history["accuracy"][-1]
        final_val_acc = history["val_accuracy"][-1]
        axes[1].text(
            0.02,
            0.98,
            f"Final Training: {final_train_acc*100:.2f}%\nFinal Validation: {final_val_acc*100:.2f}%",
            transform=axes[1].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        safe_name = name.replace("/", "_").replace(" ", "_")
        plt.savefig(
            f"{config.output_dir}/training_curves/{safe_name}_training.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    logger.info("✓ Individual training curves saved")


def plot_confusion_matrices(results: Dict, class_names: List[str], config: Config):
    """Plot individual confusion matrices for each model"""
    logger.info("Generating individual confusion matrices...")

    for name, res in results.items():
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        cm = res["cm"]
        cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".1%",
            cmap="YlGnBu",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={"label": "Percentage of Predictions"},
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="gray",
        )

        # Title with explanation
        ax.set_title(
            f"{name} - Prediction Accuracy Matrix\n"
            f"Overall Accuracy: {res['accuracy'] * 100:.2f}%\n"
            f"(Darker colors = More predictions)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_ylabel("Actual Waste Type", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Waste Type", fontsize=13, fontweight="bold")

        # Add explanation text
        explanation = "How to read: Each row shows how items of one type were classified.\nDiagonal (top-left to bottom-right) shows correct predictions."
        plt.figtext(
            0.5, 0.01, explanation, ha="center", fontsize=10, style="italic", wrap=True
        )

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        safe_name = name.replace("/", "_").replace(" ", "_")
        plt.savefig(
            f"{config.output_dir}/confusion_matrices/{safe_name}_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    logger.info("✓ Individual confusion matrices saved")


def plot_per_class_metrics(results: Dict, class_names: List[str], config: Config):
    """Plot individual per-class metrics for each model"""
    logger.info("Generating per-class metrics...")

    # Create individual charts for each model
    for name, res in results.items():
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        x = np.arange(len(class_names))
        width = 0.25

        precision = res["per_class_precision"]
        recall = res["per_class_recall"]
        f1 = res["per_class_f1"]

        # Create grouped bars
        bars1 = ax.bar(
            x - width,
            precision,
            width,
            label="Precision (Correctness)",
            color="#3498db",
            edgecolor="black",
            linewidth=1.2,
        )
        bars2 = ax.bar(
            x,
            recall,
            width,
            label="Recall (Completeness)",
            color="#2ecc71",
            edgecolor="black",
            linewidth=1.2,
        )
        bars3 = ax.bar(
            x + width,
            f1,
            width,
            label="F1-Score (Overall)",
            color="#e74c3c",
            edgecolor="black",
            linewidth=1.2,
        )

        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height*100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)

        ax.set_title(
            f"{name}\nPerformance by Waste Type", fontsize=16, fontweight="bold", pad=20
        )
        ax.set_ylabel("Score (Higher is Better)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Waste Type", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=11)
        ax.legend(fontsize=12, loc="lower right")
        ax.set_ylim([0, 1.15])
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.axhline(y=0.8, color="gray", linestyle=":", alpha=0.5, label="80% threshold")

        plt.tight_layout()
        safe_name = name.replace("/", "_").replace(" ", "_")
        plt.savefig(
            f"{config.output_dir}/per_class_metrics/{safe_name}_per_class.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    logger.info("✓ Per-class metrics saved")


def plot_model_comparison_overview(results: Dict, config: Config):
    """Create easy-to-understand comparison charts for all models"""
    logger.info("Generating model comparison overview...")

    # 1. Overall Accuracy Comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    models = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in models]

    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]
    models_sorted = [models[i] for i in sorted_indices]
    accuracies_sorted = [accuracies[i] for i in sorted_indices]

    # Create color gradient (best = green, worst = yellow)
    colors = plt.cm.RdYlGn(np.linspace(0.4, 0.9, len(models)))

    bars = ax.barh(
        models_sorted, accuracies_sorted, color=colors, edgecolor="black", linewidth=2
    )

    # Add percentage labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies_sorted)):
        ax.text(
            acc + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.2f}%",
            va="center",
            fontweight="bold",
            fontsize=12,
        )

        # Add rank medal
        rank = i + 1
        medal = (
            "🥇"
            if rank == 1
            else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        )
        ax.text(2, bar.get_y() + bar.get_height() / 2, medal, va="center", fontsize=14)

    ax.set_xlabel("Accuracy Percentage", fontsize=13, fontweight="bold")
    ax.set_title(
        "Model Accuracy Ranking\n(Which model makes the most correct predictions?)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlim([0, 105])
    ax.grid(True, alpha=0.3, axis="x", linestyle="--")

    plt.tight_layout()
    plt.savefig(
        f"{config.output_dir}/model_comparison_accuracy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Multi-metric Comparison
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(models))
    width = 0.2

    for i, model in enumerate(models):
        res = results[model]
        values = [
            res["accuracy"] * 100,
            res["weighted_avg"]["precision"] * 100,
            res["weighted_avg"]["recall"] * 100,
            res["weighted_avg"]["f1-score"] * 100,
        ]

        positions = x + (i - len(models) / 2) * width + width / 2
        bars = ax.bar(
            positions, values, width, label=model, edgecolor="black", linewidth=1
        )

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_ylabel("Score Percentage", fontsize=13, fontweight="bold")
    ax.set_title(
        "Complete Model Performance Comparison\n(All Key Metrics)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=12, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_ylim([0, 110])
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.axhline(
        y=80,
        color="red",
        linestyle=":",
        alpha=0.5,
        linewidth=2,
        label="80% Good Performance",
    )
    ax.axhline(
        y=90,
        color="green",
        linestyle=":",
        alpha=0.5,
        linewidth=2,
        label="90% Excellent Performance",
    )

    plt.tight_layout()
    plt.savefig(
        f"{config.output_dir}/model_comparison_all_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 3. Performance Summary Cards
    fig = plt.figure(figsize=(16, 10))

    for idx, (name, res) in enumerate(results.items()):
        ax = plt.subplot(2, 2, idx + 1)
        ax.axis("off")

        # Create a summary card
        summary_text = f"""
        {name}
        {'=' * 50}

        OVERALL PERFORMANCE
        ✓ Accuracy: {res['accuracy']*100:.2f}%
        ✓ Precision: {res['weighted_avg']['precision']*100:.2f}%
        ✓ Recall: {res['weighted_avg']['recall']*100:.2f}%
        ✓ F1-Score: {res['weighted_avg']['f1-score']*100:.2f}%

        WHAT THIS MEANS:
        • Out of 100 items, {int(res['accuracy']*100)} are correctly sorted
        • When it says "recyclable", it's right {res['weighted_avg']['precision']*100:.1f}% of the time
        • It finds {res['weighted_avg']['recall']*100:.1f}% of all recyclable items

        TRAINING STABILITY
        Final Training Accuracy: {res['history']['accuracy'][-1]*100:.2f}%
        Final Validation Accuracy: {res['history']['val_accuracy'][-1]*100:.2f}%
        """

        # Determine performance level
        acc = res["accuracy"]
        if acc >= 0.95:
            grade = "EXCELLENT ⭐⭐⭐⭐⭐"
            color = "#27ae60"
        elif acc >= 0.90:
            grade = "VERY GOOD ⭐⭐⭐⭐"
            color = "#2ecc71"
        elif acc >= 0.85:
            grade = "GOOD ⭐⭐⭐"
            color = "#f39c12"
        elif acc >= 0.80:
            grade = "ACCEPTABLE ⭐⭐"
            color = "#e67e22"
        else:
            grade = "NEEDS IMPROVEMENT ⭐"
            color = "#e74c3c"

        ax.text(
            0.5,
            0.95,
            grade,
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            transform=ax.transAxes,
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=color,
                alpha=0.3,
                edgecolor=color,
                linewidth=3,
            ),
        )

        ax.text(
            0.05,
            0.85,
            summary_text,
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
            transform=ax.transAxes,
            bbox=dict(
                boxstyle="round,pad=1",
                facecolor="lightgray",
                alpha=0.2,
                edgecolor="black",
                linewidth=1,
            ),
        )

    plt.suptitle(
        "Model Performance Summary Cards\n(Quick Overview of Each Model)",
        fontsize=18,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(
        f"{config.output_dir}/model_summary_cards.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    logger.info("✓ Model comparison overview saved")


# ============================================================================
# REPORTING
# ============================================================================


def save_comprehensive_report(results: Dict, class_names: List[str], config: Config):
    """Save text and JSON reports"""
    logger.info("Generating comprehensive report...")

    report_path = f"{config.output_dir}/comprehensive_report.txt"

    with open(report_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("IMPROVED WASTE SORTING SYSTEM - COMPREHENSIVE REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n\n")

        f.write("CONFIGURATION:\n")
        f.write(f"  Data Root: {config.data_root}\n")
        f.write(f"  Image Size: {config.img_size}\n")
        f.write(f"  Batch Size: {config.batch_size}\n")
        f.write(f"  Total Epochs: {config.epochs}\n")
        f.write(f"  Phase 1 Epochs: {config.phase1_epochs}\n")
        f.write(f"  Phase 2 Epochs: {config.epochs - config.phase1_epochs}\n")
        f.write(f"  Learning Rate (Phase 1): {config.learning_rate}\n")
        f.write(f"  Learning Rate (Phase 2): {config.fine_tune_lr}\n")
        f.write(f"  Dropout Rate: {config.dropout_rate}\n")
        f.write(f"  Use Focal Loss: {config.use_focal_loss}\n")
        if config.use_focal_loss:
            f.write(f"  Focal Loss Gamma: {config.focal_loss_gamma}\n")
            f.write(f"  Focal Loss Alpha: {config.focal_loss_alpha}\n")
        f.write(f"  Use Class Weights: {config.use_class_weights}\n")
        f.write(f"  Use TTA: {config.use_tta}\n")
        f.write(f"  Mixed Precision: {config.use_mixed_precision}\n")
        f.write(f"  Classes: {', '.join(class_names)}\n\n")

        f.write("MODEL COMPARISON:\n")
        f.write(
            f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n"
        )
        f.write("-" * 100 + "\n")

        sorted_results = sorted(
            results.items(), key=lambda x: x[1]["accuracy"], reverse=True
        )

        for rank, (name, res) in enumerate(sorted_results, 1):
            f.write(
                f"{rank}. {name:<22} "
                f"{res['accuracy']:<12.4f} "
                f"{res['weighted_avg']['precision']:<12.4f} "
                f"{res['weighted_avg']['recall']:<12.4f} "
                f"{res['weighted_avg']['f1-score']:<12.4f}\n"
            )

        f.write("\n" + "=" * 100 + "\n\n")

        for name, res in results.items():
            f.write(f"\n{'=' * 100}\n")
            f.write(f"MODEL: {name}\n")
            f.write(f"{'=' * 100}\n\n")

            f.write("Overall Metrics:\n")
            f.write(
                f"  Accuracy: {res['accuracy']:.4f} ({res['accuracy'] * 100:.2f}%)\n"
            )
            f.write(f"  Macro Avg Precision: {res['macro_avg']['precision']:.4f}\n")
            f.write(f"  Macro Avg Recall: {res['macro_avg']['recall']:.4f}\n")
            f.write(f"  Macro Avg F1-Score: {res['macro_avg']['f1-score']:.4f}\n")
            f.write(
                f"  Weighted Avg Precision: {res['weighted_avg']['precision']:.4f}\n"
            )
            f.write(f"  Weighted Avg Recall: {res['weighted_avg']['recall']:.4f}\n")
            f.write(
                f"  Weighted Avg F1-Score: {res['weighted_avg']['f1-score']:.4f}\n\n"
            )

            f.write("Per-Class Performance:\n")
            f.write(
                f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n"
            )
            f.write("-" * 100 + "\n")

            for i, cls in enumerate(class_names):
                f.write(
                    f"{cls:<15} "
                    f"{res['per_class_precision'][i]:<12.4f} "
                    f"{res['per_class_recall'][i]:<12.4f} "
                    f"{res['per_class_f1'][i]:<12.4f} "
                    f"{int(res['per_class_support'][i]):<10}\n"
                )

            f.write("\n")

    logger.info(f"✓ Text report saved to: {report_path}")

    # Save JSON report
    json_report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "data_root": config.data_root,
                "img_size": config.img_size,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "fine_tune_lr": config.fine_tune_lr,
                "use_focal_loss": config.use_focal_loss,
                "use_class_weights": config.use_class_weights,
                "use_tta": config.use_tta,
            },
        },
        "models": {},
    }

    for name, res in results.items():
        json_report["models"][name] = {
            "accuracy": float(res["accuracy"]),
            "macro_avg": {k: float(v) for k, v in res["macro_avg"].items()},
            "weighted_avg": {k: float(v) for k, v in res["weighted_avg"].items()},
            "per_class": {
                class_names[i]: {
                    "precision": float(res["per_class_precision"][i]),
                    "recall": float(res["per_class_recall"][i]),
                    "f1": float(res["per_class_f1"][i]),
                    "support": int(res["per_class_support"][i]),
                }
                for i in range(len(class_names))
            },
        }

    with open(f"{config.output_dir}/report.json", "w") as f:
        json.dump(json_report, f, indent=2)

    logger.info("✓ JSON report saved")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function"""
    print("\n" + "=" * 100)
    print("IMPROVED WASTE SORTING MODEL COMPARISON SYSTEM")
    print("=" * 100 + "\n")

    start_time = time.time()

    set_seed(42)

    if config.use_mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("✓ Mixed precision training enabled")

    logger.info("\n[1/8] Loading data with optimized pipeline...")
    data_loader = OptimizedDataLoader(config)
    train_ds, val_ds, test_ds, class_names, class_weights = data_loader.load_data()
    logger.info(f"✓ Classes: {class_names}")

    logger.info("\n[2/8] Training models...")
    trainer = SequentialTrainer(config)
    results = trainer.train_all_models(
        train_ds, val_ds, test_ds, class_names, class_weights
    )

    logger.info("\n[3/8] Generating training curves...")
    plot_training_curves(results, config)

    logger.info("\n[4/8] Generating confusion matrices...")
    plot_confusion_matrices(results, class_names, config)

    logger.info("\n[5/8] Generating per-class metrics...")
    plot_per_class_metrics(results, class_names, config)

    logger.info("\n[6/8] Generating model comparison overview...")
    plot_model_comparison_overview(results, config)

    logger.info("\n[7/8] Saving comprehensive report...")
    save_comprehensive_report(results, class_names, config)

    best_model_name = max(results.items(), key=lambda x: x[1]["accuracy"])[0]
    logger.info(
        f"\n[8/8] Generating XAI visualizations for best model: {best_model_name}"
    )

    test_dir = os.path.join(config.data_root, "test")
    generate_xai_visualizations_parallel(
        results[best_model_name]["model_path"],
        test_dir,
        class_names,
        best_model_name,
        config,
    )

    total_time = time.time() - start_time

    print("\n" + "=" * 100)
    print("EXECUTION SUMMARY")
    print("=" * 100)

    sorted_results = sorted(results.items(), key=lambda x: x["accuracy"], reverse=True)

    print("\nModel Rankings:")
    for rank, (name, res) in enumerate(sorted_results, 1):
        print(f"{rank}. {name}")
        print(f"   Accuracy: {res['accuracy']:.4f} ({res['accuracy'] * 100:.2f}%)")
        print(f"   Macro F1: {res['macro_avg']['f1-score']:.4f}")
        print(f"   Weighted F1: {res['weighted_avg']['f1-score']:.4f}")

    print(
        f"\n✓ Best Model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})"
    )
    print(
        f"✓ Total execution time: {total_time / 60:.1f} minutes ({total_time:.1f} seconds)"
    )
    print(f"✓ All outputs saved to: {config.output_dir}/")
    print(f"✓ Models saved to: {config.output_dir}/models/")
    print(
        f"✓ XAI visualizations: {config.output_dir}/xai_visualizations/{best_model_name}/"
    )
    print("=" * 100 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
    except Exception as e:
        logger.exception(f"\n\nError during execution: {e}")
        raise
