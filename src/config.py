"""Central configuration.

Pure data + explicit validation. Importing this module has NO side effects.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

OUTPUT_SUBDIRS = (
    "models",
    "confusion_matrices",
    "training_curves",
    "xai_visualizations",
    "per_class_metrics",
    "logs",
    "checkpoints",
)


def _default_data_root() -> str:
    return os.environ.get("WASTE_DATA_ROOT", "./data")


@dataclass
class Config:
    # FIX: data_root uses default_factory.
    # Original: `data_root: str = os.environ.get(...)` is evaluated once, when the class
    # body runs (at import). Setting WASTE_DATA_ROOT after importing had no effect.
    data_root: str = field(default_factory=_default_data_root)
    output_dir: str = "optimized_outputs"
    seed: int = 42

    # data
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    # Elements are uint8 images (~150 KB each at 224x224). Lower this if RAM is tight.
    shuffle_buffer: int = 2048

    # models
    models_to_train: List[str] = field(
        default_factory=lambda: ["ResNet50", "EfficientNetV2B0", "DenseNet121", "MobileNetV2"]
    )
    pretrained: bool = True
    # FIX (new): by default a failed ImageNet download is an error.
    # Original: after 3 failed attempts it silently trained from random weights, which
    # makes "transfer learning" results meaningless without anyone noticing.
    allow_random_init: bool = False
    dropout_rate: float = 0.3

    # training
    epochs: int = 50
    phase1_epochs: int = 15
    learning_rate: float = 1e-3
    fine_tune_lr: float = 1e-5
    fine_tune_from_layer_percent: float = 0.8  # freeze the first 80% of backbone layers
    warmup_epochs: int = 3
    gradient_clip_value: float = 1.0
    use_mixed_precision: bool = False
    use_class_weights: bool = True
    use_focal_loss: bool = False
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25
    use_tta: bool = False
    tta_steps: int = 5

    # XAI
    xai_samples_per_class: int = 3
    xai_methods: List[str] = field(
        default_factory=lambda: ["grad_cam", "lime", "occlusion", "integrated_gradients"]
    )
    occlusion_size: int = 30
    occlusion_stride: int = 15
    occlusion_batch_size: int = 32
    ig_steps: int = 50
    lime_samples: int = 300

    # FIX: validation and directory creation are explicit methods.
    # Original: __post_init__ validated the data dir and created folders, and a module-level
    # `config = Config()` ran it on import. So `import` crashed without data and made folders.
    def validate(self) -> None:
        root = Path(self.data_root)
        if not root.is_dir():
            raise FileNotFoundError(
                f"Data directory not found: {root}\n"
                "Set WASTE_DATA_ROOT (or --data-root). Expected layout:\n"
                f"  {root}/{{train,val,test}}/<class_name>/*.jpg"
            )
        for split in ("train", "val", "test"):
            if not (root / split).is_dir():
                raise FileNotFoundError(f"Missing required directory: {root / split}")
        if not 0 < self.phase1_epochs < self.epochs:
            raise ValueError("Need 0 < phase1_epochs < epochs")
        if self.warmup_epochs > self.phase1_epochs:
            raise ValueError("warmup_epochs must be <= phase1_epochs")
        if not 0.0 <= self.fine_tune_from_layer_percent <= 1.0:
            raise ValueError("fine_tune_from_layer_percent must be in [0, 1]")

    def make_output_dirs(self) -> None:
        for sub in OUTPUT_SUBDIRS:
            Path(self.output_dir, sub).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
