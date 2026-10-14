from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import Config
from data import IMG_EXTS
from xai.explainer import XAIExplainer

logger = logging.getLogger(__name__)

METHOD_ORDER = ("grad_cam", "occlusion", "integrated_gradients", "lime")


# FIX: sequential, single explainer, replaces generate_xai_visualizations_parallel and
# process_single_xai_worker (multiprocessing.Pool).
# Original problems:
#   - fork()ing a Pool after TensorFlow/CUDA is initialised commonly deadlocks or crashes;
#   - every worker re-loaded the whole model PER IMAGE and each also called clear_session;
#   - workers raced on the same GPU, so "parallel" was not faster;
#   - axes.imshow(img) was called on the axes ARRAY (needs axes[0]).
# 12 images x 4 methods is small enough that loading the model once is the real speed-up.
def generate_xai_visualizations(
    model_path: str, test_dir: str, class_names: List[str], model_name: str, config: Config
) -> int:
    logger.info("Generating XAI visualizations for %s...", model_name)
    out_dir = Path(config.output_dir, "xai_visualizations", model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    explainer = XAIExplainer(model_path, class_names, config)
    ok = total = 0
    for cls in class_names:
        cls_dir = Path(test_dir) / cls
        if not cls_dir.is_dir():
            continue
        imgs = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        for idx, path in enumerate(imgs[: config.xai_samples_per_class]):
            total += 1
            try:
                img = Image.open(path).convert("RGB").resize((config.img_size[1], config.img_size[0]))
                arr = np.asarray(img, dtype=np.float32)  # raw 0-255, same scale as training
                ex = explainer.explain_all(arr, cls)

                shown = [m for m in METHOD_ORDER if ex.get(m) is not None]
                fig, axes = plt.subplots(1, len(shown) + 1, figsize=(5 * (len(shown) + 1), 5))
                axes = np.atleast_1d(axes)
                axes[0].imshow(img)
                axes[0].set_title(
                    f"Original\nTrue: {ex['true_class']}\nPred: {ex['pred_class']}\nConf: {ex['confidence']:.3f}",
                    fontsize=10,
                )
                axes[0].axis("off")
                for ax, m in zip(axes[1:], shown):
                    ax.imshow(img)
                    ax.imshow(ex[m], cmap="jet", alpha=0.3 if m == "lime" else 0.5)
                    ax.set_title(m.replace("_", " ").title(), fontsize=10)
                    ax.axis("off")
                fig.tight_layout()
                fig.savefig(out_dir / f"{cls}_sample_{idx}_xai.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                ok += 1
            except Exception:
                logger.exception("XAI failed for %s", path)
                plt.close("all")

    del explainer
    keras.backend.clear_session()
    logger.info("XAI visualization complete: %d/%d successful", ok, total)
    return ok
