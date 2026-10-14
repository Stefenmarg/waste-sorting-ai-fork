"""XAI methods. Images are float32 arrays of RAW 0-255 pixels, shape (H, W, 3)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cv2
import keras
import numpy as np
import tensorflow as tf
from lime import lime_image

from config import Config
from modeling import preprocessing  # noqa: F401  (registers BackbonePreprocess for load_model)

logger = logging.getLogger(__name__)


class XAIExplainer:
    def __init__(self, model_path: str, class_names: List[str], config: Config):
        self.model = keras.models.load_model(model_path, compile=False)
        self.class_names = class_names
        self.config = config
        self.lime_explainer = lime_image.LimeImageExplainer(random_state=config.seed)
        self._pre, self._backbone, self._post = self._split_at_backbone()
        # FIX: no __del__. Original: __del__ called keras.backend.clear_session(), which wipes
        # global Keras state for EVERY model in the process whenever any explainer was
        # garbage-collected. Cleanup is now explicit in the caller.

    def _probs(self, img: np.ndarray) -> np.ndarray:
        # FIX: returns shape (n_classes,). Original kept the (1, n_classes) predict() output
        # and then indexed pred_proba[pred_idx] / predict(...)[class_idx], i.e. took ROW
        # pred_idx of a 1-row array -> IndexError for every image, outside any try/except.
        return self.model(img[None].astype(np.float32), training=False).numpy()[0]

    # ------------------------------------------------------------------ Grad-CAM
    def _split_at_backbone(self):
        layers = [l for l in self.model.layers if not isinstance(l, keras.layers.InputLayer)]
        idx = [i for i, l in enumerate(layers) if isinstance(l, keras.Model)]
        if not idx:
            logger.warning("No nested backbone found; Grad-CAM disabled")
            return None, None, None
        i = idx[-1]
        logger.info("Grad-CAM target: backbone layer '%s'", layers[i].name)
        return layers[:i], layers[i], layers[i + 1:]

    def grad_cam(self, img: np.ndarray, class_idx: int) -> np.ndarray:
        h, w = self.config.img_size
        if self._backbone is None:
            return np.zeros((h, w), dtype=np.float32)
        x = tf.convert_to_tensor(img[None], tf.float32)
        # FIX: run the forward pass layer-by-layer instead of building keras.Model(inputs,
        # [layer.output, model.output]) from `layer.output_shape`.
        # Original: `layer.output_shape` does not exist in Keras 3 (AttributeError), and the
        # feature map of a NESTED backbone is not a tensor of the outer graph, so the
        # two-output-model trick is fragile. Calling layers directly under the tape is robust.
        # Layers are called without `training`, so Dropout/BatchNorm run in inference mode.
        with tf.GradientTape() as tape:
            hdn = x
            for l in self._pre:
                hdn = l(hdn)
            conv = self._backbone(hdn)
            tape.watch(conv)
            y = conv
            for l in self._post:
                y = l(y)
            score = y[:, class_idx]
        grads = tape.gradient(score, conv)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heat = tf.reduce_sum(conv[0] * pooled, axis=-1)
        heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-10)
        # FIX: conv[0] drops the batch axis. Original passed a (1, h, w) array to cv2.resize,
        # which interprets it as an image with width 1 and produced a wrong/failed resize.
        # Also cv2 wants (width, height); img_size is (height, width).
        return cv2.resize(heat.numpy(), (w, h))

    # ------------------------------------------------------------------ occlusion
    def occlusion_sensitivity(self, img: np.ndarray, class_idx: int) -> np.ndarray:
        h, w = self.config.img_size
        size, stride = self.config.occlusion_size, self.config.occlusion_stride
        baseline = float(self._probs(img)[class_idx])
        fill = float(img.mean())  # neutral grey-ish patch instead of pure black

        batch, positions = [], []
        # FIX: range(..., h - size + 1, ...). Original used h - size, so the last valid
        # window position was never tested and the bottom/right edge was always blank.
        for y in range(0, h - size + 1, stride):
            for x in range(0, w - size + 1, stride):
                occ = img.copy()
                occ[y:y + size, x:x + size] = fill
                batch.append(occ)
                positions.append((y, x))

        preds = self.model.predict(
            np.asarray(batch, dtype=np.float32), batch_size=self.config.occlusion_batch_size, verbose=0
        )
        sens = np.zeros((h, w), dtype=np.float32)
        for (y, x), p in zip(positions, preds):
            # FIX: element-wise max over the whole patch.
            # Original: `region = max(sens[y, x], s)` compared against only the patch's
            # top-left pixel, then overwrote the entire patch with that scalar - overlapping
            # patches wiped out stronger values from earlier ones.
            region = sens[y:y + size, x:x + size]
            np.maximum(region, baseline - float(p[class_idx]), out=region)

        if sens.max() > sens.min():
            sens = (sens - sens.min()) / (sens.max() - sens.min())
        return sens

    # ------------------------------------------------------------------ integrated gradients
    def integrated_gradients(self, img: np.ndarray, class_idx: int) -> np.ndarray:
        steps = self.config.ig_steps
        x = tf.convert_to_tensor(img[None], tf.float32)
        baseline = tf.zeros_like(x)
        alphas = tf.linspace(0.0, 1.0, steps + 1)[:, None, None, None]
        # FIX: all interpolation steps in ONE batch with tape.watch.
        # Original: 50 separate forward+backward passes, each creating a new tf.Variable,
        # and a plain Riemann sum. One batched pass is ~an order of magnitude faster, and the
        # trapezoid rule below is more accurate for the same number of steps.
        interp = baseline + alphas * (x - baseline)
        with tf.GradientTape() as tape:
            tape.watch(interp)
            score = self.model(interp, training=False)[:, class_idx]
        grads = tape.gradient(score, interp)
        avg = tf.reduce_mean((grads[:-1] + grads[1:]) / 2.0, axis=0)
        attr = tf.reduce_sum(tf.abs((x[0] - baseline[0]) * avg), axis=-1).numpy()
        if attr.max() > attr.min():
            attr = (attr - attr.min()) / (attr.max() - attr.min())
        return attr

    # ------------------------------------------------------------------ LIME
    def lime_explain(self, img: np.ndarray, class_idx: int) -> np.ndarray:
        def predict_fn(images: np.ndarray) -> np.ndarray:
            # FIX: LIME hands over uint8 0-255 images. Original fed these straight into a model
            # that had been trained on 0-1 inputs, so LIME explained the model's behaviour on
            # inputs 255x too large (meaningless masks). The model now takes 0-255 natively,
            # so the scales match; we only cast.
            return self.model.predict(images.astype(np.float32), batch_size=32, verbose=0)

        explanation = self.lime_explainer.explain_instance(
            img.astype(np.uint8),
            predict_fn,
            # FIX: ask for the label we will read back (top_labels=None so `labels` is used).
            # Original: top_labels=1 and later get_image_and_mask(class_idx) -> KeyError
            # whenever class_idx was not LIME's own top-1 label.
            labels=(class_idx,),
            top_labels=None,
            num_samples=self.config.lime_samples,
            batch_size=32,
            random_seed=self.config.seed,
        )
        _, mask = explanation.get_image_and_mask(class_idx, positive_only=True, num_features=5, hide_rest=False)
        return mask.astype(np.float32)

    # ------------------------------------------------------------------ all
    def explain_all(self, img: np.ndarray, true_class: str) -> Dict[str, Any]:
        probs = self._probs(img)
        pred_idx = int(np.argmax(probs))
        out: Dict[str, Any] = {
            "true_class": true_class,
            "pred_class": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
        }
        methods = {
            "grad_cam": self.grad_cam,
            "occlusion": self.occlusion_sensitivity,
            "integrated_gradients": self.integrated_gradients,
            "lime": self.lime_explain,
        }
        for name, fn in methods.items():  # FIX: one loop instead of four copy-pasted try/except blocks
            if name not in self.config.xai_methods:
                continue
            try:
                out[name] = fn(img, pred_idx)
            except Exception as e:
                logger.warning("%s failed: %s", name, e)
                out[name] = None
        return out
