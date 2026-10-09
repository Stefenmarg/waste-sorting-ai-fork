from __future__ import annotations

import gc
import importlib.util
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import keras
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from waste_sorting.config import Config
from waste_sorting.data import Datasets
from waste_sorting.modeling import preprocessing  # noqa: F401  (registers BackbonePreprocess for load_model)
from waste_sorting.modeling.builder import build_model, compile_model, unfreeze_top_layers
from waste_sorting.training.callbacks import ProgressCallback, WarmUpLearningRate
from waste_sorting.training.evaluation import evaluate_model

logger = logging.getLogger(__name__)


def _tb(out: Path, arch: str, tag: str):
    # FIX: histogram_freq=0. Original used 1, which dumps weight histograms every epoch and
    # noticeably slows training for four large backbones.
    # FIX (new): TensorBoard is optional. The callback raises TBNotInstalledError inside fit()
    # when the package is absent (it is not guaranteed to come with every TensorFlow build),
    # which would abort training after the data/model were already set up.
    if importlib.util.find_spec("tensorboard") is None:
        logger.warning("tensorboard not installed; skipping TensorBoard logging")
        return None
    return TensorBoard(log_dir=str(out / "logs" / f"{arch}_{tag}_{datetime.now():%Y%m%d-%H%M%S}"), histogram_freq=0)


def train_single_model(arch: str, data: Datasets, cfg: Config) -> Dict[str, Any]:
    out = Path(cfg.output_dir)
    ckpt_p1 = out / "checkpoints" / f"{arch}_phase1_best.keras"
    ckpt_p2 = out / "checkpoints" / f"{arch}_phase2_best.keras"
    final_path = out / "models" / f"{arch}_best_model.keras"  # FIX: .keras, native Keras 3 format
    csv_path = out / "logs" / f"{arch}_training_log.csv"
    csv_path.unlink(missing_ok=True)  # CSVLogger(append=True) below must not append to an old run

    logger.info("=" * 80 + "\nTraining %s\n" + "=" * 80, arch)
    model, base_model = build_model(arch, len(data.class_names), cfg)
    progress = ProgressCallback(arch)

    # ------------------------------------------------------------------ phase 1
    logger.info("%s: PHASE 1 - classifier head (frozen backbone), up to %d epochs", arch, cfg.phase1_epochs)
    cb1 = [
        WarmUpLearningRate(cfg.warmup_epochs, cfg.learning_rate * 0.1, cfg.learning_rate, verbose=1),
        progress,
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, min_delta=1e-3, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, cooldown=2, verbose=1),
        # FIX: checkpoint monitors val_loss, same as EarlyStopping.
        # Original: ModelCheckpoint used val_accuracy while EarlyStopping restored the best
        # val_loss weights. The file on disk (used for XAI) and the in-memory model (used for
        # test metrics) could be different weights, so the explanations did not describe the
        # model whose accuracy was reported.
        ModelCheckpoint(str(ckpt_p1), monitor="val_loss", save_best_only=True, verbose=0),
        _tb(out, arch, "phase1"),
        CSVLogger(str(csv_path), append=True),
    ]
    cb1 = [c for c in cb1 if c is not None]
    h1 = model.fit(
        data.train, validation_data=data.val, epochs=cfg.phase1_epochs,
        class_weight=data.class_weights, callbacks=cb1, verbose=0,
    )

    # ------------------------------------------------------------------ phase 2
    logger.info("%s: PHASE 2 - fine-tuning top of backbone, epochs %d..%d", arch, cfg.phase1_epochs + 1, cfg.epochs)
    unfreeze_top_layers(base_model, cfg.fine_tune_from_layer_percent)
    compile_model(model, cfg.fine_tune_lr, cfg)  # must recompile after changing .trainable
    cb2 = [
        progress,
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, min_delta=1e-4, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, min_lr=1e-8, cooldown=2, verbose=1),
        ModelCheckpoint(str(ckpt_p2), monitor="val_loss", save_best_only=True, verbose=0),
        _tb(out, arch, "finetune"),
        # FIX: phase 2 also logs to the CSV. Original: only phase 1 had a CSVLogger.
        CSVLogger(str(csv_path), append=True),
    ]
    cb2 = [c for c in cb2 if c is not None]
    h2 = model.fit(
        data.train, validation_data=data.val, epochs=cfg.epochs, initial_epoch=cfg.phase1_epochs,
        class_weight=data.class_weights, callbacks=cb2, verbose=0,
    )

    # --------------------------------------------- pick the best weights, once
    v1, v2 = min(h1.history["val_loss"]), min(h2.history["val_loss"])
    src = ckpt_p2 if v2 <= v1 else ckpt_p1
    # FIX: choose between phase-1 and phase-2 checkpoints by val_loss.
    # Original: only one checkpoint path existed per phase, and phase 2's "best" was just
    # the best of phase 2 - it could be worse than what phase 1 had already reached.
    shutil.copyfile(src, final_path)
    logger.info("%s: best val_loss %.4f (%s) -> %s", arch, min(v1, v2), "phase 2" if src == ckpt_p2 else "phase 1", final_path)

    # ------------------------------------------------------------------ test evaluation
    # FIX: evaluate the exact file that will later be explained (reloaded from disk).
    best_model = keras.models.load_model(final_path, compile=False)
    logger.info("%s: evaluating on test set...", arch)
    metrics = evaluate_model(best_model, data.test, data.class_names, cfg)
    logger.info(
        "%s test: acc=%.4f macroF1=%.4f weightedF1=%.4f",
        arch, metrics["accuracy"], metrics["macro_avg"]["f1-score"], metrics["weighted_avg"]["f1-score"],
    )

    keys = ("loss", "val_loss", "accuracy", "val_accuracy")
    history = {k: h1.history[k] + h2.history[k] for k in keys}

    del model, base_model, best_model
    keras.backend.clear_session()
    gc.collect()

    return {
        "arch": arch,
        "model_path": str(final_path),
        "history": history,
        "phase1_len": len(h1.history["loss"]),
        "best_val_loss": float(min(v1, v2)),  # FIX (new): model selection key. Original picked the "best" model by TEST accuracy (test-set leakage).
        **metrics,
    }


class SequentialTrainer:
    """Trains models one after another (right for a single GPU)."""

    def __init__(self, config: Config):
        self.config = config

    def train_all_models(self, data: Datasets) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        names = self.config.models_to_train
        for i, arch in enumerate(names, 1):
            logger.info("Model %d/%d: %s", i, len(names), arch)
            try:
                results[arch] = train_single_model(arch, data, self.config)
            except Exception:
                # FIX (new): one failing architecture (e.g. GPU OOM) no longer discards the
                # hours already spent on the others. Original: any exception aborted the run.
                logger.exception("Training %s failed; continuing with the next model", arch)
                keras.backend.clear_session()
                gc.collect()
        if not results:
            raise RuntimeError("All models failed to train; see log above.")
        return results
