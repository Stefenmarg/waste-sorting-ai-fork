from __future__ import annotations

import logging
import time

from keras.callbacks import Callback

logger = logging.getLogger(__name__)


class WarmUpLearningRate(Callback):
    """Linear LR warm-up that actually reaches the target LR."""

    def __init__(self, warmup_epochs: int, initial_lr: float, target_lr: float, verbose: int = 0):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # FIX: (epoch + 1) / warmup_epochs.
            # Original: epoch / warmup_epochs. With 3 warm-up epochs that gives 0, 1/3, 2/3 of
            # the way, and nothing ever set the LR to the target afterwards, so training ran
            # at ~70% of the intended learning rate until ReduceLROnPlateau happened to fire.
            frac = (epoch + 1) / self.warmup_epochs
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * frac
            self.model.optimizer.learning_rate.assign(lr)
            if self.verbose:
                logger.info("Epoch %d: warm-up LR = %.6f", epoch + 1, lr)


class ProgressCallback(Callback):
    """Per-epoch timing and metrics to the logger (fit runs with verbose=0)."""

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self._t0 = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logger.info(
            "%s - epoch %d (%.1fs) loss=%.4f acc=%.4f val_loss=%.4f val_acc=%.4f",
            self.model_name, epoch + 1, time.time() - self._t0,
            logs.get("loss", 0), logs.get("accuracy", 0),
            logs.get("val_loss", 0), logs.get("val_accuracy", 0),
        )
