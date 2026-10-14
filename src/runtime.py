"""Process-wide setup. Every function here is called explicitly from cli.main()."""
from __future__ import annotations

import logging
import warnings
from datetime import datetime
from pathlib import Path

import keras
import tensorflow as tf


def setup_logging(output_dir: str) -> None:
    # FIX: called from main(), log file goes inside output_dir.
    # Original: logging.basicConfig ran at import time and dropped a training_*.log file
    # into whatever the current directory was, even for `import` in a test.
    log_dir = Path(output_dir, "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    warnings.filterwarnings("ignore", category=FutureWarning)


def configure_gpu() -> int:
    """Enable memory growth. Must run before any op touches the GPU."""
    logger = logging.getLogger(__name__)
    gpus = tf.config.list_physical_devices("GPU")
    # FIX: loop over devices, once, at startup.
    # Original: train_single_model called set_memory_growth(gpus, True) with the whole LIST
    # (it takes one device -> error), and after the GPU was already initialised, where it
    # can only raise RuntimeError.
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning("Could not set memory growth on %s: %s", gpu, e)
    logger.info("Found %d GPU(s)", len(gpus))
    return len(gpus)


def set_seed(seed: int = 42) -> None:
    # FIX: keras.utils.set_random_seed seeds python, numpy and the backend together.
    # Original: wrote PYTHONHASHSEED / TF_DETERMINISTIC_OPS into os.environ at runtime.
    # Both are read at interpreter / TF start-up, so mid-run they were no-ops and runs
    # were not actually reproducible.
    keras.utils.set_random_seed(seed)
    logging.getLogger(__name__).info("Random seed set to %d", seed)


def configure_matplotlib() -> None:
    """Headless-safe backend (servers / WSL without a display)."""
    import matplotlib

    matplotlib.use("Agg")
