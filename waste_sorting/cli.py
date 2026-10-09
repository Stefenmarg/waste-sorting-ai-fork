"""Entry point. The ONLY module that wires the others together."""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Dict, Optional, Sequence

import keras
from config import Config
from reporting.plots import (
    plot_confusion_matrices,
    plot_model_comparison_overview,
    plot_per_class_metrics,
    plot_training_curves,
)
from reporting.report import save_comprehensive_report
from runtime import configure_gpu, configure_matplotlib, set_seed, setup_logging
from training.trainer import SequentialTrainer
from xai.visualize import generate_xai_visualizations

from data import OptimizedDataLoader

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Waste sorting model comparison")
    p.add_argument("--data-root")
    p.add_argument("--output-dir")
    p.add_argument("--models", nargs="+")
    p.add_argument("--epochs", type=int)
    p.add_argument("--skip-xai", action="store_true")
    return p.parse_args(argv)


def run(cfg: Config, skip_xai: bool = False) -> Dict[str, dict]:
    """Full pipeline for an already-validated config (also used by the smoke test)."""
    t0 = time.time()
    cfg.make_output_dirs()
    if cfg.use_mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision enabled")

    logger.info("[1/7] Loading data")
    data = OptimizedDataLoader(cfg).load_data()

    logger.info("[2/7] Training models")
    results = SequentialTrainer(cfg).train_all_models(data)

    logger.info("[3/7] Training curves")
    plot_training_curves(results, cfg)
    logger.info("[4/7] Confusion matrices")
    plot_confusion_matrices(results, data.class_names, cfg)
    logger.info("[5/7] Per-class metrics")
    plot_per_class_metrics(results, data.class_names, cfg)
    logger.info("[6/7] Comparison overview")
    plot_model_comparison_overview(results, cfg)

    # FIX: pick the winner by VALIDATION loss. Original: max test accuracy -> the test set
    # was used for model selection, so the reported test score was optimistic.
    best = min(results, key=lambda k: results[k]["best_val_loss"])
    logger.info("[7/7] Report (selected model: %s)", best)
    save_comprehensive_report(results, data.class_names, cfg, best)

    if not skip_xai:
        generate_xai_visualizations(
            results[best]["model_path"],
            os.path.join(cfg.data_root, "test"),
            data.class_names,
            best,
            cfg,
        )

    print("\n" + "=" * 80 + "\nEXECUTION SUMMARY\n" + "=" * 80)
    # FIX: sort on x[1]["accuracy"]. Original `key=lambda x: x["accuracy"]` indexed a
    # (name, dict) TUPLE with a string -> TypeError after all training had finished.
    for rank, (name, r) in enumerate(
        sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True), 1
    ):
        print(
            f"{rank}. {name}: acc={r['accuracy']:.4f} macroF1={r['macro_avg']['f1-score']:.4f} "
            f"weightedF1={r['weighted_avg']['f1-score']:.4f} bestValLoss={r['best_val_loss']:.4f}"
        )
    print(f"\nSelected (lowest val loss): {best}")
    print(
        f"Total time: {(time.time() - t0) / 60:.1f} min | outputs: {cfg.output_dir}/\n"
    )
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    overrides = {
        k: v
        for k, v in {
            "data_root": args.data_root,
            "output_dir": args.output_dir,
            "models_to_train": args.models,
            "epochs": args.epochs,
        }.items()
        if v is not None
    }
    cfg = Config(**overrides)
    cfg.validate()
    cfg.make_output_dirs()
    setup_logging(cfg.output_dir)
    configure_matplotlib()
    configure_gpu()
    set_seed(cfg.seed)
    try:
        run(cfg, skip_xai=args.skip_xai)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Run failed")
        raise


if __name__ == "__main__":
    main()
