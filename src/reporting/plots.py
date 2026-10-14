from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import Config

logger = logging.getLogger(__name__)
Results = Dict[str, dict]


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _save(fig, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(results: Results, config: Config) -> None:
    for name, res in results.items():
        hist = res["history"]
        epochs = range(1, len(hist["loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, key, title, ylabel in (
            (axes[0], "loss", "Loss During Training", "Loss (lower is better)"),
            (axes[1], "accuracy", "Accuracy During Training", "Accuracy (higher is better)"),
        ):
            ax.plot(epochs, hist[key], "b-", label=f"Training {key}", lw=2.5, marker="o", ms=4)
            ax.plot(epochs, hist[f"val_{key}"], "r--", label=f"Validation {key}", lw=2.5, marker="s", ms=4)
            # FIX (new): mark where fine-tuning starts. Original concatenated both phases into
            # one curve with no boundary, so the LR drop / unfreeze jump looked like a bug.
            boundary = res.get("phase1_len")
            if boundary and boundary < len(hist[key]):
                ax.axvline(boundary + 0.5, color="gray", ls=":", label="fine-tuning starts")
            ax.set_title(f"{name}\n{title}", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Training epoch", fontsize=13)
            ax.set_ylabel(ylabel, fontsize=13)
            ax.grid(True, alpha=0.3, ls="--")
            ax.legend(fontsize=11)  # legend AFTER all labelled artists
        axes[1].set_ylim([0, 1.05])
        axes[1].text(
            0.02, 0.98,
            f"Final training: {hist['accuracy'][-1] * 100:.2f}%\nFinal validation: {hist['val_accuracy'][-1] * 100:.2f}%",
            transform=axes[1].transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        fig.tight_layout()
        _save(fig, Path(config.output_dir, "training_curves", f"{_safe(name)}_training.png"))
    logger.info("Training curves saved")


def plot_confusion_matrices(results: Results, class_names: List[str], config: Config) -> None:
    for name, res in results.items():
        cm = res["cm"]
        norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            norm, annot=True, fmt=".1%", cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar_kws={"label": "Share of true class"}, vmin=0, vmax=1, linewidths=0.5, linecolor="gray",
        )
        ax.set_title(
            f"{name} - Confusion Matrix (row-normalised)\nTest accuracy: {res['accuracy'] * 100:.2f}%",
            fontsize=16, fontweight="bold", pad=20,
        )
        ax.set_ylabel("Actual class", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted class", fontsize=13, fontweight="bold")
        fig.text(0.5, 0.01, "Each row sums to 100%. The diagonal is the share predicted correctly.",
                 ha="center", fontsize=10, style="italic")
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        _save(fig, Path(config.output_dir, "confusion_matrices", f"{_safe(name)}_confusion_matrix.png"))
    logger.info("Confusion matrices saved")


def plot_per_class_metrics(results: Results, class_names: List[str], config: Config) -> None:
    for name, res in results.items():
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(class_names))
        width = 0.25
        series = (
            ("Precision", res["per_class_precision"], "#3498db", -width),
            ("Recall", res["per_class_recall"], "#2ecc71", 0.0),
            ("F1-score", res["per_class_f1"], "#e74c3c", width),
        )
        for label, vals, color, offset in series:
            bars = ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="black", linewidth=1.2)
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height() * 100:.1f}%",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.axhline(0.8, color="gray", ls=":", alpha=0.6, label="80% threshold")
        ax.set_title(f"{name}\nPerformance by class", fontsize=16, fontweight="bold", pad=20)
        ax.set_ylabel("Score", fontsize=13, fontweight="bold")
        ax.set_xlabel("Class", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=11)
        ax.set_ylim([0, 1.15])
        ax.grid(True, alpha=0.3, axis="y", ls="--")
        # FIX: legend AFTER the axhline. Original called legend() first, so the "80% threshold"
        # label was attached too late and never appeared.
        ax.legend(fontsize=12, loc="lower right")
        fig.tight_layout()
        _save(fig, Path(config.output_dir, "per_class_metrics", f"{_safe(name)}_per_class.png"))
    logger.info("Per-class metrics saved")


def _grade(acc: float):
    # FIX: plain text. Original used emoji/star glyphs, which the default matplotlib font
    # (DejaVu Sans) cannot draw: they rendered as empty boxes plus "missing glyph" warnings.
    if acc >= 0.95:
        return "EXCELLENT", "#27ae60"
    if acc >= 0.90:
        return "VERY GOOD", "#2ecc71"
    if acc >= 0.85:
        return "GOOD", "#f39c12"
    if acc >= 0.80:
        return "ACCEPTABLE", "#e67e22"
    return "NEEDS IMPROVEMENT", "#e74c3c"


def plot_model_comparison_overview(results: Results, config: Config) -> None:
    out = Path(config.output_dir)
    models = list(results)

    # 1. accuracy ranking
    accs = {m: results[m]["accuracy"] * 100 for m in models}
    ranked = sorted(models, key=lambda m: accs[m], reverse=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.9, 0.4, len(ranked)))
    bars = ax.barh(ranked[::-1], [accs[m] for m in ranked][::-1], color=colors[::-1], edgecolor="black", linewidth=2)
    for b, m in zip(bars, ranked[::-1]):
        ax.text(accs[m] + 1, b.get_y() + b.get_height() / 2, f"{accs[m]:.2f}%", va="center", fontweight="bold", fontsize=12)
    ax.set_xlabel("Test accuracy (%)", fontsize=13, fontweight="bold")
    ax.set_title("Model Accuracy Ranking", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlim([0, 105])
    ax.grid(True, alpha=0.3, axis="x", ls="--")
    fig.tight_layout()
    _save(fig, out / "model_comparison_accuracy.png")

    # 2. multi-metric comparison
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    # FIX: x positions come from the NUMBER OF METRICS and bar width scales with the number of
    # models. Original: np.arange(len(models)) was used for 4 metric groups, which only worked
    # because there happened to be exactly 4 models; bars overlapped/misaligned otherwise.
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, m in enumerate(models):
        r = results[m]
        vals = [r["accuracy"] * 100, r["weighted_avg"]["precision"] * 100,
                r["weighted_avg"]["recall"] * 100, r["weighted_avg"]["f1-score"] * 100]
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width, label=m, edgecolor="black", linewidth=1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(80, color="red", ls=":", alpha=0.5, lw=2, label="80% good")
    ax.axhline(90, color="green", ls=":", alpha=0.5, lw=2, label="90% excellent")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=13, fontweight="bold")
    ax.set_title("Model Performance Comparison (weighted averages)", fontsize=16, fontweight="bold", pad=20)
    ax.set_ylim([0, 110])
    ax.grid(True, alpha=0.3, axis="y", ls="--")
    ax.legend(fontsize=11, loc="lower right")  # after axhlines so threshold labels show
    fig.tight_layout()
    _save(fig, out / "model_comparison_all_metrics.png")

    # 3. summary cards
    # FIX: grid size derived from the number of models. Original: hard-coded subplot(2, 2, i),
    # which raises ValueError for a 5th model.
    cols = 2
    rows = math.ceil(len(models) / cols)
    fig = plt.figure(figsize=(16, 5 * rows))
    for idx, m in enumerate(models):
        r = results[m]
        ax = fig.add_subplot(rows, cols, idx + 1)
        ax.axis("off")
        grade, color = _grade(r["accuracy"])
        # FIX: neutral wording. Original text said "When it says 'recyclable'..." although the
        # classes are arbitrary waste types, and reported a weighted figure as if it were
        # specific to one class.
        text = (
            f"{m}\n{'=' * 44}\n\n"
            f"TEST PERFORMANCE\n"
            f"  Accuracy:  {r['accuracy'] * 100:6.2f}%\n"
            f"  Precision: {r['weighted_avg']['precision'] * 100:6.2f}%  (weighted)\n"
            f"  Recall:    {r['weighted_avg']['recall'] * 100:6.2f}%  (weighted)\n"
            f"  F1-score:  {r['weighted_avg']['f1-score'] * 100:6.2f}%  (weighted)\n"
            f"  Macro F1:  {r['macro_avg']['f1-score'] * 100:6.2f}%\n\n"
            f"TRAINING (last epoch)\n"
            f"  Train acc: {r['history']['accuracy'][-1] * 100:6.2f}%\n"
            f"  Val acc:   {r['history']['val_accuracy'][-1] * 100:6.2f}%\n"
            f"  Best val loss: {r['best_val_loss']:.4f}"
        )
        ax.text(0.5, 0.97, grade, ha="center", va="top", fontsize=14, fontweight="bold", color=color,
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.25, edgecolor=color, linewidth=3))
        ax.text(0.05, 0.84, text, ha="left", va="top", fontsize=10, family="monospace", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=1", facecolor="lightgray", alpha=0.2, edgecolor="black"))
    fig.suptitle("Model Performance Summary", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out / "model_summary_cards.png")
    logger.info("Model comparison overview saved")
