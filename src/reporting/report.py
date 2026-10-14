from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from config import Config

logger = logging.getLogger(__name__)


def save_comprehensive_report(results: Dict[str, dict], class_names: List[str], config: Config, best_model: str) -> None:
    out = Path(config.output_dir)
    ranked = sorted(results.items(), key=lambda kv: kv[1]["accuracy"], reverse=True)

    lines: List[str] = ["=" * 100, "WASTE SORTING - COMPREHENSIVE REPORT",
                        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}", "=" * 100, "", "CONFIGURATION:"]
    for k, v in config.to_dict().items():
        lines.append(f"  {k}: {v}")
    lines += ["", f"Classes: {', '.join(class_names)}", "",
              # FIX (new): selection is by validation loss and says so. Original ranked and
              # picked the "best" model on the TEST set, so the test score was optimistically
              # biased and the report did not disclose how the model was chosen.
              f"SELECTED MODEL (lowest validation loss): {best_model}", "",
              "MODEL COMPARISON (sorted by test accuracy):",
              f"{'Model':<24}{'Accuracy':<11}{'Precision':<11}{'Recall':<11}{'F1':<11}{'ValLoss':<10}", "-" * 100]
    for rank, (name, r) in enumerate(ranked, 1):
        lines.append(
            f"{rank}. {name:<21}{r['accuracy']:<11.4f}{r['weighted_avg']['precision']:<11.4f}"
            f"{r['weighted_avg']['recall']:<11.4f}{r['weighted_avg']['f1-score']:<11.4f}{r['best_val_loss']:<10.4f}"
        )
    for name, r in results.items():
        lines += ["", "=" * 100, f"MODEL: {name}", "=" * 100, "",
                  f"  Accuracy: {r['accuracy']:.4f}",
                  f"  Macro    P/R/F1: {r['macro_avg']['precision']:.4f} / {r['macro_avg']['recall']:.4f} / {r['macro_avg']['f1-score']:.4f}",
                  f"  Weighted P/R/F1: {r['weighted_avg']['precision']:.4f} / {r['weighted_avg']['recall']:.4f} / {r['weighted_avg']['f1-score']:.4f}",
                  "", f"{'Class':<20}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Support':<10}", "-" * 66]
        for i, c in enumerate(class_names):
            lines.append(
                f"{c:<20}{r['per_class_precision'][i]:<12.4f}{r['per_class_recall'][i]:<12.4f}"
                f"{r['per_class_f1'][i]:<12.4f}{int(r['per_class_support'][i]):<10}"
            )
    (out / "comprehensive_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "metadata": {"generated_at": datetime.now().isoformat(), "config": config.to_dict(), "selected_model": best_model},
        "models": {
            name: {
                "accuracy": float(r["accuracy"]),
                "best_val_loss": float(r["best_val_loss"]),
                "macro_avg": {k: float(v) for k, v in r["macro_avg"].items()},
                "weighted_avg": {k: float(v) for k, v in r["weighted_avg"].items()},
                "per_class": {
                    c: {
                        "precision": float(r["per_class_precision"][i]),
                        "recall": float(r["per_class_recall"][i]),
                        "f1": float(r["per_class_f1"][i]),
                        "support": int(r["per_class_support"][i]),
                    }
                    for i, c in enumerate(class_names)
                },
            }
            for name, r in results.items()
        },
    }
    (out / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Reports saved to %s", out)
