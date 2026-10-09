"""
Regenerate XAI visualizations with all methods working properly.
Creates 2x3 grid: Original, Occlusion overlay, Occlusion heatmap, LIME overlay, LIME mask, Prediction info
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries

print("Loading DenseNet121 model...")
model_path = "optimized_outputs/models/DenseNet121_best_model.h5"
model = keras.models.load_model(model_path)
print(f"✓ Model loaded")

class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
data_dir = Path("data/test")
output_dir = Path("optimized_outputs/xai_visualizations")
output_dir.mkdir(parents=True, exist_ok=True)

# Find sample images
print("\nFinding sample images...")
sample_images = {}
for class_name in class_names:
    class_dir = data_dir / class_name
    if class_dir.exists():
        images = list(class_dir.glob("*.jpg"))
        if images:
            sample_images[class_name] = str(images[0])
            print(f"  ✓ {class_name}: {images[0].name}")


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess image"""
    img = keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    return img_array


def get_occlusion_sensitivity(model, img_array, patch_size=30, stride=15):
    """Generate occlusion sensitivity map"""
    print("    Generating occlusion map...")
    original_pred = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
    original_class = np.argmax(original_pred)
    original_confidence = original_pred[original_class]

    height, width = img_array.shape[:2]
    sensitivity_map = np.zeros((height, width))

    for y in range(0, height - patch_size, stride):
        for x in range(0, width - patch_size, stride):
            occluded = img_array.copy()
            occluded[y : y + patch_size, x : x + patch_size] = 0.5
            occluded_pred = model.predict(np.expand_dims(occluded, axis=0), verbose=0)[
                0
            ]
            occluded_confidence = occluded_pred[original_class]
            sensitivity = original_confidence - occluded_confidence
            sensitivity_map[y : y + patch_size, x : x + patch_size] = np.maximum(
                sensitivity_map[y : y + patch_size, x : x + patch_size], sensitivity
            )

    if np.max(sensitivity_map) > 0:
        sensitivity_map /= np.max(sensitivity_map)

    return sensitivity_map


def get_lime_explanation(model, img_array, num_samples=200, num_features=5):
    """Generate LIME explanation"""
    print("    Generating LIME explanation...")
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        return model.predict(images, verbose=0)

    explanation = explainer.explain_instance(
        img_array, predict_fn, top_labels=1, hide_color=0, num_samples=num_samples
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label, positive_only=True, num_features=num_features, hide_rest=False
    )

    return mask


def create_xai_visualization(img_path, class_name, output_path):
    """Create comprehensive 2x3 XAI visualization"""
    print(f"\nProcessing {class_name}...")

    # Load image
    img_array = load_and_preprocess_image(img_path)

    # Get prediction
    print("    Making prediction...")
    pred = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
    pred_class = np.argmax(pred)
    pred_confidence = pred[pred_class]

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Explainable AI: {class_name.upper()}\n"
        f"Predicted: {class_names[pred_class]} ({pred_confidence * 100:.1f}% confidence)",
        fontsize=16,
        fontweight="bold",
    )

    # Row 1, Col 1: Original Image
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title("Original Image", fontweight="bold", fontsize=12)
    axes[0, 0].axis("off")

    # Row 1, Col 2: Occlusion Sensitivity Overlay
    occlusion = get_occlusion_sensitivity(model, img_array, patch_size=30, stride=15)
    axes[0, 1].imshow(img_array)
    axes[0, 1].imshow(occlusion, cmap="hot", alpha=0.5)
    axes[0, 1].set_title(
        "Occlusion Sensitivity Overlay\n(bright = important)",
        fontweight="bold",
        fontsize=12,
    )
    axes[0, 1].axis("off")

    # Row 1, Col 3: Occlusion Heatmap Only
    im1 = axes[0, 2].imshow(occlusion, cmap="hot")
    axes[0, 2].set_title(
        "Occlusion Heatmap\n(blocking these areas hurts prediction)",
        fontweight="bold",
        fontsize=12,
    )
    axes[0, 2].axis("off")
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, label="Sensitivity")

    # Row 2, Col 1: LIME Overlay
    lime_mask = get_lime_explanation(model, img_array, num_samples=200, num_features=5)
    lime_img = mark_boundaries(img_array, lime_mask, color=(0, 1, 0), mode="thick")
    axes[1, 0].imshow(lime_img)
    axes[1, 0].set_title(
        "LIME Explanation\n(green = important regions)", fontweight="bold", fontsize=12
    )
    axes[1, 0].axis("off")

    # Row 2, Col 2: LIME Mask Only
    axes[1, 1].imshow(lime_mask, cmap="tab20", interpolation="nearest")
    axes[1, 1].set_title(
        "LIME Superpixel Segmentation\n(colors = identified regions)",
        fontweight="bold",
        fontsize=12,
    )
    axes[1, 1].axis("off")

    # Row 2, Col 3: Prediction Confidence Bar Chart
    axes[1, 2].barh(
        class_names,
        pred * 100,
        color=["green" if i == pred_class else "gray" for i in range(len(class_names))],
    )
    axes[1, 2].set_xlabel("Confidence (%)", fontweight="bold")
    axes[1, 2].set_title(
        "Prediction Confidence\nfor All Classes", fontweight="bold", fontsize=12
    )
    axes[1, 2].set_xlim(0, 100)
    for i, v in enumerate(pred * 100):
        axes[1, 2].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: {output_path}")


# Generate visualizations
print("\n" + "=" * 70)
print("GENERATING XAI VISUALIZATIONS")
print("=" * 70)

for class_name, img_path in sample_images.items():
    output_path = output_dir / f"xai_example_{class_name}.png"
    create_xai_visualization(img_path, class_name, output_path)

# Create summary figure
print("\n" + "=" * 70)
print("Creating summary figure...")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    "XAI Occlusion Sensitivity - All Waste Categories", fontsize=18, fontweight="bold"
)

for idx, (class_name, img_path) in enumerate(sample_images.items()):
    if idx >= 6:
        break

    row = idx // 3
    col = idx % 3

    img_array = load_and_preprocess_image(img_path)
    pred = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
    pred_class = np.argmax(pred)
    pred_confidence = pred[pred_class]

    print(f"  Processing {class_name} for summary...")
    occlusion = get_occlusion_sensitivity(model, img_array, patch_size=30, stride=15)

    axes[row, col].imshow(img_array)
    axes[row, col].imshow(occlusion, cmap="hot", alpha=0.5)

    correct = "✓" if class_names[pred_class] == class_name else "✗"
    axes[row, col].set_title(
        f"{class_name.upper()} {correct}\n"
        f"Pred: {class_names[pred_class]} ({pred_confidence * 100:.0f}%)",
        fontweight="bold",
        fontsize=12,
    )
    axes[row, col].axis("off")

plt.tight_layout()
summary_path = output_dir / "xai_summary_all_classes.png"
plt.savefig(summary_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Summary saved: {summary_path}")

print("\n" + "=" * 70)
print("✅ XAI VISUALIZATION GENERATION COMPLETE")
print("=" * 70)
print(f"\nGenerated {len(sample_images)} XAI visualizations")
print("Each visualization includes:")
print("  • Original image")
print("  • Occlusion sensitivity overlay and heatmap")
print("  • LIME explanation and superpixel segmentation")
print("  • Prediction confidence for all classes")
print("\n✓ All 6 subplots are now properly filled!")
