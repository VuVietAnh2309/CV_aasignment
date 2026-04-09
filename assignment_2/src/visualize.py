"""
Visualization tools cho Object Detection.
Hien thi bounding boxes, confusion matrix, PR curves, benchmark charts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image

from dataset import VOC_CLASSES, IDX_TO_CLASS

# Color palette cho 20 classes
COLORS = plt.cm.tab20(np.linspace(0, 1, 20))


def draw_boxes(image, boxes, labels, scores=None, gt_boxes=None, gt_labels=None,
               score_threshold=0.5, figsize=(12, 8), title=None):
    """
    Ve bounding boxes len anh.
    Xanh = prediction, Do = ground truth.

    Args:
        image: PIL Image hoac numpy array hoac torch tensor
        boxes: predicted boxes (N, 4)
        labels: predicted labels (N,)
        scores: confidence scores (N,) optional
        gt_boxes: ground truth boxes (M, 4) optional
        gt_labels: ground truth labels (M,) optional
        score_threshold: chi hien thi predictions co score >= threshold
    """
    # Convert image
    if hasattr(image, "numpy"):
        # torch tensor (C, H, W) -> (H, W, C)
        image = image.permute(1, 2, 0).numpy()
    elif isinstance(image, Image.Image):
        image = np.array(image)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    # Convert tensors to numpy
    if hasattr(boxes, "cpu"):
        boxes = boxes.cpu().numpy()
        labels = labels.cpu().numpy()
        if scores is not None:
            scores = scores.cpu().numpy()

    # Ve ground truth (do, net dut)
    if gt_boxes is not None:
        if hasattr(gt_boxes, "cpu"):
            gt_boxes = gt_boxes.cpu().numpy()
            gt_labels = gt_labels.cpu().numpy()

        for i, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
            x1, y1, x2, y2 = box
            class_name = IDX_TO_CLASS.get(int(label), f"cls_{label}")
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="red", facecolor="none", linestyle="--"
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"GT: {class_name}", fontsize=8,
                    color="white", bbox=dict(boxstyle="round,pad=0.2",
                                             facecolor="red", alpha=0.7))

    # Ve predictions (xanh)
    for i, (box, label) in enumerate(zip(boxes, labels)):
        score = scores[i] if scores is not None else 1.0
        if score < score_threshold:
            continue

        x1, y1, x2, y2 = box
        class_name = IDX_TO_CLASS.get(int(label), f"cls_{label}")
        color = COLORS[int(label) % 20]

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        label_text = f"{class_name}: {score:.2f}" if scores is not None else class_name
        ax.text(x1, y1 - 5, label_text, fontsize=8,
                color="white", bbox=dict(boxstyle="round,pad=0.2",
                                         facecolor=color, alpha=0.8))

    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_per_class_ap(results, model_name="Model", figsize=(14, 6)):
    """
    Bar chart: AP cho tung class.
    """
    per_class = results["per_class"]
    classes = []
    aps = []

    for cls in VOC_CLASSES:
        if cls in per_class and per_class[cls]["num_gt"] > 0:
            classes.append(cls)
            aps.append(per_class[cls]["ap"])

    # Sort theo AP
    sorted_indices = np.argsort(aps)[::-1]
    classes = [classes[i] for i in sorted_indices]
    aps = [aps[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.RdYlGn(np.array(aps))
    bars = ax.barh(range(len(classes)), aps, color=colors)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("AP@0.5")
    ax.set_title(f"{model_name} - Per-Class AP@0.5 (mAP={results['mAP_50']:.4f})")
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()

    # Hien thi gia tri tren bar
    for i, (bar, ap) in enumerate(zip(bars, aps)):
        ax.text(ap + 0.01, i, f"{ap:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(matrix, figsize=(14, 12), title="Confusion Matrix"):
    """
    Hien thi confusion matrix dang heatmap.
    """
    class_names = ["background"] + VOC_CLASSES

    # Chi hien thi classes co data
    row_mask = matrix.sum(axis=1) > 0
    col_mask = matrix.sum(axis=0) > 0
    mask = row_mask | col_mask
    filtered_matrix = matrix[np.ix_(mask, mask)]
    filtered_names = [class_names[i] for i in range(len(class_names)) if i < len(mask) and mask[i]]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        filtered_matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=filtered_names, yticklabels=filtered_names, ax=ax
    )
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_benchmark_comparison(all_results, figsize=(16, 10)):
    """
    So sanh benchmark giua cac model.

    Args:
        all_results: dict {model_name: {"mAP_50": ..., "mAP_50_95": ...,
                                         "fps": ..., "params": ..., "per_class": ...}}
    """
    model_names = list(all_results.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. mAP@0.5 comparison
    ax = axes[0, 0]
    maps_50 = [all_results[m]["mAP_50"] for m in model_names]
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    bars = ax.bar(model_names, maps_50, color=colors)
    ax.set_ylabel("mAP@0.5")
    ax.set_title("mAP@0.5 Comparison")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, maps_50):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", fontsize=10)

    # 2. mAP@0.5:0.95 comparison
    ax = axes[0, 1]
    maps_95 = [all_results[m]["mAP_50_95"] for m in model_names]
    bars = ax.bar(model_names, maps_95, color=colors)
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_title("mAP@0.5:0.95 Comparison")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, maps_95):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", fontsize=10)

    # 3. FPS comparison
    ax = axes[1, 0]
    fps_vals = [all_results[m].get("fps", 0) for m in model_names]
    bars = ax.bar(model_names, fps_vals, color=colors)
    ax.set_ylabel("FPS")
    ax.set_title("Inference Speed (FPS)")
    for bar, val in zip(bars, fps_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                f"{val:.1f}", ha="center", fontsize=10)

    # 4. Speed vs Accuracy scatter
    ax = axes[1, 1]
    for i, name in enumerate(model_names):
        ax.scatter(fps_vals[i], maps_50[i], s=200, c=[colors[i]],
                   edgecolors="black", zorder=5)
        ax.annotate(name, (fps_vals[i], maps_50[i]),
                    textcoords="offset points", xytext=(10, 5), fontsize=10)
    ax.set_xlabel("FPS")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Speed vs Accuracy Trade-off")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Object Detection Benchmark Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_per_class_comparison(all_results, figsize=(16, 8)):
    """
    So sanh AP per-class giua cac model (grouped bar chart).
    """
    model_names = list(all_results.keys())
    n_models = len(model_names)
    bar_width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=figsize)

    for i, model_name in enumerate(model_names):
        per_class = all_results[model_name]["per_class"]
        aps = []
        for cls in VOC_CLASSES:
            if cls in per_class and isinstance(per_class[cls], dict):
                aps.append(per_class[cls]["ap"])
            else:
                aps.append(0)

        x = np.arange(len(VOC_CLASSES))
        ax.bar(x + i * bar_width, aps, bar_width, label=model_name)

    ax.set_xticks(np.arange(len(VOC_CLASSES)) + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(VOC_CLASSES, rotation=45, ha="right")
    ax.set_ylabel("AP@0.5")
    ax.set_title("Per-Class AP Comparison")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_model_evolution_timeline(figsize=(16, 6)):
    """
    Ve timeline su phat trien cua cac model object detection.
    Dung cho phan literature review.
    """
    models = [
        ("R-CNN", 2014, "two-stage", "58.5"),
        ("Fast R-CNN", 2015, "two-stage", "70.0"),
        ("Faster R-CNN", 2015, "two-stage", "73.2"),
        ("YOLOv1", 2016, "one-stage", "63.4"),
        ("SSD", 2016, "one-stage", "74.3"),
        ("YOLOv3", 2018, "one-stage", "57.9"),
        ("YOLOv8", 2023, "one-stage", "53.9"),
    ]

    fig, ax = plt.subplots(figsize=figsize)

    for name, year, family, map_val in models:
        color = "#2196F3" if family == "two-stage" else "#FF9800"
        marker = "s" if family == "two-stage" else "o"
        ax.scatter(year, float(map_val), s=200, c=color, marker=marker,
                   edgecolors="black", zorder=5)
        ax.annotate(f"{name}\nmAP: {map_val}", (year, float(map_val)),
                    textcoords="offset points", xytext=(0, 15),
                    ha="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))

    # Legend
    ax.scatter([], [], s=100, c="#2196F3", marker="s", label="Two-Stage")
    ax.scatter([], [], s=100, c="#FF9800", marker="o", label="One-Stage")
    ax.legend(fontsize=11)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("mAP on VOC (paper-reported)", fontsize=12)
    ax.set_title("Evolution of Object Detection Models", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
