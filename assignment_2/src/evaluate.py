"""
Evaluation Metrics cho Object Detection.
Dung chung cho tat ca model: Faster R-CNN, YOLOv3, YOLOv8.

Metrics:
- mAP@0.5
- mAP@0.5:0.95
- Precision / Recall per class
- Confusion Matrix
- Inference time (FPS)
"""

import time
import torch
import numpy as np
from collections import defaultdict

from dataset import VOC_CLASSES, IDX_TO_CLASS


def compute_iou(box1, box2):
    """
    Tinh IoU giua 2 tap boxes.

    Args:
        box1: (N, 4) tensor [xmin, ymin, xmax, ymax]
        box2: (M, 4) tensor [xmin, ymin, xmax, ymax]

    Returns:
        iou: (N, M) tensor
    """
    x1 = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0].unsqueeze(0))
    y1 = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1].unsqueeze(0))
    x2 = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2].unsqueeze(0))
    y2 = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3].unsqueeze(0))

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection

    return intersection / (union + 1e-6)


def compute_ap(recall, precision):
    """
    Tinh Average Precision bang phuong phap 11-point interpolation (VOC style).

    Args:
        recall: sorted recall values
        precision: corresponding precision values

    Returns:
        ap: average precision
    """
    # Them diem (0, 1) va (1, 0)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    # Dam bao precision giam dan (monotonically decreasing)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Tim cac diem ma recall thay doi
    indices = np.where(recall[1:] != recall[:-1])[0] + 1

    # Tinh AP = tong (delta_recall * precision)
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    return ap


def evaluate_detections(predictions, ground_truths, iou_threshold=0.5, num_classes=21):
    """
    Danh gia ket qua detection.

    Args:
        predictions: list of dicts, moi dict co:
            - boxes: (N, 4) tensor
            - labels: (N,) tensor
            - scores: (N,) tensor
        ground_truths: list of dicts, moi dict co:
            - boxes: (M, 4) tensor
            - labels: (M,) tensor
            - difficulties: (M,) tensor (optional)
        iou_threshold: nguong IoU de xac dinh TP/FP
        num_classes: so classes (bao gom background)

    Returns:
        results: dict chua mAP, per-class AP, precision, recall
    """
    # Thu thap tat ca detections va ground truths theo class
    class_detections = defaultdict(list)  # class_id -> [(score, is_tp)]
    class_num_gt = defaultdict(int)       # class_id -> so ground truths

    for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]
        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]
        gt_difficulties = gt.get("difficulties", torch.zeros(len(gt_labels)))

        # Dem so GT cho moi class (bo qua difficult)
        for label, diff in zip(gt_labels, gt_difficulties):
            if diff == 0:
                class_num_gt[label.item()] += 1

        # Voi moi class, match predictions voi ground truths
        for class_id in range(1, num_classes):
            # Lay predictions cua class nay
            pred_mask = pred_labels == class_id
            if pred_mask.sum() == 0:
                continue

            c_pred_boxes = pred_boxes[pred_mask]
            c_pred_scores = pred_scores[pred_mask]

            # Lay ground truths cua class nay
            gt_mask = gt_labels == class_id
            c_gt_boxes = gt_boxes[gt_mask]
            c_gt_difficulties = gt_difficulties[gt_mask]

            # Sort predictions theo score (cao -> thap)
            sorted_indices = torch.argsort(c_pred_scores, descending=True)
            c_pred_boxes = c_pred_boxes[sorted_indices]
            c_pred_scores = c_pred_scores[sorted_indices]

            # Danh dau GT da duoc matched
            matched = torch.zeros(len(c_gt_boxes), dtype=torch.bool)

            for pred_idx in range(len(c_pred_boxes)):
                score = c_pred_scores[pred_idx].item()

                if len(c_gt_boxes) == 0:
                    class_detections[class_id].append((score, False))
                    continue

                # Tinh IoU voi tat ca GT chua match
                ious = compute_iou(
                    c_pred_boxes[pred_idx].unsqueeze(0), c_gt_boxes
                )[0]

                best_iou, best_gt_idx = ious.max(0)

                if best_iou >= iou_threshold and not matched[best_gt_idx]:
                    if c_gt_difficulties[best_gt_idx] == 0:
                        class_detections[class_id].append((score, True))
                        matched[best_gt_idx] = True
                    # Bo qua difficult objects
                else:
                    class_detections[class_id].append((score, False))

    # Tinh AP cho moi class
    results = {}
    aps = []

    for class_id in range(1, num_classes):
        class_name = IDX_TO_CLASS.get(class_id, f"class_{class_id}")
        detections = class_detections[class_id]
        num_gt = class_num_gt[class_id]

        if num_gt == 0:
            results[class_name] = {"ap": 0.0, "precision": 0.0, "recall": 0.0, "num_gt": 0}
            continue

        # Sort theo score
        detections.sort(key=lambda x: x[0], reverse=True)
        scores = [d[0] for d in detections]
        is_tp = [d[1] for d in detections]

        # Tinh cumulative TP va FP
        tp_cumsum = np.cumsum(is_tp).astype(float)
        fp_cumsum = np.cumsum([not tp for tp in is_tp]).astype(float)

        recall = tp_cumsum / num_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = compute_ap(recall, precision)
        aps.append(ap)

        # Lay precision/recall tai diem cuoi cung
        final_precision = precision[-1] if len(precision) > 0 else 0.0
        final_recall = recall[-1] if len(recall) > 0 else 0.0

        results[class_name] = {
            "ap": ap,
            "precision": float(final_precision),
            "recall": float(final_recall),
            "num_gt": num_gt,
            "num_detections": len(detections),
        }

    # Tinh mAP
    mAP = np.mean(aps) if aps else 0.0
    results["mAP"] = float(mAP)

    return results


def evaluate_at_multiple_ious(predictions, ground_truths, num_classes=21):
    """
    Tinh mAP@0.5 va mAP@0.5:0.95 (COCO style).

    Returns:
        dict voi mAP_50, mAP_50_95, va per-threshold results
    """
    # mAP@0.5
    results_50 = evaluate_detections(predictions, ground_truths, iou_threshold=0.5, num_classes=num_classes)

    # mAP@0.5:0.95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    maps = []
    for iou_thresh in iou_thresholds:
        r = evaluate_detections(predictions, ground_truths, iou_threshold=iou_thresh, num_classes=num_classes)
        maps.append(r["mAP"])

    return {
        "mAP_50": results_50["mAP"],
        "mAP_50_95": float(np.mean(maps)),
        "per_class": results_50,
        "per_threshold_mAP": {f"{t:.2f}": m for t, m in zip(iou_thresholds, maps)},
    }


def measure_inference_time(model, dataloader, device, num_warmup=10, num_runs=100):
    """
    Do thoi gian inference trung binh va tinh FPS.

    Args:
        model: model da load weights
        dataloader: DataLoader
        device: torch device
        num_warmup: so lan chay warm-up (khong tinh)
        num_runs: so lan chay de do

    Returns:
        dict voi avg_time_ms va fps
    """
    model.eval()
    times = []

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_warmup + num_runs:
                break

            images = [img.to(device) for img in images]

            start = time.time()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            if i >= num_warmup:
                times.append(end - start)

    avg_time = np.mean(times) if times else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0

    return {
        "avg_time_ms": avg_time * 1000,
        "fps": fps,
        "num_runs": len(times),
    }


def build_confusion_matrix(predictions, ground_truths, iou_threshold=0.5, num_classes=21):
    """
    Xay dung confusion matrix cho object detection.

    Matrix size: (num_classes+1) x (num_classes+1)
    Hang cuoi = background (FP: detect nhung khong co GT)
    Cot cuoi = missed (FN: co GT nhung khong detect)

    Returns:
        confusion_matrix: numpy array (num_classes+1, num_classes+1)
    """
    # +1 cho background/missed
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]
        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue

        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            ious = compute_iou(pred_boxes, gt_boxes)

            # Match predictions voi ground truths (greedy matching)
            matched_gt = set()
            matched_pred = set()

            # Sort predictions theo score
            sorted_pred = torch.argsort(pred_scores, descending=True)

            for pred_idx in sorted_pred:
                pred_idx = pred_idx.item()
                if len(gt_boxes) > 0:
                    best_iou, best_gt = ious[pred_idx].max(0)
                    best_gt = best_gt.item()

                    if best_iou >= iou_threshold and best_gt not in matched_gt:
                        # Matched: pred_label vs gt_label
                        p_label = pred_labels[pred_idx].item()
                        g_label = gt_labels[best_gt].item()
                        matrix[p_label, g_label] += 1
                        matched_gt.add(best_gt)
                        matched_pred.add(pred_idx)
                    else:
                        # False positive (predicted but no match)
                        p_label = pred_labels[pred_idx].item()
                        matrix[p_label, 0] += 1  # background column
                else:
                    p_label = pred_labels[pred_idx].item()
                    matrix[p_label, 0] += 1

            # Missed ground truths (FN)
            for gt_idx in range(len(gt_labels)):
                if gt_idx not in matched_gt:
                    g_label = gt_labels[gt_idx].item()
                    matrix[0, g_label] += 1  # background row

        elif len(pred_boxes) > 0:
            # Tat ca predictions la FP
            for p_label in pred_labels:
                matrix[p_label.item(), 0] += 1

        elif len(gt_boxes) > 0:
            # Tat ca ground truths la FN
            for g_label in gt_labels:
                matrix[0, g_label.item()] += 1

    return matrix


def print_results(results, model_name="Model"):
    """In ket qua danh gia dep."""
    print(f"\n{'='*60}")
    print(f"  {model_name} - Evaluation Results")
    print(f"{'='*60}")
    print(f"  mAP@0.5:      {results['mAP_50']:.4f}")
    print(f"  mAP@0.5:0.95: {results['mAP_50_95']:.4f}")
    print(f"{'='*60}")

    print(f"\n  {'Class':<15} {'AP@0.5':>8} {'Prec':>8} {'Recall':>8} {'#GT':>6}")
    print(f"  {'-'*47}")

    per_class = results["per_class"]
    for cls in VOC_CLASSES:
        if cls in per_class:
            info = per_class[cls]
            print(
                f"  {cls:<15} {info['ap']:>8.4f} {info['precision']:>8.4f} "
                f"{info['recall']:>8.4f} {info['num_gt']:>6}"
            )

    print(f"{'='*60}\n")


def save_results(results, save_path):
    """Luu ket qua ra file JSON."""
    import json

    # Convert numpy types sang Python types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"Results saved to {save_path}")
