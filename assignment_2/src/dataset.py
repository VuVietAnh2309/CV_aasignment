"""
PASCAL VOC Dataset Loader
Cung cap dataset chung cho tat ca cac model (Faster R-CNN, YOLOv3, YOLOv8).
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from xml.etree import ElementTree as ET

# 20 classes cua PASCAL VOC
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Map class name -> index (bat dau tu 1, 0 la background)
CLASS_TO_IDX = {cls: idx + 1 for idx, cls in enumerate(VOC_CLASSES)}
IDX_TO_CLASS = {idx + 1: cls for idx, cls in enumerate(VOC_CLASSES)}
IDX_TO_CLASS[0] = "background"

NUM_CLASSES = len(VOC_CLASSES) + 1  # +1 cho background


def parse_voc_annotation(annotation_path):
    """Parse file XML annotation cua VOC, tra ve boxes va labels."""
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = []
    labels = []
    difficulties = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()
        if class_name not in CLASS_TO_IDX:
            continue

        difficult = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_TO_IDX[class_name])
        difficulties.append(difficult)

    return {
        "boxes": np.array(boxes, dtype=np.float32),
        "labels": np.array(labels, dtype=np.int64),
        "difficulties": np.array(difficulties, dtype=np.int64),
    }


class VOCDetectionDataset(Dataset):
    """
    PASCAL VOC Detection Dataset.
    Compatible voi Faster R-CNN (torchvision).

    Args:
        root: duong dan den thu muc VOCdevkit/VOC2012
        image_set: "train", "val", hoac "trainval"
        transforms: torchvision transforms (optional)
    """

    def __init__(self, root, image_set="trainval", transforms=None):
        self.root = root
        self.transforms = transforms

        # Doc danh sach image IDs tu file split
        split_file = os.path.join(
            root, "ImageSets", "Main", f"{image_set}.txt"
        )
        with open(split_file, "r") as f:
            self.image_ids = [line.strip() for line in f.readlines() if line.strip()]

        self.image_dir = os.path.join(root, "JPEGImages")
        self.annotation_dir = os.path.join(root, "Annotations")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Load image
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        # Load annotation
        annotation_path = os.path.join(self.annotation_dir, f"{image_id}.xml")
        annotation = parse_voc_annotation(annotation_path)

        # Chuyen sang torch tensors
        target = {
            "boxes": torch.as_tensor(annotation["boxes"], dtype=torch.float32),
            "labels": torch.as_tensor(annotation["labels"], dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "iscrowd": torch.zeros(len(annotation["labels"]), dtype=torch.int64),
            "difficulties": torch.as_tensor(annotation["difficulties"], dtype=torch.int64),
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


def get_transform(train=False):
    """
    Tra ve transform pipeline.
    Faster R-CNN torchvision chi can ToTensor.
    """
    transforms = [ToTensorTransform()]
    if train:
        transforms.append(RandomHorizontalFlipTransform(prob=0.5))
    return Compose(transforms)


class Compose:
    """Compose nhieu transforms, truyen ca image va target."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensorTransform:
    """Chuyen PIL Image sang Tensor, normalize ve [0, 1]."""
    def __call__(self, image, target):
        image = T.functional.to_tensor(image)
        return image, target


class RandomHorizontalFlipTransform:
    """Random horizontal flip cho ca image va bounding boxes."""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if np.random.random() < self.prob:
            image = T.functional.hflip(image)
            width = image.shape[-1]
            boxes = target["boxes"]
            # Flip boxes: xmin, ymin, xmax, ymax -> width-xmax, ymin, width-xmin, ymax
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return image, target


def collate_fn(batch):
    """Custom collate cho DataLoader (moi image co so boxes khac nhau)."""
    return tuple(zip(*batch))


def download_voc_dataset(root="./data", year="2012"):
    """
    Download PASCAL VOC dataset bang torchvision.

    Args:
        root: thu muc goc de luu dataset
        year: "2007" hoac "2012"
    """
    from torchvision.datasets import VOCDetection

    print(f"Downloading PASCAL VOC {year}...")
    # Download trainval
    VOCDetection(root=root, year=year, image_set="trainval", download=True)
    print(f"PASCAL VOC {year} downloaded to {root}")

    # Tra ve duong dan den thu muc VOC
    voc_root = os.path.join(root, f"VOCdevkit/VOC{year}")
    return voc_root


def get_voc_datasets(root="./data", year="2012"):
    """
    Tao train/val datasets.

    Returns:
        train_dataset, val_dataset
    """
    voc_root = os.path.join(root, f"VOCdevkit/VOC{year}")

    if not os.path.exists(voc_root):
        voc_root = download_voc_dataset(root, year)

    train_dataset = VOCDetectionDataset(
        root=voc_root,
        image_set="train",
        transforms=get_transform(train=True),
    )
    val_dataset = VOCDetectionDataset(
        root=voc_root,
        image_set="val",
        transforms=get_transform(train=False),
    )

    return train_dataset, val_dataset


if __name__ == "__main__":
    # Test thu dataset
    train_ds, val_ds = get_voc_datasets()
    print(f"Train: {len(train_ds)} images")
    print(f"Val:   {len(val_ds)} images")

    # Xem 1 sample
    image, target = train_ds[0]
    print(f"Image shape: {image.shape}")
    print(f"Boxes: {target['boxes'].shape}")
    print(f"Labels: {target['labels']}")
    print(f"Classes: {[IDX_TO_CLASS[l.item()] for l in target['labels']]}")
