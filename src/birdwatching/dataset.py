from __future__ import annotations

import os
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


def _bbox_safe_random_crop(
    image: Image.Image,
    bbox_norm: torch.Tensor,
    min_scale: float = 0.6,
    max_scale: float = 1.0,
    p: float = 0.7,
) -> tuple[Image.Image, torch.Tensor]:      #random crop guaranteed to contain the GT bbox;re-normalizes bbox coords.

    if random.random() > p:
        return image, bbox_norm

    W, H = image.size
    x1, y1, x2, y2 = bbox_norm.tolist()
    bx1, by1, bx2, by2 = x1 * W, y1 * H, x2 * W, y2 * H   #calculate the bounding box coordinates
    bw, bh = bx2 - bx1, by2 - by1
    if bw <= 0 or bh <= 0:
        return image, bbox_norm

    scale = random.uniform(min_scale, max_scale)
    crop_w = min(max(int(W * scale), int(bw) + 1), W)
    crop_h = min(max(int(H * scale), int(bh) + 1), H)

    left_min = max(0, int(bx2) - crop_w)
    left_max = min(W - crop_w, int(bx1))
    top_min  = max(0, int(by2) - crop_h)
    top_max  = min(H - crop_h, int(by1))      
    if left_max < left_min or top_max < top_min:
        return image, bbox_norm

    left = random.randint(left_min, left_max)
    top  = random.randint(top_min,  top_max)
    image = image.crop((left, top, left + crop_w, top + crop_h))
    new_bbox = torch.tensor([
        (bx1 - left) / crop_w, (by1 - top) / crop_h,
        (bx2 - left) / crop_w, (by2 - top) / crop_h,
    ], dtype=torch.float32).clamp(0.0, 1.0)
    return image, new_bbox     #return the cropped image and the new bounding box coordinates


class BirdsDataset(Dataset):
    
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, root, split, img_size: int = 224,
                 augment: bool = False, aug_strength: str = 'basic'):
        self.root = root
        self.img_size = img_size 
        self.augment = augment
        self.aug_strength = aug_strength

        img_paths, class_labels, bboxes = {}, {}, {}
        with open(os.path.join(root, 'images.txt')) as f:
            for line in f:
                uid, path = line.strip().split(maxsplit=1)
                img_paths[uid] = path
        with open(os.path.join(root, 'image_class_labels.txt')) as f:
            for line in f:
                uid, lbl = line.strip().split()
                class_labels[uid] = int(lbl)
        with open(os.path.join(root, 'bounding_boxes.txt')) as f:
            for line in f:
                parts = line.strip().split()
                bboxes[parts[0]] = [float(v) for v in parts[1:5]]
        with open(os.path.join(root, f'{split}_split.txt')) as f:
            split_ids = [line.strip() for line in f]

        self.samples = [
            {'id': uid, 'path': img_paths[uid],
             'label': class_labels[uid], 'bbox': bboxes[uid]}
            for uid in split_ids if uid in class_labels and uid in bboxes
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(os.path.join(self.root, 'images', s['path'])).convert('RGB')
        W, H = image.size

        x1, y1, x2, y2 = s['bbox']
        bbox = torch.tensor(
            [x1 / W, y1 / H, x2 / W, y2 / H], dtype=torch.float32
        ).clamp(0.0, 1.0)

        if self.augment and self.aug_strength == 'strong':
            image, bbox = _bbox_safe_random_crop(image, bbox)

        image = TF.resize(image, (self.img_size, self.img_size))

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                x1n, y1n, x2n, y2n = bbox.tolist()
                bbox = torch.tensor([1.0 - x2n, y1n, 1.0 - x1n, y2n], dtype=torch.float32)
            image = TF.adjust_brightness(image, 1.0 + random.uniform(-0.3, 0.3))
            image = TF.adjust_contrast(image,   1.0 + random.uniform(-0.3, 0.3))
            image = TF.adjust_saturation(image, 1.0 + random.uniform(-0.3, 0.3))
            image = TF.adjust_hue(image, random.uniform(-0.1, 0.1))

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.MEAN, self.STD)
        return image, s['label'], bbox, s['id']


class BirdsTestDataset(Dataset):


    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, root, img_size: int = 224):
        self.root = root
        self.img_size = img_size

        img_paths = {}
        with open(os.path.join(root, 'images.txt')) as f:
            for line in f:
                uid, path = line.strip().split(maxsplit=1)   #split the line into the image id and the path
                img_paths[uid] = path
        with open(os.path.join(root, 'test_split.txt')) as f:
            test_ids = [line.strip() for line in f]   #get the test image ids

        self.samples = [{'id': uid, 'path': img_paths[uid]} for uid in test_ids]   #create the samples

    def __len__(self):
        return len(self.samples)   #return the number of samples

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(os.path.join(self.root, 'images', s['path'])).convert('RGB')
        image = TF.resize(image, (self.img_size, self.img_size))
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.MEAN, self.STD)
        return image, s['id']
