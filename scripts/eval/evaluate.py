
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from birdwatching.dataset import BirdsDataset
from birdwatching.losses import sort_bbox
from birdwatching.metrics import calculate_AP
from birdwatching.model import MODEL_NAMES, build_model, default_checkpoint_path
from birdwatching.paths import resolve_data_root

N_CLASSES = 555
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model_name, checkpoint, split='val', img_size=256,
             batch_size=32, num_workers=0, data_root=None):
    root = resolve_data_root(data_root)
    ds = BirdsDataset(root, split, img_size, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(num_workers > 0))

    model = build_model(model_name, N_CLASSES, img_size=img_size).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    det_boxes, det_labels, det_scores = [], [], []
    true_boxes, true_labels = [], []
    correct = total = 0

    with torch.no_grad():
        for imgs, labels, bboxes, _ in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            cls_out, bbox_out = model(imgs)
            bbox_out = sort_bbox(bbox_out) #sort the bounding boxes by the confidence score
            probs = torch.softmax(cls_out, dim=1)
            pred_scores, pred_labels = probs.max(1)   #get the highest probability and the corresponding label

            correct += (pred_labels.cpu() == labels).sum().item()
            total   += labels.size(0)

            for i in range(labels.size(0)): #append the bounding boxes, labels, and scores to the list
                det_boxes.append(bbox_out[i].unsqueeze(0).cpu())
                det_labels.append(pred_labels[i].unsqueeze(0).cpu())
                det_scores.append(pred_scores[i].unsqueeze(0).cpu())
                true_boxes.append(bboxes[i].unsqueeze(0))
                true_labels.append(labels[i].unsqueeze(0))

    acc = correct / total
    results = calculate_AP(det_boxes, det_labels, det_scores,
                           true_boxes, true_labels, n_classes=N_CLASSES)
    print(f"Accuracy (top-1): {acc:.4f}")
    print(f"mAP (IoU=0.5)   : {results['mAP']:.4f}")
    return acc, results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=MODEL_NAMES, required=True)
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--split', choices=('train', 'val'), default='val')
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--data_root', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ckpt = args.checkpoint or default_checkpoint_path(args.model, 'best')
    evaluate(args.model, ckpt, args.split,
             img_size=args.img_size, batch_size=args.batch_size,
             num_workers=args.num_workers, data_root=args.data_root)
