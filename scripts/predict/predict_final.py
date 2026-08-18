
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from birdwatching.dataset import BirdsTestDataset
from birdwatching.inference_crop import _fused_crop_logits
from birdwatching.losses import sort_bbox
from birdwatching.model import build_model
from birdwatching.paths import CHECKPOINTS_DIR, PREDICTIONS_DIR, resolve_data_root

N_CLASSES = 555
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLS_CKPT = str(CHECKPOINTS_DIR / 'tinyvit_micro_final.pth')
LOC_CKPT = str(CHECKPOINTS_DIR / 'final_model_efficientnet_b0.pth')


def _crop_classify(model, imgs, bbox_src):
    _, _, H, W = imgs.shape
    crops = []
    for i in range(imgs.size(0)):
        x1n, y1n, x2n, y2n = bbox_src[i].tolist()
        pad = 0.05 * min(H, W)
        cx1 = max(0, int(x1n * W - pad))
        cy1 = max(0, int(y1n * H - pad))
        cx2 = min(W, int(x2n * W + pad)) #calculate the coordinates of the crop
        cy2 = min(H, int(y2n * H + pad))
        if cx2 <= cx1: cx2 = min(W, cx1 + 2)
        if cy2 <= cy1: cy2 = min(H, cy1 + 2)
        patch = imgs[i:i+1, :, cy1:cy2, cx1:cx2]
        crops.append(F.interpolate(patch, size=(H, W), mode='bilinear', align_corners=False))
    logits, _ = model(torch.cat(crops, 0))
    return logits


def unflip_bbox(b):
    x1, y1, x2, y2 = b.unbind(1)
    return torch.stack([1 - x2, y1, 1 - x1, y2], 1)


def predict(args):
    root = resolve_data_root(args.data_root)
    ds = BirdsTestDataset(root, img_size=256)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)

    cls_model = build_model('tinyvit_micro', N_CLASSES, img_size=256).to(DEVICE)
    cls_model.load_state_dict(torch.load(CLS_CKPT, map_location=DEVICE))
    cls_model.eval()

    loc_model = build_model('efficientnet_b0', N_CLASSES, img_size=256).to(DEVICE)
    loc_model.load_state_dict(torch.load(LOC_CKPT, map_location=DEVICE))
    loc_model.eval()

    class_names = {}
    with open(os.path.join(root, 'classes.txt')) as f:
        for line in f:
            idx, name = line.strip().split(maxsplit=1)
            class_names[int(idx)] = name #map the index to the class name

    rows = []
    with torch.no_grad():
        for batch_idx, (imgs, img_ids) in enumerate(loader):
            imgs = imgs.to(DEVICE, non_blocking=True)
            imgs_f = torch.flip(imgs, dims=[3])

            lg_cls,  bbox_cls  = _fused_crop_logits(cls_model, imgs,   0.05, 0.5, 0.5)
            lg_cls_f, bbox_cls_f = _fused_crop_logits(cls_model, imgs_f, 0.05, 0.5, 0.5)

            _, bbox_loc_raw  = loc_model(imgs)
            bbox_loc = sort_bbox(bbox_loc_raw)
            _, bbox_loc_raw_f = loc_model(imgs_f)
            bbox_loc_f = sort_bbox(bbox_loc_raw_f) #sort the bounding boxes by the confidence score

            lg_crop_loc   = _crop_classify(cls_model, imgs,   bbox_loc)
            lg_crop_loc_f = _crop_classify(cls_model, imgs_f, bbox_loc_f) #crop the image and classify the crop

            lg_fused = (lg_cls + lg_cls_f + lg_crop_loc + lg_crop_loc_f) / 4.0
            probs = F.softmax(lg_fused / 0.90, dim=1) #softmax the logits
            pred_scores, pred_labels = probs.max(1)

            bbox_final = (bbox_cls + unflip_bbox(bbox_cls_f) +
                          bbox_loc + unflip_bbox(bbox_loc_f)) / 4.0

            for i, img_id in enumerate(img_ids):
                x1, y1, x2, y2 = bbox_final[i].tolist()
                rows.append({'Id': img_id, 'Predictions': json.dumps({        #append 
                    'label': class_names[int(pred_labels[i])],
                    'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                    'score': float(pred_scores[i]),
                })})

            if (batch_idx + 1) % 50 == 0:
                print(f"{min((batch_idx+1)*args.batch_size, len(ds))}/{len(ds)}")

    out = PREDICTIONS_DIR / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Id', 'Predictions'])
        writer.writeheader()
        writer.writerows(rows)  #write to the csv file
    print(f"Saved {len(rows)} rows -> {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=str, default='prediction_final.csv')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0)    
    p.add_argument('--data_root', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    predict(parse_args())
