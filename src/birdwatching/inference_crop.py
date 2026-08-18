from __future__ import annotations

import torch
import torch.nn.functional as F

from birdwatching.losses import sort_bbox


def _fused_crop_logits(model, imgs, crop_padding, weight_full, weight_crop):

    logits1, bbox1 = model(imgs)
    bbox1 = sort_bbox(bbox1)
    _, _, H, W = imgs.shape
    crops = []
    for i in range(imgs.size(0)):
        x1n, y1n, x2n, y2n = bbox1[i].tolist()
        pad = crop_padding * min(H, W)
        cx1 = max(0, int(x1n * W - pad))
        cy1 = max(0, int(y1n * H - pad))
        cx2 = min(W, int(x2n * W + pad))
        cy2 = min(H, int(y2n * H + pad))
        if cx2 <= cx1: cx2 = min(W, cx1 + 2)
        if cy2 <= cy1: cy2 = min(H, cy1 + 2)
        patch = imgs[i:i+1, :, cy1:cy2, cx1:cx2]
        crops.append(F.interpolate(patch, size=(H, W), mode='bilinear', align_corners=False))
    logits2, _ = model(torch.cat(crops, 0))
    return weight_full * logits1 + weight_crop * logits2, bbox1
