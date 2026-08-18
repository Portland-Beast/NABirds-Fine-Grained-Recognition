from __future__ import annotations




import torch
import torch.nn as nn
import torch.nn.functional as F


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p_x1, p_y1, p_x2, p_y2 = pred.unbind(-1)
    t_x1, t_y1, t_x2, t_y2 = target.unbind(-1)

    inter_w = (torch.minimum(p_x2, t_x2) - torch.maximum(p_x1, t_x1)).clamp(min=0)
    inter_h = (torch.minimum(p_y2, t_y2) - torch.maximum(p_y1, t_y1)).clamp(min=0)  #calculate the intersection height
    inter   = inter_w * inter_h

    area_p = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)
    area_t = (t_x2 - t_x1).clamp(min=0) * (t_y2 - t_y1).clamp(min=0)  #calculate the area of the target bounding box
    union  = area_p + area_t - inter + 1e-9
    iou    = inter / union  #calculate the IoU

    enc_w = (torch.maximum(p_x2, t_x2) - torch.minimum(p_x1, t_x1)).clamp(min=0)
    enc_h = (torch.maximum(p_y2, t_y2) - torch.minimum(p_y1, t_y1)).clamp(min=0)  #calculate the encoding     
    area_enc = enc_w * enc_h + 1e-9

    return (1.0 - (iou - (area_enc - union) / area_enc)).mean()    
  

def sort_bbox(bbox: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = bbox.unbind(-1)
    return torch.stack([torch.minimum(x1, x2), torch.minimum(y1, y2),
                        torch.maximum(x1, x2), torch.maximum(y1, y2)], dim=-1)   #sort the bounding boxes by the confidence score


class BBoxLoss(nn.Module):

    def __init__(self, kind: str = 'giou'):
        super().__init__()
        self.kind = kind.lower()
        if self.kind not in ('smooth_l1', 'giou', 'l1+giou'):
            raise ValueError(f"Unknown bbox loss '{kind}'.")
        self._huber = nn.SmoothL1Loss()  #Huber loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.kind == 'smooth_l1':
            return self._huber(pred, target)
        if self.kind == 'giou':
            return giou_loss(pred, target)
        return 0.5 * self._huber(pred, target) + giou_loss(pred, target)
