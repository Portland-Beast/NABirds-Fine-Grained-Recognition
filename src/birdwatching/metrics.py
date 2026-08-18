"""PASCAL VOC mAP via calculate_AP. Lists are per image; leading dim is object count.
Boxes: normalized xy; labels: class indices from 0."""

from collections import defaultdict
import torch



def _box_iou_single(box_a: torch.Tensor, box_b: torch.Tensor) -> float:   #calculate the IoU between two bounding boxes

    x1 = max(box_a[0].item(), box_b[0].item())
    y1 = max(box_a[1].item(), box_b[1].item())
    x2 = min(box_a[2].item(), box_b[2].item())
    y2 = min(box_a[3].item(), box_b[3].item())

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, (box_a[2] - box_a[0]).item()) * max(0.0, (box_a[3] - box_a[1]).item())
    area_b = max(0.0, (box_b[2] - box_b[0]).item()) * max(0.0, (box_b[3] - box_b[1]).item())
    union = area_a + area_b - inter

    return inter / (union + 1e-9)


def _voc_ap(recalls, precisions) -> float:    #11-point interpolated AP as used in PASCAL VOC 2007.
    ap = 0.0
    for thresh in [t / 10.0 for t in range(11)]:
        prec_at = [p for r, p in zip(recalls, precisions) if r >= thresh]
        ap += max(prec_at, default=0.0)
    return ap / 11.0



def calculate_AP(
    det_boxes,  
    det_labels,  
    det_scores,  
    true_boxes,  
    true_labels,  
    n_classes: int,
    iou_threshold: float = 0.5,
):

       #class_dets[c] = list of (score, is_tp) tuples across all images
    class_dets: dict[int, list] = defaultdict(list)
    class_n_gt: dict[int, int]  = defaultdict(int)

    for img_det_boxes, img_det_labels, img_det_scores, img_true_boxes, img_true_labels in zip(
        det_boxes, det_labels, det_scores, true_boxes, true_labels
    ):
        img_det_boxes   = img_det_boxes.cpu()
        img_det_labels  = img_det_labels.cpu()
        img_det_scores  = img_det_scores.cpu()
        img_true_boxes  = img_true_boxes.cpu()
        img_true_labels = img_true_labels.cpu()


        for gt_lbl in img_true_labels:
            class_n_gt[int(gt_lbl)] += 1   #count the number of ground truth objects for each class

        if img_det_scores.numel() == 0:
            continue   #if there are no detections, skip the image

        gt_matched = [False] * len(img_true_labels)


        order = img_det_scores.argsort(descending=True)    #process the detections in descending score order
        for i in order:
            det_box   = img_det_boxes[i]
            det_lbl   = int(img_det_labels[i])
            det_score = float(img_det_scores[i])

            best_iou = iou_threshold
            best_j   = -1

            for j, (gt_box, gt_lbl) in enumerate(
                zip(img_true_boxes, img_true_labels)
            ):
                if int(gt_lbl) != det_lbl:
                    continue
                if gt_matched[j]:
                    continue
                iou = _box_iou_single(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j   = j

            is_tp = best_j >= 0
            if is_tp:
                gt_matched[best_j] = True

            class_dets[det_lbl].append((det_score, is_tp))   #append the detection score and the true positive flag to the class detections

    per_class_AP: dict[int, float] = {}

    for c in range(n_classes):
        n_gt = class_n_gt[c]
        if n_gt == 0:
            continue

        dets = class_dets[c]
        if not dets:
            per_class_AP[c] = 0.0
            continue

        dets.sort(key=lambda x: x[0], reverse=True)   #sort the detections by the score

        tp_cum = fp_cum = 0
        recalls    = []
        precisions = []

        for _, is_tp in dets:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            precisions.append(tp_cum / (tp_cum + fp_cum))   #calculate the precision
            recalls.append(tp_cum / n_gt)   #calculate the recall

        per_class_AP[c] = _voc_ap(recalls, precisions)

    mAP = sum(per_class_AP.values()) / max(len(per_class_AP), 1)

    return {'mAP': mAP, 'per_class_AP': per_class_AP} 
