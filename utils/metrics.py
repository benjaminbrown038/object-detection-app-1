import torch
from utils.box_ops import box_iou

def precision_recall(preds, gts, iou_thresh=0.5):
    tp = 0
    fp = 0
    fn = len(gts)

    for p in preds:
        matched = False
        for g in gts:
            if box_iou(p, g) > iou_thresh:
                tp += 1
                fn -= 1
                matched = True
                break
        if not matched:
            fp += 1

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return precision, recall
