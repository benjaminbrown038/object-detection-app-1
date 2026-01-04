import torch
from utils.box_ops import box_iou

def nms(boxes, scores, iou_thresh=0.5):
    keep = []
    idxs = scores.argsort(descending=True)

    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.numel() == 1:
            break

        ious = torch.tensor([
            box_iou(boxes[i], boxes[j]) for j in idxs[1:]
        ])
        idxs = idxs[1:][ious < iou_thresh]

    return keep
