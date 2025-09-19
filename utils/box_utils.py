import torch
from torchvision.ops.boxes import box_area
import random
import time


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # print(box1, box1.shape)
    # print(box2, box2.shape)
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def xywh2xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def xyxy2xywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2.0, (y0 + y1) / 2.0,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    if boxes1.size()[0] == 4:
        boxes1 = boxes1.unsqueeze(dim=0)
        boxes2 = boxes2.unsqueeze(dim=0)
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def sampleNegBBox(box, CAsampleType, CAsampleNum, category=0, w=640, h=640):
    assert CAsampleType in ['random', 'attention', 'crossImage', 'crossBatch']
    index = 0
    negBox_list = []
    # ori_center = [(box[0]+box[2])/2, (box[1]+box[3])/2]
    ori_w, ori_h = box[2]-box[0], box[3]-box[1]
    flag=0
    while index < CAsampleNum:
        flag += 1
        # print(flag)
        if CAsampleType == 'random':
            xNeg = torch.randint(1, w, (1,))
            yNeg = torch.randint(1, h, (1,))
            wNeg = ori_w + random.randint(torch.round(-ori_w * 0.1), torch.round(ori_w * 0.1))
            hNeg = ori_h + random.randint(torch.round(-ori_h * 0.1), torch.round(ori_h * 0.1))
        elif CAsampleType == 'attention':
            pass

        negBox = torch.zeros([4])
        negBox[0], negBox[1], negBox[2], negBox[3] = xNeg - 0.5 * wNeg, yNeg - 0.5 * hNeg, xNeg + 0.5 * wNeg, yNeg + 0.5 * hNeg
        negBox = torch.round(negBox)
        # 加入越界条件筛选 invalid bbox
        if negBox[0] < 0 or negBox[1] < 0 or negBox[2] >= w or negBox[3] >= h:
            continue
        # 加入box冲突条件筛选 invalid bbox
        iou, union = box_iou(box.unsqueeze(0), negBox.unsqueeze(0))
        if iou > 0.25 and flag < 300:
            continue
        negBox_list.append(negBox)
        index += 1
        
    return negBox_list
