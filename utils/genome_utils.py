import json
import os
import torch

ANATOMY = ['right lung', 'right upper lung zone', 'right mid lung zone', 'right lower lung zone', 'right hilar structures', 'right apical zone',
           'right costophrenic angle', 'right cardiophrenic angle', 'right hemidiaphragm', 'right clavicle', 'right cardiac silhouette', 'right atrium',
           'right upper abdomen', 'left lung', 'left upper lung zone', 'left mid lung zone', 'left lower lung zone', 'left hilar structures',
           'left apical zone', 'left costophrenic angle', 'left cardiophrenic angle', 'left hemidiaphragm', 'left clavicle', 'left cardiac silhouette', 'left upper abdomen',
           'trachea', 'spine', 'aortic arch', 'mediastinum', 'upper mediastinum', 'svc', 'cardiac silhouette', 'cavoatrial junction', 'descending aorta', 'carina', 'abdomen']


# get anatomy classification labels
def getCLSLabel(json_name, bbox, bs=640, js=224, thr=0.8):
    labels = [0]*ANATOMY.__len__()
    with open(json_name, 'r') as f:
        sample = json.loads(f.read())
        objects = sample['objects']
        for obj in objects:
            anatomy_box = [getInt(obj['x1'],bs,js), getInt(obj['y1'],bs,js), getInt(obj['x2'],bs,js), getInt(obj['y2'],bs,js)]
            flag = isInclude(bbox, anatomy_box, thr)
            idx = ANATOMY.index(obj['bbox_name'])
            labels[idx] = flag
    return labels

# box1 is or not in box2; with form of (x1, y1, x2, y2); box1 intersect box2 / box1 > thr
def isInclude(box1, box2, thr):
    box2 = torch.FloatTensor(box2)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    lt = torch.max(box1[:2], box2[:2])
    rb = torch.min(box1[2:], box2[2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[0] * wh[1]
    if inter/area1 >= thr or inter/area2 >= thr:
        return 1
    else:
        return 0

# get the resize anatomy box
def getInt(x, bs, js):
    return max(0, min(round(x * bs / js), bs))


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