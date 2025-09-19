import torch
import numpy as np

from utils.box_utils import bbox_iou, xywh2xyxy

def trans_vg_eval_val_one(pred_boxes, gt_boxes):
    # For single_box
    # batch_size = pred_boxes.shape[0]
    # pred_boxes = xywh2xyxy(pred_boxes)
    # pred_boxes = torch.clamp(pred_boxes, 0, 1)
    # gt_boxes = xywh2xyxy(gt_boxes)
    # iou = bbox_iou(pred_boxes, gt_boxes)
    # accu = torch.sum(iou >= 0.5) / float(batch_size)
    
    # For multi_box
    num_boxes = 0
    no_more_box = 0
    iou = torch.zeros(pred_boxes.size()[0])
    accu = torch.zeros(pred_boxes.size()[0])
    for i in range(gt_boxes.size()[0]):
        temp_zeros = torch.zeros_like(gt_boxes[i])
        if torch.equal(gt_boxes[i],temp_zeros):
            no_more_box = no_more_box + 1
        else:
            num_boxes  =  num_boxes + 1
            pred_boxes[i] = xywh2xyxy(pred_boxes[i])
            pred_boxes[i] = torch.clamp(pred_boxes[i], 0, 1)
            gt_boxes[i] = xywh2xyxy(gt_boxes[i])
            iou[i] = bbox_iou(pred_boxes[i].unsqueeze(0), gt_boxes[i].unsqueeze(0))
            accu[i] = torch.sum(iou[i] >= 0.5) 
    mean_iou =  iou / float(num_boxes)
    mean_accu = accu/ float(num_boxes)

    return iou, accu

def trans_vg_eval_val(pred_boxes, gt_boxes):
    # For single_box:
    # batch_size = pred_boxes.shape[0]
    # pred_boxes = xywh2xyxy(pred_boxes)
    # pred_boxes = torch.clamp(pred_boxes, 0, 1)
    # gt_boxes = xywh2xyxy(gt_boxes)
    # iou = bbox_iou(pred_boxes, gt_boxes)
    # accu_5 = torch.sum(iou >= 0.5) / float(batch_size)
    # accu_3 = torch.sum(iou >= 0.3) / float(batch_size)
    # accu_1 = torch.sum(iou >= 0.1) / float(batch_size)
    new_pred_boxes = torch.zeros_like(pred_boxes)
    new_gt_boxes = torch.zeros_like(gt_boxes)

    # For multi_box
    num_boxes = 0
    no_more_box = 0
    iou = torch.zeros(pred_boxes.size()[0])
    accu_5 = torch.zeros(pred_boxes.size()[0])
    accu_3 = torch.zeros(pred_boxes.size()[0])
    accu_1 = torch.zeros(pred_boxes.size()[0])

    # for gt
    # for i in range(gt_boxes.size()[0]):
        # temp_zeros = torch.zeros_like(gt_boxes[i])
        # if torch.equal(gt_boxes[i],temp_zeros):
        #     no_more_box = no_more_box + 1
    # for pred
    # for i in range(pred_boxes.size()[0]):
    for i in range(1):
    
        temp_zeros = torch.zeros_like(pred_boxes[i])
        if torch.equal(gt_boxes[i],temp_zeros):
            no_more_box = no_more_box + 1
        else:
            num_boxes  =  num_boxes + 1
            new_pred_boxes[i] = xywh2xyxy(pred_boxes[i])
            # (0-1)
            new_pred_boxes[i] = torch.clamp(new_pred_boxes[i], 0, 1)
            new_gt_boxes[i] = xywh2xyxy(gt_boxes[i])
            iou[i] = bbox_iou(new_pred_boxes[i].unsqueeze(0), new_gt_boxes[i].unsqueeze(0))
            accu_5[i] = torch.sum(iou[i] >= 0.5) 
            accu_3[i] = torch.sum(iou[i] >= 0.3) 
            accu_1[i] = torch.sum(iou[i] >= 0.1) 
    mean_iou =  torch.sum(iou) / float(num_boxes)
    mean_accu_5 = torch.sum(accu_5)/ float(num_boxes)
    mean_accu_3 = torch.sum(accu_3)/ float(num_boxes)
    mean_accu_1 = torch.sum(accu_1)/ float(num_boxes)
    # return iou, accu_5,accu_3,accu_1
    return mean_iou, mean_accu_5,mean_accu_3,mean_accu_1

def trans_vg_eval_val_imgsize(pred_boxes, gt_boxes,img_size):
    # For single_box:
    # batch_size = pred_boxes.shape[0]
    # pred_boxes = xywh2xyxy(pred_boxes)
    # pred_boxes = torch.clamp(pred_boxes, 0, 1)
    # gt_boxes = xywh2xyxy(gt_boxes)
    # iou = bbox_iou(pred_boxes, gt_boxes)
    # accu_5 = torch.sum(iou >= 0.5) / float(batch_size)
    # accu_3 = torch.sum(iou >= 0.3) / float(batch_size)
    # accu_1 = torch.sum(iou >= 0.1) / float(batch_size)
    new_pred_boxes = torch.zeros_like(pred_boxes)
    new_gt_boxes = torch.zeros_like(gt_boxes)

    # For multi_box
    num_boxes = 0
    no_more_box = 0
    iou = torch.zeros(pred_boxes.size()[0])
    accu_5 = torch.zeros(pred_boxes.size()[0])
    accu_3 = torch.zeros(pred_boxes.size()[0])
    accu_1 = torch.zeros(pred_boxes.size()[0])

    # for gt
    # for i in range(gt_boxes.size()[0]):
        # temp_zeros = torch.zeros_like(gt_boxes[i])
        # if torch.equal(gt_boxes[i],temp_zeros):
        #     no_more_box = no_more_box + 1
    # for pred
    # for i in range(pred_boxes.size()[0]):
    for i in range(1):
    
        temp_zeros = torch.zeros_like(pred_boxes[i])
        if torch.equal(gt_boxes[i],temp_zeros):
            no_more_box = no_more_box + 1
        else:
            num_boxes  =  num_boxes + 1
            new_pred_boxes[i] = xywh2xyxy(pred_boxes[i])
            # 0-image_size or (0-1)
            new_pred_boxes[i] = torch.clamp(new_pred_boxes[i], 0, img_size)
            new_gt_boxes[i] = xywh2xyxy(gt_boxes[i])
            iou[i] = bbox_iou(new_pred_boxes[i].unsqueeze(0), new_gt_boxes[i].unsqueeze(0))
            accu_5[i] = torch.sum(iou[i] >= 0.5) 
            accu_3[i] = torch.sum(iou[i] >= 0.3) 
            accu_1[i] = torch.sum(iou[i] >= 0.1) 
    mean_iou =  torch.sum(iou) / float(num_boxes)
    mean_accu_5 = torch.sum(accu_5)/ float(num_boxes)
    mean_accu_3 = torch.sum(accu_3)/ float(num_boxes)
    mean_accu_1 = torch.sum(accu_1)/ float(num_boxes)
    # return iou, accu_5,accu_3,accu_1
    return mean_iou, mean_accu_5,mean_accu_3,mean_accu_1


def trans_vg_eval_test(pred_boxes, gt_boxes, sum=True):
    # For single_box:
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu = torch.sum(iou >= 0.5) if sum else iou >= 0.5

    return iou, accu

def eval_category(category_id_list, iou, accu):
    # category_id_list包含 [1,2,3...8]id子类，此处需要 -1 表示从0编码
    category_id_list = category_id_list.cpu().numpy()
    sub = list(set(category_id_list)).__len__()
    iou = iou.cpu().numpy()
    accu = accu.cpu().numpy()
    category_iou = [0] * sub
    category_accu = [0] * sub
    sub_num = [0] * sub
    for (id, iou_, accu_) in zip(category_id_list, iou, accu):
        category_iou[id-1] += iou_
        category_accu[id-1] += accu_
        sub_num[id-1] += 1
    category_iou = [i / s for i, s in zip(category_iou, sub_num)]
    category_accu = [a / s for a, s in zip(category_accu, sub_num)]
    return category_iou, category_accu


