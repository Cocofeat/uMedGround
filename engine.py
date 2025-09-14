# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
from PIL import Image
import os
import sys
import torch
import torch.distributed as dist

from tqdm import tqdm
from typing import Iterable

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils
import numpy as np
from utils.box_utils import xywh2xyxy


def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, device: torch.device, 
                    epoch: int, max_norm: float = 0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 70

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        img_data, text_data, target, info = batch

        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        
        # model forward
        output = model(img_data, text_data)

        pred_box = output['pred_box']

        loss_dict = loss_utils.trans_vg_loss(pred_box, target)
        losses = sum(loss_dict[k] for k in loss_dict.keys())
        if args.model_name == 'TransVG_ca':
            vg_visu = output['vg_visu']
            vg_text = output['vg_text']
            text_mask = output['text_mask']
            # 根据target box确定要pooling的正样本视觉特征
            if 'lcp' in args.CAMode:
                pos_pool, att_pos = loss_utils.visuPooling(vg_visu, target, output['attn_output_weights'])  # 8x256
            else:
                pos_pool = loss_utils.visuPooling(vg_visu, target)  # 8x256
            # 根据NegBBoxs确定要pooling的负样本视觉特征
            bs = text_mask.shape[0]
            NegBBoxs = torch.cat([torch.tensor([info[i]['NegBBoxs']]) for i in range(bs)])  # bs, neg_num, 4
            NegBBoxs = NegBBoxs.to(device)
            neg_pools = []
            att_negs = []
            for i in range(NegBBoxs.shape[1]):  #neg_num
                if 'lcp' in args.CAMode:
                    neg_pool, att_neg = loss_utils.visuPooling(vg_visu, NegBBoxs[:, i, :], output['attn_output_weights'])  # 8x256
                    att_negs.append(att_neg.unsqueeze(dim=1))  # 5*8x421
                else:
                    neg_pool = loss_utils.visuPooling(vg_visu, NegBBoxs[:, i, :])  # 8x256
                neg_pools.append(neg_pool.unsqueeze(dim=1))
            neg_pools = torch.cat(neg_pools, dim=1)  # 8x5x256
            if 'lcp' in args.CAMode:
                att_negs = torch.cat(att_negs, dim=1)  # 8x5x421

            # 根据text_mask确定要Pooling的文本特征或者直接取REG特征
            if 'reg' in args.CAMode:
                # 用REG的特征当作RNN_pool
                rnn_pool = output['vg_reg']
            elif 'lcpTriple' in args.CAMode:
                # 用test的特征当作RNN_pool, 并且把reg_att传出来
                rnn_pool, att_text, att_reg = loss_utils.textPooling(vg_text, text_mask, args.CATextPoolType, att_weights=output['attn_output_weights'], text_data=output['text_data'], lcpTriple="lcpTriple")
            elif 'lcp' in args.CAMode:
                # 用test的特征当作RNN_pool
                rnn_pool, att_text = loss_utils.textPooling(vg_text, text_mask, args.CATextPoolType, att_weights=output['attn_output_weights'], text_data=output['text_data'])
            else:
                # 用test的特征当作RNN_pool
                rnn_pool = loss_utils.textPooling(vg_text, text_mask, args.CATextPoolType, text_data=output['text_data'])

            # 权重调节函数，希望在中后期加入这个loss
            CAlossWeight = args.CAlossWeightBase * loss_utils.CAlossFunc(epoch, args.epochs)
            if 'ORI' in args.CAMode:
                loss_ca_text = 0
            elif 'batch' in args.CAMode:
                loss_ca_text = loss_utils.trans_vg_caloss_crossbatch(pos_pool, neg_pools, rnn_pool)
            elif 'lcpW' in args.CAMode:  # lcp with layer projection
                loss_ca_text = loss_utils.trans_vg_caloss_inimage_lcp(pos_pool, neg_pools, rnn_pool, att_pos, att_negs, att_text, output['vg_hs'],\
                    wc1=output['wc1'])  # Text-Image and lcp enhanced loss
            elif 'lcpTriple' in args.CAMode:
                loss_ca_text = loss_utils.trans_vg_caloss_inimage_lcp_triple(pos_pool, neg_pools, rnn_pool, att_pos, att_negs, att_text, att_reg, output['vg_hs'], temp3=args.CATemperature)  # Text-Image and lcp enhanced loss
            elif 'lcp' in args.CAMode:
                loss_ca_text = loss_utils.trans_vg_caloss_inimage_lcp(pos_pool, neg_pools, rnn_pool, att_pos, att_negs, att_text, output['vg_hs'], temp3=args.CATemperature)  # Text-Image and lcp enhanced loss
            elif 'image' in args.CAMode:
                loss_ca_text = loss_utils.trans_vg_caloss_inimage(pos_pool, neg_pools, rnn_pool)  # Text-Image
            else:
                raise ValueError('No this mode')
             
            # loss_ca_text = 0
            losses = losses + loss_ca_text * CAlossWeight
            loss_dict['loss_ca_text'] = loss_ca_text

            if 'conBox' in args.CAMode:
                loss_cons_bbox, loss_cons_giou = loss_utils.trans_vg_conBox(output['pred_box'], output['pred_box_fool'])
                losses = losses + (loss_cons_bbox+loss_cons_giou)*args.ConsLossWeightBase
                loss_dict['loss_cons_bbox'] = loss_cons_bbox
                loss_dict['loss_cons_giou'] = loss_cons_giou



        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v
                                      for k, v in loss_dict_reduced.items()}
        losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
        loss_value = losses_reduced_unscaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'
    print_freq = 30

    for batch in metric_logger.log_every(data_loader, print_freq, header):        
        # copy to GPU
        img_data, text_data, target, info = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        
        # model forward
        output = model(img_data, text_data)
        pred_boxes = output['pred_box']
        miou, accu = eval_utils.trans_vg_eval_val(pred_boxes, target)
        
        metric_logger.update_v2('miou', torch.mean(miou).tolist(), batch_size)
        metric_logger.update_v2('accu', accu.tolist(), batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


@torch.no_grad()
def evaluate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    category_id_list = []
    bbox_save = {}
    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target, info = batch  # info: img_path, phrase_record, anno_id, category_id
        # @3 start
        # text_data.mask = text_data.mask.flip(dims=[0])
        # text_data.tensors = text_data.tensors.flip(dims=[0])
        # @3 end

        batch_size = img_data.tensors.size(0)
        category_id = [item['category_id'] for item in info]
        category_id_list.extend(category_id)
        anno_id = [item['anno_id'] for item in info]
        
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        output = model(img_data, text_data)
        pred_boxes = output['pred_box']

        iou, _ = eval_utils.trans_vg_eval_test(pred_boxes, target)

        # 保存每一个box和对应的id
        for i in range(batch_size):
            pred_boxes_ = xywh2xyxy(pred_boxes.detach().cpu())*640
            box = list(np.array(pred_boxes_[i]))
            a_id = anno_id[i]
            attention_map = output['attn_output_weights'][i]
            attention_map_reg = np.asarray(attention_map[0].detach().cpu())
            # bbox_save[a_id] = [box, attention_map_reg]
            bbox_save[a_id] = {'pbox': box, 'attention_map': attention_map_reg}

        # visualization bbox
        if args.visualization:
            from visualization.visual_bbox import visualBBox
            from visualization.visual_MHA import visualMHA
            from utils.misc import make_dirs
            attn_output_weights = output['attn_output_weights']
            text_ids = text_data.decompose()[0]
            for i, item in enumerate(info):
                # @3 start
                # a=info[batch_size - i-1]
                # img_name, phrase_record, anno_id = item['img_path'], a['phrase_record'], item['anno_id']
                # @3 end
                img_name, phrase_record, anno_id = item['img_path'], item['phrase_record'], item['anno_id']
                img_PIL = Image.open(img_name)
                tmp = visualBBox(img_PIL, pred_boxes[i].cpu(), phrase_record, mode='output')
                imgWBBox = visualBBox(tmp, target[i].cpu(), phrase_record, mode='target', iou=iou[i].item())
                save_path = os.path.join(args.output_dir, 'visual', str(anno_id)+'_'+os.path.split(img_name)[-1])
                make_dirs(os.path.dirname(save_path))
                imgWBBox.save(save_path)

                # visual MHA
                if args.visual_MHA:
                    text_id, att = text_ids[i].cpu().numpy(), attn_output_weights[i].cpu().numpy()
                    save_path = os.path.join(args.output_dir, 'visual_MHA', str(anno_id)+'_'+os.path.split(img_name)[-1].replace('.jpg', ''))
                    make_dirs(save_path)
                    visualMHA(imgWBBox, phrase_record, anno_id, text_id, att, save_path)

        pred_box_list.append(pred_boxes.cpu())
        gt_box_list.append(target.cpu())

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    iou, accu = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes, sum=False)
    torch.save(bbox_save, './save_box_att/bbox_save.pth')

    # accu_tensor = torch.tensor(accu).to(device)
    # iou_tensor = torch.tensor(iou).to(device)
    category_id_list_tensor = torch.tensor(category_id_list).to(device)
    total_results = torch.tensor([total_num, torch.sum(accu), torch.sum(iou)]).to(device)
    iou_tensor = torch.tensor(iou).to(device)
    accu_tensor = torch.tensor(accu).to(device)
    
    torch.cuda.synchronize()
    dist.all_reduce(total_results)
    dist.all_reduce(category_id_list_tensor)
    dist.all_reduce(iou_tensor)
    dist.all_reduce(accu_tensor)
    # print(category_id_list_tensor)

    total_accuracy = float(torch.sum(total_results[1])) / float(total_results[0])
    total_iou = float(torch.sum(total_results[2])) / float(total_results[0])
    if args.dataset in ['MS_CXR', 'ChestXray8']:
        category_iou, category_accu = eval_utils.eval_category(category_id_list_tensor, iou_tensor, accu_tensor)
        results = {'total_accuracy': total_accuracy, 'total_iou': total_iou, 'category_iou':category_iou, 'category_accu':category_accu}
    else:
        results = {'total_accuracy': total_accuracy, 'total_iou': total_iou}
    return results
        