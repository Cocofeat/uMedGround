from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h,build_sam_vit_b
from .llava import conversation as conversation_lib
import transformers
from peft import LoraConfig, get_peft_model
from ..visual_model.detr import build_detr_decoder

from utils.box_utils import bbox_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou
from .sam_lora_image_encoder import _LoRA_qkv
import math

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.config.lora_sam_rank = kwargs.get("lora_sam_rank", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.config.lora_sam_rank = kwargs.get("lora_sam_rank", None)
            self.initialize_lisa_modules(self.config)
        # self.visumodel_decoder = build_detr_decoder(self.config)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def initialize_lisa_modules(self, config):
        # SAM
        if self.vision_pretrained == "./pretrained/medsam_vit_b.pth":
            self.visual_model = build_sam_vit_b(self.vision_pretrained)
        else:
            self.visual_model = build_sam_vit_h(self.vision_pretrained)

        # Lora_sam_encoder begin
        lora_sam_rank = config.lora_sam_rank
        # assert lora_sam_rank > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_sam_rank > 0 :
            self.lora_layer = list(
                    range(len(self.visual_model.image_encoder.blocks)))  # Only apply lora to the image encoder by default
            ## create for storage, then we can init them or load weights

            self.w_As = []  # These are linear layers
            self.w_Bs = []

            # freeze model parameters
            for param in self.visual_model.parameters():
                param.requires_grad = False

            # Here, we do the surgery for lora_sam_image_encoder
            for t_layer_i, blk in enumerate(self.visual_model.image_encoder.blocks):
                # If we only want few lora layer instead of all
                if t_layer_i not in self.lora_layer:
                    continue
                w_qkv_linear = blk.attn.qkv
                self.dim = w_qkv_linear.in_features
                w_a_linear_q = nn.Linear(self.dim, lora_sam_rank, bias=False)
                w_b_linear_q = nn.Linear(lora_sam_rank, self.dim, bias=False)
                w_a_linear_v = nn.Linear(self.dim, lora_sam_rank, bias=False)
                w_b_linear_v = nn.Linear(lora_sam_rank, self.dim, bias=False)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                blk.attn.qkv = _LoRA_qkv(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            self.reset_parameters()
            # Lora_sam_encoder end

        else:
            ## freeze model parameters
            for param in self.visual_model.parameters():
                param.requires_grad = False

        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        # if not hasattr(config, "train_mask_decoder"):
        #     config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
        #     config.mm_vision_tower = kwargs.get(
        #         "vision_tower", "openai/clip-vit-large-patch14"
        #     )
        #     self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        #     self.reg_loss_weight = kwargs.pop("reg_loss_weight", None)
        config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
        config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.reg_loss_weight = kwargs.pop("reg_loss_weight", None)
        self.evidential_loss_weight = kwargs.pop("evidential_loss_weight", None)
        self.uncertain_box_loss_weight = kwargs.pop("uncertain_box_loss_weight", None)

        self.llava_head = kwargs.pop("llava_head")
        self.llava_sam_head = kwargs.pop("llava_sam_head")
        self.uncertain_box_head = kwargs.pop("uncertain_box_head")
        self.mhp_box_head = kwargs.pop("mhp_box_head")
        self.Umhp_box_head = kwargs.pop("Umhp_box_head")

        # config.lora_sam_rank = kwargs.get("lora_sam_rank","0")
        self.box_token_idx = kwargs.pop("box_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bbox_prediction_head = MLP(256, 256, 4, 3)
        self.post_init()

    def evidence(self, x):
        return F.softplus(x)
    
    def infer(self, input):
        """
        :param input: feature
        :return: evidence 
        """
        inc_gamma = self.transform_gamma(input)
        logsigma = self.transform_sigma(input)
        logalpha = self.transform_alpha(input)
        logbeta = self.transform_beta(input)

        gamma = inc_gamma
        sigma_v = self.evidence(logsigma)  # + 1.0
        alpha = self.evidence(logalpha)
        alpha = alpha + 1
        beta = self.evidence(logbeta)

        # The constraints
        _ev_dec_alpha_max = 20.0
        _ev_dec_sigma_max = 20.0
        _ev_dec_beta_min = 0.2
        alpha_thr = _ev_dec_alpha_max * torch.ones(alpha .shape).to(alpha .device)
        alpha = torch.min(alpha, alpha_thr)
        sigma_v_thr = _ev_dec_sigma_max * torch.ones(sigma_v.shape).to(sigma_v.device)
        sigma_v = torch.min(sigma_v, sigma_v_thr)
        beta_min = _ev_dec_beta_min * torch.ones(beta.shape).to(beta.device)
        beta = beta + beta_min

        return gamma, sigma_v, alpha, beta
    
    def trans_evi_vg_loss(self, batch_pred, batch_target):
        """Compute the losses related to the bounding boxes, 
           including the L1 regression loss and the GIoU loss
        """
        batch_size = batch_pred.shape[0]
        # world_size = get_world_size()
        num_boxes = batch_size

        loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(
            xywh2xyxy(batch_pred),
            xywh2xyxy(batch_target)
            # batch_pred,
            # batch_target
        ))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes * 0.01
        losses['loss_giou'] = loss_giou.sum() / num_boxes 

        return losses

    def trans_vg_loss(self, batch_pred, batch_target):
        """Compute the losses related to the bounding boxes, 
           including the L1 regression loss and the GIoU loss
        """
        # for single_box
        # batch_size = batch_pred.shape[0]
        # # world_size = get_world_size()
        # num_boxes = batch_size

        # for multi_box
        num_boxes = 0
        no_more_box = 0
        loss_bbox = torch.zeros_like(batch_target[0])
        loss_giou = torch.zeros_like(batch_target[0])
        # for i in range(batch_target.size()[0]): # For multi
        for i in range(1): # For single
            temp_zeros = torch.zeros_like(batch_target[i])
            if torch.equal(batch_target[i],temp_zeros):
                no_more_box = no_more_box + 1
            else:
                num_boxes  =  num_boxes + 1
                loss_bbox += F.l1_loss(batch_pred[i], batch_target[i], reduction='none')
                loss_giou += 1 - torch.diag(generalized_box_iou(
                xywh2xyxy(batch_pred[i]),
                xywh2xyxy(batch_target[i])
                # batch_pred,
                # batch_target
                ))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        # losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['loss_giou'] = loss_giou[0].sum() / num_boxes

        return losses
    
    def Cal_distance(self, batch_pred, batch_target):
        """Compute the losses related to the bounding boxes, 
           including the L1 regression loss and the GIoU loss
        """
        # for single_box
        # batch_size = batch_pred.shape[0]
        # # world_size = get_world_size()
        # num_boxes = batch_size

        # for multi_box
        num_boxes = 0
        no_more_box = 0
        distance_bbox = torch.zeros_like(batch_target[0])
        # for i in range(batch_target.size()[0]): # For multi
        for i in range(1): # For single
            temp_zeros = torch.zeros_like(batch_target[i])
            if torch.equal(batch_target[i],temp_zeros):
                no_more_box = no_more_box + 1
            else:
                num_boxes  =  num_boxes + 1
                distance_bbox += F.l1_loss(batch_pred[i], batch_target[i], reduction='none')

        return distance_bbox
    
    def trans_uncertain_box_loss(self, batch_pred, batch_pred1, batch_pred2):
        """Compute the losses related to the bounding boxes, 
           including the L1 regression loss and the GIoU loss
        """
        batch_size = batch_pred.shape[0]
        # world_size = get_world_size()
        num_boxes = batch_size
        hinge_loss = nn.MarginRankingLoss(margin=0.01)
        # hinge_loss = nn.MarginRankingLoss(margin=0.001)
        # hinge_loss = nn.MarginRankingLoss(margin=0.005)

        loss_bbox_12 = hinge_loss(torch.abs(batch_pred - batch_pred1), torch.zeros_like(batch_pred), torch.ones_like(batch_pred))
        loss_bbox_13 = hinge_loss(torch.abs(batch_pred - batch_pred2), torch.zeros_like(batch_pred), torch.ones_like(batch_pred))
        loss_bbox_23 = hinge_loss(torch.abs(batch_pred1 - batch_pred2), torch.zeros_like(batch_pred), torch.ones_like(batch_pred))

        # loss_bbox_12 = max(0.01- torch.mean(torch.abs(batch_pred - batch_pred1)), 0)
        # loss_bbox_13 = max(0.01- torch.mean(torch.abs(batch_pred - batch_pred2)), 0)
        # loss_bbox_23 = max(0.01- torch.mean(torch.abs(batch_pred1 - batch_pred2)), 0)
        loss_bbox = loss_bbox_12 + loss_bbox_13 + loss_bbox_23
        # loss_giou = 1 - torch.diag(generalized_box_iou(
        #     xywh2xyxy(batch_pred),
        #     xywh2xyxy(batch_pred1)
        # ))

        losses = loss_bbox / num_boxes
        # losses['loss_bbox'] = loss_bbox / num_boxes
        # losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses
    
    def trans_uncertain_box_loss2(self, batch_pred, batch_pred1, batch_pred2):
        """Compute the losses related to the bounding boxes, 
           including the L1 regression loss and the GIoU loss
        """
        batch_size = batch_pred.shape[0]
        # world_size = get_world_size()
        num_boxes = batch_size
        hinge_loss = nn.MarginRankingLoss(margin=0.01)
        # hinge_loss = nn.MarginRankingLoss(margin=0.001)
        # hinge_loss = nn.MarginRankingLoss(margin=0.005)

        loss_bbox_12 = hinge_loss(torch.abs(batch_pred - batch_pred1), torch.zeros_like(batch_pred), torch.ones_like(batch_pred))
        loss_bbox_13 = hinge_loss(torch.abs(batch_pred - batch_pred2), torch.zeros_like(batch_pred), torch.ones_like(batch_pred))
        loss_bbox_23 = hinge_loss(torch.abs(batch_pred1 - batch_pred2), torch.zeros_like(batch_pred), torch.ones_like(batch_pred))

        # loss_bbox_12 = max(0.01- torch.mean(torch.abs(batch_pred - batch_pred1)), 0)
        # loss_bbox_13 = max(0.01- torch.mean(torch.abs(batch_pred - batch_pred2)), 0)
        # loss_bbox_23 = max(0.01- torch.mean(torch.abs(batch_pred1 - batch_pred2)), 0)
        loss_bbox = loss_bbox_12 + loss_bbox_13 + loss_bbox_23
        # loss_giou = 1 - torch.diag(generalized_box_iou(
        #     xywh2xyxy(batch_pred),
        #     xywh2xyxy(batch_pred1)
        # ))

        losses = loss_bbox / num_boxes
        # losses['loss_bbox'] = loss_bbox / num_boxes
        # losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses
    
    def trans_MHP_loss(self, batch_gt, batch_pred1, batch_pred2, sigma):
        """Compute the losses related to the bounding boxes, 
           including the L1 regression loss and the GIoU loss
        """
        M_Class = 3
        loss1 = self.trans_vg_loss(batch_gt, batch_pred1)
        loss2 = self.trans_vg_loss(batch_gt, batch_pred2)

        loss1 = {key: value * sigma/(M_Class-1) for key, value in self.trans_vg_loss(batch_gt, batch_pred1).items()}
        loss2 = {key: value * sigma/(M_Class-1) for key, value in self.trans_vg_loss(batch_gt, batch_pred2).items()}
        losses = {key: loss1[key] + loss2[key] for key in loss1}
                
        return losses

    def trans_UMHP_loss(self, batch_gt, batch_pred1, batch_pred2, sigma1, sigma2):
        """Compute the losses related to the bounding boxes, 
           including the L1 regression loss and the GIoU loss
        """
        loss1 = self.trans_vg_loss(batch_gt, batch_pred1)
        loss2 = self.trans_vg_loss(batch_gt, batch_pred2)

        loss1 = {key: value * sigma1 for key, value in self.trans_vg_loss(batch_gt, batch_pred1).items()}
        loss2 = {key: value * sigma2 for key, value in self.trans_vg_loss(batch_gt, batch_pred2).items()}
        losses = {key: loss1[key] + loss2[key] for key in loss1}
                
        return losses

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor, # image after clip_preprocess
        input_ids: torch.LongTensor, # tokenizer
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        bboxes: torch.FloatTensor,
        inference: bool = False,
        conversation_list: list=[],
        **kwargs,
    ):
        # print(bboxes.size())
        image_embeddings = self.get_visual_embs(images) 
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        box_token_mask = input_ids[:, 1:] == self.box_token_idx
        
        # Add one tensor [bs,1]
        box_token_mask = torch.cat(
            [
                box_token_mask,
                torch.zeros((box_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        box_token_mask = torch.cat(
            [torch.zeros((box_token_mask.shape[0], 255)).bool().cuda(), box_token_mask],
            dim=1,
        )

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            # output_ids = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                # output_ids.append(output_i.sequences)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings_hidden = last_hidden_state[box_token_mask]
        box_token_counts = box_token_mask.int().sum(-1)  # [bs, ]

        # box_token_counts

        box_token_offset = box_token_counts.cumsum(-1)
        box_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), box_token_offset], dim=0
        )

        # Box number leijia's tensoer
        box_token_offset = box_token_offset[offset] 

        pred_embeddings_ = []
        for i in range(len(box_token_offset) - 1):
            start_i, end_i = box_token_offset[i], box_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings_hidden[start_i:end_i])
        # pred_embeddings = pred_embeddings_

        multimask_output = False
        mean_boxes = []
        uncertainty_boxes = []
        if self.uncertain_box_head:
            boxes_2 = []
            boxes_1 = []
            uncertain_box_losses_list = []
            uncertain_box_loss = dict()
            all_boxes_list = []
            Auncertainty_boxes = []
        elif self.mhp_box_head or self.Umhp_box_head:
            boxes_2 = []
            boxes_1 = []
            all_boxes_list = []
            mhp_box_losses_list = []
            Auncertainty_boxes = []
            mhp_box_loss_dict = dict()
        pred_boxes = []
        pred_llava_boxes = []
        att_weights = []
        att_llava_weights = []
        losses_list = []
        losses_llava_list = []
        loss_dict = dict()
        loss_llava_dict = dict()

        for i in range(len(pred_embeddings_)):
            cur_pred_embeddings = pred_embeddings_[i]
            ## without maskdecoder
            if self.llava_head:
                pred_box = self.bbox_prediction_head(cur_pred_embeddings)
                att_weight = torch.zeros_like(pred_box) # no attention weigts
                pred_boxes.append(pred_box)
                att_weights.append(att_weight)
                ## for Sinle-box
                # loss_dict[i] = self.trans_vg_loss(pred_box, bboxes[i])
            # print("loss_bbox: %s; loss_giou: %s"%(loss_dict[i]['loss_bbox'].item(),loss_dict[i]['loss_giou'].item()))  
            # losses = sum(loss_dict[i][k] for k in loss_dict[i].keys())
            # pred_boxes.append(pred_box)
            # losses_list.append(losses)

            # with maskdecoder
            else:

                if min(cur_pred_embeddings.shape) != 0:
                    (
                        sparse_embeddings,
                        dense_embeddings,
                    ) = self.model.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        # text_embeds=pred_embeddings[i][0,:].unsqueeze(0).unsqueeze(0), # for first predembedding
                        text_embeds=cur_pred_embeddings.unsqueeze(1), # for all predembedding
                    )
                    sparse_embeddings = sparse_embeddings.to(cur_pred_embeddings.dtype)
                    pred_box, box_token, att_weight = self.model.visual_model.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=multimask_output,
                    )
                    pred_llava_box = self.bbox_prediction_head(cur_pred_embeddings)
                    pred_llava_boxes.append(pred_llava_box)
                    if self.uncertain_box_head or self.mhp_box_head or self.Umhp_box_head:
                        two_box = []
                        for box_id in box_token:
                            two_box.append(box_id)
                        # if len(two_box[0])>1:
                        #     boxes_1.append(two_box[0][0])
                        #     boxes_2.append(two_box[1][0])
                        #     all_boxes = torch.cat((pred_box, two_box[0][0], two_box[1][0]), 0)
                        #     all_boxes_list.append(all_boxes)
                        # else:
                        boxes_1.append(two_box[0])
                        boxes_2.append(two_box[1])
                        all_boxes = torch.cat((pred_box, two_box[0], two_box[1]), 0)
                        all_boxes_list.append(all_boxes)
                        mean_box =  torch.mean(all_boxes,0)
                        uncertainty_box = torch.var(all_boxes,0)
                        mean_boxes.append(mean_box.unsqueeze(0))
                        uncertainty_boxes.append(uncertainty_box.unsqueeze(0))
                        mean_all_boxes =  torch.cat((mean_box.unsqueeze(0), all_boxes), 0)
                        Auncertainty_box = torch.var(mean_all_boxes,0)
                        
                    pred_boxes.append(pred_box)
                    att_weights.append(att_weight)

        model_output = output
        gt_boxes = bboxes

        if inference:
            if self.uncertain_box_head or self.mhp_box_head or self.Umhp_box_head:
                return { 
                "mean_boxes":mean_boxes,
                "Auncertainty_boxes":Auncertainty_boxes,
                "uncertainty_boxes":uncertainty_boxes,
                "all_boxes":all_boxes_list,
                "att_weights":att_weights,
                "pred_boxes": pred_boxes,
                "gt_boxes": gt_boxes,}            
            else:
                if self.llava_sam_head:
                    return { 
                    "pred_llava_boxes": pred_llava_boxes,
                    "pred_boxes": pred_boxes,
                    "att_weights":att_weights,
                    "gt_boxes": gt_boxes,}
                else:
                    return { 
                    "pred_boxes": pred_boxes,
                    "att_weights":att_weights,
                    "gt_boxes": gt_boxes,}

        
        for i in range(len(pred_embeddings_)):
            # first llava-output-supervised:
            if self.llava_sam_head:
                loss_llava_dict[i] = self.trans_vg_loss(pred_llava_boxes[i], bboxes[i])
            ##for multi-boxes
            ##loss_dict = self.trans_vg_loss(pred_box, bboxes[i,:,:])

            ##for Sinle-box
            if self.uncertain_box_head:
                loss_dict[i] = self.trans_vg_loss(mean_boxes[i], bboxes[i])
                uncertain_box_loss[i] = self.trans_uncertain_box_loss(pred_boxes[i], boxes_1[i], boxes_2[i])   
                uncertain_box_losses_list.append(uncertain_box_loss[i])
            elif self.mhp_box_head:
                sigma_hyper_parameter = torch.tensor(0.05)
                loss_dict[i] = {key: value *  (1 - sigma_hyper_parameter) for key, value in self.trans_vg_loss(pred_boxes[i], bboxes[i]).items()}
                mhp_box_loss_dict[i] = self.trans_MHP_loss(bboxes[i], boxes_1[i], boxes_2[i], sigma_hyper_parameter)
            elif self.Umhp_box_head:
                Distance_1 = self.Cal_distance(bboxes[i],pred_boxes[i])
                Distance_2 = self.Cal_distance(bboxes[i],boxes_1[i])
                Distance_3 = self.Cal_distance(bboxes[i],boxes_2[i])
                norm_v1 = torch.tensor.linalg.norm(Distance_1)
                norm_v2 = torch.tensor.linalg.norm(Distance_2)
                norm_v3 = torch.tensor.linalg.norm(Distance_3)
                total_norm = norm_v1 + norm_v2 + norm_v3
                sigma_hyper_parameter_1 = norm_v1 / total_norm
                sigma_hyper_parameter_2 = norm_v2 / total_norm
                sigma_hyper_parameter_3 = norm_v3 / total_norm

                loss_dict[i] = {key: value *  sigma_hyper_parameter_1 for key, value in self.trans_vg_loss(pred_boxes[i], bboxes[i]).items()}
                mhp_box_loss_dict[i] = self.trans_UMHP_loss(bboxes[i], boxes_1[i], boxes_2[i], sigma_hyper_parameter_2,sigma_hyper_parameter_3)   
            else:
                loss_dict[i] = self.trans_vg_loss(pred_boxes[i], bboxes[i])
                    
            if self.llava_sam_head:
                losses_llava = sum(loss_llava_dict[i][k_m] for k_m in loss_llava_dict[i].keys())
                losses_llava_list.append(losses_llava)
            losses = sum(loss_dict[i][k] for k in loss_dict[i].keys())
            losses_list.append(losses)
            if self.mhp_box_head or self.Umhp_box_head:
                mhp_box_losses = sum(mhp_box_loss_dict[i][k] for k in mhp_box_loss_dict[i].keys())
                mhp_box_losses_list.append(mhp_box_losses)

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        box_reg_loss = sum(losses_list) * self.reg_loss_weight

        box_reg_llava_loss = sum(losses_llava_list)* self.reg_loss_weight

        if self.uncertain_box_head:
            box_uncertain_box_loss = sum(uncertain_box_losses_list) * self.uncertain_box_loss_weight
            total_loss = ce_loss + box_reg_loss + box_reg_llava_loss + box_uncertain_box_loss
        elif self.mhp_box_head or self.Umhp_box_head:
            mhp_box_loss = sum(mhp_box_losses_list) * self.uncertain_box_loss_weight
            total_loss = ce_loss + box_reg_loss + box_reg_llava_loss + mhp_box_loss
        else:
            if self.llava_sam_head:
                total_loss = ce_loss + box_reg_llava_loss + box_reg_loss
            else:
                total_loss = ce_loss + box_reg_loss

        if box_reg_loss == 0:
            print("K=1")

        if self.uncertain_box_head:
            return {
            "ce_loss": ce_loss,
            "box_reg_loss": box_reg_loss,
            "box_uncertain_box_loss": box_uncertain_box_loss,
            "box_reg_llava_loss": box_reg_llava_loss,
            "total_loss": total_loss,
            }
        elif self.mhp_box_head or self.Umhp_box_head:
            return {
            "ce_loss": ce_loss,
            "box_reg_loss": box_reg_loss,
            "box_mhp_box_loss": mhp_box_loss,
            "box_reg_llava_loss": box_reg_llava_loss,
            "total_loss": total_loss,
            }
        elif self.llava_sam_head:
            return {
            "ce_loss": ce_loss,
            "box_reg_loss": box_reg_loss,
            "box_reg_llava_loss": box_reg_llava_loss,
            "total_loss": total_loss,
            }
        else:
            return {
            "ce_loss": ce_loss,
            "box_reg_loss": box_reg_loss,
            "total_loss": total_loss,
            }

    def evaluate(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor, # image after clip_preprocess
        input_ids: torch.LongTensor, # tokenizer
        max_new_tokens: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        bboxes: torch.FloatTensor,
        inference: bool = False,
        conversation_list: list=[],
        **kwargs,
    ):
        output_dict=dict()
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            box_token_mask = output_ids[:, 1:] == self.box_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            box_token_mask = torch.cat(
                [
                    torch.zeros((box_token_mask.shape[0], 255)).bool().cuda(),
                    box_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings_hidden = last_hidden_state[box_token_mask]

            box_token_counts = box_token_mask.int().sum(-1)  # [bs, ]
            box_token_offset = box_token_counts.cumsum(-1)
            box_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), box_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(box_token_offset) - 1):
                start_i, end_i = box_token_offset[i], box_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings_hidden[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_boxes = []
            mean_boxes = []
            att_weights = []
            uncertainty_boxes = []
            Auncertainty_boxes = []
            if self.uncertain_box_head or self.mhp_box_head or self.Umhp_box_head:
                boxes_1 = []
                boxes_2 = []
                all_boxes_list = []
            for i in range(len(pred_embeddings)):
                # without maskdecoder
                # pred_box = self.bbox_prediction_head(pred_embeddings[i])
                # pred_boxes.append(pred_box)
                # with maskdecoder
                if min(pred_embeddings[i].shape) != 0:
                    (
                    sparse_embeddings,
                    dense_embeddings,
                    ) = self.model.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i].unsqueeze(1),
                    )
                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                    pred_box,box_token,att_weight = self.model.visual_model.mask_decoder(
                    # pred_box,box_token = self.model.visual_model.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=multimask_output,
                        )
                    if self.uncertain_box_head or self.mhp_box_head or self.Umhp_box_head:
                        two_box = []
                        for box_id in box_token:
                            two_box.append(box_id)
                        boxes_1.append(two_box[0])
                        boxes_2.append(two_box[1])
                        # if pred_box.size()[0] == 2: 
                        #     all_boxes = torch.cat((pred_box[-1].unsqueeze(0), two_box[0][-1].unsqueeze(0), two_box[1][-1].unsqueeze(0)), 0)
                        # else:
                        all_boxes = torch.cat((pred_box, two_box[0], two_box[1]), 0)

                        mean_box =  torch.mean(all_boxes,0)
                        uncertainty_box = torch.var(all_boxes,0)
                        mean_all_boxes =  torch.cat((mean_box.unsqueeze(0), all_boxes), 0)
                        Auncertainty_box = torch.var(mean_all_boxes,0)
                        mean_boxes.append(mean_box.unsqueeze(0))
                        uncertainty_boxes.append(uncertainty_box.unsqueeze(0))
                        Auncertainty_boxes.append(Auncertainty_box)
                        all_boxes_list.append(all_boxes)

                    pred_boxes.append(pred_box)
                    att_weights.append(att_weight)

        if self.uncertain_box_head or self.mhp_box_head or self.Umhp_box_head:
            return { 
            "output_ids": output_ids,
            "mean_boxes":mean_boxes,
            "Auncertainty_boxes":Auncertainty_boxes,
            "uncertainty_boxes":uncertainty_boxes,
            "all_boxes":all_boxes_list,
            "pred_boxes": pred_boxes,
            "att_weights":att_weights,
            "gt_boxes": bboxes,}            
        else:
            return {"pred_boxes": pred_boxes,
            "gt_boxes": bboxes,
            "att_weights":att_weights,
            "output_ids": output_ids,
            }
          

def build_LISA(args):

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    # num_added_tokens = tokenizer.add_tokens("[BOX]")
    args.box_token_idx = tokenizer("[BOX]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "box_token_idx": args.box_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    model.get_model().initialize_lisa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False
    
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    return model

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x