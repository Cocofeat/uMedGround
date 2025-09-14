import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# from pytorch_pretrained_bert.modeling import BertModel
from .visual_model.detr import build_detr_decoder
from .vlm_model.LISA import build_LISA
import copy
# from utils.box_utils import xywh2xyxy


class TransVG_ca(nn.Module):
    def __init__(self, args):
        super(TransVG_ca, self).__init__()
        hidden_dim = args.vl_hidden_dim
        divisor = 16 if args.dilation else 32
        self.num_visu_token = int((args.imsize / divisor) ** 2)
        self.num_text_token = args.max_query_len

        self.vis_vlmmodel = build_LISA(args)

        num_total = self.num_visu_token + self.num_text_token + 1
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)

        # self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        # self.text_proj = nn.Linear(self.vlmmodel.num_channels, hidden_dim)

        # self.num_classes = args.num_classes
        # self.num_queries = len(num_boxes)
        self.precision = args.precision
        self.vl_transformer = build_detr_decoder(args)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)


    def forward(self, input_dict):

        # LISA
        # Output: Visual_embeddings, pred_embeddings, ce_loss
        # Input: Visual_text dict
        
        LISA_output_dict = self.vis_vlmmodel(input_dict) 
        # Q-decoder
        output_dict = self.vl_transformer(LISA_output_dict)

        pred_box = self.bbox_embed(output_dict["pred_ems"]).sigmoid()
        return {'pred_box': pred_box}


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
