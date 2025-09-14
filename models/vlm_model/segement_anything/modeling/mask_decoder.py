# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        # num_multimask_outputs: int = 1,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1

        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.bbox_prediction_head = MLP(transformer_dim, transformer_dim, 4, 3)
        self.bbox_prediction_head_drop = MLP_drop(transformer_dim, transformer_dim, 4, 3)

        # self.output_upscaling = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
        #     ),
        #     LayerNorm2d(transformer_dim // 4),
        #     activation(),
        #     nn.ConvTranspose2d(
        #         transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
        #     ),
        #     activation(),
        # )
        # self.output_hypernetworks_mlps = nn.ModuleList(
        #     [
        #         MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        #         for i in range(self.num_mask_tokens)
        #     ]
        # )
        self.drop0_1 = nn.Dropout(0.1)
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        box = self.predict_box(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # # Select the correct mask or masks for output
        # if multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        # return masks, iou_pred
        return box


    def predict_box(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )

        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # image_embeddings: [1, C, H, W], tokens: [B, N, C]
        # dense_prompt_embeddings: [B, C, H, W]
        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src, att_weights = self.transformer(src, pos_src, tokens)
        # evidential_head
        reg_token = hs[:, 0, :]
        weights = att_weights[:, 0, :]
        weights = weights.view(b, int(h/4), int(w/4))
        # uncertain_box 
        reg_token2 = hs[:, 1, :]
        reg_token3 = hs[:, 2, :]

        pred_box = self.bbox_prediction_head(reg_token).sigmoid()
        pred_box2 = self.bbox_prediction_head(reg_token2).sigmoid()
        pred_box3 = self.bbox_prediction_head(reg_token3).sigmoid()
        
        # add drppout layer
        # pred_box = self.drop0_1(pred_box)
        # pred_box2 = self.drop0_1(pred_box2)
        # pred_box3 = self.drop0_1(pred_box3)

        # pred_box = self.bbox_prediction_head_drop(reg_token).sigmoid()
        # pred_box2 = self.bbox_prediction_head_drop(reg_token2).sigmoid()
        # pred_box3 = self.bbox_prediction_head_drop(reg_token3).sigmoid()

        pred_boxes = {pred_box2,pred_box3}
        
        # iou_token_out = hs[:, 0, :]
        # mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # # # Upscale mask embeddings and predict masks using the mask tokens
        # src = src.transpose(1, 2).view(b, c, h, w)
        # upscaled_embedding = self.output_upscaling(src)
        # hyper_in_list: List[torch.Tensor] = []
        # for i in range(self.num_mask_tokens):
        #     hyper_in_list.append(
        #         self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
        #     )
        # hyper_in = torch.stack(hyper_in_list, dim=1)
        # b, c, h, w = upscaled_embedding.shape
        # masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
        #     b, self.num_mask_tokens, h, w
        # )

        # # Generate mask quality predictions
        # iou_pred = self.iou_prediction_head(iou_token_out)

        # return masks, iou_pred
        # return pred_box,pred_boxes  # Uncertain box
        return pred_box,pred_boxes,weights  # Uncertain box

        # return pred_box,reg_token,weights  # Evidental box


# class MaskDecoder_Evidential(nn.Module):
#     def __init__(
#         self,
#         *,
#         transformer_dim: int,
#         transformer: nn.Module,
#         num_multimask_outputs: int = 3,
#         # num_multimask_outputs: int = 1,
#         activation: Type[nn.Module] = nn.GELU,
#         iou_head_depth: int = 3,
#         iou_head_hidden_dim: int = 256,
#     ) -> None:
#         """
#         Predicts masks given an image and prompt embeddings, using a
#         transformer architecture.

#         Arguments:
#           transformer_dim (int): the channel dimension of the transformer
#           transformer (nn.Module): the transformer used to predict masks
#           num_multimask_outputs (int): the number of masks to predict
#             when disambiguating masks
#           activation (nn.Module): the type of activation to use when
#             upscaling masks
#           iou_head_depth (int): the depth of the MLP used to predict
#             mask quality
#           iou_head_hidden_dim (int): the hidden dimension of the MLP
#             used to predict mask quality
#         """
#         super().__init__()
#         self.transformer_dim = transformer_dim
#         self.transformer = transformer

#         self.num_multimask_outputs = num_multimask_outputs

#         self.iou_token = nn.Embedding(1, transformer_dim)
#         self.num_mask_tokens = num_multimask_outputs + 1

#         self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
#         self.bbox_prediction_head = MLP(transformer_dim, transformer_dim, 4, 3)
#         # ---Evidential
#         self.transform_gamma = nn.Sequential(nn.ReLU(), nn.Linear(transformer_dim, transformer_dim), nn.ReLU(), nn.Linear(transformer_dim, 64), nn.ReLU(),
#                                              nn.Linear(64, 4))
#         self.transform_v = nn.Sequential(nn.ReLU(), nn.Linear(transformer_dim, transformer_dim), nn.ReLU(), nn.Linear(transformer_dim, 64), nn.ReLU(),
#                                              nn.Linear(64, 4))
#         self.transform_alpha = nn.Sequential(nn.ReLU(), nn.Linear(transformer_dim, transformer_dim), nn.ReLU(), nn.Linear(transformer_dim, 64), nn.ReLU(),
#                                              nn.Linear(64, 4))
#         self.transform_beta = nn.Sequential(nn.ReLU(), nn.Linear(transformer_dim, transformer_dim), nn.ReLU(), nn.Linear(transformer_dim, 64), nn.ReLU(),
#                                              nn.Linear(64, 4))

#         # self.output_upscaling = nn.Sequential(
#         #     nn.ConvTranspose2d(
#         #         transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
#         #     ),
#         #     LayerNorm2d(transformer_dim // 4),
#         #     activation(),
#         #     nn.ConvTranspose2d(
#         #         transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
#         #     ),
#         #     activation(),
#         # )
#         # self.output_hypernetworks_mlps = nn.ModuleList(
#         #     [
#         #         MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
#         #         for i in range(self.num_mask_tokens)
#         #     ]
#         # )

#         self.iou_prediction_head = MLP(
#             transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
#         )

#     def evidence(self, x):
#         return F.softplus(x)
    
#     def infer(self, input):
#         """
#         :param input: feature
#         :return: evidence 
#         """
#         inc_gamma = self.transform_gamma(input)
#         logv = self.transform_v(input)
#         logalpha = self.transform_alpha(input)
#         logbeta = self.transform_beta(input)

#         gamma = inc_gamma
#         v = self.evidence(logv)  # + 1.0
#         alpha = self.evidence(logalpha)
#         alpha = alpha + 1
#         beta = self.evidence(logbeta)

#         # The constraints
#         _ev_dec_alpha_max = 20.0
#         _ev_dec_v_max = 20.0
#         _ev_dec_beta_min = 0.2
#         alpha_thr = _ev_dec_alpha_max * torch.ones(alpha .shape).to(alpha .device)
#         alpha = torch.min(alpha, alpha_thr)
#         v_thr = _ev_dec_v_max * torch.ones(v.shape).to(v.device)
#         v = torch.min(v, v_thr)
#         beta_min = _ev_dec_beta_min * torch.ones(beta.shape).to(beta.device)
#         beta = beta + beta_min

#         return gamma, v, alpha, beta
    
#     def forward(
#         self,
#         image_embeddings: torch.Tensor,
#         image_pe: torch.Tensor,
#         sparse_prompt_embeddings: torch.Tensor,
#         dense_prompt_embeddings: torch.Tensor,
#         multimask_output: bool,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Predict masks given image and prompt embeddings.

#         Arguments:
#           image_embeddings (torch.Tensor): the embeddings from the image encoder
#           image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
#           sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
#           dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
#           multimask_output (bool): Whether to return multiple masks or a single
#             mask.

#         Returns:
#           torch.Tensor: batched predicted masks
#           torch.Tensor: batched predictions of mask quality
#         """
#         box, evidential_paras = self.predict_box(
#             image_embeddings=image_embeddings,
#             image_pe=image_pe,
#             sparse_prompt_embeddings=sparse_prompt_embeddings,
#             dense_prompt_embeddings=dense_prompt_embeddings,
#         )

#         # # Select the correct mask or masks for output
#         # if multimask_output:
#         #     mask_slice = slice(1, None)
#         # else:
#         #     mask_slice = slice(0, 1)
#         # masks = masks[:, mask_slice, :, :]
#         # iou_pred = iou_pred[:, mask_slice]

#         # Prepare output
#         # return masks, iou_pred
#         return box, evidential_paras


#     def predict_box(
#         self,
#         image_embeddings: torch.Tensor,
#         image_pe: torch.Tensor,
#         sparse_prompt_embeddings: torch.Tensor,
#         dense_prompt_embeddings: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Predicts masks. See 'forward' for more details."""
#         # Concatenate output tokens
#         output_tokens = torch.cat(
#             [self.iou_token.weight, self.mask_tokens.weight], dim=0
#         )
#         output_tokens = output_tokens.unsqueeze(0).expand(
#             sparse_prompt_embeddings.size(0), -1, -1
#         )

#         tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

#         # image_embeddings: [1, C, H, W], tokens: [B, N, C]
#         # dense_prompt_embeddings: [B, C, H, W]
#         # Expand per-image data in batch direction to be per-mask
#         src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
#         src = src + dense_prompt_embeddings
#         pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
#         b, c, h, w = src.shape

#         # Run the transformer
#         hs, src = self.transformer(src, pos_src, tokens)
#         # reg_token = hs[:, 0, :]
#         reg_token = hs[:, 1, :]
#         # reg_token = hs[:, 2, :]

#         pred_box = self.bbox_prediction_head(reg_token).sigmoid()

#         # if evidential_head == True:
#         gamma, v, alpha, beta = self.infer(reg_token)
#         evidential_output = dict()
#         evidential_output["gamma"] = gamma
#         evidential_output["v"] = v
#         evidential_output["alpha"] = alpha
#         evidential_output["beta"] = beta

#         # iou_token_out = hs[:, 0, :]
#         # mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

#         # # Upscale mask embeddings and predict masks using the mask tokens
#         # src = src.transpose(1, 2).view(b, c, h, w)
#         # upscaled_embedding = self.output_upscaling(src)
#         # hyper_in_list: List[torch.Tensor] = []
#         # for i in range(self.num_mask_tokens):
#         #     hyper_in_list.append(
#         #         self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
#         #     )
#         # hyper_in = torch.stack(hyper_in_list, dim=1)
#         # b, c, h, w = upscaled_embedding.shape
#         # masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
#         #     b, self.num_mask_tokens, h, w
#         # )

#         # # Generate mask quality predictions
#         # iou_pred = self.iou_prediction_head(iou_token_out)

#         # return masks, iou_pred
#         return pred_box,evidential_output 


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
# class MLP(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         output_dim: int,
#         num_layers: int,
#         sigmoid_output: bool = False,
#     ) -> None:
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(
#             nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
#         )
#         self.sigmoid_output = sigmoid_output

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         if self.sigmoid_output:
#             x = F.sigmoid(x)
#         return x
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

class MLP_drop(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.drop0_1 = nn.Dropout(0.1)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            x = self.drop0_1(x)
        return x