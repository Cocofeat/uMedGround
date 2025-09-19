import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from models.vlm_model.llava import conversation as conversation_lib
from models.vlm_model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)

def xywh2xyxy_tuple(x):
    x_LeftTop, y_LeftTop, w, h = x
    b = [(x_LeftTop), (y_LeftTop),
         (x_LeftTop + w), (y_LeftTop + h)]
    return b[0:2],b[2:4]

def xycwh2xyxy_tuple(x):
    x_c, y_c, w, h = x
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return b[0:2],b[2:4]

def xyxy2xywh_tuple(x):
    x0, y0, x1, y1 = x
    b = [(x0 + x1) / 2.0, (y0 + y1) / 2.0,
         (x1 - x0), (y1 - y0)]
    return b

def xy2xywh_tuple(x):
    x0, y0, w, h = x
    b = [int(x0 + 0.5 * w), int(y0 + 0.5 * h),
         (w), (h)]
    return b

def save_box_image(box , image , image_name, class_name = "patient"):

    # tuple_ptLeftTop,tuple_ptRightBottom = xycwh2xyxy_tuple(box)
    # tuple_ptLeftTop,tuple_ptRightBottom = box[0:2],box[2:4]
    tuple_ptLeftTop,tuple_ptRightBottom = xywh2xyxy_tuple(box)
    ptLeftTop = np.array(tuple_ptLeftTop)
    ptRightBottom = np.array(tuple_ptRightBottom)

    # 框的颜色
    point_color = (0, 255, 0)
    # 线的厚度
    thickness = 2
    # 线的类型
    lineType = 4

    src = image
    # cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    src = np.array(src)
    # 画 b_box
    cv2.rectangle(src, tuple(ptLeftTop), tuple(ptRightBottom), point_color, thickness, lineType)

    # 获取文字区域框大小
    t_size = cv2.getTextSize(class_name, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
    # 获取 文字区域右下角坐标
    textlbottom = ptLeftTop + np.array(list(t_size))
    # 绘制文字区域矩形框
    cv2.rectangle(src, tuple(ptLeftTop), tuple(textlbottom),  point_color, -1)
    # 计算文字起始位置偏移
    ptLeftTop[1] = ptLeftTop[1] + (t_size[1]/2 + 4)
    # 绘字
    cv2.putText(src, class_name , tuple(ptLeftTop), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)
    # 打印图片的shape
    print(src.shape)

    cv2.imwrite("./box_data/" + "box_"+ image_name, src)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


class ReasonGroDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024 # img_size = 600 or 1024
    ignore_label = 255

    def __init__(
        self,
        train_data_type,
        phrase_data,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        reason_gro_data="ReasonGro|train",
        explanatory=0.1,
    ):
        self.train_data_type = train_data_type
        self.phrase_data = phrase_data
        self.exclude_val = exclude_val
        self.reason_gro_data = reason_gro_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        reason_gro_data, data_name, splits = reason_gro_data.split("|")
        splits = splits.split("_")
        images = []
        for split in splits:
            if self.train_data_type == "CheXpert":
                    images_split_path = os.path.join(

                    # base_image_dir, reason_gro_data, data_name, "ChestXray8_llama_position_new_resized_" + split +".pth" # resized Llama report data
                    base_image_dir, reason_gro_data, data_name, "ChestXray8_gpt_position_new_resized_" + split +".pth"

                    )
            else:
                    images_split_path = os.path.join(
                    base_image_dir, reason_gro_data, data_name, data_name + '_S_' + split +".pth" 
                    )
            images = torch.load(images_split_path)["image_paths"]
            all_reports = torch.load(images_split_path)["all_reports"]
            # all_reports = torch.load(images_split_path)["all_report_answer"]
            bboxes = torch.load(images_split_path)["bboxes"]
            phrases = torch.load(images_split_path)["phrases"]
        # del(images[1:-1])
        # del(bboxes[1:-1])
        # del(all_reports[1:-1])
        # del(phrases[1:-1])
        self.reason_gro_data = (images, all_reports, bboxes,phrases)
        print("number of reason_gro samples: ", len(images))

        # if explanatory != -1:
        #     self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
        #     self.img_to_explanation = {}
        #     for item in all_reports:
        #         self.img_to_explanation = {
        #             "outputs": item,
        #         }

        # print("len(self.img_to_explanation): ", len(self.img_to_explanation))

    def __len__(self):
        return self.samples_per_epoch

    def preprocess_box(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""

        x = x / self.image_size
        return x
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        images, all_reports, bboxes, phrases = self.reason_gro_data

        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        phrase = phrases[idx]
        if len(bboxes[idx]) == 4: 
            # For Single_input
            bbox = [bboxes[idx]]
        else:
            # For Multiple_input
            bbox = bboxes[idx]

        # For input box and phrase equal to the first one.
        # if len(bbox)>1:
        #     for i in range(len(bbox)):
        #         bbox[i] = bbox[0]
        #         phrase[i] = phrase[0]

        # Phrase or Report as input        
        if self.phrase_data:
            # phrase as input
            sents = list([phrases[idx]+"."])
        else:
            # report as input
            sents = all_reports[idx]    

        image_path_add = "./ln_data/reason_gro/MS_CXR/" + image_path

        image = cv2.imread(image_path_add)
        # cv2.imwrite("/data0/zouke/projects/MedRPG/box_data/" + image_path.split("/")[-1],image)
        # save_box_image(bbox,image,image_path.split("/")[-1],"patient")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # ori_size = image.shape[:2]
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # mask, sents, is_sentence = get_mask_from_json(json_path, image)

        is_sentence = True

        if len(sents) >= self.num_classes_per_sample:
            # original
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
            # for phrase
            # sampled_inds = list(range(len(sents)))

        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()

        image = self.transform.apply_image(image)  # preprocess image for sam

        resize = image.shape[:2]

        # if self.explanatory != -1:
        #     if random.random() < self.explanatory:
        #         choice = 1
        #     else:
        #         choice = random.randint(0, 1)
        choice = 1
        
        questions = []
        # answers = []
        # phrase_answers=[0 for x in range(0,len(phrase))] # for multi-conversations
        answers =[]
        phrase_answers=[]
        for text in sampled_sents:
            if is_sentence:
                question_template = random.choice(self.long_question_list)
                questions.append(question_template.format(sent=text))
            else:
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))

            # add explanation if applicable
            img_name = image_path.split("/")[-1]
            if self.explanatory != -1:
                if choice == 0:  # [BOX] token
                    answers.append(random.choice(self.answer_list))
                elif choice == 1:  # [BOX] token + text answer
                    
                    # for i in range(len(phrase)):
                    #     # for multi-conversations
                    #     # answers[i] = []
                    #     # answers[i].append(" {}".format(phrase[i]) + '.' +  random.choice(self.answer_list))
                    #     # for one-conversation
                    #     answers.append(" {}".format(phrase) + '. ' +  random.choice(self.answer_list))
                    # For single_phrase:
                    # answers.append("{}".format(phrase) + '. ' +  random.choice(self.answer_list))
                    if isinstance(phrase,list):
                        # for multi-pharse:
                        for i in range(len(phrase)):
                            phrase_answers.append("{}".format(phrase[i]) + '. ' +  random.choice(self.answer_list))
                        # Construct answers
                        if len(phrase_answers)==2:
                            answers.append(phrase_answers[0]+' '+phrase_answers[1])
                        elif len(phrase_answers)==3:
                            answers.append(phrase_answers[0]+' '+phrase_answers[1]+' '+phrase_answers[2])
                        else:
                            answers = phrase_answers
                    else:
                        answers.append("{}".format(phrase) + '. ' +  random.choice(self.answer_list))

                    # questions[-1] = (
                    #     DEFAULT_IMAGE_TOKEN
                    #     + "\n"
                    #     + text
                    #     + " {}".format(random.choice(self.explanatory_question_list))
                    # )
                elif choice == 2:  # vanilla text answer
                    phrase = phrases[idx]
                    for i in range(len(phrase)):
                        answers[i] = []
                        answers[i].append( " {}".format(answers[i]) + '.')
                else:
                    raise ValueError("Not implemented yet.")
            else:
                answers.append(random.choice(self.answer_list))

            conversations = []
            conv = conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], answers[i])
                conversations.append(conv.get_prompt())
                i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # for multi-boxes begin
        # print("Box number:%d, idx:%d"%(len(bbox),idx))
        tensor_bbox = []

        for i in range(len(bbox)):
            if len(bbox[i])< 4:
                k = 1
            else:
                bbox[i] = xy2xywh_tuple(bbox[i])
                tensor_bbox.append(self.preprocess_box(torch.from_numpy(np.array(bbox[i]).reshape(-1, 2, 2))).reshape(-1, 4))  # preprocess box for decoder
            # bbox[i] = self.transform.apply_boxes(torch.from_numpy(np.array(bbox[i])),(self.image_size,self.image_size))  # preprocess box for decoder
        tensor_bbox = torch.stack(tensor_bbox).squeeze(1) 
        # for multi-boxes end
        
        # for single-box
        # bbox = self.preprocess_box(torch.from_numpy(np.array(bbox).reshape(-1, 2, 2))).reshape(-1, 4).unsqueeze(0) 
        # cv2.imwrite("zero_ChestXray.jpg", np.array(image.detach().cpu().numpy()))
        return (
            image_path,
            image,
            image_clip,
            conversations,
            tensor_bbox,
            resize,
            questions,
            sampled_sents,
        )
