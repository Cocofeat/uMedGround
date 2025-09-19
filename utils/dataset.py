import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from models.vlm_model.llava import conversation as conversation_lib
from models.vlm_model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from models.vlm_model.llava.mm_utils import tokenizer_image_token
from models.vlm_model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .reason_gro_dataset import ReasonGroDataset,save_box_image
# from .refer import REFER
# from .refer_seg_dataset import ReferSegDataset
# from .sem_seg_dataset import SemSegDataset
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)
from .reason_gro_dataset import xy2xywh_tuple
# from .vqa_dataset import VQADataset


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    temp_bboxes_list = []
    bboxes_list = []
    max_box_number = 3
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        bbox,
        resize,
        questions,
        sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        # for single-box
        # bboxes_list.append(bbox[0]) 
        
        # for multi-box begin
        # print(bbox.size())
        if bbox.size()[0] < max_box_number:
            zero_box_tensor = torch.zeros((max_box_number-bbox.size()[0],4),dtype=bbox.dtype)
            bbox = torch.cat([bbox,zero_box_tensor], dim=0)
        bboxes_list.append(bbox) 
        # for multi-box end

        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    box_token_idx = tokenizer("[BOX]", add_special_tokens=False).input_ids[0]
    box_token_mask = input_ids == box_token_idx
    for i in range(len(input_ids)):
        if len(input_ids[i])>512:
            print("input_id_lenth > 512")
    if box_token_mask.sum() < 2:
        print("k=1")
    # for i in range(len(input_ids)):
    #     ids_batch = input_ids[i]
    #     for j in range(ids_batch.shape[0]):


    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    # if inferences[0] == False:
    #     truncate_len = tokenizer.model_max_length - 255

    #     if input_ids.shape[1] > truncate_len:
    #         input_ids = input_ids[:, :truncate_len]
    #         targets = targets[:, :truncate_len]
    #         attention_masks = attention_masks[:, :truncate_len]

    # for i in range(len(temp_bboxes_list)):
    #     for j in range(len(temp_bboxes_list[i])):
    #         bboxes_list.append(temp_bboxes_list[i][j])

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "bboxes": torch.stack(bboxes_list, dim=0),
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
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
        dataset="vqa||reason_gro",
        sample_rate=[3, 1],
        reason_gro_data="ReasonGro|MS_CXR|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "reason_gro":
                self.all_datasets.append(
                    ReasonGroDataset(
                        train_data_type,
                        phrase_data,
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_gro_data,
                        explanatory,
                    )
                )
            # elif dataset == "vqa":
            #     self.all_datasets.append(
            #         VQADataset(
            #             base_image_dir,
            #             tokenizer,
            #             vision_tower,
            #             samples_per_epoch,
            #             precision,
            #             image_size,
            #             num_classes_per_sample,
            #             exclude_val,
            #             vqa_data,
            #         )
            #     )
            # elif dataset == "refer_seg":
            #     self.all_datasets.append(
            #         ReferSegDataset(
            #             base_image_dir,
            #             tokenizer,
            #             vision_tower,
            #             samples_per_epoch,
            #             precision,
            #             image_size,
            #             num_classes_per_sample,
            #             exclude_val,
            #             refer_seg_data,
            #         )
            #     )
            # elif dataset == "sem_seg":
            #     self.all_datasets.append(
            #         SemSegDataset(
            #             base_image_dir,
            #             tokenizer,
            #             vision_tower,
            #             samples_per_epoch,
            #             precision,
            #             image_size,
            #             num_classes_per_sample,
            #             exclude_val,
            #             sem_seg_data,
            #         )
            #     )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        ind = 0
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference

class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        train_data_type,
        phrase_data,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=600,
    ):
        self.train_data_type = train_data_type
        self.base_image_dir = base_image_dir
        self.phrase_data = phrase_data

        reason_gro_data, data_name, splits = val_dataset.split("|")

        splits = splits.split("_")
        images = []
        for split in splits:
            if self.train_data_type == "CheXpert":
                    images_split_path = os.path.join(
                    # base_image_dir, reason_gro_data, data_name, "ChestXray8_resized_" + split +".pth"
                    # base_image_dir, reason_gro_data, data_name, "ChestXray8_llama_position_new_resized_" + split +".pth"
                    base_image_dir, reason_gro_data, data_name, "ChestXray8_gpt_position_new_resized_" + split +".pth"

                    )
            else:
                    images_split_path = os.path.join(
                    base_image_dir, reason_gro_data, data_name, data_name + "_S_" + split +".pth"
                    )
            self.images = torch.load(images_split_path)["image_paths"]
            self.all_reports = torch.load(images_split_path)["all_reports"]
            # self.all_reports = torch.load(images_split_path)["all_report_answer"]
            self.bboxes = torch.load(images_split_path)["bboxes"]
            self.phrases = torch.load(images_split_path)["phrases"]        
        self.data_type = "reason_gro"
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        return len(self.images)

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
    
    def preprocess_box(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""

        x = x / self.image_size
        return x
    
    def __getitem__(self, idx):
        # Reson grounding box
        image_path = self.images[idx]

        image_path_add = "./ln_data/reason_gro/MS_CXR/" + image_path

        image = cv2.imread(image_path_add)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        all_reports, bboxes, phrases = self.all_reports, self.bboxes, self.phrases
        phrase = phrases[idx]
        if len(bboxes[idx]) == 4:
            bbox = [bboxes[idx]]
        else:
            bbox = bboxes[idx]
        
        # For input box and phrase equal to the first one.
        # if len(bbox)>1:
        #     for i in range(len(bbox)):
        #         bbox[i] = bbox[0]
        #         phrase[i] = phrase[0]    

        if self.phrase_data:
            # phrase as input
            sampled_sents = list([phrases[idx]+"."])
        else:
            # report as input
            sampled_sents = all_reports[idx]

        is_sentence  = True
        answers =[]
        phrase_answers=[]
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        text = sampled_sents[0]

        if is_sentence:
            conv.append_message(
                conv.roles[0],
                DEFAULT_IMAGE_TOKEN
                # Single-box from one report
                # + "\n {} Please extract the phrase and output bounding box from this report.".format(text),
                # Multi-box to multi-phrases
                + "\n {} Please divide this report into phrases and output bounding boxes.".format(text),

            )

            # For Multi
            if isinstance(phrase,list):
                # For Multi
                for i in range(len(phrase)):
                    phrase_answers.append("{}".format(phrase[i]) + '. ' +  "It is [BOX].")
                if len(phrase_answers)==2:
                    answers.append(phrase_answers[0]+' '+phrase_answers[1])
                elif len(phrase_answers)==3:
                    answers.append(phrase_answers[0]+' '+phrase_answers[1]+' '+phrase_answers[2])
                else:
                    answers = phrase_answers
                conv.append_message(conv.roles[1], answers[0])
            else:
                # For single
                conv.append_message(conv.roles[1], " {}".format(phrase) + '. ' + "It is [BOX].")
        else:
            conv.append_message(
                conv.roles[0],
                DEFAULT_IMAGE_TOKEN
                + "\n What is {} in this image? Please output grounding box.".format(
                    text
                ),
            )
            conv.append_message(conv.roles[1], "[BOX].")
        conversations.append(conv.get_prompt())

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        # Multi_box
        tensor_bbox = []
        for i in range(len(bbox)):
            if len(bbox[i])< 4:
                k = 1
            else:
                bbox[i] = xy2xywh_tuple(bbox[i])
                tensor_bbox.append(self.preprocess_box(torch.from_numpy(np.array(bbox[i]).reshape(-1, 2, 2))).reshape(-1, 4))  # preprocess box for decoder
            # bbox[i] = self.transform.apply_boxes(torch.from_numpy(np.array(bbox[i])),(self.image_size,self.image_size))  # preprocess box for decoder
        tensor_bbox = torch.stack(tensor_bbox).squeeze(1) 
        # bbox = torch.stack(bbox).squeeze(1) 

        # Single_box
        # bbox = xy2xywh_tuple(bbox)    
        # bbox = self.preprocess_box(torch.from_numpy(np.array(bbox).reshape(-1, 2, 2))).reshape(-1, 4).unsqueeze(0) 
        
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            tensor_bbox,
            resize,
            None,
            None,
            inference,
        )

class TestDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        phrase_data,
        base_image_dir,
        tokenizer,
        vision_tower,
        test_dataset,
        image_size=640,
    ):
        self.base_image_dir = base_image_dir
        self.phrase_data = phrase_data

        reason_gro_data, data_name, splits = test_dataset.split("|")

        splits = splits.split("_")
        images = []
        for split in splits:
            if split == "test": 
                images_split_path = os.path.join(
                    base_image_dir, reason_gro_data, data_name, data_name + '_S_' + split +".pth" # mimic
                    # base_image_dir, reason_gro_data, data_name, "ChestXray8_gpt_position_new_resized_" + split +".pth" # ChestXray8_gpt
                    # base_image_dir, reason_gro_data, data_name, "ChestXray8_llama_position_new_resized_" + split +".pth" # ChestXray8_llama

                )
            else:
                images_split_path = os.path.join(
                    base_image_dir, reason_gro_data, data_name, data_name + '_S_' + split +".pth"
                    # base_image_dir, reason_gro_data, data_name, "ChestXray8_gpt_position_new_resized_" + split +".pth"
                    # base_image_dir, reason_gro_data, data_name, "ChestXray8_llama_position_new_resized_" + split +".pth"

                )
            self.images = torch.load(images_split_path)["image_paths"]
            self.all_reports = torch.load(images_split_path)["all_reports"] # for report & report_class_answer & _S_report_Answer_Class_gpt_
            # self.all_reports = torch.load(images_split_path)["all_report_answer"] # QA input _S_report_Answer_gpt_
            # self.all_reports = torch.load(images_split_path)["all_report_class"] # class input _S_Class_

            self.bboxes = torch.load(images_split_path)["bboxes"]
            self.phrases = torch.load(images_split_path)["phrases"]        
        self.data_type = "reason_gro"
        
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        
        # for test the 0:1 two cases
        # del self.images[1:-1] 
        # del self.bboxes[1:-1]
        # del self.all_reports[1:-1]
        # del self.phrases[1:-1]

    def __len__(self):
        return len(self.images)

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
    
    def gaussian_noise(self, image, mean, var):

        image = np.array(image / 255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise

        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.

        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out * 255)

        return out

    def preprocess_box(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""

        x = x / self.image_size
        return x
    
    def __getitem__(self, idx):
        # Reson grounding box
        image_path = self.images[idx]

        image_path_add = "./ln_data/reason_gro/MS_CXR/" + image_path
        # image_path_add = image_path

        image = cv2.imread(image_path_add)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # add noise
        # noisy_image = self.gaussian_noise(image, mean=0, var=0.01)
        # noisy_image = self.gaussian_noise(image, mean=0, var=0.1)
        # np_image = noisy_image
        # image = np_image
        
        # normal
        np_image = image
        
        
        all_reports,  bboxes, phrases = self.all_reports, self.bboxes, self.phrases
        phrase = phrases[idx]
        if len(bboxes[idx]) == 4:
            bbox = [bboxes[idx]]
        else:
            bbox = bboxes[idx]
        
        # save_box_image(bbox,image,image_path.split("/")[-1],"patient")

        if self.phrase_data:
            # phrase as input
            sampled_sents = list([phrases[idx]+"."])
        else:
            # report as input
            sampled_sents = all_reports[idx]

        # For input box and phrase equal to the first one.
        # if len(bbox)>1:
        #     for i in range(len(bbox)):
        #         bbox[i] = bbox[0]
        #         phrase[i] = phrase[0]

        is_sentence  = True
        
        answers =[]
        phrase_answers=[]
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        if isinstance(sampled_sents,list):
            text = sampled_sents[0]
        else:
            text = sampled_sents

        if is_sentence:
            conv.append_message(
                conv.roles[0],
                DEFAULT_IMAGE_TOKEN
                # Single-box from one report
                # + "\n {} Please extract the phrase and output bounding box from this report.".format(text),
                # Multi-box to multi-phrases
                + "\n {} Please divide this report into phrases and output bounding boxes.".format(text),
                # + "\n {} Please divide this report into phrases and output bounding boxes.".format(text),

            )
            # conv.append_message(conv.roles[1], " {}".format(phrase) + '. ' + "It is [BOX].")
            # conv.append_message(conv.roles[1], " {}".format(phrase) + '. ')

            if isinstance(phrase,list):
                # For Multi
                for i in range(len(phrase)):
                    phrase_answers.append("{}".format(phrase[i]) + '. ' +  "It is [BOX].")
                if len(phrase_answers)==2:
                    answers.append(phrase_answers[0]+' '+phrase_answers[1])
                elif len(phrase_answers)==3:
                    answers.append(phrase_answers[0]+' '+phrase_answers[1]+' '+phrase_answers[2])
                else:
                    answers = phrase_answers
                conv.append_message(conv.roles[1], answers[0])
            else:
                # For single
                conv.append_message(conv.roles[1], " {}".format(phrase) + '. ' + "It is [BOX].")

        else:
            conv.append_message(
                conv.roles[0],
                DEFAULT_IMAGE_TOKEN
                + "\n What is {} in this image? Please output grounding box.".format(
                    text
                ),
            )
            conv.append_message(conv.roles[1], "[BOX].")
        conversations.append(conv.get_prompt())

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # Multi_box
        tensor_bbox = []
        for i in range(len(bbox)):
            if len(bbox[i])< 4:
                k = 1
            else:
                bbox[i] = xy2xywh_tuple(bbox[i])
                tensor_bbox.append(self.preprocess_box(torch.from_numpy(np.array(bbox[i]).reshape(-1, 2, 2))).reshape(-1, 4))  # preprocess box for decoder
            # bbox[i] = self.transform.apply_boxes(torch.from_numpy(np.array(bbox[i])),(self.image_size,self.image_size))  # preprocess box for decoder
        tensor_bbox = torch.stack(tensor_bbox).squeeze(1) 

        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            tensor_bbox,
            resize,
            None,
            np_image,
            inference,
        )
