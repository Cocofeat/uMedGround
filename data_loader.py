# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import re
# import cv2
import sys
import json
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
sys.path.append('.')

from PIL import Image
from transformers import AutoTokenizer, AutoModel
# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from transformers import BertTokenizer
from utils.word_utils import Corpus
from utils.box_utils import sampleNegBBox
from utils.genome_utils import getCLSLabel


def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_examples_to_features(examples, seq_length, tokenizer, usemarker=None):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            if usemarker is not None:
                # tokens_a = ['a', 'e', 'b', '*', 'c', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', '*', 'u']
                marker_idx = [i for i,x in enumerate(tokens_a) if x=='*']
                if marker_idx[1] > seq_length - 3 and len(tokens_a) - seq_length+1 < marker_idx[0]: #第二个*的下标不能大于17，且从后往前数第一个*不能溢出
                    tokens_a = tokens_a[-(seq_length-2):]
                    new_marker_idx = [i for i,x in enumerate(tokens_a) if x=='*']
                    if len(new_marker_idx) < 2:  #说明第一个marker被删掉了
                        pass
                elif len(tokens_a) - seq_length+1 >= marker_idx[0]:
                    max_len = min(marker_idx[1]-marker_idx[0]+1, seq_length-2)
                    tokens_a = tokens_a[marker_idx[0]: marker_idx[0]+max_len]
                    tokens_a[-1] = '*' #如果**的内容超出范围，强行把最后一位置为*
                elif marker_idx[1]-marker_idx[0]<2:
                    tokens_a = [i for i in tokens_a if i != '*']
                    tokens_a = ['*'] + tokens_a + ['*'] #如果**连在一起，把**放到首尾两端
                else:
                    if len(tokens_a) > seq_length - 2:
                        tokens_a = tokens_a[0:(seq_length - 2)]
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > seq_length - 2:
                    tokens_a = tokens_a[0:(seq_length - 2)]
                
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

class DatasetNotFoundError(Exception):
    pass

class TransVGDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test')
        },
        'MS_CXR': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'MS_CXR', 'split_by': 'MS_CXR'}
        },
        'ChestXray8': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'ChestXray8', 'split_by': 'ChestXray8'}
        },
        'SGH_CXR_V1': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'SGH_CXR_V1', 'split_by': 'SGH_CXR_V1'}
        }

    }

    def __init__(self, args, data_root, split_root='data', dataset='referit',
                 transform=None, return_idx=False, testmode=False,
                 split='train', max_query_len=128, lstm=False, 
                 bert_model='bert-base-uncased'):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.lstm = lstm
        self.transform = transform
        self.testmode = testmode
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.return_idx=return_idx
        self.args = args
        self.ID_Categories = {1: 'Cardiomegaly', 2: 'Lung Opacity', 3:'Edema', 4: 'Consolidation', 5: 'Pneumonia', 6:'Atelectasis', 7: 'Pneumothorax', 8:'Pleural Effusion'}

        assert self.transform is not None

        if split == 'train':
            self.augment = True
        else:
            self.augment = False

        if self.dataset == 'MS_CXR':
            self.dataset_root = osp.join(self.data_root, 'MS_CXR')
            self.im_dir = self.dataset_root  # 具体的图片路径保存在split中
        elif self.dataset == 'ChestXray8':
            self.dataset_root = osp.join(self.data_root, 'ChestXray8')
            self.im_dir = self.dataset_root  # 具体的图片路径保存在split中
        elif self.dataset == 'SGH_CXR_V1':
            self.dataset_root = osp.join(self.data_root, 'SGH_CXR_V1')
            self.im_dir = self.dataset_root  # 具体的图片路径保存在split中
        elif self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        elif  self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k_images')
        else:   ## refcoco, etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            self.split_dir = osp.join(self.dataset_root, 'splits')

        if not self.exists_dataset():
            # self.process_dataset()
            print('Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ')
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if self.lstm:
            self.corpus = Corpus()
            corpus_path = osp.join(dataset_path, 'corpus.pth')
            self.corpus = torch.load(corpus_path)

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item(self, idx):
        info = {}
        if self.dataset == 'MS_CXR':
            # anno_id, image_id, category_id, img_file, bbox, width, height, phrase, phrase_marker = self.images[idx]  # 核心三要素 img_file, bbox, phrase
            anno_id, image_id, category_id, img_file, bbox, width, height, phrase = self.images[idx]  # 核心三要素 img_file, bbox, phrase
            info['anno_id'] = anno_id
            info['category_id'] = category_id
        elif self.dataset == 'ChestXray8':
            anno_id, image_id, category_id, img_file, bbox, phrase, prompt_text = self.images[idx]  # 核心三要素 img_file, bbox, phrase
            info['anno_id'] = anno_id
            info['category_id'] = category_id
            # info['img_file'] = img_file
        elif self.dataset == 'SGH_CXR_V1':
            anno_id, image_id, category_id, img_file, bbox, phrase, patient_id = self.images[idx]  # 核心三要素 img_file, bbox, phrase
            info['anno_id'] = anno_id
            info['category_id'] = category_id
        elif self.dataset == 'flickr':
            img_file, bbox, phrase = self.images[idx]
        else:
            img_file, _, bbox, phrase, attri = self.images[idx]
        ## box format: to x1y1x2y2
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)

        # img_file = 'files/p12/p12423759/s53349935/b8c7a778-2f7f712d-5c598645-6aeebbb3-66ffbcc7.jpg'  # Experiments @fixImage
        if self.args.ablation == 'onlyText':
            img_file = 'files/p12/p12423759/s53349935/b8c7a778-2f7f712d-5c598645-6aeebbb3-66ffbcc7.jpg'

        img_path = osp.join(self.im_dir, img_file)
        info['img_path'] = img_path
        img = Image.open(img_path).convert("RGB")
        
        # img = cv2.imread(img_path)
        # ## duplicate channel if gray image
        # if img.shape[-1] > 1:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # else:
        #     img = np.stack([img] * 3)

        bbox = torch.tensor(bbox)
        bbox = bbox.float()
        # info['phrase_marker'] = phrase_marker
        return img, phrase, bbox, info

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox, info = self.pull_item(idx)
        # phrase = phrase.decode("utf-8").encode().lower()
        phrase = phrase.lower()
        if hasattr(self.args, 'CATextPoolType') and self.args.CATextPoolType == 'marker':
            # TODO
            phrase = info['phrase_marker']
        info['phrase_record'] = phrase  # for visualization  # info: img_path, phrase_record, anno_id, category_id
        input_dict = {'img': img, 'box': bbox, 'text': phrase}

        if self.args.model_name == 'TransVG_ca' and self.split == 'train':
            NegBBoxs = sampleNegBBox(bbox, self.args.CAsampleType, self.args.CAsampleNum)  # negative bbox
            
            input_dict = {'img': img, 'box': bbox, 'text': phrase, 'NegBBoxs': NegBBoxs}
        if self.args.model_name == 'TransVG_gn' and self.split == 'train':
            json_name = os.path.splitext(os.path.basename(info['img_path']))[0]+'_SceneGraph.json'
            json_name = os.path.join(self.args.GNpath, json_name)
            # 解析json, 得到所有的anatomy-level的分类label
            gnLabel = getCLSLabel(json_name, bbox)
            info['gnLabel'] = gnLabel

        input_dict = self.transform(input_dict)
        img = input_dict['img']
        bbox = input_dict['box']
        phrase = input_dict['text']
        img_mask = input_dict['mask']
        if self.args.model_name == 'TransVG_ca' and self.split == 'train':
            info['NegBBoxs'] = [np.array(negBBox, dtype=np.float32) for negBBox in input_dict['NegBBoxs']]
        
        if self.lstm:
            phrase = self.tokenize_phrase(phrase)
            word_id = phrase
            word_mask = np.array(word_id>0, dtype=int)
        else:
            ## encode phrase to bert input
            examples = read_examples(phrase, idx)
            if hasattr(self.args, 'CATextPoolType') and self.args.CATextPoolType == 'marker':
                use_marker = 'yes'
            else:
                use_marker = None
            features = convert_examples_to_features(
                examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer, usemarker=use_marker)
            word_id = features[0].input_ids
            word_mask = features[0].input_mask
            if self.args.ablation == 'onlyImage':
                word_mask = [0] * word_mask.__len__()  # experiments @2
            # if self.args.ablation == 'onlyText':
            #     img_mask = np.ones_like(np.array(img_mask))

        if self.testmode:
            return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
                np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0]
        else:
            return img, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32), info