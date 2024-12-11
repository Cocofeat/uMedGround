# uMedGround
* This repository provides the code for our submission paper "Uncertainty-aware Medical Diagnostic Phrase Identifying and Grounding" 
* Current Pre-implementation of [uMedGround](https://arxiv.org/abs/2404.06798)

## Requirment
```pip install requirements.txt```
- torch==1.13.1
- torchvision==0.14.1
- packaging
- sentencepiece
- peft==0.4.0
- einops==0.4.1
- ...

## Public Datasets
* [MIMIC-CXR dataset](https://physionet.org/content/mimic-cxr/2.0.0/)
* [ChestX-ray8 dataset](https://ar5iv.labs.arxiv.org/html/1705.02315)
* [Medical-Diff-VQA dataset](https://physionet.org/content/medical-diff-vqa/1.0.0/)

## Curated Datasets
If the paper goes to substantive review, we promise to disclose all the datasets.

* [MRG-MS-CXR dataset]
* [MRG-ChestX-ray8 dataset]
* [MRG-MIMIC-VQA]
* [MRG-MIMIC-Class dataset]

## Code Usage
For now, we've only exposed the main_train and main_test code to get a feel for this manuscript.
If the paper goes to substantive review, we promise to disclose all the codes and pre-trained weights.

### Train
- Run the script ```python main_train.py```
### Test
- Run the script ```python main_test.py```

## Citation
If you find uMedGround helps your research, please cite our paper:
```
@InProceedings{uMedGround_Zou_2024,
author="Zou, Ke
and Bai, Yang
and Chen, Zhihao
and Chen, Yidi
and Ren, Kai
and Wang, Meng
and Yuan, Xuedong
and Shen, Xiaojing
and Fu, Huazhu",
title="MedRG: Medical Report Grounding with Multi-modal Large Language Model",
journal={arXiv preprint arXiv:2404.06798},
year={2024}
}
```
