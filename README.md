# uMedGround
* This repository provides the code for our submission paper "Uncertainty-aware Medical Diagnostic Phrase Identifying and Grounding" 
* Current Pre-implementation of [uMedGround](https://arxiv.org/abs/2404.06798)
* Abstrat: Medical phrase grounding is crucial for identifying relevant regions in medical images based on phrase queries, facilitating accurate image analysis and diagnosis. However, current methods rely on manual extraction of key phrases from medical reports, reducing efficiency and increasing the workload for clinicians.  Additionally, the lack of model confidence estimation limits clinical trust and usability. In this paper, we introduce a novel task—Medical Report Grounding (MRG)—which aims to directly identify diagnostic phrases and their corresponding grounding boxes from medical reports in an end-to-end manner. To address this challenge, we propose uMedGround, a reliable framework that leverages a multimodal large language model (LLM) to predict diagnostic phrases by embedding a unique token, BOX, into the vocabulary to enhance detection capabilities. The embedded token, together with the input medical image, is decoded by a vision encoder-decoder to generate the corresponding grounding box. Critically, uMedGround incorporates an uncertainty-aware prediction model, significantly improving the robustness and reliability of grounding predictions. Experimental results demonstrate that uMedGround outperforms state-of-the-art medical phrase grounding methods and fine-tuned large visual-language models, validating its effectiveness and reliability. This study represents a pioneering exploration of the MRG task, marking the first-ever endeavor in this domain. Additionally, we explore the potential of uMedGround in grounded medical visual question answering and class-based localization applications.
  
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
