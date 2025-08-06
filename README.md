# uMedGround
* This repository provides the code for our submission paper "Uncertainty-aware Medical Diagnostic Phrase Identifying and Grounding" 
* Current Pre-implementation of [uMedGround](https://arxiv.org/abs/2404.06798)

## Introduction
Medical phrase grounding is crucial for identifying relevant regions in medical images based on phrase queries, facilitating accurate image analysis and diagnosis. However, current methods rely on manual extraction of key phrases from medical reports, reducing efficiency and increasing the workload for clinicians.  Additionally, the lack of model confidence estimation limits clinical trust and usability. In this paper, we introduce a novel task—Medical Report Grounding (MRG)—which aims to directly identify diagnostic phrases and their corresponding grounding boxes from medical reports in an end-to-end manner. To address this challenge, we propose uMedGround, a reliable framework that leverages a multimodal large language model (LLM) to predict diagnostic phrases by embedding a unique token, BOX, into the vocabulary to enhance detection capabilities. The embedded token, together with the input medical image, is decoded by a vision encoder-decoder to generate the corresponding grounding box. Critically, uMedGround incorporates an uncertainty-aware prediction model, significantly improving the robustness and reliability of grounding predictions. Experimental results demonstrate that uMedGround outperforms state-of-the-art medical phrase grounding methods and fine-tuned large visual-language models, validating its effectiveness and reliability. This study represents a pioneering exploration of the MRG task, marking the first-ever endeavor in this domain. Additionally, we explore the potential of uMedGround in grounded medical visual question answering and class-based localization applications.

<p align="center"> <img src="image/Visio-Method.png" width="100%"> </p>

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

##  :fire: NEWS :fire:

* [2025/08/06] We will release all codes and datasets as soon as possible.
* [2025/08/06] Our camera-ready paper was released first on the [arixv](https://arxiv.org/abs/2404.06798). 
* [2025/08/04] Our paper was accepted by IEEE TPAMI 2025, thanks all my collobrators. 
* [2025/06/26] Our related paer accepted by **ICCV 2025**: "GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis"[[Paper]](https://arxiv.org/abs/2411.16778)[[Link]](https://www.med-vqa.com/GEMeX/), Congrats to Bo ! !
* [2024/12/02] Our paper submitted to IEEE TPAMI. 

## Citation
If you find uMedGround helps your research, please cite our paper:
```
@InProceedings{uMedGround_Zou_2024,
author="Zou, Ke
and Bai, Yang
and Bo, Liu
and Yidi, Chen,
and Chen, Zhihao
and Chen, Yidi
and Yuan, Xuedong
and Wang, Meng
and Shen, Xiaojing
and Xiaochun Cao
and Yih Chung Tham
and Fu, Huazhu",
title="MedRG: Medical Report Grounding with Multi-modal Large Language Model",
journal={arXiv preprint arXiv:2404.06798},
year={2024}
}
```

## Acknowledgement
-  This work is built upon the  [LISA](https://github.com/dvlab-research/LISA)

## Contact
* If you have any problems about our work, please contact [me](kezou8@gmail.com) 
* Project Link: [UMedGround](https://github.com/Cocofeat/UMedGround/)
