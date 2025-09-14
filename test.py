import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import shutil
import sys
import time
from functools import partial
import cv2
import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from models.vlm_model.segment_anything.utils.transforms import ResizeLongestSide
from models.vlm_model.LISA import LISAForCausalLM
from models.vlm_model.llava import conversation as conversation_lib
from utils.dataset import TestDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from utils.prompter import Prompter
import utils.eval_utils as eval_utils
from torch.utils.data import DataLoader, DistributedSampler
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
# from pycocoevalcap.eval import calculate_metrics
from matplotlib import cm
# import matplotlib

# local_env = os.environ.copy()
# local_env["PATH"]="/home/zou_ke/miniconda3/envs/coco/bin/" + local_env["PATH"]
# os.environ.update(local_env)



def parse_args(args):
    parser = argparse.ArgumentParser(description="uMedGround Model Testing/Training")
    parser.add_argument(
        # "--version", default="xinlai/LISA-7B-v1"
        # "--version", default="xinlai/LISA-13B-llama2-v1"
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
        # "--version", default="./LISA-13B/liuhaotian/llava-llama-2-13b-chat-lightning-preview"

    )
    # parser.add_argument("--vis_save_path", default="vis_output/ChestXray8/", type=str)
    parser.add_argument("--vis_save_path", default="vis_output/ChestXray8/MRG", type=str)

    parser.add_argument(
        "--precision",
        default="bf16", # fp32
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=640, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_sam_rank", default=0, type=int)    

    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )

    parser.add_argument(
        "--dataset", default="vqa||reason_gro", type=str
    )
    # parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument("--sample_rates", default="3,1", type=str)

    parser.add_argument("--reason_gro_data", default="ReasonGro|MS_CXR|train", type=str)
    parser.add_argument("--phrase_data", action="store_true", default=False)

    parser.add_argument("--test_dataset", default="ReasonGro|MS_CXR|test", type=str)
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="uMedGround", type=str)
    # parser.add_argument(
    #     "--batch_size", default=1, type=int, help="batch size per device per step"
    # )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    # parser.add_argument("--explanatory", default=-1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    # parser.add_argument("--vision_pretrained", default="/data0/zouke/projects/MedRPG/pretrained/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--vision_pretrained", default="/data0/zouke/projects/MedRPG/pretrained/medsam_vit_b.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    # parser.add_argument("--weight", default="./runs7/lisa/ckpt_model_UMRG-MHP_bs8/pytorch_model.bin", type=str) # UMRG mimic Best 47.65
    # parser.add_argument("--weight", default="./runs11/lisa/ckpt_model_MRG_lr0003_MIMIC/pytorch_model.bin", type=str) # MRG mimic Best miou: 0.4604,
    parser.add_argument("--weight", default="./runs_UMRG_Chest_gpt_po_BS2_LR0003/lisa/ckpt_model/pytorch_model.bin", type=str) # UMRG Chest_gpt_po Best : 0.3849, accu_5: 0.3838, accu_3: 0.5657, accu_1: 0.8434

    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    # parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--uncertain_box_head", action="store_true", default=False)
    parser.add_argument("--llava_head", action="store_true", default=False)
    parser.add_argument("--llava_sam_head", action="store_true", default=False)
    parser.add_argument("--mhp_box_head", action="store_true", default=False)
    parser.add_argument("--Umhp_box_head", action="store_true", default=False)

    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

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

    # prompter =  Prompter("")
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "lora_sam_rank": args.lora_sam_rank,
        "vision_pretrained": args.vision_pretrained,
        "llava_sam_head":args.llava_sam_head,
        "llava_head":args.llava_head,
        "uncertain_box_head":args.uncertain_box_head,
        "mhp_box_head":args.mhp_box_head,
        "Umhp_box_head":args.Umhp_box_head,
        "box_token_idx": args.box_token_idx,
        "vision_tower": args.vision_tower,
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
    model.config.lora_sam_rank = args.lora_sam_rank
    model.config.vision_pretrained = args.vision_pretrained

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)
    model.get_model().initialize_lisa_modules(model.get_model().config)
    prompter = Prompter("")

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

    state_dict = torch.load(args.weight, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    # model.load_state_dict(state_dict, strict=False)

    model.to(device=args.local_rank)

    test_dataset = TestDataset(
            args.phrase_data,
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            args.test_dataset,
            args.image_size,
        )
    print(
            f"Testing with {len(test_dataset)} examples "
        )

    # resume deepspeed checkpoint
    # if args.auto_resume and len(args.resume) == 0:
    #     resume = os.path.join(args.log_dir, "ckpt_model")
    #     if os.path.exists(resume):
    #         args.resume = resume

    # if args.resume:
    #     load_path, client_state = model_engine.load_checkpoint(args.resume)
    #     with open(os.path.join(args.resume, "latest"), "r") as f:
    #         ckpt_dir = f.readlines()[0].strip()
    #     args.start_epoch = (
    #         int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
    #     )
    #     print(
    #         "resume training from {}, start from epoch {}".format(
    #             args.resume, args.start_epoch
    #         )
    #     )

    # world_size = torch.cuda.device_count()
    args.distributed = False
    if args.distributed:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
        # test_sampler = torch.utils.data.BatchSampler(test_dataset)

    # test dataset
    if test_dataset is not None:
        assert args.test_batch_size == 1
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            # num_workers=args.workers,
            pin_memory=False,
            sampler=test_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )
    # model = model_engine.module
        test(test_loader, model, tokenizer, writer, prompter,args)
        exit()

def test(test_loader, model_engine, tokenizer, writer, prompter, args):
    miou_meter = AverageMeter("miou", ":6.3f", Summary.SUM)
    accu_meter_5 = AverageMeter("accu_5", ":6.3f", Summary.SUM)
    accu_meter_3 = AverageMeter("accu_3", ":6.3f", Summary.SUM)
    accu_meter_1 = AverageMeter("accu_1", ":6.3f", Summary.SUM)

    model_engine.eval()
    text_output_list = []
    att_weights_list = []

    pred_boxes_list = []
    all_boxes_list = []
    uncertainty_boxes_list = []
    np_size_pred_boxes_list =[]
    np_size_gt_boxes_list = []
    i = 0
    ref_text = dict()
    gt_text = dict()
    weight_name = args.weight.split('/')[-2] 
    for input_dict in tqdm.tqdm(test_loader):
        test_loader.dataset
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
        input_dict["max_new_tokens"] = 512

        with torch.no_grad():
            model_engine = model_engine.to(torch.bfloat16)
            output_dict = model_engine.evaluate(**input_dict)
        
        np_image = input_dict["sampled_classes_list"][0]
        phrase = test_loader.dataset.phrases[i]
        image_name = input_dict["image_paths"][0].split('/')[-1]

        if args.uncertain_box_head or args.mhp_box_head or args.Umhp_box_head:
            pred_boxes = output_dict["mean_boxes"][0]
            uncertainty_boxes = output_dict["uncertainty_boxes"][0]
            att_weights = output_dict["att_weights"][0]
            if args.uncertain_box_head or args.mhp_box_head:
                all_boxes = output_dict["all_boxes"][0]
        else:
            pred_boxes = output_dict["pred_boxes"][0]
            att_weights = output_dict["att_weights"][0]

        gt_boxes = output_dict["gt_boxes"][0]
        output_ids = output_dict["output_ids"]
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        output_tokens = tokenizer.decode(output_ids, skip_special_tokens=False)
        output_tokens = prompter.get_response(output_tokens)
        text_output = output_tokens.split('</s>')[0]
        text_output = text_output.replace("\n", "").replace("  ", " ")
        phrase_pred = text_output.replace("It is [BOX]."," ")
        print("image name:%s"%image_name)
        print("phrase_pred:%s,phrase_gt:%s"%(phrase_pred,phrase))
        
        # all are list
        # if isinstance(phrase_pred,list):
        #     ref_text[str(i)] = [phrase_pred]
        # else:
        #     ref_text[str(i)] = [[phrase_pred]]
        # gt_text[str(i)] = [phrase]

        # all are list
        if not isinstance(phrase_pred,list):
            ref_text[str(i)] = [phrase_pred]
        # gt_text[str(i)] = phrase # for the char
        gt_text[str(i)] = [phrase] # for the list

        # ref_text.append ({
        #     'image_id': i,
        #     'grounding': phrase_pred
        # })
        # gt_text.append ({
        #     'image_id': i,
        #     'grounding': phrase
        # })


        if args.uncertain_box_head or args.mhp_box_head:
            pred_boxes_list.append(pred_boxes)
            uncertainty_boxes_list.append(uncertainty_boxes)
            att_weights_list.append(att_weights)
            if args.uncertain_box_head or args.mhp_box_head:
                all_boxes_list.append(all_boxes)
                size_three_boxes = all_boxes * args.image_size
                np_size_three_boxes = size_three_boxes.detach().to(torch.int).cpu().numpy()
        else:
            pred_boxes_list.append(pred_boxes)
            att_weights_list.append(att_weights)

        size_pred_boxes = pred_boxes * args.image_size
        size_gt_boxes = gt_boxes * args.image_size
        np_size_pred_boxes = size_pred_boxes.detach().to(torch.int).cpu().numpy()
        np_size_gt_boxes = size_gt_boxes.detach().to(torch.int).cpu().numpy()
        
        np_size_pred_boxes_list.append(np_size_pred_boxes)
        np_size_gt_boxes_list.append(np_size_gt_boxes)

        miou, accu_5, accu_3, accu_1 = eval_utils.trans_vg_eval_val_imgsize(size_pred_boxes, size_gt_boxes,args.image_size)

        intersection, acc_5, acc_3, acc_1 = torch.mean(miou).cpu().numpy(), accu_5.cpu().numpy(), accu_3.cpu().numpy(), accu_1.cpu().numpy()
        print("image name:%s"%image_name)
        print("one_iou: {:.4f}, one_accu_5: {:.4f}, one_accu_3: {:.4f}, one_accu_1: {:.4f}".format(intersection, acc_5,acc_3,acc_1))

        miou_meter.update(intersection), accu_meter_5.update(acc_5),accu_meter_3.update(acc_3),accu_meter_1.update(acc_1)
        text_output_list.append(text_output)

        # np_image = image.squeeze().detach().to(torch.float).cpu().numpy()
        # np_image = np.array(np_image)
        if not os.path.exists(r'/data0/zou_ke/projects/MedRPG/'+args.vis_save_path):
            os.makedirs(r'/data0/zou_ke/projects/MedRPG/'+args.vis_save_path)
        if not os.path.isfile(r'/data0/zou_ke/projects/MedRPG/'+args.vis_save_path + "original/" + image_name):
            cv2.imwrite(r'/data0/zou_ke/projects/MedRPG/'+args.vis_save_path + "original/" + image_name,np_image)
        # if not os.path.isfile(r'/data0/zou_ke/projects/MedRPG/'+args.vis_save_path + "noise50_" + image_name):
        #     cv2.imwrite(r'/data0/zou_ke/projects/MedRPG/'+args.vis_save_path + "noise50_" + image_name,np_image)
        ### save_box_image(list(map(tuple,np_size_gt_boxes))[0],np_image,args.vis_save_path,"gt_" + image_name,phrase)
        ### save_box_image(list(map(tuple,np_size_pred_boxes))[0],np_image,args.vis_save_path,"pred_" + image_name,text_output)
        ### save_box_image_all(list(map(tuple,np_size_pred_boxes))[0],list(map(tuple,np_size_gt_boxes))[0],np_image,args.vis_save_path,"pred_gt_" + image_name,text_output,phrase)
        # save_box_image_gt(list(map(tuple,np_size_gt_boxes))[0],np_image,args.vis_save_path + weight_name + '/' ,"gt_" + image_name)
        # save_box_image_pred(list(map(tuple,np_size_pred_boxes))[0],np_image,args.vis_save_path + weight_name + '/' ,"pred_" + image_name)
        if args.uncertain_box_head or args.mhp_box_head:
            save_box_image_all_iou(list(map(tuple,np_size_pred_boxes))[0],list(map(tuple,np_size_gt_boxes))[0],np_image,"./"+ args.vis_save_path + weight_name + '/' , "pred_gt_" + image_name,"Pred","GT",intersection,acc_5)
            save_box_image_all_three_iou(list(map(tuple,np_size_pred_boxes))[0],list(map(tuple,np_size_three_boxes)),list(map(tuple,np_size_gt_boxes))[0],np_image,"./"+ args.vis_save_path + weight_name + '/' , "pred_gt_three" + image_name,"Pred","GT",intersection,acc_5)
            # save_box_image_all_iou(list(map(tuple,np_size_pred_boxes))[0],list(map(tuple,np_size_gt_boxes))[0],np_image,"./"+ args.vis_save_path + weight_name + '/' ,"noise50_" + "pred_gt_" + image_name,"Pred","GT",intersection,acc_5)
            # save_box_image_all_three_iou(list(map(tuple,np_size_pred_boxes))[0],list(map(tuple,np_size_three_boxes)),list(map(tuple,np_size_gt_boxes))[0],np_image,"./"+ args.vis_save_path + weight_name + '/' ,"noise50_" + "pred_gt_three" + image_name,"Pred","GT",intersection,acc_5)
        else:
            # save_box_image_all_iou(list(map(tuple,np_size_pred_boxes))[0],list(map(tuple,np_size_gt_boxes))[0],np_image,"./"+ args.vis_save_path + weight_name + '/' ,"noise50_"  + "pred_gt_" + image_name,"Pred","GT",intersection,acc_5)
            save_box_image_all_iou(list(map(tuple,np_size_pred_boxes))[0],list(map(tuple,np_size_gt_boxes))[0],np_image,"./"+ args.vis_save_path + weight_name + '/' ,"pred_gt_" + image_name,"Pred","GT",intersection,acc_5)
        # original_image = cv2.imread("./"+ args.vis_save_path + weight_name + '/' + "noise50_"  + "pred_gt_" + image_name)
        original_image = cv2.imread("./"+ args.vis_save_path + weight_name + '/' + "pred_gt_" + image_name)

        # tensor_weights = torch.nn.functional.interpolate(att_weights.unsqueeze(0), (1024, 1024), mode='bilinear', align_corners=False).squeeze(0)
        # # padding_left, padding_right, padding_top, padding_bottom = (192, 192, 192, 192)
        # # original_height, original_width = 640, 640
        # # restored_tensor_weights = tensor_weights[:, padding_top:padding_top + original_height, padding_left:padding_left + original_width]
        
        # restored_tensor_weights = tensor_weights[:, :640, :640] 

        # np_weights1 = restored_tensor_weights[0,:,:].squeeze().cpu().numpy()
        # np_weights1 = np.clip(np_weights1, 0, 1)
        # cmap = cm.get_cmap('jet')
        # # cmap = matplotlib.pyplot.get_cmap('jet')
        # attention_colormap1 = (cmap(np_weights1)[:, :, :3] * 255).astype('uint8')
        # result_image1 = cv2.addWeighted(original_image, 0.7, attention_colormap1, 0.3, 0)
        # cv2.imwrite("./"+ args.vis_save_path + weight_name + '/' + "colormap1_nonnoise_"  + "pred_gt_" + image_name, result_image1)
        # cv2.imwrite("./"+ args.vis_save_path + weight_name + '/' + "colormap1_noise50_"  + "pred_gt_" + image_name, result_image1)

        # np_weights2 = restored_tensor_weights[1,:,:].squeeze().cpu().numpy()
        # np_weights2 = np.clip(np_weights2, 0, 1)
        # cmap = cm.get_cmap('jet')
        # attention_colormap2 = (cmap(np_weights2)[:, :, :3] * 255).astype('uint8')
        # result_image2 = cv2.addWeighted(original_image, 0.7, attention_colormap2, 0.3, 0)
        # cv2.imwrite("./"+ args.vis_save_path + weight_name + '/' + "colormap2_noise_"  + "pred_gt_" + image_name, result_image2)

        i+=1
    # miou_meter.all_reduce()
    # accu_meter.all_reduce()
    avg_accu_5 = accu_meter_5.avg
    avg_accu_3 = accu_meter_3.avg
    avg_accu_1 = accu_meter_1.avg
    avg_miou = miou_meter.avg
    print("miou: {:.4f}, accu_5: {:.4f}, accu_3: {:.4f}, accu_1: {:.4f}".format(avg_miou, avg_accu_5, avg_accu_3, avg_accu_1))

    pred_output_all = dict()
    pred_output_all['accu_meter_5'] = accu_meter_5
    pred_output_all['accu_meter_3'] = accu_meter_3
    pred_output_all['accu_meter_1'] = accu_meter_1
    pred_output_all['miou_meter'] = miou_meter
    pred_output_all['avg_miou'] = avg_miou
    pred_output_all['avg_accu_5'] = avg_accu_5
    pred_output_all['avg_accu_3'] = avg_accu_3
    pred_output_all['avg_accu_1'] = avg_accu_1
    pred_output_all['text_output_list'] = text_output_list
    pred_output_all['np_size_pred_boxes_list'] = np_size_pred_boxes_list
    pred_output_all['np_size_gt_boxes_list'] = np_size_gt_boxes_list

    if args.uncertain_box_head or args.mhp_box_head:
        pred_output_all['pred_boxes_list'] = pred_boxes_list
        pred_output_all['uncertainty_boxes_list'] = uncertainty_boxes_list
        pred_output_all['att_weights_list'] = att_weights_list
        if args.uncertain_box_head or args.mhp_box_head:
            pred_output_all['all_boxes_list'] = all_boxes_list
    else:
        pred_output_all['pred_boxes_list'] = pred_boxes_list
        pred_output_all['att_weights_list'] = att_weights_list
    score_text = Scorer(ref_text,gt_text)
    text_scores = score_text.compute_scores()
    pred_output_all['text_scores'] = text_scores
    
    if not os.path.exists("./"+"p_" + args.vis_save_path): #判断所在目录下是否有该文件名的文件夹
        os.makedirs("./"+"p_" + args.vis_save_path)  
    # torch.save(pred_output_all,"./"+"p_" + args.vis_save_path + weight_name + "noise50_" + "results_all.pth")
    torch.save(pred_output_all,"./"+"p_" + args.vis_save_path + weight_name + "nonnoise_" + "results_all.pth")

    if args.local_rank == 0:
        writer.add_scalar("test/accu_5", avg_accu_5)
        writer.add_scalar("test/accu_3", avg_accu_3)
        writer.add_scalar("test/accu_1", avg_accu_1)

        writer.add_scalar("test/miou", avg_miou)
        # writer.add_scalar("test/phrase", text_output_list)
        print("miou: {:.4f}, accu_5: {:.4f}, accu_3: {:.4f}, accu_1: {:.4f}".format(avg_miou, avg_accu_5, avg_accu_3, avg_accu_1))
    
    for key,value in text_scores.items():
        print('{}:{}'.format(key,value))

    return avg_miou, avg_accu_5, avg_accu_3, avg_accu_1, pred_boxes_list, text_output_list


def save_box_image(box , image , save_path, image_name,  class_name = "patient"):

    # tuple_ptLeftTop,tuple_ptRightBottom = xycwh2xyxy_tuple(box)
    # tuple_ptLeftTop,tuple_ptRightBottom = box[0:2],box[2:4]
    tuple_ptLeftTop,tuple_ptRightBottom = xycwh2xyxy_tuple(box)
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

    if not os.path.exists(save_path): #判断所在目录下是否有该文件名的文件夹
        os.makedirs(save_path)  
    cv2.imwrite(save_path + image_name, src)

def save_box_image_all(box, gt_box, image , save_path, image_name,  class_name, class_name_gt):

    # tuple_ptLeftTop,tuple_ptRightBottom = xycwh2xyxy_tuple(box)
    # tuple_ptLeftTop,tuple_ptRightBottom = box[0:2],box[2:4]
    tuple_ptLeftTop,tuple_ptRightBottom = xycwh2xyxy_tuple(box)
    tuple_ptLeftTop_gt,tuple_ptRightBottom_gt = xycwh2xyxy_tuple(gt_box)

    ptLeftTop = np.array(tuple_ptLeftTop)
    ptRightBottom = np.array(tuple_ptRightBottom)

    ptLeftTop_gt = np.array(tuple_ptLeftTop_gt)
    ptRightBottom_gt = np.array(tuple_ptRightBottom_gt)

    # 框的颜色
    point_color = (0, 255, 0)
    point_color_gt = (192, 0, 0)

    # 线的厚度
    thickness = 2
    # 线的类型
    lineType = 4

    src = image
    # cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    src = np.array(src)
    # 画 b_box
    cv2.rectangle(src, tuple(ptLeftTop), tuple(ptRightBottom), point_color, thickness, lineType)
    cv2.rectangle(src, tuple(ptLeftTop_gt), tuple(ptRightBottom_gt), point_color_gt, thickness, lineType)

    # 获取文字区域框大小
    t_size = cv2.getTextSize(class_name, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
    t_size_gt = cv2.getTextSize(class_name_gt, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]

    # 获取 文字区域右下角坐标
    textlbottom = ptLeftTop + np.array(list(t_size))
    textlbottom_gt = ptLeftTop_gt + np.array(list(t_size_gt))

    # 绘制文字区域矩形框
    cv2.rectangle(src, tuple(ptLeftTop), tuple(textlbottom),  point_color, -1)
    cv2.rectangle(src, tuple(ptLeftTop_gt), tuple(textlbottom_gt),  point_color_gt, -1)

    # 计算文字起始位置偏移
    ptLeftTop[1] = ptLeftTop[1] + (t_size[1]/2 + 4)
    ptLeftTop_gt[1] = ptLeftTop_gt[1] + (t_size_gt[1]/2 + 4)

    # 绘字
    cv2.putText(src, class_name , tuple(ptLeftTop), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)
    cv2.putText(src, class_name_gt , tuple(ptLeftTop_gt), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)

    # 打印图片的shape
    print(src.shape)

    cv2.imwrite(save_path + image_name, src)

def save_box_image_all_three_iou(box,three_box, gt_box, image , save_path, image_name,  class_name, class_name_gt,iou, acc):
    class_name_iou_acc = "iou: "+ str(np.around(iou,4)) +"  " + "acc: "+ str(np.around(acc,4))
    # tuple_ptLeftTop,tuple_ptRightBottom = xycwh2xyxy_tuple(box)
    # tuple_ptLeftTop,tuple_ptRightBottom = box[0:2],box[2:4]
    tuple_ptLeftTop_0,tuple_ptRightBottom_0 = xycwh2xyxy_tuple(three_box[0])
    tuple_ptLeftTop_1,tuple_ptRightBottom_1 = xycwh2xyxy_tuple(three_box[1])
    tuple_ptLeftTop_2,tuple_ptRightBottom_2 = xycwh2xyxy_tuple(three_box[2])

    tuple_ptLeftTop,tuple_ptRightBottom = xycwh2xyxy_tuple(box)
    tuple_ptLeftTop_gt,tuple_ptRightBottom_gt = xycwh2xyxy_tuple(gt_box)

    ptLeftTop = np.array(tuple_ptLeftTop)
    ptRightBottom = np.array(tuple_ptRightBottom)

    ptLeftTop_0 = np.array(tuple_ptLeftTop_0)
    ptRightBottom_0 = np.array(tuple_ptRightBottom_0)

    ptLeftTop_1 = np.array(tuple_ptLeftTop_1)
    ptRightBottom_1 = np.array(tuple_ptRightBottom_1)

    ptLeftTop_2 = np.array(tuple_ptLeftTop_2)
    ptRightBottom_2 = np.array(tuple_ptRightBottom_2)

    ptLeftTop_gt = np.array(tuple_ptLeftTop_gt)
    ptRightBottom_gt = np.array(tuple_ptRightBottom_gt)

    size_range = image.shape
    tuple_ptLeftTop_iou_acc = (size_range[0]-200,20)
    ptLeftTop_iou_acc = np.array(tuple_ptLeftTop_iou_acc)

    # 框的颜色
    point_color = (0, 255, 0)
    point_color_c = (255, 0, 0)

    point_color_gt = (0, 0, 255)
    point_color_iou = (255, 255, 255)

    # 线的厚度
    thickness = 2
    # 线的类型
    lineType = 4

    src = image
    # cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    src = np.array(src)
    # 画 b_box
    cv2.rectangle(src, tuple(ptLeftTop_0), tuple(ptRightBottom_0), point_color, thickness, lineType)
    cv2.rectangle(src, tuple(ptLeftTop_1), tuple(ptRightBottom_1), point_color, thickness, lineType)
    cv2.rectangle(src, tuple(ptLeftTop_2), tuple(ptRightBottom_2), point_color, thickness, lineType)

    cv2.rectangle(src, tuple(ptLeftTop), tuple(ptRightBottom), point_color_c, thickness, lineType)
    cv2.rectangle(src, tuple(ptLeftTop_gt), tuple(ptRightBottom_gt), point_color_gt, thickness, lineType)

    # 获取文字区域框大小
    t_size = cv2.getTextSize(class_name, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
    t_size_gt = cv2.getTextSize(class_name_gt, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
    t_size_iou_acc = cv2.getTextSize(class_name_iou_acc, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]

    # 获取 文字区域右下角坐标
    textlbottom = ptLeftTop + np.array(list(t_size)) + (0,5)
    textlbottom_0 = ptLeftTop_0 + np.array(list(t_size)) + (0,5)
    textlbottom_1 = ptLeftTop_1 + np.array(list(t_size)) + (0,5)
    textlbottom_2 = ptLeftTop_2 + np.array(list(t_size)) + (0,5)

    textlbottom_gt = ptLeftTop_gt + np.array(list(t_size_gt)) + (0,5)
    textlbottom_iou_acc = ptLeftTop_iou_acc + np.array(list(t_size_iou_acc)) + (0,5)

    # 绘制文字区域矩形框
    cv2.rectangle(src, tuple(ptLeftTop), tuple(textlbottom),  point_color_c, -1)
    cv2.rectangle(src, tuple(ptLeftTop_0), tuple(textlbottom_0),  point_color, -1)
    cv2.rectangle(src, tuple(ptLeftTop_1), tuple(textlbottom_1),  point_color, -1)
    cv2.rectangle(src, tuple(ptLeftTop_2), tuple(textlbottom_2),  point_color, -1)

    cv2.rectangle(src, tuple(ptLeftTop_gt), tuple(textlbottom_gt),  point_color_gt, -1)
    cv2.rectangle(src, tuple(ptLeftTop_iou_acc), tuple(textlbottom_iou_acc),  point_color_iou, -1)

    # 计算文字起始位置偏移
    ptLeftTop_0[1] = ptLeftTop_0[1] + (t_size[1]/2 + 4)
    ptLeftTop_1[1] = ptLeftTop_1[1] + (t_size[1]/2 + 4)
    ptLeftTop_2[1] = ptLeftTop_2[1] + (t_size[1]/2 + 4)
    ptLeftTop[1] = ptLeftTop[1] + (t_size[1]/2 + 4)
    ptLeftTop_gt[1] = ptLeftTop_gt[1] + (t_size_gt[1]/2 + 4)
    ptLeftTop_iou_acc[1] = ptLeftTop_iou_acc[1] + (t_size_iou_acc[1]/2 + 4)

    # 绘字 BGR
    # cv2.putText(src, class_name , tuple(ptLeftTop), cv2.FONT_HERSHEY_PLAIN, 1.0, (247, 9, 66), 2)
    # cv2.putText(src, class_name_gt , tuple(ptLeftTop_gt), cv2.FONT_HERSHEY_PLAIN, 1.0, (247, 9, 66), 2)
    cv2.putText(src, class_name , tuple(ptLeftTop), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2)
    cv2.putText(src, class_name , tuple(ptLeftTop_0), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2)
    cv2.putText(src, class_name , tuple(ptLeftTop_1), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2)
    cv2.putText(src, class_name , tuple(ptLeftTop_2), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2)

    cv2.putText(src, class_name_gt , tuple(ptLeftTop_gt), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2)
    cv2.putText(src, class_name_iou_acc , tuple(ptLeftTop_iou_acc), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)

    # 打印图片的shape
    # print(src.shape)
    if not os.path.exists(save_path): #判断所在目录下是否有该文件名的文件夹
        os.makedirs(save_path)
    cv2.imwrite(save_path + image_name, src)


def save_box_image_all_iou(box, gt_box, image , save_path, image_name,  class_name, class_name_gt,iou, acc):
    class_name_iou_acc = "iou: "+ str(np.around(iou,4)) +"  " + "acc: "+ str(np.around(acc,4))
    # tuple_ptLeftTop,tuple_ptRightBottom = xycwh2xyxy_tuple(box)
    # tuple_ptLeftTop,tuple_ptRightBottom = box[0:2],box[2:4]
    tuple_ptLeftTop,tuple_ptRightBottom = xycwh2xyxy_tuple(box)
    tuple_ptLeftTop_gt,tuple_ptRightBottom_gt = xycwh2xyxy_tuple(gt_box)

    ptLeftTop = np.array(tuple_ptLeftTop)
    ptRightBottom = np.array(tuple_ptRightBottom)

    ptLeftTop_gt = np.array(tuple_ptLeftTop_gt)
    ptRightBottom_gt = np.array(tuple_ptRightBottom_gt)

    size_range = image.shape
    tuple_ptLeftTop_iou_acc = (size_range[0]-200,20)
    ptLeftTop_iou_acc = np.array(tuple_ptLeftTop_iou_acc)

    # 框的颜色
    point_color = (0, 255, 0)
    point_color_gt = (0, 0, 255)
    point_color_iou = (255, 255, 255)

    # 线的厚度
    thickness = 2
    # 线的类型
    lineType = 4

    src = image
    # cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    src = np.array(src)
    # 画 b_box
    cv2.rectangle(src, tuple(ptLeftTop), tuple(ptRightBottom), point_color, thickness, lineType)
    cv2.rectangle(src, tuple(ptLeftTop_gt), tuple(ptRightBottom_gt), point_color_gt, thickness, lineType)

    # 获取文字区域框大小
    t_size = cv2.getTextSize(class_name, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
    t_size_gt = cv2.getTextSize(class_name_gt, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
    t_size_iou_acc = cv2.getTextSize(class_name_iou_acc, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]

    # 获取 文字区域右下角坐标
    textlbottom = ptLeftTop + np.array(list(t_size)) + (0,5)
    textlbottom_gt = ptLeftTop_gt + np.array(list(t_size_gt)) + (0,5)
    textlbottom_iou_acc = ptLeftTop_iou_acc + np.array(list(t_size_iou_acc)) + (0,5)

    # 绘制文字区域矩形框
    cv2.rectangle(src, tuple(ptLeftTop), tuple(textlbottom),  point_color, -1)
    cv2.rectangle(src, tuple(ptLeftTop_gt), tuple(textlbottom_gt),  point_color_gt, -1)
    cv2.rectangle(src, tuple(ptLeftTop_iou_acc), tuple(textlbottom_iou_acc),  point_color_iou, -1)

    # 计算文字起始位置偏移
    ptLeftTop[1] = ptLeftTop[1] + (t_size[1]/2 + 4)
    ptLeftTop_gt[1] = ptLeftTop_gt[1] + (t_size_gt[1]/2 + 4)
    ptLeftTop_iou_acc[1] = ptLeftTop_iou_acc[1] + (t_size_iou_acc[1]/2 + 4)

    # 绘字 BGR
    # cv2.putText(src, class_name , tuple(ptLeftTop), cv2.FONT_HERSHEY_PLAIN, 1.0, (247, 9, 66), 2)
    # cv2.putText(src, class_name_gt , tuple(ptLeftTop_gt), cv2.FONT_HERSHEY_PLAIN, 1.0, (247, 9, 66), 2)
    cv2.putText(src, class_name , tuple(ptLeftTop), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2)
    cv2.putText(src, class_name_gt , tuple(ptLeftTop_gt), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2)
    cv2.putText(src, class_name_iou_acc , tuple(ptLeftTop_iou_acc), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)

    # 打印图片的shape
    # print(src.shape)
    if not os.path.exists(save_path): #判断所在目录下是否有该文件名的文件夹
        os.makedirs(save_path)
    cv2.imwrite(save_path + image_name, src)

def save_box_image_gt(gt_box, image , save_path, image_name):

    tuple_ptLeftTop_gt,tuple_ptRightBottom_gt = xycwh2xyxy_tuple(gt_box)

    ptLeftTop_gt = np.array(tuple_ptLeftTop_gt)
    ptRightBottom_gt = np.array(tuple_ptRightBottom_gt)

    size_range = image.shape
    tuple_ptLeftTop_iou_acc = (size_range[0]-200,20)
    ptLeftTop_iou_acc = np.array(tuple_ptLeftTop_iou_acc)

    # 框的颜色
    point_color = (0, 255, 0)
    point_color_gt = (0, 0, 255)
    point_color_iou = (255, 255, 255)

    # 线的厚度
    thickness = 2
    # 线的类型
    lineType = 4

    src = image
    # cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    src = np.array(src)
    # 画 b_box
    cv2.rectangle(src, tuple(ptLeftTop_gt), tuple(ptRightBottom_gt), point_color_gt, thickness, lineType)

    # 打印图片的shape
    # print(src.shape)

    cv2.imwrite(save_path + image_name, src)

def save_box_image_pred(gt_box, image , save_path, image_name):

    tuple_ptLeftTop_gt,tuple_ptRightBottom_gt = xycwh2xyxy_tuple(gt_box)

    ptLeftTop_gt = np.array(tuple_ptLeftTop_gt)
    ptRightBottom_gt = np.array(tuple_ptRightBottom_gt)

    size_range = image.shape
    tuple_ptLeftTop_iou_acc = (size_range[0]-200,20)
    ptLeftTop_iou_acc = np.array(tuple_ptLeftTop_iou_acc)

    # 框的颜色
    point_color_gt = (0, 255, 0)
    # point_color_gt = (0, 0, 255)
    point_color_iou = (255, 255, 255)

    # 线的厚度
    thickness = 2
    # 线的类型
    lineType = 4

    src = image
    # cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    src = np.array(src)
    # 画 b_box
    cv2.rectangle(src, tuple(ptLeftTop_gt), tuple(ptRightBottom_gt), point_color_gt, thickness, lineType)

    # 打印图片的shape
    # print(src.shape)

    cv2.imwrite(save_path + image_name, src)

def xycwh2xyxy_tuple(x):
    x_c, y_c, w, h = x
    b = [int(x_c- w*0.5), int(y_c-h*0.5),
         int(x_c + w*0.5), int(y_c + h*0.5)]
    return b[0:2],b[2:4]

class Scorer():
    def __init__(self,ref,gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
        ]
    
    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.4f"%(m, sc))
                total_scores["Bleu"] = score
            else:
                print("%s: %0.4f"%(method, score))
                total_scores[method] = score
        
        print('*****DONE*****')

        return total_scores
    
if __name__ == "__main__":
    main(sys.argv[1:])
