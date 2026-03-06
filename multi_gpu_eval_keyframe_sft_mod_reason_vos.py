import torch
import json
import os
import re
import math
import argparse
import shutil
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
import time

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# from revos_eval_dataset import ReVOSDataset, collate_fn
from batch_eval_keyframe_sft_dataset_ import ReVOSDataset, collate_fn
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_video_predictor

from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer


def extract_bbox_points_think(output_text, x_factor, y_factor):
    json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text[0], re.DOTALL)
    if json_match:
        data = json.loads(json_match.group(1).replace("'", '"'))
        keyframe_id = int(data.get("keyframe_timestamp", -1))
        if isinstance(data["bbox_2d_list"], list):
            pred_bboxes = [[
                int(item[0] * x_factor + 0.5),
                int(item[1] * y_factor + 0.5),
                int(item[2] * x_factor + 0.5),
                int(item[3] * y_factor + 0.5)
            ] for item in data["bbox_2d_list"]]
        else:
            pred_bboxes = [int(data['bbox_2d'][0] * x_factor + 0.5),
                int(data['bbox_2d'][1] * y_factor + 0.5),
                int(data['bbox_2d'][2] * x_factor + 0.5),
                int(data['bbox_2d'][3] * y_factor + 0.5)]
    
    think_pattern = r'<think>([^<]+)</think>'
    think_text = ""
    think_match = re.search(think_pattern, output_text[0])
    if think_match:
        think_text = think_match.group(1)
    
    return pred_bboxes, keyframe_id, think_text


def main(args):
            
    valid_json_path = os.path.join(args.data_path, "meta_expressions_.json")
    video_data = json.load(open(valid_json_path, "r"))["videos"]
    
    video_metas = []
    for video_name, vid_data in video_data.items():
        vid_frames = sorted(vid_data['frames'])  # 00000, 00001, ...
        vid_len = len(vid_frames)
        src_dataset = vid_data['source']
        for exp_dict in vid_data["expressions"]:
            meta = {}
            # meta['video'] = video_name  # 377b1c5f365c
            meta['exp'] = exp_dict['exp_text']  # 4 lizards moving around
            meta['obj_id'] = [int(x) for x in exp_dict['obj_id']]  # [0, 1, 2, 3, ]
            # meta['anno_id'] = [str(x) for x in exp_dict['anno_id']]  # [2, 3, 4, 5, ]
            meta['frames'] = vid_frames  # ['00000', '00001', ...]
            meta['exp_id'] = exp_dict["exp_id"]  # '0'
            meta['category'] = 0
            meta['length'] = vid_len
            meta['frame_path'] = [os.path.join(args.data_path, "JPEGImages", video_name, frame_name + '.jpg') for frame_name in vid_frames]
            meta['video'] = f"{src_dataset}_{video_name}_{exp_dict['obj_id']}"
            # meta['tp'] = exp_dict["tp"]

            video_metas.append(meta)
    
    print(f"Total samples: {len(video_metas)}")
    
    # 性能指标统计
    inference_times = []  # 存储每次推理的时间（秒）
    max_memory_allocated = 0  # 最大显存占用（MB）
    flops_calculated = False
    model_flops = None
    
    # default processer
    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left", use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(args.reasoning_model_path)
    revos_dataset = ReVOSDataset(data_path=args.data_path,
                                 tokenizer=tokenizer,
                                 processor=processor,
                                 prompt_key="prompt",
                                 max_prompt_length=7100,
                                 truncation="right",
                                 system_prompt=r"You are a helpful assistant.",
                                 max_pixels=12845056,
                                 min_pixels=3136,
                                 video_metas=video_metas, 
                                 )
    
    # 设置dataloader
    sampler = SequentialSampler(data_source=revos_dataset)
    test_dataloader = DataLoader(
            dataset=revos_dataset,
            batch_size=args.batch_size,
            num_workers=8,
            drop_last=False,
            collate_fn=collate_fn,
            sampler=sampler,
        )
    
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    print("Loading reasoning model...")
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    reasoning_model.eval()
    
    # 重置显存统计
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    # load sam2 models
    model_cfg = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2_video_predictor(model_cfg, "/SAM2Long-main/checkpoints/sam2.1_hiera_large.pt", device='cuda')
    
    # model_cfg = "sam2/configs/sam2/sam2_hiera_l.yaml"
    # sam2_model = build_sam2_video_predictor(model_cfg, "/sam2_large/sam2_hiera_large.pt", device='cuda')
    
    segmentation_model = SAM2ImagePredictor(sam2_model)
    resize_size = 560
    output_video_dir = args.output_path
    
    keyframe_id_list = []
    
    think_dict = dict()
    
    for num, batch_sample in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing"):
        
        if os.path.exists(os.path.join(output_video_dir, "Annotations", batch_sample["video_name"][0], str(batch_sample["exp_id"][0]))):
            continue
        
        
        frame_path_list = batch_sample["image_path_list"][0]
        # frame_sample = frame_path_list[0].split('/')[-1]
        # video_copy_path = os.path.join(copy_path, batch_sample["video_name"][0])
        # os.makedirs(video_copy_path, exist_ok=True)
        video_dir = os.path.dirname(frame_path_list[0])
        
        
        # keyframe_idx = batch_sample["keyframe_id"][0]
        # keyframe_idx = 3
        prompt_inputs = dict()
        prompt_inputs["input_ids"] = batch_sample["input_ids"][0].unsqueeze(0).to("cuda")
        prompt_inputs["attention_mask"] = batch_sample["attention_mask"][0].unsqueeze(0).to("cuda")
        prompt_inputs["image_grid_thw"] = batch_sample["image_grid_thw"][0].to("cuda")
        prompt_inputs["pixel_values"] = batch_sample["pixel_values"][0].to("cuda")
        frame_ids = batch_sample["frame_ids"]
        
        # 计算FLOPs（只计算一次，使用第一个样本）
        if not flops_calculated:
            print("Calculating FLOPs...")
            try:
                # 尝试使用thop计算FLOPs
                from thop import profile
                # 创建dummy输入用于计算FLOPs
                dummy_inputs = {
                    "input_ids": prompt_inputs["input_ids"],
                    "attention_mask": prompt_inputs["attention_mask"],
                    "image_grid_thw": prompt_inputs["image_grid_thw"],
                    "pixel_values": prompt_inputs["pixel_values"]
                }
                macs, params = profile(reasoning_model, inputs=(dummy_inputs,), verbose=False)
                model_flops = macs / 1e9  # 转换为GFlops
                print(f"Model FLOPs: {model_flops:.2f} GFlops")
                print(f"Model Parameters: {params / 1e6:.2f} M")
            except ImportError:
                print("Warning: thop not installed. Install with: pip install thop")
                print("Skipping FLOPs calculation...")
                model_flops = None
            except Exception as e:
                print(f"Warning: FLOPs calculation failed: {e}")
                print("This is normal for generation models. FLOPs will be set to None.")
                model_flops = None
            flops_calculated = True
        
        # 测量推理时间
        torch.cuda.synchronize()
        start_time = time.time()
        
        generated_ids = reasoning_model.generate(**prompt_inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
        
        torch.cuda.synchronize()
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # 更新最大显存占用
        current_memory = torch.cuda.max_memory_allocated() / 1024**2  # 转换为MB
        max_memory_allocated = max(max_memory_allocated, current_memory)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(batch_sample["input_ids"][0].unsqueeze(0), generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16): 
            try:
                bbox, keyframe_idx, think = extract_bbox_points_think(
                                            batch_output_text, 
                                            batch_sample["width"][0] / resize_size,
                                            batch_sample["height"][0] / resize_size,
                                        )
                # keyframe_idx = frame_ids[0][keyframe_idx]
                
                think_dict[batch_sample["video_name"][0] + "_" + str(batch_sample["exp_id"][0])] = {
                    "bboxes": bbox, 
                    "keyframe_idx": keyframe_idx, 
                    "think_process": think, 
                }
                
            except Exception as e:
                keyframe_idx = 0
                bbox = [[0, 0, 0, 0]]
                
            if bbox == []:
                keyframe_idx = 0
                bbox = [[0, 0, 0, 0]]
                
            if keyframe_idx != 0:
                keyframe_id_list.append(keyframe_idx)
                pass
            
            state = sam2_model.init_state(batch_sample["raw_images"][0])
            sam2_model.reset_state(state)
            # add new prompts and instantly get the output on the same frame
            if isinstance(bbox[0], list):
                for num, box in enumerate(bbox):
                    frame_idx, object_ids, masks = sam2_model.add_new_points_or_box(state, 
                                                                                    frame_idx=keyframe_idx,
                                                                                    obj_id=num,
                                                                                    box=box)
            else:
                frame_idx, object_ids, masks = sam2_model.add_new_points_or_box(state, 
                                                                        frame_idx=keyframe_idx,
                                                                        obj_id=0,
                                                                        box=bbox)

            # propagate the prompts to get masklets throughout the video
            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in sam2_model.propagate_in_video(state, 
                                                                                            start_frame_idx=keyframe_idx, 
                                                                                            reverse=False):
                video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
                
            for out_frame_idx, out_obj_ids, out_mask_logits in sam2_model.propagate_in_video(state, 
                                                                                            start_frame_idx=keyframe_idx, 
                                                                                            reverse=True):
                video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
                
            mask_shape = out_mask_logits[0].shape[-2:]
            video_masks = []    # num_frames, h, w
            sorted_video_segments = dict(sorted(video_segments.items()))
            for frame_idx, frame_segment_dict in sorted_video_segments.items():
                frame_mask = np.zeros(mask_shape, dtype=np.uint8)        # h, w
                for obj_id, mask_logits in frame_segment_dict.items():
                    frame_mask = frame_mask | mask_logits
                video_masks.append(frame_mask)
            
            video_masks = np.stack(video_masks, axis=0)[:, 0]      # num_frame, h, w
                    
        # visualize and save the results
        for frame_idx, frame_path in enumerate(frame_path_list):
            frame_name = os.path.basename(frame_path)           # e.g., img_00001.jpg
            output_frame_path = os.path.join(output_video_dir, "Annotations", batch_sample["video_name"][0], str(batch_sample["exp_id"][0]), frame_name.replace("jpg", "png"))
            os.makedirs(os.path.dirname(output_frame_path), exist_ok=True)
            mask = video_masks[frame_idx]
            mask = mask.astype(np.float32)
            mask = Image.fromarray(mask * 255).convert('L')
            mask.save(output_frame_path)
    
    print(keyframe_id_list)
    
    # 计算性能指标
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    min_inference_time = np.min(inference_times) if inference_times else 0
    max_inference_time = np.max(inference_times) if inference_times else 0
    std_inference_time = np.std(inference_times) if inference_times else 0
    
    # 获取最终的最大显存占用
    final_max_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    peak_memory = torch.cuda.max_memory_reserved() / 1024**2  # MB
    
    # 构建性能报告
    performance_metrics = {
        "total_samples": len(video_metas),
        "processed_samples": len(inference_times),
        "inference_time": {
            "average_seconds": float(avg_inference_time),
            "min_seconds": float(min_inference_time),
            "max_seconds": float(max_inference_time),
            "std_seconds": float(std_inference_time),
            "total_seconds": float(sum(inference_times)),
            "samples_per_second": float(len(inference_times) / sum(inference_times)) if sum(inference_times) > 0 else 0
        },
        "memory": {
            "max_allocated_mb": float(max_memory_allocated),
            "final_max_allocated_mb": float(final_max_memory),
            "peak_reserved_mb": float(peak_memory)
        },
        "flops": {
            "gflops": float(model_flops) if model_flops is not None else None,
            "note": "FLOPs calculation may not be accurate for generation models"
        }
    }
    
    print("\n" + "="*50)
    print("Performance Metrics Summary")
    print("="*50)
    print(f"Total samples processed: {len(inference_times)}")
    print(f"\nInference Time:")
    print(f"  Average: {avg_inference_time:.4f} seconds")
    print(f"  Min: {min_inference_time:.4f} seconds")
    print(f"  Max: {max_inference_time:.4f} seconds")
    print(f"  Std: {std_inference_time:.4f} seconds")
    print(f"  Throughput: {performance_metrics['inference_time']['samples_per_second']:.2f} samples/second")
    print(f"\nMemory Usage:")
    print(f"  Max Allocated: {max_memory_allocated:.2f} MB ({max_memory_allocated/1024:.2f} GB)")
    print(f"  Peak Reserved: {peak_memory:.2f} MB ({peak_memory/1024:.2f} GB)")
    if model_flops is not None:
        print(f"\nFLOPs:")
        print(f"  Model FLOPs: {model_flops:.2f} GFlops")
    print("="*50)
    
    # 保存性能指标
    metrics_path = os.path.join(output_video_dir, "performance_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(performance_metrics, f, indent=4)
    print(f"\nPerformance metrics saved to: {metrics_path}")
    
    # 将所有 numpy 类型递归转换为原生 Python 类型
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.integer, pd.Int64Dtype)):
            return int(obj)
        elif isinstance(obj, (np.floating, pd.Float64Dtype)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # 示例：think_dict 是你的字典
    think_dict = convert(think_dict)

    with open(os.path.join(output_video_dir, "think_results.json"), "w") as f:
        json.dump(think_dict, f, indent=4) 
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="/Seg-Zero/workdir/run_qwen2_5_3b_revos_keyframe_sam_sft_mod/global_step_625/actor/huggingface")
    parser.add_argument("--data_path", type=str, default="/ReasonVOS")
    parser.add_argument("--output_path", type=str, default="/Seg-Zero/runs/keyframe_sam_sft_mod_reavos_625")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--num_parts", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    main(args)