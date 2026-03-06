# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import random
import pycocotools.mask as maskUtils
import math
import pdb
import json
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from typing import Any, Dict, List, Optional
from transformers import AutoProcessor

import torch
from datasets import load_dataset, load_from_disk
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.models.transformers.qwen2_5_vl import get_rope_index
from scipy import ndimage


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    # tensors: pixel_values, image_grid_thw, input_ids, attention_mask, position_ids, gt_masks, raw_prompt_ids
    # non_tensors: height, width, keyframe_id, solution, images
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
                # print(key, value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        if key not in ["pixel_values", "image_grid_thw", "gt_masks"]:
            tensors[key] = torch.stack(value, dim=0)      # num_batch, num_frames, N, D

    return {**tensors, **non_tensors}


def process_image(image: ImageObject, max_pixels: int, min_pixels: int) -> ImageObject:
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class MultiImage_Dataset_mod(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key="prompt",
        max_prompt_length=1024,
        truncation="error",
        system_prompt=None,
        max_pixels=None,
        min_pixels=None,
        num_sampled_frames=10,
        keyframe_resize=560,
        reference_image_resize=560,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.rvos_sampled_frames = num_sampled_frames
        self.keyframe_resize = keyframe_resize
        self.reference_image_resize = reference_image_resize
        
        
        train_json_path = os.path.join(data_path, "stage2_rl_sampled.json")
        self.video_metas = json.load(open(train_json_path, "r"))
 
        self.user_prompt = "<video> " + "\n" + \
                            "You are given {num} images, each image is preceded by its index." + "\n" + \
                            "Please follow the instruction: '{Question}' and locate the target object in the designated image." + "\n" + \
                            "First, provide your reasoning process within <think> </think> tags."  + "\n" + \
                            "Then, output the index of the designated image (keyframe_timestamp) and bounding box (bbox_2d_list) within <answer> </answer> tags."       
                                                
    def __len__(self):
        return len(self.video_metas)

    def __getitem__(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        meta_info = self.video_metas[index]
        frame_path_list = meta_info['images']
        video_length = len(frame_path_list)
        exp = meta_info['question']
            
        # 首先选择出整个视频所有帧的mask
        image_shape = cv2.imread(frame_path_list[0]).shape[:2]
        height, width = image_shape

        sampled_video_frames = frame_path_list
        
        # extract bbox of the keyframe
        reference_resize = self.reference_image_resize
        x_factor = reference_resize / 1000
        y_factor = reference_resize / 1000
        
        gt_bboxes_ = meta_info["answer"]
        gt_bboxes = [gt_bboxes_[0] * x_factor + 0.5, gt_bboxes_[1] * y_factor + 0.5, gt_bboxes_[2] * x_factor + 0.5, gt_bboxes_[3] * y_factor + 0.5]
        
        # 将参考帧转换为图像格式
        replace_sent = ", ".join(f'({i+1}){"<image>"}' for i in range(video_length))
        user_prompt = self.user_prompt.replace('<video>', replace_sent)
        
        ################ New Version ################
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt.format(num=str(video_length), 
                                                           Question=exp.lower().strip("."),
                                                           )},
        ]
        ################ New Version ################
        
        row_dict = {}
        # tokenize the messages
        row_dict["video_info"] = (meta_info["images"], meta_info["question"])
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        raw_prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        resized_images = [Image.open(image_path).convert("RGB") for image_path in sampled_video_frames]
        sampled_video_frames = [img.resize((reference_resize, reference_resize), Image.BILINEAR) for img in resized_images]
        
        # print(prompt)
        # row_dict["images"] = [process_image(resized_keyframe, self.max_pixels, self.min_pixels)]
        row_dict["raw_images"] = resized_images
        row_dict["images"] = [process_image(image, self.max_pixels, self.min_pixels) for image in sampled_video_frames]
        
        # self.processor.image_processor.do_rescale = False
        image_inputs = self.processor.image_processor(row_dict["images"], return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        row_dict.update(image_inputs)
        
        if image_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size ** 2
            index = 0
            while "<image>" in prompt:
                prompt = prompt.replace(
                    "<image>",
                    "<|vision_start|>"
                    + "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length)
                    + "<|vision_end|>",
                    1,
                )
                index += 1

            prompt = prompt.replace("<|placeholder|>", self.processor.image_token)

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # print(prompt)
        # print(input_ids.shape, attention_mask.shape, image_grid_thw)
        if len(sampled_video_frames) > 1:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )  # (3, seq_len)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seqlen,)
        
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids     # 3, seq_len
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        row_dict["solution"] = gt_bboxes
        # row_dict["keyframe_id"] = keyframe_id
        # row_dict["gt_masks"] = resized_masks     # num_frames, H, W
        row_dict["ori_height"] = image_shape[0]
        row_dict["ori_width"] = image_shape[1]
        row_dict["height"] = reference_resize
        row_dict["width"] = reference_resize
        # row_dict["frame_ids"] = frame_ids
        row_dict["task_type"] = "multi_image"
        
        return row_dict
