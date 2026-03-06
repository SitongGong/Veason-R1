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
from qwen_vl_utils import process_vision_info


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

    # for key, value in tensors.items():
    #     if key not in ["pixel_values", "image_grid_thw", "gt_masks"]:
    #         tensors[key] = torch.stack(value, dim=0)      # num_batch, num_frames, N, D

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


class ReVOSDataset(Dataset):
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
        num_sampled_frames=8,
        keyframe_resize=560,
        reference_image_resize=560,
        video_metas=None, 
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
                
        self.video_metas = video_metas

        self.user_prompt =  "<video> " + "\n" + \
                            "You are given {num} frames uniformly sampled from a 1 FPS video. Each image is preceded by its timestamp." + "\n" + \
                            "Given the following expression: '{Question}', identify the timestamp where the target is best represented. Then, locate the target in that frame by predicting the bounding box(es)." + "\n" + \
                            "First, output the thinking process in <think> </think> tags, and then output the result in the following format within <answer> </answer> tags."
                           
    def __len__(self):
        return len(self.video_metas)

    def __getitem__(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        meta_info = self.video_metas[index]
        frame_path_list = meta_info['frame_path']
        video_length = meta_info['length']
        # anno_ids = meta_info['anno_id']
        obj_ids = meta_info['obj_id']
        exp = meta_info['exp']
        # height, width = meta_info["height"], meta_info["width"]

        # 首先对视频帧进行均匀采样
        if video_length > self.rvos_sampled_frames:
            # split_point = np.linspace(0, video_length, num=self.rvos_sampled_frames + 1, dtype=int)  # 从0开始均匀采样num_frames_per_sample + 1帧
            frame_ids = list(np.linspace(0, video_length - 1, num=self.rvos_sampled_frames, dtype=int))
            # frame_ids = [np.random.randint(split_point[i], split_point[i + 1]) for i in range(self.rvos_sampled_frames)]  # 每两个数字之间随机采样一帧，从而得到均匀分布的采样帧
        else:
            frame_ids = list(np.arange(video_length))
            
        # 选择出视频中第一帧作为关键帧
        sampled_video_frames = [frame_path_list[idx] for idx in frame_ids]
        keyframe = sampled_video_frames[0]
        image_shape = cv2.imread(keyframe).shape[:2]
            
        # 将参考帧转换为图像格式
        video_len = len(frame_ids)
        replace_sent = ", ".join(f'({frame_ids[i]}s){"<image>"}' for i in range(video_len))
        user_prompt = self.user_prompt.replace('<video>', replace_sent)
        
        ################ Old Version ################
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt.format(num=str(video_len), 
                                                           Question=exp.lower().strip("."),
                                                        #    Answer="{\"keyframe_timestamp\": N, \"bbox_2d_list\": [[x1,y1,x2,y2], [x1,y1,x2,y2]]}"
                                                           )},
        ]
        ################ Old Version ################
        
        row_dict = {}
        # keyframe_resize = self.keyframe_resize
        reference_image_resize = self.reference_image_resize
        # tokenize the messages
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # raw_prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        
        # resize the video frames
        # resized_keyframe = Image.open(keyframe).convert("RGB").resize((keyframe_resize, keyframe_resize), Image.BILINEAR)
        resized_images = [Image.open(image_path).convert("RGB") for image_path in sampled_video_frames]
        sampled_video_frames = [img.resize((reference_image_resize, reference_image_resize), Image.BILINEAR) for img in resized_images]
        row_dict["raw_images"] = [Image.open(img_path).convert("RGB") for img_path in frame_path_list]
        # row_dict["images"] = [process_image(resized_keyframe, self.max_pixels, self.min_pixels)]
        row_dict["images"] = [process_image(image, self.max_pixels, self.min_pixels) for image in sampled_video_frames]
        # print(len(row_dict["images"]))
        # print(np.array(row_dict["images"][0]))
        
        # self.processor.image_processor.do_rescale = False
        image_inputs = self.processor.image_processor(row_dict["images"], return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        row_dict.update(image_inputs)
        
        # print(row_dict["pixel_values"])
        
        # print(prompt)
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

        # tokenize the prompt
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

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
        row_dict["height"] = image_shape[0]
        row_dict["width"] = image_shape[1]
        row_dict["image_path_list"] = frame_path_list
        row_dict["exp_id"] = meta_info['exp_id']
        row_dict["video_name"] = meta_info["video"]
        row_dict["frame_ids"] = frame_ids
        # row_dict["keyframe_id"] = meta_info["tp"] * video_len
        
        return row_dict
