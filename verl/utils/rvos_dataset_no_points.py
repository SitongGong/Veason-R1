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


class ReVOSDataset_(Dataset):
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
        keyframe_resize=840,
        reference_image_resize=840,
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
        
        
        train_json_path = os.path.join(data_path, "meta_expressions_train_filtered.json")
        f = open(train_json_path, "r")
        video_data = json.load(f)["videos"]
        video_metas = []
        
        mask_dict_path = os.path.join(data_path, "mask_dict.json")
        self.mask_dict = json.load(open(mask_dict_path, "r"))
        for video_name, vid_data in video_data.items():
            vid_frames = sorted(vid_data['frames'])  # 00000, 00001, ...
            vid_len = len(vid_frames)
            for exp_id, exp_dict in vid_data["expressions"].items():
                meta = {}
                meta['video'] = video_name  # 377b1c5f365c
                meta['exp'] = exp_dict['exp']  # 4 lizards moving around
                meta['obj_id'] = [int(x) for x in exp_dict['obj_id']]  # [0, 1, 2, 3, ]
                meta['anno_id'] = [str(x) for x in exp_dict['anno_id']]  # [2, 3, 4, 5, ]
                meta['frames'] = vid_frames  # ['00000', '00001', ...]
                meta['exp_id'] = exp_id  # '0'
                meta['category'] = 0
                meta['length'] = vid_len
                meta['frame_path'] = [os.path.join(data_path, "JPEGImages", video_name, frame_name + '.jpg') for frame_name in vid_frames]

                video_metas.append(meta)
                
        self.video_metas = video_metas

        self.user_prompt = "Keyframe: <image> " \
                           "Reference Video Frames: <video> " \
                           "Given the Reference Video Frames and one Keyframe, please identify '{Question}' and respond with its bounding box in the Keyframe." \
                           "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
                           "Output the only one bbox of the referred object in JSON format." \
                           "If the object appears in the keyframe, provide the output in the format:" \
                           "<think> thinking process here </think>" \
                           "<answer>{Answer1}</answer>" \
                           "If the object is not found in the keyframe, provide the output in the format:" \
                           "<think> thinking process here </think>" \
                           "<answer>{Answer2}</answer>" \
                           "Do not copy or imitate the examples. Think based on the video content."
                           

    def __len__(self):
        # return len(self.dataset)
        return len(self.video_metas)

    def __getitem__(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        # row_dict = self.dataset[index]
        # pdb.set_trace()
        meta_info = self.video_metas[index]
        frame_path_list = meta_info['frame_path']
        video_length = meta_info['length']
        # height, width = meta_info["height"], meta_info["width"]

        # uniform sampling
        if video_length > self.rvos_sampled_frames:
            split_point = np.linspace(0, video_length, num=self.rvos_sampled_frames + 1, dtype=int)  # 从0开始均匀采样num_frames_per_sample + 1帧
            frame_ids = [np.random.randint(split_point[i], split_point[i + 1]) for i in range(self.rvos_sampled_frames)]  # 每两个数字之间随机采样一帧，从而得到均匀分布的采样帧
        else:
            frame_ids = list(np.arange(video_length))
            
        # extract the mask from the mask_dict
        # print(frame_ids)
        sampled_video_frames = [frame_path_list[idx] for idx in frame_ids]
        # print(sampled_video_frames)
        image_shape = cv2.imread(sampled_video_frames[0]).shape[:2]
        height, width = image_shape

        anno_ids = meta_info['anno_id']
        obj_ids = meta_info['obj_id']
        exp = meta_info['exp']

        m_final_list = []
        for seg_frame_id in frame_ids:
            m_final = np.zeros(image_shape, dtype=np.uint8)
            for x, obj_id in zip(anno_ids, obj_ids):  # 对于每个物体的标注索引和类别索引
                segm = self.mask_dict[x][seg_frame_id]  # x表示视频索引，seg_frame_id表示帧索引，对应帧的mask标注
                if segm is not None:
                    m = maskUtils.decode(segm)
                    if m.ndim == 3:
                        m = m.sum(axis=2).astype(np.uint8)
                    else:
                        m = m.astype(np.uint8)
                    m_final = m_final | m  # 对应某一帧的mask标注
            m_final_list.append(m_final)

        m_final_list = np.stack(m_final_list, axis=0)  # (num_frame, H, W)
        # resized_masks = [cv2.resize(mask, (840, 840), interpolation=cv2.INTER_NEAREST) for mask in m_final_list]
        # m_final_list = np.stack(resized_masks, axis=0)
        
        # 首先采样关键帧 (要确保采样帧中存在前景目标，如果前景目标不存在，则随机采样视频帧)
        fg_ids = [i for i in range(len(m_final_list)) if m_final_list[i].sum() > 0]
        if fg_ids:
            keyframe_id = random.choice(fg_ids)
        else:
            keyframe_id = random.choice(range(len(m_final_list)))

        keyframe = sampled_video_frames[keyframe_id]
        keyframe_mask = m_final_list[keyframe_id]     # H, W
        
        # extract bbox of the keyframe
        # print(np.where(keyframe_mask == 1))
        keyframe_resize = self.keyframe_resize
        if np.any(keyframe_mask == 1):
            left = np.where(keyframe_mask == 1)[1].min()
            top = np.where(keyframe_mask == 1)[0].min()
            right = np.where(keyframe_mask == 1)[1].max()
            bottom = np.where(keyframe_mask == 1)[0].max()
            bbox_2d = [left, top, right, bottom]
        else:
            bbox_2d = [0, 0, 0, 0]
        # print(bbox_2d)
        
        # resize the bbox
        x_factor = keyframe_resize / width
        y_factor = keyframe_resize / height
        solution = f"<box>({int(bbox_2d[0] * x_factor + 0.5)},{int(bbox_2d[1] * y_factor + 0.5)})," \
                   f"({int(bbox_2d[2] * x_factor + 0.5)},{int(bbox_2d[3] * y_factor + 0.5)})</box>"
            
            
        ########## 用来验证bbox是否正确 ##########
        # x1 = int(bbox_2d[0] * x_factor + 0.5)
        # y1 = int(bbox_2d[1] * y_factor + 0.5)
        # x2 = int(bbox_2d[2] * x_factor + 0.5)
        # y2 = int(bbox_2d[3] * y_factor + 0.5)

        # # Step 2: 在图像上画出 BBox
        # from PIL import ImageDraw
        # resized_keyframe = Image.open(keyframe).convert("RGB").resize((keyframe_resize, keyframe_resize), Image.BILINEAR)
        # draw = ImageDraw.Draw(resized_keyframe)
        # draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)

        # # Step 3: 保存图像，供检查
        # save_path = os.path.join('/18515601223/Seg-Zero/results_validate', str(index) + "_keyframe_with_bbox.jpg")
        # resized_keyframe.save(save_path)
        # print(solution)
        #####################################
        
        
        # 将参考帧转换为图像格式
        video_len = len(frame_ids)
        replace_sent = ", ".join(f'({i}){"<image>"}' for i in range(video_len))
        user_prompt = self.user_prompt.replace('<video>', replace_sent)
        
        ################ Old Version ################
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt.format(Question=exp.lower().strip("."),
                                                           Answer1="{'bbox': [x1,y1,x2,y2]}",
                                                           Answer2="{'bbox': [0,0,0,0]}")},
        ]
        ################ Old Version ################
        
        row_dict = {}
        reference_image_resize = self.reference_image_resize
        # tokenize the messages
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        raw_prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        # print(raw_prompt)
        # resized_keyframe = cv2.resize(cv2.cvtColor(cv2.imread(keyframe), cv2.COLOR_BGR2RGB), (keyframe_resize, keyframe_resize))
        # resize the reference video frames
        # resized_images = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) for image_path in sampled_video_frames]
        # sampled_video_frames = [cv2.resize(img, (reference_image_resize, reference_image_resize), interpolation=cv2.INTER_AREA) for img in resized_images]
        resized_keyframe = Image.open(keyframe).convert("RGB").resize((keyframe_resize, keyframe_resize), Image.BILINEAR)
        resized_images = [Image.open(image_path).convert("RGB") for image_path in sampled_video_frames]
        sampled_video_frames = [img.resize((reference_image_resize, reference_image_resize), Image.BILINEAR) for img in resized_images]
        
        row_dict["images"] = [process_image(resized_keyframe, self.max_pixels, self.min_pixels)]
        row_dict["images"].extend([process_image(image, self.max_pixels, self.min_pixels) for image in sampled_video_frames])
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

        # print(prompt)

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

        gt_masks = torch.from_numpy(m_final_list)
        resized_masks = gt_masks.unsqueeze(1)     # num_frames, 1, H, W
        resized_masks = F.interpolate(resized_masks, 
                                      size=(keyframe_resize, keyframe_resize), 
                                      mode="nearest").squeeze(1)  # num_frames, H, W
        
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids     # 3, seq_len
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        row_dict["solution"] = solution
        row_dict["keyframe_id"] = keyframe_id
        row_dict["gt_masks"] = resized_masks     # num_frames, H, W
        row_dict["height"] = image_shape[0]
        row_dict["width"] = image_shape[1]
        
        # print(list(row_dict.keys()))
        
        return row_dict
