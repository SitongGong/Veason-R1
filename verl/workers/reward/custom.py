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

import pdb
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import math_compute_score, r1v_compute_score, seg_compute_score, seg_strict_compute_score, video_match_keyframe_sft_external_compute_score, multi_image_compute_score, \
                                    video_seg_strict_compute_score, video_seg_strict_compute_score_sam2, video_seg_strict_compute_score_sam2_no_points, video_match_keyframe_compute_score, \
                                    video_match_sam_seg_compute_score, video_match_seg_compute_score, video_match_keyframe_sam_compute_score, video_match_external_seg_compute_score, \
                                    video_match_keyframe_external_compute_score, video_match_keyframe_sft_sam_compute_score, video_match_keyframe_sft_sam_mod_compute_score, ablation_keyframe_sft_sam_mod_compute_score, \
                                    ablation_keyframe_sft_only_external_compute_score, ablation_keyframe_sft_only_sam_mod_compute_score, ablation_no_keyframe_compute_score, video_match_keyframe_sft_sam3_mod_compute_score, video_match_keyframe_sft_sam3_mod_compute_score_text


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.score_type = compute_score
        if compute_score == "math":
            self.compute_score = math_compute_score
        elif compute_score == "r1v":
            self.compute_score = r1v_compute_score
        elif compute_score == "seg":
            self.compute_score = seg_compute_score
        elif compute_score == "seg_strict":
            self.compute_score = seg_strict_compute_score
        elif compute_score == "video_seg_strict":
            self.compute_score = video_seg_strict_compute_score
        elif compute_score == "video_seg_sam_strict":
            self.compute_score = video_seg_strict_compute_score_sam2
        elif compute_score == "video_seg_sam_strict_no_points":
            self.compute_score = video_seg_strict_compute_score_sam2_no_points
        elif compute_score == "video_seg_match":
            self.compute_score = video_match_seg_compute_score
        elif compute_score == "video_seg_match_sam":
            self.compute_score = video_match_sam_seg_compute_score
        elif compute_score == "video_seg_match_keyframe":
            self.compute_score = video_match_keyframe_compute_score
        elif compute_score == "video_seg_match_keyframe_sam":
            self.compute_score = video_match_keyframe_sam_compute_score
        elif compute_score == "video_seg_match_external_sam":
            self.compute_score = video_match_external_seg_compute_score
        elif compute_score == "video_match_keyframe_external_sam":
            self.compute_score = video_match_keyframe_external_compute_score
        elif compute_score == "video_match_keyframe_sft_sam":
            self.compute_score = video_match_keyframe_sft_sam_compute_score
        elif compute_score == "video_match_keyframe_sft_external":
            self.compute_score = video_match_keyframe_sft_external_compute_score
        elif compute_score == "video_match_keyframe_sft_sam_mod":
            self.compute_score = video_match_keyframe_sft_sam_mod_compute_score
        elif compute_score == "ablation_keyframe_sft_sam_mod_compute_score":
            self.compute_score = ablation_keyframe_sft_sam_mod_compute_score
        elif compute_score == "ablation_keyframe_sft_only_external":
            self.compute_score = ablation_keyframe_sft_only_external_compute_score
        elif compute_score == "ablation_keyframe_sft_only_sam":
            self.compute_score = ablation_keyframe_sft_only_sam_mod_compute_score
        elif compute_score == "ablation_no_keyframe_reward":
            self.compute_score = ablation_no_keyframe_compute_score
        elif compute_score == "video_match_keyframe_sft_sam3_mod":
            self.compute_score = video_match_keyframe_sft_sam3_mod_compute_score
        elif compute_score == "video_match_keyframe_sft_sam3_mod_text":
            self.compute_score = video_match_keyframe_sft_sam3_mod_compute_score_text
        elif compute_score == "multi_image":
            self.compute_score = multi_image_compute_score
        elif compute_score == "mix_training":
            self.compute_score = video_match_keyframe_sft_sam_mod_compute_score
            self.compute_score2 = multi_image_compute_score
        else:
            raise NotImplementedError()

    def __call__(self, data: DataProto, segmentation_model=None) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print = 0

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()     # 有效答案的总长度
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # ground_truth = data_item.non_tensor_batch["answer"]
            ground_truth = data_item.non_tensor_batch["solution"]
            # print(ground_truth,response_str)
            
            ############ For video segmentation task ############
            # get the mask ground truth
            if "gt_masks" in data_item.non_tensor_batch.keys():
                gt_mask = data_item.non_tensor_batch["gt_masks"]
            else:
                gt_mask = None
                
            if "images" in data_item.non_tensor_batch.keys():
                images = data_item.non_tensor_batch["images"]
            else:
                images = None
                
            if "keyframe_id" in data_item.non_tensor_batch.keys():
                keyframe_id = data_item.non_tensor_batch["keyframe_id"]
            else:
                keyframe_id = None
            ############ For video segmentation task ############
            # import pdb 
            # pdb.set_trace()
            if "frame_ids" in data_item.non_tensor_batch.keys():
                frame_ids = data_item.non_tensor_batch["frame_ids"]
            else:
                frame_ids = None
                
            video_path = data_item.non_tensor_batch.get("video_path", None) # 增加
            task_type = data_item.non_tensor_batch.get("task_type", None)

            if task_type == "vrs":
                if "frame_ids" in data_item.non_tensor_batch.keys():
                    score = self.compute_score(response_str, ground_truth, gt_mask, segmentation_model, keyframe_id, images, frame_ids) # , video_path)
                else:
                    score = self.compute_score(response_str, ground_truth, gt_mask, segmentation_model, keyframe_id, images) # , video_path)
            elif task_type == "multi_image":
                # pdb.set_trace()
                score = self.compute_score(response_str, ground_truth, 
                                        #    data_item.non_tensor_batch["height"], 
                                        #    data_item.non_tensor_batch["width"],
                                        #    data_item.non_tensor_batch["ori_height"],
                                        #    data_item.non_tensor_batch["ori_width"],
                                           )
                
                
            # pdb.set_trace()
                
            reward_tensor[i, valid_response_length - 1] = score

            if already_print < self.num_examine:
                already_print += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        return reward_tensor
