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


from .math import math_compute_score
from .r1v import r1v_compute_score
from .seg import seg_compute_score
from .seg_restrict import seg_strict_compute_score
from .video_seg import video_seg_strict_compute_score
from .video_sam_seg import video_seg_strict_compute_score_sam2
from .video_sam_no_points import video_seg_strict_compute_score_sam2_no_points
from .video_match_seg import video_match_seg_compute_score
from .video_match_sam_seg import video_match_sam_seg_compute_score
from .video_match_keyframe import video_match_keyframe_compute_score
from .video_match_keyframe_sam import video_match_keyframe_sam_compute_score
from .video_match_external_seg import video_match_external_seg_compute_score
from .video_match_keyframe_external import video_match_keyframe_external_compute_score
from .video_match_keyframe_sam_sft import video_match_keyframe_sft_sam_compute_score
from .video_match_keyframe_external_sft import video_match_keyframe_sft_external_compute_score
from .video_match_keyframe_sam_sft_mod import video_match_keyframe_sft_sam_mod_compute_score
from .ablation_keyframe_no_sam import ablation_keyframe_sft_sam_mod_compute_score
from .ablation_keyframe_external_sft import ablation_keyframe_sft_only_external_compute_score
from .ablation_keyframe_sam_sft import ablation_keyframe_sft_only_sam_mod_compute_score
from .ablation_no_keyframe_reward import ablation_no_keyframe_compute_score
from .video_match_keyframe_sam_sft_mod_sam3 import video_match_keyframe_sft_sam3_mod_compute_score
from .video_match_keyframe_sam_sft_mod_sam3_text import video_match_keyframe_sft_sam3_mod_compute_score_text
# from .multi_image import multi_image_compute_score
from .video_match_keyframe_sam_sft_mod_multiimage import multi_image_compute_score


__all__ = ["math_compute_score", "r1v_compute_score", "seg_compute_score", "seg_strict_compute_score", "video_seg_strict_compute_score", "video_seg_strict_compute_score_sam2", "video_match_keyframe_sft_sam_mod_compute_score", 
           "video_seg_strict_compute_score_sam2_no_points", "video_match_sam_seg_compute_score", "video_match_seg_compute_score", "video_match_keyframe_compute_score", "video_match_keyframe_sam_compute_score", 
           "video_match_keyframe_sft_external_compute_score", "video_match_external_seg_compute_score", "video_match_keyframe_external_compute_score", "video_match_keyframe_sft_sam_compute_score",
           "ablation_keyframe_sft_sam_mod_compute_score", "ablation_keyframe_sft_only_sam_mod_compute_score", "ablation_keyframe_sft_only_external_compute_score",
           "ablation_no_keyframe_compute_score", "video_match_keyframe_sft_sam3_mod_compute_score", "video_match_keyframe_sft_sam3_mod_compute_score_text", "multi_image_compute_score"]
