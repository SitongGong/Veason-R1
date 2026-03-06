import re
import json
import math
import pdb
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


# 形式奖励函数
def vision_reasoner_format_reward(predict_str: str) -> float:
    """
    - thinking_format_reward : 若 <think>…</think><answer>…</answer> 格式完整则记 1 分
    - segmentation_format_reward :
        • 解析 <answer>{...}</answer> 里的 JSON
        • 必须含 "keyframe_id"(int) 和 "bbox_2d_list"(list)
        • list 中每个元素应是长度 4 的数字序列
        • 每个合法 bbox 得 1/n 分，n 为 list 长度
    """
    # 1) 检查总体标签格式
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    thinking_format_reward = 1.0 if re.fullmatch(pattern, predict_str, re.DOTALL) else 0.0

    # 2) 分割格式奖励
    # pdb.set_trace()
    segmentation_format_reward = 0.0
    try:
        json_match = re.search(r'<answer>\s*(\{.*?\})\s*</answer>', predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))

            if (isinstance(data, dict) and
                isinstance(data.get("keyframe_id"), int) and
                isinstance(data.get("bbox_2d_list"), list)):

                bboxes = data["bbox_2d_list"]
                n = len(bboxes) if bboxes else 1   # 防止除 0

                for bbox in bboxes:
                    # 新格式：bbox 直接是 list/tuple 长度 4
                    if (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and
                        all(isinstance(v, (int, float)) for v in bbox)):
                        segmentation_format_reward += 1.0 / n
    except Exception:
        pass

    return thinking_format_reward + segmentation_format_reward


# 关键帧的representative奖励函数
def vision_reasoner_keyframe_reward(predict_str: str, gt_mask: torch.Tensor):
    """
    Args:
        predict_str   : 模型完整输出字符串，内部包含 <answer>{...}</answer>
        max_mask_area : 整个视频中最大 GT mask 面积（像素数）
        gt_mask       : Tensor，形状 [T, H, W] 或 [T, 1, H, W]，二值 / 概率 mask

    Returns:
        reward ∈ [0, 1]：关键帧 mask 面积 / max_mask_area
    """
    max_keyframe_reward = 0.0
    
    # ↓ 1. 没有目标：reward 设 0
    gt_areas = (gt_mask > 0).sum(axis=(1, 2))   # shape (T,)
    max_mask_area = max(gt_areas)
    if max_mask_area <= 0:
        return max_keyframe_reward
    # pdb.set_trace()
    try:
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if not json_match:
            return max_keyframe_reward                      # 格式不对 → 0

        data = json.loads(json_match.group(1))
        keyframe_id = int(data.get("keyframe_id", -1))
        
        # 2. 检查 keyframe_id 是否有效
        if keyframe_id < 0 or keyframe_id >= gt_mask.shape[0]:
            return max_keyframe_reward
        
        # 3. 取出关键帧 mask，并计算其面积
        frame_mask = gt_mask[keyframe_id]
        if frame_mask.dim() == 3:              # 若形状 [C, H, W] 取第 0 通道
            frame_mask = frame_mask[0]
        keyframe_area = (frame_mask > 0).sum().item()
        
        # 4. 归一化，得到奖励
        if max_mask_area > 0:
            reward = keyframe_area / max_mask_area
            max_keyframe_reward = max(0.0, min(1.0, reward))  # 限定 [0,1]
            
    except Exception:
        pass
    return max_keyframe_reward


def batch_iou(boxes1, boxes2):
    # boxes1: (M,4), boxes2: (N,4)
    # 广播机制自动扩展维度
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)  # (M,1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)  # (N,1)
    
    xA = np.maximum(x11, np.transpose(x21))  # (M,N)
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    box1Area = (x12 - x11 + 1) * (y12 - y11 + 1)  # (M,1)
    box2Area = (x22 - x21 + 1) * (y22 - y21 + 1)  # (N,1)
    
    unionArea = box1Area + np.transpose(box2Area) - interArea
    iou = interArea / unionArea  # (M,N)
    return iou


def compute_iou(mask1, mask2):
        
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = 1.0 if union == 0 else intersection / union
    
    return iou


# 关键帧的iou匹配reward + mask传播reward
def vision_reasoner_accuracy_reward(predict_str: str, ground_truth: str, gt_mask: torch.Tensor=None, seg_model=None, images=None) -> float:
    max_accuracy_reward = 0.0
    avg_iou_reward = 0.0
    MAX_OBJECTS = 120  # 设置上限
    
    try:
        gt_data = json.loads(ground_truth)
        # gt_bboxes = [item['bbox_2d'] for item in gt_data]
        # pdb.set_trace()
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            keyframe_id = int(data.get("keyframe_id", -1))
            if keyframe_id < 0 or keyframe_id >= gt_mask.shape[0]:
                return 0.0
            
            # 模型预测的关键帧对应的gt_mask
            gt_bboxes = gt_data[str(keyframe_id)]
            gt_bboxes = [box["bbox_2d"] for box in gt_bboxes]
            # 模型预测的bbox
            pred_bbox_list = data["bbox_2d_list"]
            pred_bboxes = [item for item in pred_bbox_list]
            
            # 只有当预测或真实值超过上限时才截断
            if len(pred_bboxes) > MAX_OBJECTS:
                pred_bboxes = pred_bboxes[:MAX_OBJECTS]
            
            if len(gt_bboxes) > MAX_OBJECTS:
                gt_bboxes = gt_bboxes[:MAX_OBJECTS]
            
            # 预处理数据为numpy数组
            pred_bboxes = np.array(pred_bboxes)  # (M,4)
            gt_bboxes = np.array(gt_bboxes)    # (N,4)
            
            # 并行计算所有指标
            iou_matrix = batch_iou(pred_bboxes, gt_bboxes)  # (M,N)
            
            # pdb.set_trace()
            
            # 计算reward矩阵
            # iou_reward = (iou_matrix > 0.5).astype(float)
            iou_reward = iou_matrix.astype(float)
            
            # 构建最终的cost矩阵
            # cost_matrix = 3.0 - (iou_reward + bbox_l1_reward + point_reward)
            cost_matrix = 1.0 - iou_reward
            
            # 使用匈牙利算法找最优匹配
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # 直接从cost_matrix计算总reward
            total_reward = len(row_indices) * 1.0 - cost_matrix[row_indices, col_indices].sum()
            
            # 计算平均reward
            max_length = max(len(pred_bboxes), len(gt_bboxes))
            max_accuracy_reward = total_reward / max_length
            
    except Exception:
        pass
    return max_accuracy_reward + avg_iou_reward


def vision_reasoner_seg_reward(predict_str: str, gt_mask: torch.Tensor=None, seg_model=None, images=None) -> float:
    avg_iou_reward = 0.0
    MAX_OBJECTS = 120  # 设置上限
    try:
        # gt_data = json.loads(ground_truth)
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
        
            keyframe_id = int(data.get("keyframe_id", -1))
            if keyframe_id < 0 or keyframe_id >= gt_mask.shape[0]:
                return 0.0
            
            # 模型预测的bbox
            # pdb.set_trace()
            pred_bbox_list = data["bbox_2d_list"]
            pred_bboxes = [item for item in pred_bbox_list]
            
            # 只有当预测或真实值超过上限时才截断
            if len(pred_bboxes) > MAX_OBJECTS:
                pred_bboxes = pred_bboxes[:MAX_OBJECTS]
            
            # 预处理数据为numpy数组
            pred_bboxes = np.array(pred_bboxes)  # (M,4)
            # pdb.set_trace()
            results = seg_model.get_sam2_video_predict(video_frame_list=images, 
                                                        boxes=list(pred_bboxes), 
                                                        keyframe_idx=keyframe_id)
            
            # 计算输出和GT的mask的iou值
            # pdb.set_trace()
            pred_masks = results[0]["video_masks"]      # num_frame, h, w
            total_iou_score = 0.0
            for pred_mask, mask in zip(pred_masks, gt_mask):
                total_iou_score += compute_iou(pred_mask, mask)
            # pdb.set_trace()
            avg_iou_reward = total_iou_score / len(pred_masks)
        
    except Exception:
        pass
    return avg_iou_reward


def video_match_keyframe_external_compute_score(predict_str: str, ground_truth: str, gt_mask: torch.Tensor=None, seg_model=None, keyframe_id=None, images=None) -> float:
    # print(predict_str, ground_truth)
    format_reward = vision_reasoner_format_reward(predict_str)
    # keyframe_reward = vision_reasoner_keyframe_reward(predict_str, gt_mask)
    accuracy_reward = vision_reasoner_accuracy_reward(predict_str, ground_truth, gt_mask, seg_model, images)
    seg_reward = vision_reasoner_seg_reward(predict_str, gt_mask, seg_model, images)
    
    reward = format_reward + accuracy_reward + seg_reward
    return reward