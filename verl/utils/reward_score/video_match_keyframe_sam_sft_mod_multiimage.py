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
                isinstance(data.get("keyframe_timestamp"), int) and
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
    
    # pdb.set_trace()
    
    unionArea = box1Area + np.transpose(box2Area) - interArea
    iou = interArea / unionArea  # (M,N)
    return iou


def compute_iou(mask1, mask2):
        
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = 1.0 if union == 0 else intersection / union
    
    return iou


def vision_reasoner_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    max_accuracy_reward = 0.0
    MAX_OBJECTS = 120  # 设置上限
    
    try:
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            
            # pdb.set_trace()
            
            # 模型预测的关键帧对应的gt_mask
            gt_bboxes = ground_truth
            # 模型预测的bbox
            pred_bboxes = data["bbox_2d_list"]
            
            # 只有当预测或真实值超过上限时才截断
            if len(pred_bboxes) > MAX_OBJECTS:
                pred_bboxes = pred_bboxes[:MAX_OBJECTS]
            
            # 预处理数据为numpy数组
            pred_bboxes = np.array(pred_bboxes)  # (M,4)
            gt_bboxes = np.array([gt_bboxes])    # (N,4)
            
            # 并行计算所有指标
            iou_matrix = batch_iou(pred_bboxes, gt_bboxes)  # (M,N)
            
            # pdb.set_trace()
            
            # 计算reward矩阵
            iou_reward = iou_matrix.astype(float)
            
            # 构建最终的cost矩阵
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
    return max_accuracy_reward


def multi_image_compute_score(
    predict_str: str, 
    ground_truth: list, 
    # height: int, 
    # width: int, 
    # ori_height: int, 
    # ori_width: int
) -> float:
    """
    计算综合奖励分数
    - format_reward: 格式奖励（0或1）
    - accuracy_reward: IOU奖励（0-1之间的浮点数）
    
    Args:
        predict_str: 模型预测的字符串
        ground_truth: ground truth的bbox坐标列表，格式为 [[x1, y1, x2, y2], ...]
        height: 预测时使用的图像高度
        width: 预测时使用的图像宽度
        ori_height: 原始图像高度
        ori_width: 原始图像宽度
    
    Returns:
        float: 总奖励分数
    """
    # 计算格式奖励
    format_reward = vision_reasoner_format_reward(predict_str)
    
    # 计算IOU奖励（包含坐标转换）
    accuracy_reward = vision_reasoner_accuracy_reward(
        predict_str, ground_truth, # height, width, ori_height, ori_width
    )
    
    # 总奖励 = 格式奖励 + IOU奖励
    reward = format_reward + accuracy_reward
    return reward