import re
import json
import math
import pdb
import torch
import numpy as np


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
    
    unionArea = box1Area + np.transpose(box2Area) - interArea
    iou = interArea / unionArea  # (M,N)
    return iou


def compute_iou(mask1, mask2):
        
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = 1.0 if union == 0 else intersection / union
    
    return iou


# 从answer中提取坐标并计算IOU reward
def vision_reasoner_accuracy_reward(
    predict_str: str, 
    ground_truth: list, 
    height: int, 
    width: int, 
    ori_height: int, 
    ori_width: int
) -> float:
    """
    从predict_str的answer中提取单个bbox坐标，转换到原始图像尺寸，与ground_truth中的bbox坐标进行IOU计算
    IOU直接作为最终reward
    
    Args:
        predict_str: 模型预测的字符串，包含<answer>标签，应该包含单个bbox [x1, y1, x2, y2]
        ground_truth: ground truth的bbox坐标，格式为 [x1, y1, x2, y2]（单个bbox列表）
        height: 预测时使用的图像高度
        width: 预测时使用的图像宽度
        ori_height: 原始图像高度
        ori_width: 原始图像宽度
    
    Returns:
        float: IOU奖励值（0-1之间）
    """
    try:
        # 1. 从predict_str中提取answer中的坐标
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if not answer_match:
            return 0.0
        
        pdb.set_trace()
        
        answer_content = answer_match.group(1).strip()
        
        # 提取 [x1, y1, x2, y2] 格式的坐标（只取第一个匹配）
        coord_pattern = r'\[\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\]'
        coord_match = re.search(coord_pattern, answer_content)
        
        if not coord_match:
            return 0.0
        
        # 提取第一个（也是唯一一个）bbox坐标
        try:
            pred_bbox = [float(x) for x in coord_match.groups()]
            if len(pred_bbox) != 4:
                return 0.0
        except (ValueError, TypeError):
            return 0.0
        
        # 2. 将预测坐标从(height, width)转换到(ori_height, ori_width)
        # 计算缩放比例
        scale_x = ori_width / width if width > 0 else 1.0
        scale_y = ori_height / height if height > 0 else 1.0
        
        # 转换预测bbox坐标到原始图像尺寸
        x1, y1, x2, y2 = pred_bbox
        x1_ori = x1 * scale_x
        y1_ori = y1 * scale_y
        x2_ori = x2 * scale_x
        y2_ori = y2 * scale_y
        pred_bbox_scaled = [x1_ori, y1_ori, x2_ori, y2_ori]
        
        # 3. 处理ground_truth，确保格式正确（应该是单个bbox列表）
        if not isinstance(ground_truth, list) or len(ground_truth) != 4:
            return 0.0
        
        try:
            gt_bbox = [float(x) for x in ground_truth]
        except (ValueError, TypeError):
            return 0.0
        
        # 4. 计算单个IOU
        pred_bbox_array = np.array([pred_bbox_scaled])  # (1, 4)
        gt_bbox_array = np.array([gt_bbox])  # (1, 4)
        
        # 使用batch_iou计算，但只取第一个（也是唯一一个）IOU值
        iou_matrix = batch_iou(pred_bbox_array, gt_bbox_array)  # (1, 1)
        iou = float(iou_matrix[0, 0]) if iou_matrix.size > 0 else 0.0
        
        return iou
            
    except Exception as e:
        # 如果出现任何错误，返回0
        print(f"Error in vision_reasoner_accuracy_reward: {e}")
        return 0.0


def multi_image_compute_score(
    predict_str: str, 
    ground_truth: list, 
    height: int, 
    width: int, 
    ori_height: int, 
    ori_width: int
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
        predict_str, ground_truth, height, width, ori_height, ori_width
    )
    
    # 总奖励 = 格式奖励 + IOU奖励
    reward = format_reward + accuracy_reward
    return reward