import re
import json
import math
import pdb
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


# 形式奖励函数
def vision_reasoner_format_reward(predict_str: str) -> float:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    thinking_format_reward = 1.0 if match else 0.0 
    
    def segmentation_format(predict_str: str) -> float:
        segmentation_format_reward = 0.0
        try:
            # pdb.set_trace()
            json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
            if not json_match:
                return segmentation_format_reward
            data = json.loads(json_match.group(1))
            
            data_cnt = len(data)
            # pdb.set_trace()
            
            for item in data:
                cur_reward = 0.0

                if 'bbox_2d' in item:
                    bbox_2d = item['bbox_2d']
                    if isinstance(bbox_2d, list) and len(bbox_2d) == 4:
                        cur_reward += 1.0
                
                segmentation_format_reward += cur_reward / data_cnt
        except Exception:
            pass
        return segmentation_format_reward
        
    segmentation_format_reward = segmentation_format(predict_str)      # 对于N个bbox_2d，格式每对一个加一分，最后取平均
    
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
    iou = intersection / union if union > 0 else 0
    
    return iou


def vision_reasoner_accuracy_reward(predict_str: str, ground_truth: str, gt_mask: torch.Tensor=None, seg_model=None, keyframe_id=None, images=None) -> float:
    max_accuracy_reward = 0.0
    mask_iou_reward = 0.0
    MAX_OBJECTS = 120  # 设置上限
    
    try:
        gt_data = json.loads(ground_truth)
        gt_bboxes = [item['bbox_2d'] for item in gt_data]
        
        # pdb.set_trace()
            
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            pred_bboxes = [item['bbox_2d'] for item in data]
            
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
            
            # 计算reward矩阵
            # iou_reward = (iou_matrix > 0.5).astype(float)
            iou_reward = iou_matrix.astype(float)
            
            # 构建最终的cost矩阵
            # cost_matrix = 3.0 - (iou_reward + bbox_l1_reward + point_reward)
            cost_matrix = 1.0 - iou_reward
            
            # 使用匈牙利算法找最优匹配
            # pdb.set_trace()
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # 直接从cost_matrix计算总reward
            total_reward = len(row_indices) * 1.0 - cost_matrix[row_indices, col_indices].sum()
            
            # 计算平均reward
            max_length = max(len(pred_bboxes), len(gt_bboxes))
            max_accuracy_reward = total_reward / max_length
            
            
            # mask iou reward计算
            bboxes = pred_bboxes[row_indices]   # N, 4
            results = seg_model.get_sam2_multiple_predict(images[0], 
                                                          boxes=list(bboxes))
            
            # 计算输出和GT的mask值
            # pdb.set_trace()
            masks = results[0]["masks"]
            mask_iou_reward = compute_iou(masks, gt_mask[keyframe_id].cpu().numpy())
        
            # if iou > 0.5:   
            #     return 1.0
            # else:
            #     return iou
            
    except Exception:
        pass
    return max_accuracy_reward + mask_iou_reward


def video_match_sam_seg_compute_score(predict_str: str, ground_truth: str, gt_mask: torch.Tensor=None, seg_model=None, keyframe_id=None, images=None) -> float:
    # print(predict_str, ground_truth)
    format_reward = vision_reasoner_format_reward(predict_str)
    accuracy_reward = vision_reasoner_accuracy_reward(predict_str, ground_truth, gt_mask, seg_model, keyframe_id, images)
    
    reward = format_reward + accuracy_reward
    return reward