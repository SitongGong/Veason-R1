import os
import re
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from openai import OpenAI
import base64
import numpy as np
import cv2
from PIL import Image, ImageTk
import multiprocessing as mp
from multiprocessing import Pool
import pycocotools.mask as maskUtils
import ast
import math
import random


def init_worker(results_dict_shared, lock_shared):
    global results_dict
    global lock
    results_dict = results_dict_shared
    lock = lock_shared


def encode_video(frame_list):
    base64Frames = []
    for frame in frame_list:
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    return base64Frames


class GPT4Agent():
    def __init__(self, ):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", # "sk-a2ca842073434fde9e713ae5acc6a2a8"), 
                                                    ""),
                            base_url=""
                            # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                             )
    
    def get_response(self, timestamps, keyframe_id, img_with_mask_list, exp):
        example1 = f"""Example:   \
                       Object description: 'identify the man standing beside the white truck.' \
                       Keyframe timestamp: 2      \
                       Step 1: I analyze the video by reviewing all frames. It shows a street scene with a white truck parked on the right. A man gradually approaches the truck from the left across the frames.    \
                       Step 2: Based on the expression 'identify the man standing beside the white truck', frame at timestamp 2 is the most suitable keyframe. In this frame, the man is standing right next to the truck, and both are clearly visible.     \
                       Step 3: In frame 2, the man is located on the right side of the image, beside the truck, standing upright and facing forward. """

        example2 = f"""Example:   \
                       Object description: 'locate the person who is starting to wave their hand.' \
                       Keyframe timestamp: 4      \
                       Step 1: The video shows a person standing still and then gradually raising one hand.   \
                       Step 2: Frame at timestamp 4 is the best keyframe because it captures the moment the hand is lifted above shoulder level, clearly initiating a waving gesture.    \
                       Step 3: In frame 4, the person is centered in the image, standing upright with their right arm partially raised in a wave. """

        example3 = f"""Example:   \
                       Object description: 'identify the child jumping off the bench.' \
                       Keyframe timestamp: 3      \
                       Step 1: The frames show a child climbing onto a bench, standing, and then jumping down.    \
                       Step 2: Considering the expression 'identify the child jumping off the bench', frame at timestamp 3 is the best keyframe since it captures the child mid-air, just after leaving the bench.       \
                       Step 3: In frame 3, the child is above ground level, positioned in the center, with bent knees and arms raised to balance. """

        examples = [example1, example2, example3]
        select_example = random.choice(examples)

        prompt = f"""You are given a sequence of video frames (1 frame per second). One or more objects are marked with red rectangles in each frame — these red boxes are **only for your reference** and **should not be mentioned or relied upon** in your answer.    \
                 Your task is to analyze the visual content of the video and determine how the provided expression relates to the frames. Use the following structured reasoning process:    \
                 Given the object description:  **"{exp}"**        \
                 Keyframe Timestamp: '{timestamps[keyframe_id]}' \
                 We have selected the frame at timestamp **{timestamps[keyframe_id]}** as the keyframe. You must base your reasoning on this frame, and explain why it best matches the target described.     \
                 ### Please follow this reasoning process step-by-step:       \
                 1. Briefly describe what is happening across all frames in the video.        \
                 2. Justify why the frame at timestamp {timestamps[keyframe_id]} best represents the described target.      \
                 3. Describe what the target is doing and precisely where it is located in the selected frame.        \
                 **Important:**         \
                 - You must analyze the provided keyframe '{timestamps[keyframe_id]}' — do not choose a different one.       \
                 - Do not refer to or describe the red rectangles.         \
                 - Write naturally, but follow the reasoning steps clearly.  \
                """ + select_example

        # try:
        content = []
        for idx, img_with_mask in enumerate(img_with_mask_list):
            if timestamps is not None:
                # add timestamp for each frame
                content.append({
                    "type": "text",
                    "text": f'[{timestamps[idx]} second]'
                })
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(img_with_mask)}"
                    },
                }
            )
        content.append(
            {
                "type": "text",
                "text": prompt,
        })
        
        response = self.client.chat.completions.create(
            model="doubao-1-5-thinking-vision-pro-250428",
            # model="qwen2.5-vl-72b-instruct", 
            messages=[{
                "role": "user",
                "content": content, 
                }],
        )
        return response.choices[0].message.content


def resize(image):
    height, width = image.shape[:2]
    if height < width:
        target_height, target_width = 480, 640
    else:
        target_height, target_width = 640, 480
    if height <= target_height and width <= target_width:
        return image
    if height / target_height < width / target_width:
        new_width = target_width
        new_height = int(height * (new_width / width))
    else:
        new_height = target_height
        new_width = int(width * (new_height / height))
    return cv2.resize(image, (new_width, new_height))


def encode_image(image: str) -> str:
    # image = cv2.imread(image_path)
    image_resized = resize(image)
    _, encoded_image = cv2.imencode(".jpg", image_resized)
    return base64.b64encode(encoded_image).decode("utf-8")


def preprocess_video(video_meta, sam2_predictor=None, keyframe_h=560, keyframe_w=560, rvos_sampled_frames=8, fps=1):
    frame_path_list = video_meta["frame_path"]
    exp_text = video_meta["exp"]
    obj_ids = video_meta["obj_id"]
    anno_ids = video_meta["anno_id"]
    video_length = video_meta['length']
    
    all_frames = [cv2.imread(frame_path) for frame_path in frame_path_list]
    processed_images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in all_frames]
    processed_images = [cv2.resize(img, (keyframe_h, keyframe_w)) for img in processed_images]
    image_shape = all_frames[0].shape[:2]
    height, width = image_shape
    
    frame_ids = list(range(len(processed_images)))
    m_final_list = []
    m_final_obj_list = []
    for seg_frame_id in frame_ids:
        m_final = np.zeros(image_shape, dtype=np.uint8)
        m_object = np.zeros(image_shape, dtype=np.uint8)
        m_object_list = []
        for x, obj_id in zip(anno_ids, obj_ids):  
            segm = mask_dict[str(x)][seg_frame_id]  
            if segm is not None:
                m = maskUtils.decode(segm)
                if m.ndim == 3:
                    m = m.sum(axis=2).astype(np.uint8)
                else:
                    m = m.astype(np.uint8)
                m_final = m_final | m  
                m_object = m_object | m
            m_object_list.append(m_object)
            
        m_final_obj_list.append(np.stack(m_object_list, axis=0))      # num_objects, H, W
        m_final_list.append(m_final)
        
    m_final_obj_list = np.stack(m_final_obj_list, axis=0)         # num_frames, num_objects, H, W
    m_final_list = np.stack(m_final_list, axis=0)             # num_frames, H, W
    
    areas = (m_final_list > 0).sum(axis=(1, 2))       # shape (T,)
    # first_valid_idx = int(areas.argmax())     
    top_k_indices = areas.argsort()[-5:][::-1]
    first_valid_idx = int(random.choice(top_k_indices))
        
    if rvos_sampled_frames != 1:  
        to_devide = (rvos_sampled_frames - 1)
        step_size = math.ceil(video_length / to_devide)
        idx_start = first_valid_idx % step_size
        idx_select = list(range(idx_start, video_length, step_size))  
        timestamps = [int(round(frame_idx / fps, 1)) for frame_idx in idx_select]    
    else:
        idx_select = [first_valid_idx, ]
    assert first_valid_idx in idx_select
    
    keyframe_id = idx_select.index(first_valid_idx)
    frame_ids = idx_select
    keyframe_mask = m_final_obj_list[first_valid_idx]     # num_obj, H, W
    if keyframe_mask.shape[0] > 1:
        pass
    sampled_video_frames = [frame_path_list[idx] for idx in idx_select]
    keyframe = sampled_video_frames[keyframe_id]
    m_final_list = m_final_list[idx_select]        # num_frames, H, W
    m_final_obj_list = m_final_obj_list[idx_select]       # num_frames, num_obj, H, W
    
    # 将mask覆盖在原图上
    output_dir = "output_with_bbox"
    os.makedirs(output_dir, exist_ok=True)
    img_with_mask_list = []
    for idx, img_path in enumerate(sampled_video_frames):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        masks = m_final_obj_list[idx]  # shape: (num_obj, H, W)

        for obj_mask in masks:
            obj_mask = (obj_mask > 0).astype(np.uint8)
            # 找mask区域坐标（非零点）
            points = cv2.findNonZero(obj_mask)
            if points is not None:
                # x, y, bw, bh = cv2.boundingRect(points)
                left = np.where(obj_mask == 1)[1].min()
                top = np.where(obj_mask == 1)[0].min()
                right = np.where(obj_mask == 1)[1].max()
                bottom = np.where(obj_mask == 1)[0].max()
                cv2.rectangle(img, (left, top), (right, bottom), color=(0, 0, 255), thickness=2)

        img_with_mask_list.append(img)
        
        save_path = os.path.join(output_dir, f"frame_{idx}.jpg")
        cv2.imwrite(save_path, img)
    
    # extract bbox of each frame to get the ground truth answer
    x_factor = keyframe_w / width
    y_factor = keyframe_h / height
    bbox_2d_list = dict()
    for idx, frame_mask in enumerate(m_final_obj_list): 
        if np.any(frame_mask == 1):
            bbox_2d_list[idx] = []
            for obj_mask in frame_mask:
                if np.all(obj_mask==0):
                    bbox_2d_list[idx].append([0, 0, 0, 0])
                else:
                    left = np.where(obj_mask == 1)[1].min()
                    top = np.where(obj_mask == 1)[0].min()
                    right = np.where(obj_mask == 1)[1].max()
                    bottom = np.where(obj_mask == 1)[0].max()
                    # resize the bbox
                    bbox_2d_list[idx].append([round(left * x_factor + 0.5, 2), round(top * y_factor + 0.5, 2), 
                                              round(right * x_factor + 0.5, 2), round(bottom * y_factor + 0.5, 2)])
        else:
            bbox_2d_list[idx] = [[0, 0, 0, 0]]    
    

    return sampled_video_frames, timestamps, keyframe_id, frame_ids, bbox_2d_list, img_with_mask_list


def split_list_evenly(lst, num_parts):
    avg = len(lst) // num_parts
    remainder = len(lst) % num_parts
    result = []
    start = 0
    for i in range(num_parts):
        end = start + avg + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end
    return result


def sub_processor(video_metas):
    # video_metas = video_metas
    new_video_metas = []
    gpt4_agent = GPT4Agent()      # 构建API
    for video_meta in video_metas:
        sampled_video_frames, timestamps, keyframe_id, frame_ids, bbox_2d_list, img_with_mask_list = preprocess_video(video_meta)
        print(timestamps, frame_ids)
        response = gpt4_agent.get_response(timestamps, keyframe_id, img_with_mask_list, video_meta['exp'])
        gt_bbox_2d = [bbox_2d for bbox_2d in bbox_2d_list[keyframe_id]]
        keyframe_timestamp = timestamps[keyframe_id]
        video_meta["CoT"] = "<think>" + response + "</think>"
        result_dict = {
                        "keyframe_timestamp": keyframe_timestamp,
                        "bbox_2d_list": gt_bbox_2d
                    }

        video_meta["sampled_video_frames"] = sampled_video_frames
        video_meta["frame_ids"] = frame_ids
        video_meta["solution"] = "<answer>" + json.dumps(result_dict) + "</answer>"
        video_meta["conversation"] = [{
                                          "from": "human",
                                          "value": "<video>" + "\n" + "Given a video sequence and the following expression: '" + video_meta['exp'] + "', identify the frame where the target is best represented. Then, locate the target in that frame by predicting the bounding box(es). First output the thinking process in <think> </think> tags and then output the timestamp of the keyframe and the bounding box(es) of the target(s) in <answer> </answer> tags."
                                      },
                                      {
                                          "from": "gpt",
                                          "value": "<think>" + response + "</think>" + "\n" + "<answer>" + json.dumps(result_dict) + "</answer>"
                                      }]
        new_video_metas.append(video_meta)
        
    return new_video_metas


if __name__ == "__main__":
        
    import time

    start_time = time.time()
    
    meta_json_file = "/dataset/rvos_root/ReVOS/meta_expressions_train_select.json"
    mask_dict_file = "/dataset/rvos_root/ReVOS/mask_dict.json"
    
    video_meta_dict = json.load(open(meta_json_file, 'r'))["videos"]
    mask_dict = json.load(open("/dataset/rvos_root/ReVOS/mask_dict.json", "r"))
    video_metas = []
    data_path = "/dataset/rvos_root/ReVOS"
    max_samples_per_video = 10
    for video_name, vid_data in video_meta_dict.items():
        vid_frames = sorted(vid_data['frames'])  # 00000, 00001, ...
        vid_len = len(vid_frames)
        
        limited_expressions = list(vid_data["expressions"].items())[:max_samples_per_video]
        for exp_id, exp_dict in limited_expressions:
            meta = {
            'video': video_name,
            'exp': exp_dict['exp'],
            'obj_id': exp_dict['obj_id'],
            'anno_id': exp_dict['anno_id'],
            'frames': vid_frames,
            'exp_id': exp_id,
            'category': 0,
            'length': vid_len,
            'frame_path': [os.path.join(data_path, "JPEGImages", video_name, frame_name + '.jpg') for frame_name in vid_frames]
        }

            video_metas.append(meta)

    num_sub_processor = 32
    splits = split_list_evenly(video_metas, num_sub_processor)    
    
    with Pool(processes=num_sub_processor) as pool:
        splits = [(result) for result in splits]
        results = pool.map(sub_processor, splits)
        
    # pool.close()
    # pool.join()  
        
    merged_results = [item for sublist in results for item in sublist]
    with open("/dataset/rvos_root/ReVOS/video_metas_subset_6k_.json", "w", encoding="utf-8") as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)

    print(time.time() - start_time)