import os
import json
import random


referring_num = 0
reasoning_num = 0
single_instance = 0
multiple_instance = 0
json_file = "/dataset/rvos_root/ReVOS/meta_expressions_train_.json"
json_dict = json.load(open(json_file, "r"))["videos"]
for video_name, video_dict in json_dict.items():
    for exp_id, exp_dict in video_dict["expressions"].items():
        
        if exp_dict["type_id"] == 0:
            referring_num += 1
        elif exp_dict["type_id"] == 1:
            reasoning_num += 1
        if len(exp_dict["obj_id"]) == 1:
            single_instance += 1
        elif len(exp_dict["obj_id"]) > 1:
            multiple_instance += 1
    
referring_ratio = referring_num / (referring_num + reasoning_num)
reasoning_ratio = reasoning_num / (referring_num + reasoning_num)
single_ratio = single_instance / (single_instance + multiple_instance)
multiple_ratio = multiple_instance / (single_instance + multiple_instance)
    
ref_list = []
reason_list = []
single_list = []
multi_list = []
json_file = "/dataset/rvos_root/ReVOS/meta_expressions_train_.json"
video_data = json.load(open(json_file, "r"))["videos"]
for video_name, vid_data in video_data.items():
    for exp_id, exp_dict in vid_data["expressions"].items():
        
        if exp_dict['type_id'] == 0:
            ref_list.append((video_name, exp_id))
        elif exp_dict['type_id'] == 1:
            reason_list.append((video_name, exp_id))
            
        if len(exp_dict["obj_id"]) == 1:
            single_list.append((video_name, exp_id))
        elif len(exp_dict["obj_id"]) > 1:
            multi_list.append((video_name, exp_id))


def sample_by_overlap(type_list, obj_list, n):
    candidates = list(set(type_list) & set(obj_list))
    return random.sample(candidates, min(n, len(candidates)))

target_size = 10000
samples = []
samples += sample_by_overlap(ref_list, single_list, round(referring_ratio * single_ratio * target_size))   # ref + single
samples += sample_by_overlap(ref_list, multi_list, round(referring_ratio * multiple_ratio * target_size))    # ref + multi
samples += sample_by_overlap(reason_list, single_list, round(reasoning_ratio * single_ratio * target_size))# reason + single
samples += sample_by_overlap(reason_list, multi_list, round(reasoning_ratio * multiple_ratio * target_size)) # reason + multi

# 若不足6000，随机补足
samples = list(set(samples))
if len(samples) < target_size:
    remaining = list(set((video_name, exp_id)
                         for video_name, vdict in json_dict.items()
                         for exp_id in vdict["expressions"].keys()) - set(samples))
    samples += random.sample(remaining, target_size - len(samples))
else:
    samples = random.sample(samples, target_size)

new_json_dict = {} 
for video_name, video_dict in json_dict.items():
    new_expressions_dict = {}
    for exp_id, exp_dict in video_dict["expressions"].items():
        if (video_name, exp_id) in samples:
            new_expressions_dict[exp_id] = exp_dict
    new_video_dict = {
        "expressions": new_expressions_dict,
        "frames": video_dict["frames"],
        "vid_id": video_dict["vid_id"],
        "height": video_dict["height"],
        "width": video_dict["width"]
    }
    new_json_dict[video_name] = new_video_dict
    
output_dict = {"videos": new_json_dict}
output_file = "/dataset/rvos_root/ReVOS/meta_expressions_train_select.json"
with open(output_file, "w") as f:
    json_string = json.dumps(output_dict, indent=4)
    f.write(json_string)
