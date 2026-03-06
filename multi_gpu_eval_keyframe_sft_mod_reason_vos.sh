#!/usr/bin/env bash
source ~/miniconda3/bin/activate video_seg_zero
conda env list
cd /Seg-Zero/evaluation_scripts

# Adjust to your gpu num
GPU_IDS=(0 1)
SPLIT_NUM=2

for i in "${!GPU_IDS[@]}"; do
    GPU_ID=${GPU_IDS[$i]}
    SPLIT=$i
    echo "Launching on GPU=$GPU_ID with SPLIT=$SPLIT"
    
    # 显式设置每个子进程只使用对应 GPU
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    SPLIT=$SPLIT \
    SPLIT_NUM=$SPLIT_NUM \
    python multi_gpu_eval_keyframe_sft_mod_reason_vos.py \
        --reasoning_model_path "/Seg-Zero/workdir/run_qwen2_5_3b_revos_keyframe_sam_sft_mod/global_step_625/actor/huggingface" \
        --data_path "/ReasonVOS" \
        --output_path "/Seg-Zero/runs/keyframe_sam_sft_mod_reavos_sam_2_1" &

    sleep 1
done

echo "All tasks completed!"