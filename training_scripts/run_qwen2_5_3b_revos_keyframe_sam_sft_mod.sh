#!/usr/bin/env bash
source ~/miniconda3/bin/activate video_seg_zero
conda env list
cd /18515601223/Seg-Zero

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export MKL_SERVICE_FORCE_INTEL=1
export RAY_memory_monitor_refresh_ms=0

MODEL_PATH=/18515601223/LLaMA-Factory/output/qwen2_5vl_lora_sft_6k_keyframe_  # replace it with your local file path

RUN_NAME=$(basename "$0" .sh)

python3 -m verl.trainer.main \
    config=/18515601223/Seg-Zero/training_scripts/video_seg_zero_3b_keyframe_sam_sft_mod.yaml \
    data.val_files=None \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.kl_loss_coef=5.0e-3 \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.n=8 \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=2 \
    trainer.total_episodes=1 \
    trainer.save_checkpoint_path=/18515601223/Seg-Zero/workdir/${RUN_NAME}
